"""Step 4 LLM-generator diagnostic. REQUIRES A GPU.

Loads the 4-bit model via Unsloth, generates programs from the objective prompt,
and verifies they are valid, executable (under the REAL restricted execution
semantics), diverse, and that the pipeline is robust to invalid generations.

Realistic expectation: a single generation is often invalid (the paper reports
~44% "Program Error Percent" for Qwen), so we sample several and require that
the generator is well-formed, the execution path never crashes, and at least
one sample is fully valid end-to-end.

Prints PASS/FAIL per check; prints generated programs; exits non-zero on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.execution.runner import SAFE_BUILTINS, strip_imports  # noqa: E402
from src.llm import LLMGenerator, OBJECTIVE_PROMPT, get_task_description  # noqa: E402
from src.primitives import get_api_spec  # noqa: E402
from src.primitives.namespace import create_namespace  # noqa: E402

N_SAMPLES = 5


class Reporter:
    def __init__(self) -> None:
        self.ok = True

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        line = f"  [{status}] {label}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not passed:
            self.ok = False
        return passed


def wellformed(code: str) -> tuple[bool, str]:
    """Generator-quality check: defines estimator and compiles."""
    if "def estimator" not in code:
        return False, "no 'def estimator'"
    try:
        compile(strip_imports(code), "<gen>", "exec")
    except SyntaxError as e:
        return False, f"compile error: {e}"
    return True, "ok"


def run_restricted(code: str, dataset, data_dir: str, image):
    """Run code on ONE observation using the real restricted semantics.

    Returns (status, detail): status in {"valid", "rejected", "crashed"}.
    "rejected" = the engine handled a bad program gracefully (expected).
    "crashed"  = the engine itself raised (should never happen).
    """
    try:
        ns = create_namespace(0, dataset, data_dir)
        ns["__builtins__"] = SAFE_BUILTINS
        src = strip_imports(code)
        try:
            exec(src, ns)
            est = ns.get("estimator")
            if not callable(est):
                return "rejected", "no callable estimator"
            ret = est(image)
            if not isinstance(ret, (tuple, list)) or len(ret) == 0:
                return "rejected", f"returned {type(ret).__name__}"
            return "valid", f"{len(ret)} features"
        except Exception as e:  # bad program -> graceful rejection
            return "rejected", f"{type(e).__name__}: {e}"
    except Exception as e:  # engine-level failure -> real problem
        return "crashed", f"{type(e).__name__}: {e}"


def main() -> int:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_dir = str(ROOT / config.paths.data_dir)
    rep = Reporter()

    # 1. Load model, verify 4-bit footprint.
    print("Loading model (Unsloth, 4-bit)...")
    gen = LLMGenerator(config)
    gen.load()
    import torch
    vram_gb = torch.cuda.memory_allocated(0) / 1e9
    rep.check("model loaded in 4-bit (VRAM < 10 GB)", 2.0 < vram_gb < 10.0,
              f"allocated={vram_gb:.2f} GB")

    # 2. Objective prompt for population_density.
    pop = load_benchmark("population_density", config)
    prompt = OBJECTIVE_PROMPT.format(
        descr=get_task_description("population_density"),
        api_spec=get_api_spec("population_density"),
    )
    image = np.load(Path(data_dir) / "population_density" / "images" / f"{pop.ids[0]}.npy")

    # Sample N programs.
    print(f"Generating {N_SAMPLES} programs...")
    codes = [LLMGenerator.extract_code(gen.generate(prompt)) for _ in range(N_SAMPLES)]

    # 3. Generator quality: at least one well-formed (def estimator + compiles).
    # Not all samples are expected to be valid -- the paper reports ~44% buggy.
    wf = [wellformed(c) for c in codes]
    n_wf = sum(ok for ok, _ in wf)
    rep.check("at least one sample well-formed (def estimator + compile)",
              n_wf >= 1, f"wellformed={n_wf}/{N_SAMPLES}")

    # 4. Robustness + at-least-one-valid.
    statuses = [run_restricted(c, pop, data_dir, image) for c in codes]
    kinds = [s for s, _ in statuses]
    n_valid = kinds.count("valid")
    n_crashed = kinds.count("crashed")
    rep.check("execution path never crashes the engine", n_crashed == 0,
              f"crashed={n_crashed} details={[d for s,d in statuses if s=='crashed']}")
    rep.check("at least one sample is fully valid end-to-end", n_valid >= 1,
              f"valid={n_valid}/{N_SAMPLES} (paper Qwen error ~44%)")

    # 5. Diversity.
    rep.check("samples are diverse (>=2 distinct programs)",
              len({c.strip() for c in codes}) >= 2,
              f"distinct={len({c.strip() for c in codes})}/{N_SAMPLES}")

    # 6. Batched generation.
    batch = gen.generate_batch([prompt, prompt])
    batch_codes = [LLMGenerator.extract_code(b) for b in batch]
    rep.check("generate_batch returns 2 responses, >=1 well-formed",
              len(batch) == 2 and any(wellformed(c)[0] for c in batch_codes),
              f"wellformed={[wellformed(c)[0] for c in batch_codes]}")

    # Report sampled programs + outcomes.
    print(f"\n  outcomes: {[s for s,_ in statuses]}  (valid={n_valid}/{N_SAMPLES})")
    for i, (c, (s, d)) in enumerate(zip(codes, statuses)):
        print(f"\n----- sample {i} [{s}: {d}] -----\n{c}")

    print("\n" + "=" * 52)
    if rep.ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
