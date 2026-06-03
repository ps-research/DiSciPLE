"""Pre-flight validation: GO/NO-GO before committing full-scale compute to a model.

Model-agnostic — reads the model from the given config. Runs 6 sequential checks
(load+generate, code quality, flat returns, feature-count distribution, simplifier
effectiveness, mini-evolution) and prints a GO/NO-GO verdict.

Usage:
    python diagnostics/preflight.py                          # configs/default.yaml (llama)
    python diagnostics/preflight.py --config configs/qwen.yaml

Exit 0 on GO, non-zero on NO-GO. Set CUDA_VISIBLE_DEVICES before running.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.evolution.loop import run_evolution  # noqa: E402
from src.evolution.simplifier import count_return_features, simplify_program  # noqa: E402
from src.execution.evaluator import evaluate_program  # noqa: E402
from src.execution.runner import SAFE_BUILTINS, has_flat_tuple_return, strip_imports  # noqa: E402
from src.llm import LLMGenerator, build_objective_prompt  # noqa: E402
from src.primitives.namespace import create_namespace  # noqa: E402

N = 10  # programs to generate for checks 1-5
DATA = str(ROOT / "data")
BENCH = "population_density"


class Preflight:
    def __init__(self):
        self.lines = []      # (label, status, detail)
        self.fail = False

    def record(self, label, status, detail=""):
        self.lines.append((label, status, detail))
        if status == "FAIL":
            self.fail = True
        print(f"  -> {status}: {label}" + (f" | {detail}" if detail else ""))


def _validate(code, dataset, image):
    """Return (ok, failure_mode). ok=True iff def estimator + compiles + execs + returns tuple."""
    if "def estimator" not in code:
        return False, "no_estimator"
    try:
        compile(strip_imports(code), "<gen>", "exec")
    except SyntaxError:
        return False, "syntax_error"
    ns = create_namespace(0, dataset, DATA)
    ns["__builtins__"] = SAFE_BUILTINS
    try:
        exec(strip_imports(code), ns)
        est = ns.get("estimator")
        if not callable(est):
            return False, "no_estimator"
        ret = est(image)
        if not isinstance(ret, (tuple, list)) or len(ret) == 0:
            return False, "non_tuple_return"
        return True, "ok"
    except Exception:
        return False, "runtime_error"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    config = load_config(cfg_path)
    config.paths.data_dir = DATA

    pf = Preflight()
    print(f"\nPreflight — model: {config.llm.model}  (config: {cfg_path.name})\n")

    # Shared data: population dataset + one image.
    pop = load_benchmark(BENCH, config)
    image = np.load(ROOT / "data" / BENCH / "images" / f"{pop.ids[0]}.npy")
    prompt = build_objective_prompt(BENCH)

    # ---- Check 1: model loads + generates --------------------------------- #
    print("[Check 1] model loads + generates 10 programs")
    gen = LLMGenerator(config)
    try:
        gen.load()
    except Exception as e:
        pf.record("Check 1 (model loads + generates)", "FAIL", f"load error: {type(e).__name__}: {e}")
        return _verdict(pf, config)
    t0 = time.time()
    try:
        responses = gen.generate_batch([prompt] * N)
    except Exception as e:
        pf.record("Check 1 (model loads + generates)", "FAIL", f"gen error: {type(e).__name__}: {e}")
        return _verdict(pf, config)
    gtime = time.time() - t0
    nonempty = sum(1 for r in responses if r.strip())
    pf.record("Check 1 (model loads + generates)",
              "PASS" if (nonempty >= 8 and gtime < 60) else "FAIL",
              f"{nonempty}/{N} non-empty, gen {gtime:.0f}s")

    codes = [LLMGenerator.extract_code(r) for r in responses]

    # ---- Check 2: code quality -------------------------------------------- #
    print("[Check 2] code quality (extractable, compilable, executable)")
    modes = {}
    valid_codes = []
    for c in codes:
        ok, mode = _validate(c, pop, image)
        if ok:
            valid_codes.append(c)
        else:
            modes[mode] = modes.get(mode, 0) + 1
    nv = len(valid_codes)
    pf.record("Check 2 (code quality, >=5/10 valid)",
              "PASS" if nv >= 5 else "FAIL",
              f"{nv}/{N} valid" + (f"; failures={modes}" if modes else ""))

    # ---- Check 3: flat-return compliance ---------------------------------- #
    print("[Check 3] flat-return compliance")
    flat = [c for c in valid_codes if has_flat_tuple_return(c)]
    frac = (len(flat) / nv) if nv else 0.0
    st3 = "PASS" if frac == 1.0 else ("WARN" if frac >= 0.8 else "FAIL")
    pf.record("Check 3 (flat returns)", st3, f"{len(flat)}/{nv} flat ({frac*100:.0f}%)")

    # ---- Check 4: feature-count distribution ------------------------------ #
    print("[Check 4] feature-count distribution")
    feat_counts, results = [], []
    for c in valid_codes:
        r = evaluate_program(c, pop, DATA, config, BENCH, eval_splits=["train"])
        if r.success:
            feat_counts.append(r.n_features)
            results.append((c, r))
    if feat_counts:
        med, mx = int(np.median(feat_counts)), int(np.max(feat_counts))
        le10 = np.mean([f <= 10 for f in feat_counts])
        st4 = "PASS" if (med <= 12 and mx <= 25) else ("WARN" if med <= 15 else "FAIL")
        pf.record("Check 4 (feature count, median<=12)", st4,
                  f"median={med} max={mx} frac<=10feat={le10:.0%} dist={sorted(feat_counts)}")
    else:
        pf.record("Check 4 (feature count, median<=12)", "FAIL", "no evaluable programs")

    # ---- Check 5: simplifier effectiveness -------------------------------- #
    print("[Check 5] simplifier effectiveness")
    befores, afters = [], []
    for c, r in results:
        b = count_return_features(c)
        if b is None:
            continue
        s = simplify_program(c, r.weights)
        a = count_return_features(s)
        if a is None:
            continue
        befores.append(b)
        afters.append(a)
    if befores:
        avg_b, avg_a = np.mean(befores), np.mean(afters)
        ratio = 1 - (avg_a / avg_b) if avg_b else 0.0
        st5 = "PASS" if (avg_a <= 8 and ratio >= 0.15) else ("WARN" if avg_a <= 12 else "FAIL")
        pf.record("Check 5 (simplifier, avg after<=8)", st5,
                  f"avg {avg_b:.1f}->{avg_a:.1f} feats, pruning {ratio*100:.0f}%")
    else:
        pf.record("Check 5 (simplifier, avg after<=8)", "FAIL", "no flat programs to simplify")

    # ---- Check 6: mini evolution ------------------------------------------ #
    print("[Check 6] mini evolution (T=2, M=10, full variant)")
    c6 = config.model_copy(deep=True)
    c6.evolution.generations = 2
    c6.evolution.population_size = 10
    c6.evolution.use_critic = True
    c6.evolution.use_simplifier = True
    ck = ROOT / "checkpoints" / "_preflight"
    if ck.exists():
        shutil.rmtree(ck)
    ck.mkdir(parents=True, exist_ok=True)
    c6.paths.checkpoint_dir = str(ck)
    hist = []
    best = run_evolution(BENCH, c6, history=hist, generator=gen)
    r0 = hist[0]["best_r2"] if hist else float("-inf")
    r2 = hist[-1]["best_r2"] if hist else float("-inf")
    nfeat = count_return_features(best.program_str)
    test = best.result.scores.get("test", {}).get("l2_log", float("nan"))
    ood = best.result.scores.get("ood", {}).get("l2_log", float("nan"))
    ratio = ood / test if test and np.isfinite(test) and test > 0 else float("inf")
    improved = r2 >= r0 - 1e-9
    compact = nfeat is not None and nfeat <= 10
    no_blowup = ratio < 3.0
    st6 = "PASS" if (improved and compact and no_blowup) else "FAIL"
    pf.record("Check 6 (mini evolution)", st6,
              f"R2 {r0:.3f}->{r2:.3f}, {nfeat} feats, test={test:.3f} ood={ood:.3f} ood/test={ratio:.2f}")
    if ck.exists():
        shutil.rmtree(ck)
    print("\n--- mini-evolution best program ---\n" + best.program_str)

    return _verdict(pf, config)


def _verdict(pf, config) -> int:
    print("\n" + "=" * 51)
    print(f"  PREFLIGHT REPORT — model: {config.llm.model}")
    print("=" * 51)
    for label, status, detail in pf.lines:
        print(f"  {label:<40s} {status}" + (f"  ({detail})" if detail else ""))
    print("=" * 51)
    if pf.fail:
        print("  VERDICT: NO-GO  — a check FAILED (see above).")
        print("  -> If feature-count/simplifier failed: model generates programs too")
        print("     heavy for the simplifier. Try a simpler/different model or stronger steering.")
        print("=" * 51)
        return 1
    print("  VERDICT: GO  — model is compatible, proceed to full-scale runs.")
    print("=" * 51)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
