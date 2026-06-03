"""Step 8 pre-launch smoke check (1 GPU). Run BEFORE launch_full_scale.sh.

Verifies:
  1. config loads with gen_batch_size.
  2. train-only eval gives identical train fitness as full eval (reference program).
  3. run_experiment.py runs 1 generation (M=5) without crash and writes all outputs.
  4. checkpoint + output files exist.

Prints PASS/FAIL per check; exits non-zero on failure.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.execution.evaluator import evaluate_program  # noqa: E402

REF = ("def estimator(im):\n"
       "    a = get_average(segment(im, 'highway'))\n"
       "    b = get_average(segment(im, 'forest'))\n"
       "    c = get_average(segment(im, 'residential building'))\n"
       "    return (a, b, c)\n")


class Reporter:
    def __init__(self):
        self.ok = True

    def check(self, label, passed, detail=""):
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not passed:
            self.ok = False
        return passed


def main() -> int:
    rep = Reporter()

    # 1. config + gen_batch_size
    cfg = load_config(ROOT / "configs" / "default.yaml")
    rep.check("config loads with gen_batch_size", isinstance(cfg.llm.gen_batch_size, int),
              f"gen_batch_size={cfg.llm.gen_batch_size}")

    # 2. train-only vs full eval -> identical train fitness
    pop = load_benchmark("population_density", cfg)
    r_train = evaluate_program(REF, pop, str(ROOT / "data"), cfg, "population_density",
                               eval_splits=["train"])
    r_full = evaluate_program(REF, pop, str(ROOT / "data"), cfg, "population_density",
                              eval_splits=None)
    rep.check("train-only fitness == full-eval train fitness",
              r_train.success and r_full.success and np.isclose(r_train.fitness, r_full.fitness),
              f"train-only={r_train.fitness:.6f} full={r_full.fitness:.6f}")

    # 3 + 4. run_experiment.py for 1 gen, M=5, outputs exist
    tmp = ROOT / "results" / "_smoke_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    print("  running run_experiment.py (1 gen, M=5, GPU) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / "run_experiment.py"),
         "--benchmark", "population_density", "--variant", "full", "--gpu", "0",
         "--output_dir", str(tmp), "--generations", "1", "--population_size", "5", "--seed", "42"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    rep.check("run_experiment.py (1 gen, M=5) exits 0", proc.returncode == 0,
              f"returncode={proc.returncode}")

    expected = ["best_program.py", "scores.json", "convergence.csv", "config_used.yaml",
                "checkpoints/evolution_state.pkl"]
    missing = [f for f in expected if not (tmp / f).exists()]
    rep.check("all output artifacts written", not missing, f"missing={missing}")

    if (tmp / "scores.json").exists():
        sc = json.load(open(tmp / "scores.json"))
        has_all = set(sc.get("scores", {})) == {"train", "val", "test", "ood"}
        rep.check("scores.json has all 4 splits (final full eval)", has_all,
                  f"splits={sorted(sc.get('scores', {}))}")

    if tmp.exists():
        shutil.rmtree(tmp)

    print("\n" + "=" * 52)
    print("ALL CHECKS PASSED" if rep.ok else "SOME CHECKS FAILED")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
