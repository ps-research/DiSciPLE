"""Step 7 verification (per benchmark) of the flat-return steering + simplifier.

Usage: python diagnostics/step7_verify.py <benchmark>

Runs reduced-scale (T=3, M=10) evolution with critic + simplifier enabled and
reports whether the best program is FLAT (no starred/comprehension return) and
ACTUALLY SIMPLIFIED (post-simplifier feature count < pre), plus per-split scores.
For population_density it also compares to the Step 5/6 baselines and prints a
GATE line. Designed to be run concurrently across GPUs (one benchmark per GPU).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evolution.loop import run_evolution  # noqa: E402
from src.evolution.simplifier import count_return_features  # noqa: E402

BASELINES = {
    "population_density": {
        "step5": {"test": 0.2767, "ood": 0.2953},
        "step6": {"test": 0.2310, "ood": 0.2804},
    },
}
# Which reported metric to print per benchmark.
METRIC = {"population_density": "l2_log", "poverty": "rmse", "agb": "rmse"}


def main() -> int:
    bench = sys.argv[1]
    metric = METRIC.get(bench, "rmse")

    cfg = load_config(ROOT / "configs" / "default.yaml").model_copy(deep=True)
    cfg.evolution.generations = 3
    cfg.evolution.population_size = 10
    cfg.evolution.mutation_prob = 0.5
    cfg.evolution.use_critic = True
    cfg.evolution.use_simplifier = True
    cfg.paths.data_dir = str(ROOT / "data")          # absolute -> cwd-independent

    # Keep any existing checkpoint so a relaunch RESUMES (the node gets
    # reassigned and kills jobs; per-generation checkpoints let us continue).
    ck = ROOT / "checkpoints" / f"_step7v_{bench}"
    ck.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoint_dir = str(ck)

    hist: list = []
    best = run_evolution(bench, cfg, history=hist)

    nfeat = count_return_features(best.program_str)
    flat = nfeat is not None
    simpl = best.simplification or {}
    pre, post = simpl.get("pre_features"), simpl.get("post_features")
    simplified = (pre is not None and post is not None and post < pre)

    print(f"\n######## RESULT [{bench}] ########")
    print("R2 per gen: " + ", ".join(f"g{h['generation']}={h['best_r2']:.4f}" for h in hist))
    print(f"best flat-return: {flat}  (return elements = {nfeat})")
    print(f"best simplification: pre={pre} -> post={post}  (reduced={simplified})")
    print("scores: " + ", ".join(
        f"{s}={best.result.scores.get(s, {}).get(metric, float('nan')):.4f}"
        for s in ("train", "val", "test", "ood")))
    print("----- BEST PROGRAM -----")
    print(best.program_str)

    if bench == "population_density":
        b = BASELINES[bench]
        test = best.result.scores.get("test", {}).get("l2_log", float("nan"))
        ood = best.result.scores.get("ood", {}).get("l2_log", float("nan"))
        competitive = test <= b["step5"]["test"]
        print(f"baselines: Step5 test={b['step5']['test']} ood={b['step5']['ood']} | "
              f"Step6 test={b['step6']['test']} ood={b['step6']['ood']}")
        print(f"GATE: flat={flat} simplified={simplified} competitive(test<=Step5)={competitive} "
              f"=> {'PASS' if (flat and simplified and competitive) else 'CHECK'}")
    print(f"######## END [{bench}] ########")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
