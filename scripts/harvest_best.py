"""Harvest a converged best program from an evolution checkpoint and write a
scores.json identical in format to run_experiment.py, without finishing the run.

Used when a no-simplifier run has converged (best fitness flat for several gens)
but is crawling on program-bloat CPU eval. Re-evaluates the checkpoint's best
program on ALL splits (the same final re-eval run_evolution does at T) and writes
scores.json with explicit _harvested provenance fields.

Usage:
    python scripts/harvest_best.py --benchmark population_density \
        --variant critic_only --output_dir results/population_density_critic_only
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

VARIANTS = {
    "full": (True, True),
    "critic_only": (True, False),
    "simplifier_only": (False, True),
    "base": (False, False),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; no GPU/LLM needed

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import numpy as np  # noqa: E402

    from src.config import load_config  # noqa: E402
    from src.data.loader import load_benchmark  # noqa: E402
    from src.execution.evaluator import _transform_target, evaluate_program  # noqa: E402
    from src.execution.runner import preload_images  # noqa: E402

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path).model_copy(deep=True)
    cfg.seed = args.seed
    cfg.evolution.use_critic, cfg.evolution.use_simplifier = VARIANTS[args.variant]
    cfg.sample_frac = None
    cfg.paths.data_dir = str(root / "data")

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = root / outdir
    ckpt = outdir / "checkpoints" / "evolution_state.pkl"
    if not ckpt.exists():
        print(f"ERROR: no checkpoint at {ckpt}", file=sys.stderr)
        return 1

    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    best = state["best"]
    gen = state["generation"]
    print(f"loaded checkpoint: gen={gen} best.r2(train-only)={best.r2_score:.4f} "
          f"n_feat={best.result.n_features}", flush=True)

    benchmark = args.benchmark
    data_dir = cfg.paths.data_dir
    dataset = load_benchmark(benchmark, cfg)
    y = _transform_target(benchmark, dataset.targets)
    train_var = float(np.var(y[dataset.splits == "train"]))
    image_cache = (preload_images(dataset, data_dir)
                   if benchmark == "population_density" else None)

    print("re-evaluating best on all splits (train,val,test,ood)...", flush=True)
    full = evaluate_program(
        best.program_str, dataset, data_dir, cfg, benchmark,
        image_cache=image_cache, eval_splits=["train", "val", "test", "ood"],
    )
    if not full.success:
        print(f"ERROR: re-eval failed: {full.error_msg}", file=sys.stderr)
        return 2

    def _r2(fitness, var):
        if not np.isfinite(fitness) or var <= 0:
            return float("-inf")
        return 1.0 - fitness / var

    r2_train = _r2(full.fitness, train_var)

    out = {
        "benchmark": benchmark,
        "variant": args.variant,
        "use_critic": cfg.evolution.use_critic,
        "use_simplifier": cfg.evolution.use_simplifier,
        "seed": args.seed,
        "r2_train": r2_train,
        "fitness": full.fitness,
        "n_features": full.n_features,
        "simplification": best.simplification,
        "scores": full.scores,
        "_harvested": True,
        "_harvested_generation": gen,
        "_harvest_note": (f"Converged best harvested from gen-{gen} checkpoint "
                          f"(run killed mid-T=15; best fitness flat since gen 7 due to "
                          f"no-simplifier program bloat slowing CPU eval). Re-evaluated "
                          f"on all splits exactly as run_evolution's final step."),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "scores.json", "w") as f:
        json.dump(out, f, indent=2)

    print("WROTE", outdir / "scores.json", flush=True)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
