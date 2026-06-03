"""Run a single full-scale DiSciPLE experiment (one benchmark x one variant).

Example:
    python scripts/run_experiment.py \
        --benchmark population_density --variant full --gpu 1 \
        --output_dir results/population_density_full \
        --generations 15 --population_size 100 --seed 42

Saves to output_dir/: best_program.py, scores.json, convergence.csv,
config_used.yaml, checkpoints/ (resume-compatible).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ablation variants -> (use_critic, use_simplifier).
VARIANTS = {
    "full": (True, True),
    "critic_only": (True, False),
    "simplifier_only": (False, True),
    "base": (False, False),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=["population_density", "poverty", "agb"])
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--population_size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/default.yaml",
                    help="config file (default.yaml=llama, qwen.yaml=Qwen)")
    args = ap.parse_args()

    # Pin the GPU BEFORE importing torch. Respect a launcher-set
    # CUDA_VISIBLE_DEVICES (which already remaps the physical GPU to index 0).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Reduce CUDA memory fragmentation (recommended by PyTorch's own OOM message).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import yaml  # noqa: E402

    from src.config import load_config  # noqa: E402
    from src.evolution.loop import run_evolution  # noqa: E402

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path).model_copy(deep=True)
    cfg.evolution.generations = args.generations
    cfg.evolution.population_size = args.population_size
    cfg.seed = args.seed
    cfg.evolution.use_critic, cfg.evolution.use_simplifier = VARIANTS[args.variant]
    cfg.paths.data_dir = str(root / "data")            # absolute -> cwd-independent

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoint_dir = str(outdir / "checkpoints")
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== {args.benchmark} / {args.variant} "
          f"(critic={cfg.evolution.use_critic}, simplifier={cfg.evolution.use_simplifier}) "
          f"T={args.generations} M={args.population_size} seed={args.seed} "
          f"gen_batch_size={cfg.llm.gen_batch_size} ===", flush=True)

    history: list = []
    best = run_evolution(args.benchmark, cfg, history=history)

    # ---- save artifacts ---------------------------------------------------- #
    (outdir / "best_program.py").write_text(best.program_str.rstrip() + "\n")

    with open(outdir / "scores.json", "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "variant": args.variant,
            "use_critic": cfg.evolution.use_critic,
            "use_simplifier": cfg.evolution.use_simplifier,
            "seed": args.seed,
            "r2_train": best.r2_score,
            "fitness": best.result.fitness,
            "n_features": best.result.n_features,
            "simplification": best.simplification,
            "scores": best.result.scores,
        }, f, indent=2)

    with open(outdir / "convergence.csv", "w") as f:
        f.write("generation,best_r2,mean_r2,valid_count,best_fitness\n")
        for h in history:
            f.write(f"{h['generation']},{h['best_r2']},{h['mean_r2']},"
                    f"{h['valid_count']},{h['best_fitness']}\n")

    with open(outdir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)

    # ---- summary ----------------------------------------------------------- #
    print("\n===== SUMMARY =====", flush=True)
    print(f"{args.benchmark} / {args.variant}: R2(train)={best.r2_score:.4f} "
          f"fitness={best.result.fitness:.4f} n_features={best.result.n_features} "
          f"simplification={best.simplification}")
    for split in ("train", "val", "test", "ood"):
        if split in best.result.scores:
            s = best.result.scores[split]
            print(f"  {split:5s}: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items()))
    print(f"saved -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
