"""Step 6 critic diagnostic. REQUIRES A GPU.

Runs the evolutionary loop with the critic ENABLED at reduced scale (T=3, M=10)
and validates stratified analysis, the critic LLM step, and the resulting scores
against the Step 5 (no-critic) baseline.

Prints PASS/FAIL per check. Exits non-zero only on HARD failures (stratification
or critic crash); score comparisons are informational at reduced scale.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.evolution.critic import apply_critic, stratified_analysis  # noqa: E402
from src.evolution.loop import run_evolution  # noqa: E402
from src.execution.evaluator import evaluate_program  # noqa: E402
from src.llm import LLMGenerator  # noqa: E402

# Step 5 (no-critic) reduced-scale reference numbers, for informational comparison.
STEP5_BASELINE_R2 = 0.7028
STEP5_BASELINE_OOD_L2LOG = 0.2953


class Reporter:
    def __init__(self) -> None:
        self.ok = True

    def check(self, label: str, passed: bool, hard: bool = True, detail: str = "") -> bool:
        status = "PASS" if passed else ("FAIL" if hard else "WARN")
        line = f"  [{status}] {label}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not passed and hard:
            self.ok = False
        return passed


def main() -> int:
    config = load_config(ROOT / "configs" / "default.yaml")
    config = config.model_copy(deep=True)
    config.evolution.generations = 3
    config.evolution.population_size = 10
    config.evolution.mutation_prob = 0.5
    config.evolution.use_critic = True

    ckpt_dir = ROOT / "checkpoints" / "_step6_diag"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoint_dir = str(ckpt_dir)

    rep = Reporter()
    data_dir = config.paths.data_dir
    dataset = load_benchmark("population_density", config)

    # Reuse one loaded generator for the run and the standalone critic checks.
    gen = LLMGenerator(config)
    gen.load()

    history: list = []
    best = run_evolution("population_density", config, history=history, generator=gen)

    print("\n==================== checks ====================")

    # 1. Stratification on the gen-0 best program -> non-empty worst categories.
    gen0_prog = history[0]["best_program"]
    pr0 = evaluate_program(gen0_prog, dataset, data_dir, config, "population_density",
                           eval_splits=["train", "val", "test", "ood"])
    worst = stratified_analysis(pr0, dataset, pr0.predictions) if pr0.success else []
    rep.check("stratified_analysis returns non-empty worst categories",
              len(worst) > 0, detail=f"worst={worst}")

    # 2. Critic produces a different program.
    if worst:
        improved = apply_critic(gen0_prog, worst, gen, "population_density", config)
        rep.check("critic produces a program different from its input",
                  improved.strip() != gen0_prog.strip(),
                  detail=f"len_in={len(gen0_prog)} len_out={len(improved)}")
    else:
        rep.check("critic produces a program different from its input", False,
                  detail="no worst categories to critique")

    # 3. R^2 vs Step 5 no-critic baseline (informational).
    r2_by_gen = [(h["generation"], h["best_r2"]) for h in history]
    print("  best R^2 per generation: " + ", ".join(f"g{g}={r:.4f}" for g, r in r2_by_gen))
    rep.check(f"gen-3 best R^2 >= Step5 baseline ({STEP5_BASELINE_R2})",
              history[-1]["best_r2"] >= STEP5_BASELINE_R2, hard=False,
              detail=f"critic gen3={history[-1]['best_r2']:.4f} vs baseline={STEP5_BASELINE_R2}")

    # 4. OOD score (informational; critic is expected to help OOD most).
    ood = best.result.scores.get("ood", {}).get("l2_log", float("nan"))
    rep.check(f"OOD L2-log <= Step5 baseline ({STEP5_BASELINE_OOD_L2LOG})",
              ood <= STEP5_BASELINE_OOD_L2LOG, hard=False,
              detail=f"critic ood={ood:.4f} vs baseline={STEP5_BASELINE_OOD_L2LOG}")

    # 5. Final best program + scores.
    print("\n----- FINAL BEST PROGRAM (with critic) -----")
    print(best.program_str)
    print(f"\nR2(train) = {best.r2_score:.4f}  fitness = {best.result.fitness:.4f}")
    for split in ("train", "val", "test", "ood"):
        if split in best.result.scores:
            s = best.result.scores[split]
            print(f"  {split:5s}: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items()))

    print("\n" + "=" * 52)
    if rep.ok:
        print("ALL HARD CHECKS PASSED")
        return 0
    print("SOME HARD CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
