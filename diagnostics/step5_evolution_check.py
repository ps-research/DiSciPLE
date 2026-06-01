"""Step 5 evolutionary-loop diagnostic. REQUIRES A GPU.

Runs the base evolutionary loop (no critic/simplifier) at REDUCED scale
(T=3, M=10, rho_m=0.5) on population_density, then validates initialization,
per-generation improvement, diversity, checkpoint round-trip, and the final
best program.

Prints PASS/FAIL per check + the best program and scores. Exits non-zero on failure.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evolution.loop import (  # noqa: E402
    load_evolution_state,
    run_evolution,
    save_evolution_state,
)


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


def main() -> int:
    config = load_config(ROOT / "configs" / "default.yaml")

    # Reduced hyperparams for a fast diagnostic (do NOT touch default.yaml).
    config = config.model_copy(deep=True)
    config.evolution.generations = 3
    config.evolution.population_size = 10
    config.evolution.mutation_prob = 0.5

    # Fresh, isolated checkpoint dir (avoid resuming from a stale checkpoint).
    ckpt_dir = ROOT / "checkpoints" / "_step5_diag"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoint_dir = str(ckpt_dir)

    rep = Reporter()
    history: list = []
    best = run_evolution("population_density", config, history=history)

    print("\n==================== checks ====================")

    # 1. Initialization: >=3 of 10 valid.
    init_valid = history[0]["valid_count"]
    rep.check("init: >=3 of 10 programs valid", init_valid >= 3,
              f"valid={init_valid}/{config.evolution.population_size}")

    # 2. Generation progress: best R^2 non-decreasing gen 0 -> gen 3.
    r2_by_gen = [(h["generation"], h["best_r2"]) for h in history]
    print("  best R^2 per generation: " +
          ", ".join(f"g{g}={r:.4f}" for g, r in r2_by_gen))
    rep.check("best R^2 improves or holds (gen0 -> gen3)",
              history[-1]["best_r2"] >= history[0]["best_r2"] - 1e-9,
              f"gen0={history[0]['best_r2']:.4f} gen{history[-1]['generation']}="
              f"{history[-1]['best_r2']:.4f}")

    # 3. Diversity: best program changed across the search.
    rep.check("best program at last gen differs from gen 0",
              history[0]["best_program"].strip() != history[-1]["best_program"].strip())

    # 4. Checkpoint round-trip.
    loaded = load_evolution_state(config)
    rep.check("checkpoint loads", loaded is not None)
    if loaded is not None:
        bank1, best1, gen1 = loaded
        save_evolution_state(bank1, best1, gen1, config)
        bank2, best2, gen2 = load_evolution_state(config)
        same = (
            gen1 == gen2
            and len(bank1.entries) == len(bank2.entries)
            and all(e1.program_str == e2.program_str
                    and e1.result.fitness == e2.result.fitness
                    for e1, e2 in zip(bank1.entries, bank2.entries))
        )
        rep.check("round-tripped bank has identical entries", same,
                  f"n1={len(bank1.entries)} n2={len(bank2.entries)} gen={gen1}->{gen2}")

    # 5. Final best program.
    rep.check("final best: fitness finite and R^2 > 0 (beats mean)",
              np.isfinite(best.result.fitness) and best.r2_score > 0,
              f"fitness={best.result.fitness:.4f} r2={best.r2_score:.4f}")
    rep.check("final best has all 4 splits scored",
              set(best.result.scores.keys()) == {"train", "val", "test", "ood"},
              f"splits={sorted(best.result.scores.keys())}")

    print("\n----- FINAL BEST PROGRAM -----")
    print(best.program_str)
    print(f"\nR2(train) = {best.r2_score:.4f}")
    print(f"fitness (train L2-log) = {best.result.fitness:.4f}")
    for split in ("train", "val", "test", "ood"):
        if split in best.result.scores:
            s = best.result.scores[split]
            print(f"  {split:5s}: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items()))

    print("\n" + "=" * 52)
    if rep.ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
