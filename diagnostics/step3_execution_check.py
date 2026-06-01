"""Step 3 execution-engine diagnostic.

Exercises the full evaluate pipeline on population_density with a hand-written
reference program (no LLM), plus timeout and bad-return robustness tests.

Prints PASS/FAIL per check and exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.execution.evaluator import evaluate_program  # noqa: E402
from src.execution.runner import execute_program  # noqa: E402

REFERENCE_PROGRAM = '''
def estimator(im):
    highway = segment(im, 'highway')
    residential = segment(im, 'residential building')
    feature1 = get_average(highway)
    feature2 = get_average(residential)
    feature3 = min_pixel_distance_to_mask(highway)
    return feature1, feature2, feature3
'''


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
    data_dir = str(ROOT / config.paths.data_dir)
    rep = Reporter()

    print("\n==================== population_density ====================")
    pop = load_benchmark("population_density", config)
    n = len(pop.ids)

    # 1 + 2. execute_program -> features (N,3), finite
    exec_res = execute_program(REFERENCE_PROGRAM, pop, data_dir, "population_density")
    rep.check(
        "execute_program success, features shape (N,3)",
        exec_res.success and exec_res.features is not None
        and exec_res.features.shape == (n, 3),
        f"success={exec_res.success} shape="
        f"{None if exec_res.features is None else exec_res.features.shape} err={exec_res.error_msg}",
    )
    rep.check(
        "feature values finite (no NaN/inf)",
        exec_res.features is not None and np.isfinite(exec_res.features).all(),
    )

    # 3. evaluate_program -> success
    res = evaluate_program(REFERENCE_PROGRAM, pop, data_dir, config, "population_density")
    rep.check("evaluate_program returns success=True",
              res.success, f"err={res.error_msg}")

    # 4. fitness finite, positive, < 1.0 (beats Mean baseline 0.6696)
    rep.check(
        "fitness (train L2-log) finite, 0 < f < 1.0",
        res.success and np.isfinite(res.fitness) and 0.0 < res.fitness < 1.0,
        f"fitness={res.fitness:.4f}",
    )

    # 5. scores for all splits
    rep.check(
        "scores present for train/val/test/ood",
        set(res.scores.keys()) == {"train", "val", "test", "ood"},
        f"splits={sorted(res.scores.keys())}",
    )
    if res.success:
        print(f"  (per-split l2_log/l1_log: "
              + ", ".join(f"{s}={res.scores[s]['l2_log']:.3f}/{res.scores[s]['l1_log']:.3f}"
                          for s in ('train', 'val', 'test', 'ood')) + ")")

    # 6. weights length 3, intercept float
    rep.check(
        "OLS weights length 3 + float intercept",
        res.weights is not None and len(res.weights) == 3 and isinstance(res.intercept, float),
        f"n_weights={None if res.weights is None else len(res.weights)} intercept={res.intercept}",
    )

    # 7. timeout test (use a short timeout to keep the diagnostic fast)
    print("\n==================== robustness ====================")
    inf_prog = "def estimator(im):\n while True: pass"
    t0 = time.time()
    inf_res = execute_program(inf_prog, pop, data_dir, "population_density", timeout_seconds=5)
    elapsed = time.time() - t0
    rep.check(
        "infinite-loop program fails via timeout (<10s here)",
        (not inf_res.success) and elapsed < 10,
        f"success={inf_res.success} elapsed={elapsed:.1f}s err={inf_res.error_msg}",
    )

    # 8. bad-return test
    bad_prog = "def estimator(im):\n return 42"
    bad_res = execute_program(bad_prog, pop, data_dir, "population_density")
    rep.check(
        "non-tuple return fails gracefully",
        not bad_res.success,
        f"success={bad_res.success} err={bad_res.error_msg}",
    )

    print("\n" + "=" * 52)
    if rep.ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
