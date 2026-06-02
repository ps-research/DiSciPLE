"""Step 7 simplifier diagnostic.

Part A (CPU): unit tests for dead-code elimination, weight pruning, safety,
              and round-trip execution.
Part B (GPU): full-loop integration with critic + simplifier at reduced scale,
              compared to the Step 5/6 baselines.

Usage:
    python diagnostics/step7_simplifier_check.py A     # Part A only (CPU)
    python diagnostics/step7_simplifier_check.py B     # Part B only (GPU)
    python diagnostics/step7_simplifier_check.py       # both

Part A failures are hard (exit non-zero). Part B scores are informational.
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_benchmark  # noqa: E402
from src.evolution.simplifier import simplify_program  # noqa: E402
from src.execution.runner import SAFE_BUILTINS, strip_imports  # noqa: E402
from src.primitives.namespace import create_namespace  # noqa: E402

STEP5 = {"test": 0.2767, "ood": 0.2953}   # no critic, no simplifier
STEP6 = {"test": 0.2310, "ood": 0.2804}   # critic, no simplifier


class Reporter:
    def __init__(self) -> None:
        self.ok = True

    def check(self, label, passed, hard=True, detail=""):
        status = "PASS" if passed else ("FAIL" if hard else "WARN")
        print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not passed and hard:
            self.ok = False
        return passed


def _assign_targets(src: str) -> set[str]:
    tree = ast.parse(src)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    out = set()
    for s in func.body:
        if isinstance(s, ast.Assign):
            out |= {t.id for t in s.targets if isinstance(t, ast.Name)}
    return out


def _return_names(src: str) -> list[str]:
    tree = ast.parse(src)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    ret = next(s for s in func.body if isinstance(s, ast.Return))
    return [e.id for e in ret.value.elts if isinstance(e, ast.Name)]


# --------------------------------------------------------------------------- #
def part_a(rep: Reporter) -> None:
    print("\n==================== Part A: unit tests (CPU) ====================")

    # 1. Dead-code elimination.
    prog1 = (
        "def estimator(im):\n"
        "    a = segment(im, 'highway')\n"
        "    b = segment(im, 'forest')\n"
        "    c = get_average(a)\n"
        "    d = get_average(b)\n"
        "    unused = 42\n"
        "    return (c,)\n"
    )
    s1 = simplify_program(prog1, np.array([1.0]))
    tgts = _assign_targets(s1)
    rep.check("dead-code: b/d/unused removed, a/c kept",
              tgts == {"a", "c"}, detail=f"surviving assigns={sorted(tgts)}")

    # 2. Weight-based pruning.
    prog2 = (
        "def estimator(im):\n"
        "    f1 = get_average(segment(im, 'highway'))\n"
        "    f2 = get_average(segment(im, 'forest'))\n"
        "    f3 = get_average(segment(im, 'residential building'))\n"
        "    f4 = get_average(segment(im, 'lake'))\n"
        "    return (f1, f2, f3, f4)\n"
    )
    rpt: dict = {}
    s2 = simplify_program(prog2, np.array([0.8, 0.01, 0.5, 0.02]), report=rpt)
    ret2 = _return_names(s2)
    rep.check("weight-prune: return keeps (f1, f3)", ret2 == ["f1", "f3"],
              detail=f"return={ret2} pruned_idx={rpt.get('features_pruned')}")
    rep.check("weight-prune: f2/f4 upstream code removed",
              _assign_targets(s2) == {"f1", "f3"},
              detail=f"surviving assigns={sorted(_assign_targets(s2))}")

    # 3. Safety: complex control flow must not crash; loop preserved; compiles.
    prog3 = (
        "def estimator(im):\n"
        "    concepts = ['highway', 'forest', 'lake']\n"
        "    feats = []\n"
        "    for c in concepts:\n"
        "        m = segment(im, c)\n"
        "        feats.append(get_average(m))\n"
        "    total = sum(feats)\n"
        "    return (total, *feats)\n"
    )
    try:
        s3 = simplify_program(prog3, np.array([1.0, 0.5, 0.3, 0.2]))
        compile(s3, "<s3>", "exec")
        rep.check("safety: control-flow program simplifies w/o crash, still compiles + loop kept",
                  "for " in s3, detail="for-loop preserved")
    except Exception as e:
        rep.check("safety: control-flow program simplifies w/o crash", False, detail=repr(e))

    # 4. Round-trip: simplified program execs and returns features.
    config = load_config(ROOT / "configs" / "default.yaml")
    pop = load_benchmark("population_density", config)
    image = np.load(ROOT / "data" / "population_density" / "images" / f"{pop.ids[0]}.npy")
    ns = create_namespace(0, pop, str(ROOT / "data"))
    ns["__builtins__"] = SAFE_BUILTINS
    try:
        exec(strip_imports(s2), ns)
        out = ns["estimator"](image)
        rep.check("round-trip: simplified program runs -> tuple of 2 features",
                  isinstance(out, tuple) and len(out) == 2, detail=f"n_features={len(out)}")
    except Exception as e:
        rep.check("round-trip: simplified program runs", False, detail=repr(e))


# --------------------------------------------------------------------------- #
def part_b(rep: Reporter) -> None:
    print("\n==================== Part B: integration (GPU) ====================")
    from src.evolution.loop import run_evolution  # noqa: E402
    from src.execution.evaluator import evaluate_program  # noqa: E402

    config = load_config(ROOT / "configs" / "default.yaml").model_copy(deep=True)
    config.evolution.generations = 3
    config.evolution.population_size = 10
    config.evolution.mutation_prob = 0.5
    config.evolution.use_critic = True
    config.evolution.use_simplifier = True

    ckpt = ROOT / "checkpoints" / "_step7_diag"
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoint_dir = str(ckpt)

    history: list = []
    best = run_evolution("population_density", config, history=history)

    print("\n--------- comparison (reduced scale T=3, M=10) ---------")
    test = best.result.scores.get("test", {}).get("l2_log", float("nan"))
    ood = best.result.scores.get("ood", {}).get("l2_log", float("nan"))
    print(f"  {'variant':28s} {'test L2-log':>12s} {'ood L2-log':>12s}")
    print(f"  {'Step5 (no critic/simpl)':28s} {STEP5['test']:>12.4f} {STEP5['ood']:>12.4f}")
    print(f"  {'Step6 (critic only)':28s} {STEP6['test']:>12.4f} {STEP6['ood']:>12.4f}")
    print(f"  {'Step7 (critic+simplifier)':28s} {test:>12.4f} {ood:>12.4f}")

    rep.check("test L2-log <= Step6 baseline", test <= STEP6["test"], hard=False,
              detail=f"{test:.4f} vs {STEP6['test']}")
    rep.check("ood L2-log <= Step6 baseline", ood <= STEP6["ood"], hard=False,
              detail=f"{ood:.4f} vs {STEP6['ood']}")

    # Final best program + feature count.
    tree = ast.parse(best.program_str)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "estimator")
    ret = next((s for s in func.body if isinstance(s, ast.Return)), None)
    n_elts = len(ret.value.elts) if ret and isinstance(ret.value, ast.Tuple) else "?"
    print(f"\n----- FINAL BEST PROGRAM (critic + simplifier), return elements = {n_elts} -----")
    print(best.program_str)
    for split in ("train", "val", "test", "ood"):
        if split in best.result.scores:
            s = best.result.scores[split]
            print(f"  {split:5s}: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items()))

    # Simplification log: demonstrate on the verbose gen-0 best program.
    print("\n----- simplification log (on gen-0 best, verbose) -----")
    g0 = history[0]["best_program"]
    pr0 = evaluate_program(g0, pop_dataset(config), str(ROOT / "data"), config,
                           "population_density", eval_splits=["train", "val", "test", "ood"])
    if pr0.success:
        rpt: dict = {}
        simp = simplify_program(g0, pr0.weights, report=rpt)
        print(f"  return elements: {rpt['return_elements_before']} -> {rpt['return_elements_after']}")
        print(f"  assignments removed: {rpt['assignments_removed']}")
        print(f"  weight pruning applied: {rpt['weight_pruning_applied']}, "
              f"features pruned (idx): {rpt['features_pruned']}")
        print(f"  changed: {rpt['changed']}")
    else:
        print("  (gen-0 best did not evaluate; skipping log)")


def pop_dataset(config):
    return load_benchmark("population_density", config)


# --------------------------------------------------------------------------- #
def main() -> int:
    part = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    rep = Reporter()
    if part in ("A", "ALL"):
        part_a(rep)
    if part in ("B", "ALL"):
        part_b(rep)

    print("\n" + "=" * 60)
    if rep.ok:
        print("ALL HARD CHECKS PASSED")
        return 0
    print("SOME HARD CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
