"""Program simplifier (§4.5): AST dead-code elimination + weight-based pruning.

Successive crossover/mutation produce verbose programs with dead code and
redundant features. This analytical (no-LLM, no-GPU) simplifier:

1. **Dead-code elimination** -- on the estimator's AST, drop assignments whose
   targets are not reachable from the return statement (recursively, to a fixed
   point). Constants/args are roots; the return and unused vars are leaves; any
   leaf that isn't the return is removable.
2. **Weight-based feature pruning** -- drop returned features whose OLS weight is
   below ``weight_threshold`` x the largest weight (paper: 5%), then re-run
   dead-code elimination on the newly-orphaned assignments.

Robustness: anything it can't analyze safely is left untouched, and any failure
returns the original program unchanged.
"""
from __future__ import annotations

import ast

import numpy as np


def _loaded_names(node: ast.AST) -> set[str]:
    """All variable names *read* (ctx=Load) anywhere within ``node``."""
    return {
        n.id for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _removable_targets(stmt: ast.stmt):
    """Target name(s) if ``stmt`` is a simple removable assignment, else None.

    Only plain ``x = ...`` / ``x: T = ...`` with Name targets are considered
    removable. Everything else (for/if/while/expr/aug-assign/tuple-unpack) is
    treated as non-removable (kept) -- conservative and safe.
    """
    if isinstance(stmt, ast.Assign):
        ids = []
        for t in stmt.targets:
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                ids.append(t.id)
            else:
                return None
        return ids
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
        return [stmt.target.id]
    return None


def _eliminate_dead(body: list[ast.stmt]) -> tuple[list[ast.stmt], int]:
    """Remove assignments not reachable from the return / non-removable stmts.

    Returns (new_body, n_removed). Iterates to a fixed point.
    """
    stmts = list(body)
    removed_total = 0
    while True:
        # Seed `needed` with names used by every non-removable statement
        # (this includes the return statement and any loops/conditionals).
        needed: set[str] = set()
        for s in stmts:
            if _removable_targets(s) is None:
                needed |= _loaded_names(s)

        # Propagate liveness backward through removable assignments.
        changed = True
        while changed:
            changed = False
            for s in stmts:
                tg = _removable_targets(s)
                if tg is not None and any(t in needed for t in tg):
                    uses = _loaded_names(s.value)
                    if not uses <= needed:
                        needed |= uses
                        changed = True

        # Drop removable assignments whose targets are all unneeded.
        kept, removed_round = [], 0
        for s in stmts:
            tg = _removable_targets(s)
            if tg is not None and not any(t in needed for t in tg):
                removed_round += 1
            else:
                kept.append(s)
        stmts = kept
        removed_total += removed_round
        if removed_round == 0:
            break
    return stmts, removed_total


def _find_return(func: ast.FunctionDef):
    """First top-level Return in the function body, or None."""
    for s in func.body:
        if isinstance(s, ast.Return):
            return s
    return None


def count_return_features(program_str: str):
    """Number of return-tuple elements for a FLAT tuple return, else None.

    Returns None if the program can't be parsed, the return isn't a tuple, or
    the tuple uses starred unpacking (feature count not statically known).
    """
    try:
        tree = ast.parse(program_str)
        func = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "estimator"),
            None,
        )
        if func is None:
            return None
        ret = _find_return(func)
        if ret is None or not isinstance(ret.value, ast.Tuple):
            return None
        if any(isinstance(e, ast.Starred) for e in ret.value.elts):
            return None
        return len(ret.value.elts)
    except Exception:
        return None


def simplify_program(
    program_str: str,
    weights: np.ndarray,
    weight_threshold: float = 0.05,
    report: dict | None = None,
) -> str:
    """Simplify a program by removing dead code and low-weight features.

    Returns the simplified program string, or the original if simplification is
    not applicable or fails. ``report`` (optional dict) is populated with stats.
    """
    rep = {
        "changed": False,
        "assignments_removed": 0,
        "return_elements_before": None,
        "return_elements_after": None,
        "weight_pruning_applied": False,
        "features_pruned": [],
        "error": None,
    }
    try:
        tree = ast.parse(program_str)
        func = next(
            (n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "estimator"),
            None,
        )
        ret = _find_return(func) if func is not None else None
        if func is None or ret is None or not isinstance(ret.value, ast.Tuple):
            # Step A: can't prune a non-tuple (or missing) return -> unchanged.
            if report is not None:
                report.update(rep)
            return program_str

        rep["return_elements_before"] = len(ret.value.elts)

        # Step B: dead-code elimination (round 1).
        func.body, removed1 = _eliminate_dead(func.body)
        rep["assignments_removed"] += removed1
        ret = _find_return(func)

        # Step C: weight-based feature pruning (only when features map 1:1 to
        # return elements -- no Starred unpacking, matching lengths).
        elts = ret.value.elts
        w = np.abs(np.asarray(weights, dtype=float).ravel())
        mappable = (
            len(w) > 0
            and len(elts) == len(w)
            and not any(isinstance(e, ast.Starred) for e in elts)
        )
        if mappable:
            thresh = weight_threshold * float(w.max())
            keep = [i for i in range(len(elts)) if w[i] >= thresh]
            if not keep:                       # pathological: keep the strongest
                keep = [int(np.argmax(w))]
            if len(keep) < len(elts):
                rep["weight_pruning_applied"] = True
                rep["features_pruned"] = [i for i in range(len(elts)) if i not in keep]
                ret.value.elts = [elts[i] for i in keep]
                # Step D: dead-code elimination (round 2).
                func.body, removed2 = _eliminate_dead(func.body)
                rep["assignments_removed"] += removed2

        ret = _find_return(func)
        rep["return_elements_after"] = len(ret.value.elts)

        # Step E: unparse + compile safety net.
        new_src = ast.unparse(tree)
        compile(new_src, "<simplified>", "exec")
        rep["changed"] = new_src.strip() != program_str.strip()
        if report is not None:
            report.update(rep)
        return new_src if rep["changed"] else program_str

    except Exception as e:  # never crash -> return original
        rep["error"] = f"{type(e).__name__}: {e}"
        if report is not None:
            report.update(rep)
        return program_str
