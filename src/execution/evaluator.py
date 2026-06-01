"""Program evaluation: feature extraction -> OLS regression -> scoring.

Implements the score ``s(P; D)`` from the paper (§4.3): the program produces a
list of predictive features, a linear regressor is fit on top (train split
only), and the program is scored by its error. Fitness (what evolution
minimizes) is the train-split error; per-split metrics are reported for results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from src.config import Config
from src.data.loader import BenchmarkDataset
from src.execution.metrics import get_metrics
from src.execution.runner import execute_program


@dataclass
class ProgramResult:
    success: bool
    fitness: float                          # evolution-driving score (lower = better)
    scores: dict                            # {split: {metric_name: value}}
    weights: np.ndarray | None              # OLS coefficients (used by simplifier)
    intercept: float | None                 # OLS intercept
    n_features: int
    error_msg: str | None
    predictions: np.ndarray | None = None   # per-obs OLS predictions, full-N aligned
                                            # (NaN where not executed); used by the critic


def _transform_target(benchmark_name: str, raw: np.ndarray) -> np.ndarray:
    """Population regresses in log10-space; poverty/agb use the raw target."""
    if benchmark_name == "population_density":
        return np.log10(np.clip(raw, 1e-10, None))
    return raw.astype(float)


def evaluate_program(
    program_str: str,
    dataset: BenchmarkDataset,
    data_dir: str,
    config: Config,
    benchmark_name: str,
    image_cache: list | None = None,
    eval_splits: list | None = None,
) -> ProgramResult:
    """Run, fit OLS on train, score the requested splits, return a ``ProgramResult``.

    ``eval_splits`` restricts which splits are executed/scored. The evolutionary
    loop passes ``['train']`` (fitness only needs train) to avoid running every
    program over val/test/ood; the final best program is re-evaluated with
    ``None`` (all splits) for reporting. ``'train'`` must be included so OLS can fit.
    """
    splits = dataset.splits
    if eval_splits is None:
        obs_indices = None
        sub_idx = np.arange(len(splits))
    else:
        sub_idx = np.where(np.isin(splits, list(eval_splits)))[0]
        obs_indices = sub_idx

    exec_res = execute_program(
        program_str, dataset, data_dir, benchmark_name,
        image_cache=image_cache, obs_indices=obs_indices,
    )
    if not exec_res.success:
        return ProgramResult(
            success=False, fitness=float("inf"), scores={}, weights=None,
            intercept=None, n_features=exec_res.n_features, error_msg=exec_res.error_msg,
        )

    features = exec_res.features                       # (len(sub_idx), k)
    y = _transform_target(benchmark_name, dataset.targets)
    y_sub = y[sub_idx]
    sub_splits = splits[sub_idx]
    train_mask = sub_splits == "train"

    if train_mask.sum() == 0:
        return ProgramResult(False, float("inf"), {}, None, None,
                             exec_res.n_features, "no training observations in eval subset")

    try:
        model = LinearRegression()
        model.fit(features[train_mask], y_sub[train_mask])
        preds = model.predict(features)               # (len(sub_idx),)
    except Exception as e:
        return ProgramResult(False, float("inf"), {}, None, None,
                             exec_res.n_features, f"OLS error: {type(e).__name__}: {e}")

    metrics = get_metrics(benchmark_name)
    fitness = metrics["fitness"](preds[train_mask], y_sub[train_mask])

    scores: dict = {}
    for split in ("train", "val", "test", "ood"):
        mask = sub_splits == split
        if mask.sum() == 0:
            continue
        scores[split] = {
            name: fn(preds[mask], y_sub[mask]) for name, fn in metrics["reported"].items()
        }

    # Full-N prediction array (NaN where not executed) for the critic's stratification.
    preds_full = np.full(len(splits), np.nan, dtype=float)
    preds_full[sub_idx] = preds

    return ProgramResult(
        success=True,
        fitness=float(fitness),
        scores=scores,
        weights=np.asarray(model.coef_, dtype=float),
        intercept=float(model.intercept_),
        n_features=exec_res.n_features,
        error_msg=None,
        predictions=preds_full,
    )
