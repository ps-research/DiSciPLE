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
    feature_stds: np.ndarray | None = None  # per-feature std over train (for the simplifier's
                                            # scale-robust contribution = |weight| * std)


def _transform_target(benchmark_name: str, raw: np.ndarray) -> np.ndarray:
    """Population regresses in log10-space; poverty/agb use the raw target."""
    if benchmark_name == "population_density":
        return np.log10(np.clip(raw, 1e-10, None))
    return raw.astype(float)


# Largest standardized train condition number a feature subset may have before
# we start rejecting columns. The paper fits plain linear regression and treats
# the weights as feature-importance signals (5%-weight pruning). That premise
# silently breaks when the LLM emits a degenerate/collinear feature list (e.g. a
# duplicated column, or features all derived from a couple of variables): the
# design matrix becomes near-singular, OLS produces huge cancelling weights, and
# predictions explode under any distribution shift (catastrophic OOD on AGB).
#
# Empirically (per-benchmark threshold sweep on evolved programs), a strict
# cap of 3 -- i.e. keep only near-decorrelated features -- gives the best OOD:
# it costs little in-distribution (~4-15% test RMSE) but roughly HALVES AGB's
# OOD error (ood/test 4.5 -> 1.9), while leaving the already-clean population /
# poverty programs unharmed (poverty even improves). This favours the compact,
# decorrelated, OOD-robust programs the paper's interpretability thesis targets.
MAX_TRAIN_COND = 3.0


def _select_well_conditioned(X_train: np.ndarray, y_train: np.ndarray,
                             max_cond: float = MAX_TRAIN_COND) -> list[int]:
    """Greedily pick a well-conditioned column subset using TRAIN data only.

    Columns are considered most-predictive-first (by |correlation with the
    target|) and a column is kept only if adding it keeps the standardized train
    condition number <= ``max_cond``. This preserves the paper's plain-OLS
    regressor but rejects collinear columns that would make its weights (and the
    weight-based simplifier) meaningless. No OOD information is used. On a
    well-conditioned program every column is kept (a no-op).
    """
    X_train = np.asarray(X_train, dtype=float)
    k = X_train.shape[1]
    if k <= 1:
        return list(range(k))
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X_train - mu) / sd                       # standardize for a scale-free condition test
    znorm = np.linalg.norm(Z, axis=0)
    yc = np.asarray(y_train, dtype=float) - float(np.mean(y_train))
    denom = znorm * (np.linalg.norm(yc) + 1e-12)
    corr = np.where(denom > 0, np.abs(Z.T @ yc) / np.where(denom > 0, denom, 1.0), 0.0)
    order = [int(j) for j in np.argsort(-corr) if znorm[j] > 0]
    selected: list[int] = []
    for j in order:
        trial = selected + [j]
        if len(trial) == 1 or np.linalg.cond(Z[:, trial]) <= max_cond:
            selected = trial
    return sorted(selected) if selected else [int(np.argmax(znorm))]


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
        # Reject collinear/degenerate columns (train-only) so plain OLS stays
        # well-posed; the paper's regressor is unchanged on clean programs.
        keep = _select_well_conditioned(features[train_mask], y_sub[train_mask])
        model = LinearRegression()
        model.fit(features[train_mask][:, keep], y_sub[train_mask])
        # Weight vector aligned 1:1 to the program's returned features: kept
        # columns get their OLS coefficient, rejected columns get 0 (so the
        # weight-based simplifier prunes them). preds via the full vector are
        # identical to predicting on the kept subset (rejected coefs are 0).
        coef_full = np.zeros(features.shape[1], dtype=float)
        coef_full[keep] = model.coef_
        preds = features @ coef_full + model.intercept_   # (len(sub_idx),)
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

    # Per-feature std over the train rows (scale info for the simplifier).
    feature_stds = np.std(features[train_mask], axis=0)

    return ProgramResult(
        success=True,
        fitness=float(fitness),
        scores=scores,
        weights=coef_full,
        intercept=float(model.intercept_),
        n_features=exec_res.n_features,
        error_msg=None,
        predictions=preds_full,
        feature_stds=np.asarray(feature_stds, dtype=float),
    )
