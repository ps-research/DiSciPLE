"""Metric functions and per-benchmark metric selection.

All metrics operate on 1D arrays of predictions vs targets. The ``*_log``
variants assume their inputs are ALREADY in log10-space (the population
regression is fit in log10-space), so they are plain MSE/MAE on those inputs.
"""
from __future__ import annotations

import numpy as np


def mse(pred, true) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean((pred - true) ** 2))


def mae(pred, true) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred - true)))


def rmse(pred, true) -> float:
    return float(np.sqrt(mse(pred, true)))


def l2_log(pred, true) -> float:
    """MSE in log10-space (inputs already log-transformed)."""
    return mse(pred, true)


def l1_log(pred, true) -> float:
    """MAE in log10-space (inputs already log-transformed)."""
    return mae(pred, true)


def get_metrics(benchmark_name: str) -> dict:
    """Return the fitness metric (drives evolution) and the reporting metrics.

    Shape:
        {
          "fitness_name": str,
          "fitness": callable(pred, true) -> float,   # minimized on train split
          "reported": {metric_name: callable(pred, true) -> float, ...},
        }
    """
    if benchmark_name == "population_density":
        return {
            "fitness_name": "l2_log",
            "fitness": l2_log,
            "reported": {"l2_log": l2_log, "l1_log": l1_log},
        }
    # poverty and agb: raw-space L2 fitness; report L1 + RMSE.
    return {
        "fitness_name": "l2",
        "fitness": mse,
        "reported": {"l1": mae, "rmse": rmse},
    }
