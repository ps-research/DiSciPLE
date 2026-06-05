"""Benchmark data loaders for the DiSciPLE reproduction.

Perception is precomputed: each observation carries a stack of 42 binary
OSM concept masks (one channel per concept in ``data/concepts.txt`` order).
Images are intentionally NOT loaded -- the evolutionary loop operates over
the precomputed masks (and, for some benchmarks, scalar environment vars).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Scalar environment variables available for benchmarks with ``has_env: true``.
ENV_COLUMNS = ["temperature", "precipitation", "nightlight", "elevation"]


@dataclass
class BenchmarkDataset:
    """Holds all (non-image) data for one benchmark."""

    name: str                                # benchmark name (subfolder under data/)
    ids: list[str]
    targets: np.ndarray                      # raw target values, (N,)
    masks: dict[str, np.ndarray]             # concept_name -> (N, H, W) uint8
    env: dict[str, np.ndarray] | None        # env_var_name -> (N,) float, or None
    splits: np.ndarray                       # (N,) str: 'train'/'val'/'test'/'ood'
    concepts: list[str]                      # the 42 concept names, in order
    lat: np.ndarray                          # (N,) float
    lon: np.ndarray                          # (N,) float


def _load_concepts(data_dir: Path) -> list[str]:
    with open(data_dir / "concepts.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_benchmark(name: str, config) -> BenchmarkDataset:
    """Load a benchmark by name into a :class:`BenchmarkDataset`.

    Reads ``data/<name>/manifest.csv`` and the per-observation mask stacks
    from ``data/<name>/masks/<id>.npz`` (single key ``masks`` of shape
    ``(42, H, W)``). Targets are stored RAW; metric-side transforms (e.g.
    log10 for population) are applied later by the metric functions.
    """
    data_dir = Path(config.paths.data_dir)
    bench_dir = data_dir / name
    concepts = _load_concepts(data_dir)

    # Read 'id' as string: some benchmarks use zero-padded codes (e.g. US
    # census FIPS like '010010210001') whose leading zeros must be preserved
    # to match the mask filenames.
    df = pd.read_csv(bench_dir / "manifest.csv", dtype={"id": str})

    # Optional deterministic per-split subsample (fast validation runs). Done
    # BEFORE loading masks so a 10% run also reads only ~10% of the .npz files.
    sample_frac = getattr(config, "sample_frac", None)
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        all_splits = df["split"].to_numpy(dtype=object).astype(str)
        rng = np.random.default_rng(int(getattr(config, "seed", 0)))
        keep = np.zeros(len(df), dtype=bool)
        for sp in np.unique(all_splits):
            idx = np.where(all_splits == sp)[0]
            n = max(1, int(round(len(idx) * sample_frac)))
            sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
            keep[sel] = True
        df = df.iloc[np.where(keep)[0]].reset_index(drop=True)

    ids = df["id"].tolist()
    targets = df["target"].to_numpy(dtype=np.float64)
    splits = df["split"].to_numpy(dtype=object).astype(str)
    lat = df["lat"].to_numpy(dtype=np.float64)
    lon = df["lon"].to_numpy(dtype=np.float64)

    # Stack masks in manifest row order -> per-concept (N, H, W) arrays.
    masks_dir = bench_dir / "masks"
    stacks = []
    for obs_id in ids:
        with np.load(masks_dir / f"{obs_id}.npz") as z:
            stacks.append(z["masks"])           # (C, H, W) uint8
    all_masks = np.stack(stacks, axis=0)         # (N, C, H, W)

    masks = {concept: all_masks[:, c] for c, concept in enumerate(concepts)}

    has_env = config.benchmarks[name].has_env
    env: dict[str, np.ndarray] | None = None
    if has_env:
        env = {col: df[col].to_numpy(dtype=np.float64) for col in ENV_COLUMNS}

    return BenchmarkDataset(
        name=name,
        ids=ids,
        targets=targets,
        masks=masks,
        env=env,
        splits=splits,
        concepts=concepts,
        lat=lat,
        lon=lon,
    )
