"""Step 1 data integrity diagnostic.

Validates, for each of the three benchmarks:
  * manifest loads and row count matches the number of mask files (no
    missing / orphan files);
  * every mask stack has shape (42, 224, 224) with values in {0, 1};
  * split distribution (train/val/test/ood) with a non-empty OOD split;
  * geographic split-leakage:
      - population_density & poverty: all OOD strictly WEST of all
        train/val/test (max OOD lon < min non-OOD lon);
      - agb: OOD exclusively WA, train/val/test exclusively MA+ME;
  * target range sanity (population > 0, AGB >= 0, poverty in a plausible
    range);
  * environment-variable ranges when has_env (temp/precip/elev in [0, 255],
    nightlight in [0, 1], no NaNs).

Prints PASS/FAIL per check and exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import ENV_COLUMNS  # noqa: E402

EXPECTED_SHAPE = (42, 224, 224)
EXPECTED_SPLITS = {"train", "val", "test", "ood"}


class Reporter:
    """Accumulates PASS/FAIL results and tracks overall success."""

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


def check_manifest_and_masks(rep: Reporter, bench_dir: Path, df: pd.DataFrame) -> None:
    ids = df["id"].tolist()
    masks_dir = bench_dir / "masks"
    mask_files = {p.stem for p in masks_dir.glob("*.npz")}
    manifest_ids = set(ids)

    missing = manifest_ids - mask_files  # manifest rows without a mask file
    orphan = mask_files - manifest_ids   # mask files without a manifest row
    rep.check(
        "row count == mask file count (no missing/orphan)",
        len(missing) == 0 and len(orphan) == 0 and len(ids) == len(mask_files),
        f"rows={len(ids)} masks={len(mask_files)} missing={len(missing)} orphan={len(orphan)}",
    )

    # Validate shape + value range for EVERY mask stack.
    bad_shape = 0
    bad_values = 0
    for obs_id in ids:
        with np.load(masks_dir / f"{obs_id}.npz") as z:
            arr = z["masks"]
        if arr.shape != EXPECTED_SHAPE:
            bad_shape += 1
        else:
            u = np.unique(arr)
            if not np.isin(u, (0, 1)).all():
                bad_values += 1
    rep.check(
        f"all masks shape {EXPECTED_SHAPE}",
        bad_shape == 0,
        f"{bad_shape} files with wrong shape",
    )
    rep.check(
        "all mask values in {0, 1}",
        bad_values == 0,
        f"{bad_values} files with out-of-range values",
    )


def check_splits(rep: Reporter, df: pd.DataFrame) -> None:
    counts = df["split"].value_counts().to_dict()
    print(f"  split counts: {counts}")
    present = set(counts)
    rep.check(
        "splits are exactly {train,val,test,ood}",
        present == EXPECTED_SPLITS,
        f"found {sorted(present)}",
    )
    rep.check(
        "OOD split exists and is non-empty",
        counts.get("ood", 0) > 0,
        f"ood={counts.get('ood', 0)}",
    )


def check_leakage_lon(rep: Reporter, df: pd.DataFrame) -> None:
    ood = df[df["split"] == "ood"]["lon"].to_numpy()
    rest = df[df["split"] != "ood"]["lon"].to_numpy()
    # "Easternmost two-thirds" -> train/val/test; westernmost third -> ood.
    # West = smaller longitude, so every OOD lon must be < every other lon.
    passed = ood.max() < rest.min()
    rep.check(
        "OOD strictly west of train/val/test (no lon overlap)",
        passed,
        f"max(ood_lon)={ood.max():.4f} min(rest_lon)={rest.min():.4f}",
    )


def check_leakage_state(rep: Reporter, df: pd.DataFrame) -> None:
    ood_states = set(df[df["split"] == "ood"]["state"].unique())
    rest_states = set(df[df["split"] != "ood"]["state"].unique())
    rep.check(
        "OOD is exclusively WA",
        ood_states == {"WA"},
        f"ood states={sorted(ood_states)}",
    )
    rep.check(
        "train/val/test is exclusively MA+ME",
        rest_states <= {"MA", "ME"} and "WA" not in rest_states,
        f"non-ood states={sorted(rest_states)}",
    )


def check_targets(rep: Reporter, name: str, df: pd.DataFrame) -> None:
    t = df["target"].to_numpy(dtype=np.float64)
    rep.check(f"{name} targets finite (no NaN/inf)", np.isfinite(t).all())
    if name == "population_density":
        rep.check("population targets > 0", (t > 0).all(),
                  f"min={t.min():.4f}")
    elif name == "agb":
        rep.check("AGB targets >= 0", (t >= 0).all(), f"min={t.min():.4f}")
    elif name == "poverty":
        # Wealth index is a standardized score; expect a modest finite range.
        plausible = np.abs(t).max() < 10.0
        rep.check("poverty targets in plausible range (|t| < 10)", plausible,
                  f"range=[{t.min():.4f}, {t.max():.4f}]")


def check_env(rep: Reporter, df: pd.DataFrame) -> None:
    for col in ENV_COLUMNS:
        if col not in df.columns:
            rep.check(f"env column '{col}' present", False)
            continue
        v = df[col].to_numpy(dtype=np.float64)
        no_nan = np.isfinite(v).all()
        if col == "nightlight":
            in_range = (v >= 0).all() and (v <= 1).all()
            bounds = "[0, 1]"
        else:
            in_range = (v >= 0).all() and (v <= 255).all()
            bounds = "[0, 255]"
        rep.check(
            f"env '{col}' no NaN and in {bounds}",
            no_nan and in_range,
            f"range=[{v.min():.4f}, {v.max():.4f}]",
        )


def main() -> int:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_dir = Path(config.paths.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    rep = Reporter()
    for name, bcfg in config.benchmarks.items():
        print(f"\n==================== {name} ====================")
        bench_dir = data_dir / name
        df = pd.read_csv(bench_dir / "manifest.csv", dtype={"id": str})

        check_manifest_and_masks(rep, bench_dir, df)
        check_splits(rep, df)

        if name in ("population_density", "poverty"):
            check_leakage_lon(rep, df)
        elif name == "agb":
            check_leakage_state(rep, df)

        check_targets(rep, name, df)
        if bcfg.has_env:
            check_env(rep, df)

    print("\n" + "=" * 52)
    if rep.ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
