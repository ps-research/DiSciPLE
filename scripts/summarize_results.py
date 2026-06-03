"""Summarize Step 8 results into paper-comparison tables.

Reads results/<benchmark>_<variant>/scores.json and prints:
  Table 1 - main results (full system) vs paper
  Table 2 - population ablation vs paper (Table 3 in the paper)
  Table 3 - poverty + AGB ablation (bonus; not in the paper)

"In-distribution" uses the held-out test split; "OOD" uses the ood split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Paper reference numbers (from the Step 8 spec).
PAPER_MAIN = {
    ("population_density", "l2_log"): {"in": "0.2626*", "ood": "—"},
    ("population_density", "l1_log"): {"in": "0.4426", "ood": "0.4426"},
    ("poverty", "l1"): {"in": "1.077", "ood": "1.134"},
    ("poverty", "rmse"): {"in": "1.314", "ood": "1.420"},
    ("agb", "l1"): {"in": "24.79", "ood": "31.10"},
    ("agb", "rmse"): {"in": "32.99", "ood": "42.93"},
}
PAPER_POP_ABLATION = {
    "base": {"test": "0.2906", "ood": "0.4258"},
    "critic_only": {"test": "0.2873", "ood": "0.4184"},
    "full": {"test": "0.2607", "ood": "0.3807"},
}
METRICS = {"population_density": ("l2_log", "l1_log"),
           "poverty": ("l1", "rmse"), "agb": ("l1", "rmse")}


def load(bench: str, variant: str):
    p = RESULTS / f"{bench}_{variant}" / "scores.json"
    if not p.exists():
        return None
    return json.load(open(p))


def cell(d, split, metric):
    if d is None:
        return "...."
    v = d.get("scores", {}).get(split, {}).get(metric)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "...."


def table1():
    print("\n=== Table 1 - Main results (full system) ===")
    print(f"{'Benchmark':<20}{'Metric':<8}{'Ours(in)':>10}{'Paper(in)':>11}"
          f"{'Ours(ood)':>11}{'Paper(ood)':>12}")
    print("-" * 72)
    for bench in ("population_density", "poverty", "agb"):
        d = load(bench, "full")
        for metric in METRICS[bench]:
            pap = PAPER_MAIN.get((bench, metric), {"in": "—", "ood": "—"})
            print(f"{bench:<20}{metric:<8}{cell(d,'test',metric):>10}{pap['in']:>11}"
                  f"{cell(d,'ood',metric):>11}{pap['ood']:>12}")


def ablation_table(title, benches, paper=None):
    print(f"\n=== {title} ===")
    print(f"{'Benchmark':<20}{'Variant':<14}{'Test':>10}{'OOD':>10}"
          f"{'PaperTest':>12}{'PaperOOD':>10}")
    print("-" * 76)
    for bench in benches:
        metric = METRICS[bench][0]
        for variant in ("base", "critic_only", "full"):
            d = load(bench, variant)
            pt = po = "—"
            if paper and variant in paper:
                pt, po = paper[variant]["test"], paper[variant]["ood"]
            print(f"{bench:<20}{variant:<14}{cell(d,'test',metric):>10}"
                  f"{cell(d,'ood',metric):>10}{pt:>12}{po:>10}")


def main() -> int:
    n_done = len(list(RESULTS.glob("*/scores.json"))) if RESULTS.exists() else 0
    print(f"Found {n_done} completed runs in {RESULTS}")
    table1()
    ablation_table("Table 2 - Population ablation (paper Table 3)",
                   ["population_density"], PAPER_POP_ABLATION)
    ablation_table("Table 3 - Poverty + AGB ablation (bonus, not in paper)",
                   ["poverty", "agb"])
    print("\n(.... = run not finished yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
