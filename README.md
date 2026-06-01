# DiSciPLE (reproduction)

A faithful reproduction of **DiSciPLE** (*Discovering Scientific Programs using
LLMs and Evolution*, CVPR 2025): an evolutionary program-search framework that
uses an LLM to discover interpretable programs predicting geospatial indicators
from precomputed visual concepts.

This is a **pure DiSciPLE reproduction** — no CANON extensions, no Bayesian
regression, no invariants.

## Environment

```bash
conda activate /nfs-stor/salem.lahlou/sandeep/WACV/envs/unsloth-venv
```

- LLM inference uses **Unsloth** (not vLLM).
- GPUs: pin to the two free devices on the shared node (`smart-gpu`).

## Layout

```
configs/      # YAML configs (default.yaml)
src/
  config.py   # pydantic config loading + validation
  data/       # benchmark data loaders
  primitives/ # primitive function library (Step 2)
  execution/  # program execution sandbox (Step 3)
  evolution/  # evolutionary search loop (Step 5)
  utils/      # seeding, etc.
diagnostics/  # standalone data/sanity checks
data/         # benchmarks (gitignored): population_density, poverty, agb
```

## Benchmarks

| Benchmark            | Metric   | Env vars | Target               |
|----------------------|----------|----------|----------------------|
| `population_density` | L2 (log) | no       | people / sq mile     |
| `poverty`            | L2       | yes      | wealth index         |
| `agb`                | L2       | yes      | above-ground biomass |

Each observation carries 42 binary OSM concept masks (`(42, 224, 224)`),
listed in `data/concepts.txt`. Data is split geographically: the two-thirds
easternmost observations form train/val/test; the westernmost one-third is the
out-of-distribution (`ood`) reliability split.

## Diagnostics

```bash
python diagnostics/step1_data_check.py
```
Validates manifests, mask shapes/values, split distributions, geographic
split-leakage, target ranges, and environment-variable ranges.
