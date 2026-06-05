"""Configuration loading and validation for the DiSciPLE reproduction.

Uses pydantic for typed, validated nested config. ``load_config`` reads the
YAML file, validates it, and ensures the checkpoint / log directories exist.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    data_dir: str
    checkpoint_dir: str
    log_dir: str


class BenchmarkConfig(BaseModel):
    metric: str          # 'l2' or 'l2_log'
    has_env: bool


class EvolutionConfig(BaseModel):
    generations: int     # T
    population_size: int  # M
    mutation_prob: float  # rho_m
    use_critic: bool = True       # Step 6 critic (disable for the no-critic ablation)
    use_simplifier: bool = True   # Step 7 simplifier (disable for the no-simplifier ablation)


class LLMConfig(BaseModel):
    model: str
    max_seq_length: int
    load_in_4bit: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    dtype: str | None = None   # None -> let Unsloth auto-detect (bf16 on A100)
    gen_batch_size: int = 16   # batched-generation chunk size (16 avoids OOM on long/bloated programs)


class Config(BaseModel):
    paths: PathsConfig
    benchmarks: dict[str, BenchmarkConfig]
    evolution: EvolutionConfig
    llm: LLMConfig
    seed: int
    sample_frac: float | None = None   # if set (0,1): deterministically subsample each split (fast validation runs)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load, validate, and return the config; create output dirs on load."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config(**raw)

    # Ensure output directories exist (relative paths resolved against CWD).
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)

    return config
