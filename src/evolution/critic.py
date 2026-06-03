"""Program critic (§4.4): stratified evaluation + LLM-guided improvement.

After crossover/mutation, the critic partitions the (training) observations by
land-use category using the 42 OSM concept masks, computes the program's
per-stratum error, finds the categories it does worst on, and prompts the LLM to
generate an improved program targeting those weak categories.
"""
from __future__ import annotations

import numpy as np

from src.config import Config
from src.data.loader import BenchmarkDataset
from src.execution.evaluator import ProgramResult, _transform_target
from src.llm import CRITIC_PROMPT, LLMGenerator, build_objective_prompt

_PRESENCE_ATTR = "_concept_presence_cache"


def _concept_presence(dataset: BenchmarkDataset, presence_threshold: float) -> dict:
    """Per-concept boolean presence mask over observations (cached on dataset).

    A concept is "present" in an observation if the fraction of mask pixels set
    exceeds ``presence_threshold``. This depends only on the data, so it is
    computed once and cached on the dataset object.
    """
    cache = getattr(dataset, _PRESENCE_ATTR, None)
    if cache is not None and cache[0] == presence_threshold:
        return cache[1]

    presence = {}
    for concept in dataset.concepts:
        m = dataset.masks[concept]                      # (N, H, W) uint8
        frac = m.reshape(m.shape[0], -1).mean(axis=1)   # fraction present per obs
        presence[concept] = frac > presence_threshold
    setattr(dataset, _PRESENCE_ATTR, (presence_threshold, presence))
    return presence


def stratified_analysis(
    program_result: ProgramResult,
    dataset: BenchmarkDataset,
    predictions: np.ndarray,
    n_worst: int = 5,
    presence_threshold: float = 0.01,
) -> list[str]:
    """Return up to ``n_worst`` concept names where the program errs most (worst first).

    Error is the mean squared error in the metric's space (log10 for population,
    raw otherwise) over TRAIN observations in which the concept is present.
    Concepts present in fewer than 20 train observations are skipped.
    """
    y = _transform_target(dataset.name, dataset.targets)
    train_mask = dataset.splits == "train"
    finite = np.isfinite(predictions)
    presence = _concept_presence(dataset, presence_threshold)

    errors: list[tuple[str, float]] = []
    for concept in dataset.concepts:
        sel = presence[concept] & train_mask & finite
        if sel.sum() < 20:
            continue
        err = float(np.mean((predictions[sel] - y[sel]) ** 2))
        errors.append((concept, err))

    errors.sort(key=lambda ce: ce[1], reverse=True)   # worst (highest error) first
    return [c for c, _ in errors[:n_worst]]


def build_critic_prompt(program_str: str, worst_categories: list[str], benchmark_name: str) -> str:
    """Objective prompt + the program + the critic instruction (Appendix D)."""
    objective = build_objective_prompt(benchmark_name)
    critic = CRITIC_PROMPT.format(bad_categories=", ".join(worst_categories))
    return f"{objective}\n\n{program_str}\n\n{critic}"


def apply_critic(
    program_str: str,
    worst_categories: list[str],
    generator: LLMGenerator,
    benchmark_name: str,
    config: Config,
) -> str:
    """Prompt the LLM to improve the program on its weak categories.

    Returns the improved program string, or the original if extraction fails.
    """
    prompt = build_critic_prompt(program_str, worst_categories, benchmark_name)
    improved = LLMGenerator.extract_code(generator.generate(prompt))
    return improved if improved.strip() else program_str
