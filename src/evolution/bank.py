"""Population storage and parent selection for the evolutionary loop."""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.execution.evaluator import ProgramResult


@dataclass
class BankEntry:
    program_str: str
    result: ProgramResult     # from Step 3's evaluator
    r2_score: float           # R^2 on training data (shown to the LLM; higher = better)


class ProgramBank:
    """A generation's population of programs."""

    def __init__(self, benchmark_name: str = ""):
        self.benchmark_name = benchmark_name
        self.entries: list[BankEntry] = []

    def add(self, entry: BankEntry) -> None:
        self.entries.append(entry)

    def valid_entries(self) -> list[BankEntry]:
        return [e for e in self.entries if e.result.success]

    def valid_count(self) -> int:
        return len(self.valid_entries())

    def best(self) -> BankEntry:
        """Entry with the lowest fitness (error). Failed entries have inf fitness."""
        if not self.entries:
            raise ValueError("cannot take best() of an empty bank")
        return min(self.entries, key=lambda e: e.result.fitness)

    def sample_parents(self, k: int = 2, tournament_size: int = 3) -> list[BankEntry]:
        """Tournament selection over valid entries.

        Each parent: draw ``tournament_size`` random valid entries, keep the one
        with the lowest fitness. With fewer than 2 valid entries, fall back to
        random selection from whatever valid entries exist (empty list if none).
        """
        valid = self.valid_entries()
        if len(valid) == 0:
            return []
        if len(valid) < 2:
            return [random.choice(valid) for _ in range(k)]

        parents = []
        for _ in range(k):
            competitors = random.sample(valid, min(tournament_size, len(valid)))
            parents.append(min(competitors, key=lambda e: e.result.fitness))
        return parents
