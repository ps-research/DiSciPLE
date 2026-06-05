"""The DiSciPLE evolutionary search loop (Algorithm 1), base version.

Initializes a population of LLM-generated programs from the objective prompt,
then evolves them with fitness-weighted (tournament) selection + LLM crossover
and probabilistic mutation, tracking the best program ``P*`` across generations
and checkpointing after each generation for cluster survivability.

This is the BASE loop: the critic (Step 6) and simplifier (Step 7) steps of
Algorithm 1 are skipped (no-ops) here.

Evolution evaluates programs train-only (fitness needs only the train split);
the returned best program is re-evaluated on all splits for reporting.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

from src.config import Config
from src.data.loader import load_benchmark
from src.evolution.bank import BankEntry, ProgramBank
from src.evolution.critic import build_critic_prompt, stratified_analysis
from src.evolution.simplifier import count_return_features, simplify_program
from src.execution.evaluator import _transform_target, evaluate_program
from src.execution.runner import preload_images
from src.llm import (
    CROSSOVER_PROMPT,
    MUTATION_PROMPT,
    LLMGenerator,
    build_objective_prompt,
)
from src.primitives import get_api_spec
from src.utils import set_seed

CHECKPOINT_NAME = "evolution_state.pkl"
_GEN_CHUNK = 8   # LLM batch chunk size (keeps VRAM bounded for large M)


# --------------------------------------------------------------------------- #
# Checkpointing                                                               #
# --------------------------------------------------------------------------- #
def _checkpoint_path(config: Config) -> Path:
    return Path(config.paths.checkpoint_dir) / CHECKPOINT_NAME


def save_evolution_state(bank: ProgramBank, best: BankEntry, generation: int, config: Config) -> str:
    """Pickle {bank, best, generation} to the checkpoint dir; return the path."""
    path = _checkpoint_path(config)
    with open(path, "wb") as f:
        pickle.dump({"bank": bank, "best": best, "generation": generation}, f)
    return str(path)


def load_evolution_state(config: Config):
    """Return (bank, best, generation) from the checkpoint, or None if absent."""
    path = _checkpoint_path(config)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state["bank"], state["best"], state["generation"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _r2(fitness: float, train_var: float) -> float:
    if not np.isfinite(fitness) or train_var <= 0:
        return float("-inf")
    return 1.0 - fitness / train_var


def _make_entry(code: str, result, train_var: float, simplification: dict | None = None) -> BankEntry:
    r2 = _r2(result.fitness, train_var) if result.success else float("-inf")
    return BankEntry(program_str=code, result=result, r2_score=r2, simplification=simplification)


def _chunked_generate(gen: LLMGenerator, prompts: list[str], chunk: int = _GEN_CHUNK) -> list[str]:
    """Batched generation in fixed-size chunks (bounds VRAM for large M)."""
    out: list[str] = []
    for s in range(0, len(prompts), chunk):
        out.extend(gen.generate_batch(prompts[s:s + chunk]))
    return out


_FULL_SPLITS = ("train", "val", "test", "ood")


def _critic_phase(offspring_codes, dataset, benchmark_name, gen, evaluate_fn, gen_chunk):
    """Run the critic on each offspring; return a list of (code, ProgramResult).

    For each offspring: preliminary TRAIN-only evaluation (fitness + stratified
    analysis only need train predictions) -> stratified analysis -> LLM critic
    call -> re-evaluate the improved program, keeping whichever of
    (improved, original) has the lower fitness. Critic LLM calls are batched.
    """
    prelim = [evaluate_fn(c, ("train",)) for c in offspring_codes]

    crit_idx, crit_prompts = [], []
    for i, (code, pr) in enumerate(zip(offspring_codes, prelim)):
        if pr.success and pr.predictions is not None:
            worst = stratified_analysis(pr, dataset, pr.predictions)
            if worst:
                crit_idx.append(i)
                crit_prompts.append(build_critic_prompt(code, worst, benchmark_name))

    improved: dict = {}
    if crit_prompts:
        gens = _chunked_generate(gen, crit_prompts, gen_chunk)
        improved = {i: LLMGenerator.extract_code(g) for i, g in zip(crit_idx, gens)}

    results = []
    for i, code in enumerate(offspring_codes):
        pr = prelim[i]
        if i in improved:
            fin = evaluate_fn(improved[i], ("train",))
            # Keep the critic's program only if it strictly improves fitness.
            if fin.success and fin.fitness < pr.fitness:
                results.append((improved[i], fin))
                continue
        results.append((code, pr))
    return results


def _simplifier_phase(offspring, evaluate_fn, splits):
    """Analytically simplify each offspring; re-evaluate and keep if it still runs.

    Simplification (dead-code + low-weight feature pruning) is near-lossless, so
    the simplified program is kept whenever it re-evaluates successfully. Returns
    (code, result, simplification) triples, where simplification records the
    pre/post return-feature counts.
    """
    out = []
    for code, result in offspring:
        simpl = None
        if result.success and result.weights is not None:
            pre = count_return_features(code)
            simplified = simplify_program(code, result.weights, feature_stds=result.feature_stds)
            if simplified.strip() != code.strip():
                fin = evaluate_fn(simplified, splits)
                if fin.success:
                    out.append((simplified, fin,
                                {"pre_features": pre, "post_features": count_return_features(simplified)}))
                    continue
                # Loud warning: a simplification was produced but failed re-eval, so we
                # fall back to the un-simplified program (the feature cap does NOT apply
                # to it). Silent fallback here previously masked a dead-code bug.
                print(f"[simplifier] WARNING: discarded simplification "
                      f"({pre}->{count_return_features(simplified)} feats) -- re-eval failed: "
                      f"{(fin.error_msg or '')[:120]}; keeping un-simplified program", flush=True)
            simpl = {"pre_features": pre, "post_features": pre}
        out.append((code, result, simpl))
    return out


def _log_gen(generation: int, bank: ProgramBank, best: BankEntry, history) -> None:
    valid = bank.valid_entries()
    r2s = [e.r2_score for e in valid if np.isfinite(e.r2_score)]
    mean_r2 = float(np.mean(r2s)) if r2s else float("-inf")
    print(
        f"[gen {generation:>2}] valid={bank.valid_count()}/{len(bank.entries)} "
        f"best_r2(P*)={best.r2_score:.4f} mean_r2={mean_r2:.4f} "
        f"best_fitness={best.result.fitness:.4f}"
    )
    if history is not None:
        history.append({
            "generation": generation,
            "best_r2": best.r2_score,
            "best_fitness": best.result.fitness,
            "best_program": best.program_str,
            "valid_count": bank.valid_count(),
            "mean_r2": mean_r2,
        })


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def run_evolution(
    benchmark_name: str,
    config: Config,
    history: list | None = None,
    generator: LLMGenerator | None = None,
) -> BankEntry:
    """Run the full evolutionary search; return the best program found.

    ``history`` (optional) is appended one dict per generation for introspection.
    ``generator`` (optional) reuses an already-loaded LLM (avoids a second load).
    """
    set_seed(config.seed)
    data_dir = config.paths.data_dir
    dataset = load_benchmark(benchmark_name, config)

    # Training-target variance (in the metric's space) for R^2.
    y = _transform_target(benchmark_name, dataset.targets)
    train_var = float(np.var(y[dataset.splits == "train"]))

    image_cache = (
        preload_images(dataset, data_dir) if benchmark_name == "population_density" else None
    )

    objective = build_objective_prompt(benchmark_name)

    M = config.evolution.population_size
    T = config.evolution.generations
    rho_m = config.evolution.mutation_prob
    use_critic = config.evolution.use_critic
    use_simplifier = config.evolution.use_simplifier
    gen_chunk = config.llm.gen_batch_size

    gen = generator if generator is not None else LLMGenerator(config)
    if gen.model is None:
        gen.load()

    def evaluate(code: str, splits=("train",)):
        return evaluate_program(
            code, dataset, data_dir, config, benchmark_name,
            image_cache=image_cache, eval_splits=list(splits),
        )

    # -- Resume or initialize ------------------------------------------------ #
    resume = load_evolution_state(config)
    if resume is not None and resume[0].benchmark_name == benchmark_name:
        bank, best, completed = resume
        print(f"[resume] loaded checkpoint at generation {completed}")
    else:
        # Phase 1: initialization from the objective prompt.
        bank = ProgramBank(benchmark_name)
        init_codes = [LLMGenerator.extract_code(r)
                      for r in _chunked_generate(gen, [objective] * M, gen_chunk)]
        for code in init_codes:
            bank.add(_make_entry(code, evaluate(code), train_var))
        best = bank.best()
        completed = 0
        save_evolution_state(bank, best, 0, config)
        _log_gen(0, bank, best, history)

    # -- Phase 2: evolution -------------------------------------------------- #
    for t in range(completed + 1, T + 1):
        new_bank = ProgramBank(benchmark_name)

        # (a) sample parents + (b) build crossover prompts for M offspring.
        cross_prompts = []
        for _ in range(M):
            parents = bank.sample_parents(k=2, tournament_size=3)
            if len(parents) < 2:
                cross_prompts.append(objective)   # fallback: regenerate from objective
            else:
                p1, p2 = parents
                cp = CROSSOVER_PROMPT.format(
                    program1=p1.program_str, score1=round(p1.r2_score, 4),
                    program2=p2.program_str, score2=round(p2.r2_score, 4),
                )
                cross_prompts.append(f"{objective}\n\n{cp}")

        cross_codes = [LLMGenerator.extract_code(r) for r in _chunked_generate(gen, cross_prompts, gen_chunk)]
        cross_results = [evaluate(c) for c in cross_codes]
        cross_r2 = [_r2(r.fitness, train_var) if r.success else float("-inf") for r in cross_results]

        # (c) mutation with probability rho_m (uses crossover offspring's score).
        mutate = [random.random() < rho_m for _ in range(M)]
        mut_idx = [i for i in range(M) if mutate[i]]
        mut_prompts = [
            f"{objective}\n\n" + MUTATION_PROMPT.format(
                program=cross_codes[i], score=round(cross_r2[i], 4))
            for i in mut_idx
        ]
        mut_codes = {}
        if mut_prompts:
            gens = _chunked_generate(gen, mut_prompts, gen_chunk)
            mut_codes = {i: LLMGenerator.extract_code(r) for i, r in zip(mut_idx, gens)}

        # assemble post-mutation offspring.
        offspring_codes = [mut_codes[i] if mutate[i] else cross_codes[i] for i in range(M)]

        # (d) critic (Step 6). Evolution evaluates train-only for speed.
        if use_critic:
            offspring = _critic_phase(offspring_codes, dataset, benchmark_name, gen, evaluate, gen_chunk)
        else:
            # No critic: reuse the crossover eval for non-mutated, eval mutated (train-only).
            offspring = [
                (offspring_codes[i], cross_results[i] if not mutate[i] else evaluate(offspring_codes[i]))
                for i in range(M)
            ]

        # (e) simplifier (Step 7) -> (code, result, simplification) triples.
        if use_simplifier:
            offspring = _simplifier_phase(offspring, evaluate, ("train",))
        else:
            offspring = [(code, result, None) for code, result in offspring]

        # (f) track P*, fill the next bank.
        for code, result, simpl in offspring:
            entry = _make_entry(code, result, train_var, simplification=simpl)
            new_bank.add(entry)
            if result.success and result.fitness < best.result.fitness:
                best = entry

        bank = new_bank
        completed = t
        save_evolution_state(bank, best, t, config)
        _log_gen(t, bank, best, history)

    # -- Phase 3: re-evaluate the best program on all splits for reporting --- #
    full = evaluate(best.program_str, splits=("train", "val", "test", "ood"))
    if full.success:
        best = BankEntry(best.program_str, full, _r2(full.fitness, train_var),
                         simplification=best.simplification)

    print("\n===== BEST PROGRAM =====")
    print(best.program_str)
    print(f"\nR2(train)={best.r2_score:.4f}  fitness={best.result.fitness:.4f}")
    print(f"per-split scores: {best.result.scores}")
    return best
