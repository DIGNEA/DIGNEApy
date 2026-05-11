#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   llm.py
@Time    :   2026/05/11 14:08:29
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   LLM-driven Evolutionary Instance Generator.
             Uses a Large Language Model (via Anthropic API) as the
             variation operator instead of classic crossover / mutation,
             while keeping Novelty Search as the diversity objective.
"""

import json
import textwrap
from typing import Iterable, Optional, Sequence

import httpx
import numpy as np
import ollama

from .._core import NS, Domain, Instance, P, SupportsSolve
from .._core.descriptors import DESCRIPTORS, describe
from .._core.scores import PerformanceFn, max_gap_target
from ..archives import Archive
from ..operators import Replacement, generational_replacement
from ..transformers import SupportsTransform
from ._base_generator import GenResult
from .evolutionary import Evolutionary

# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert in combinatorial optimisation and algorithm benchmarking.
    Your task is to propose NEW problem instances for a given domain so that a
    portfolio of solvers produces maximally *diverse* and *discriminatory*
    results — i.e. different solvers should win on different instances.

    You will be given:
    • A JSON description of the domain (variable count, bounds, semantics).
    • A sample of the current population with their performance-bias scores
      (higher = more discriminatory) and novelty scores (higher = more diverse).
    • Elite examples: the most discriminatory AND the most novel instances seen
      so far, as positive reference points.

    You must return ONLY a valid JSON array of {n} new instances.
    Each instance is itself a JSON array of numbers respecting the domain bounds.
    No markdown, no explanation — pure JSON.
""").strip()

_USER_TEMPLATE = textwrap.dedent("""
    ## Domain
    {domain_description}

    ## Current population sample (up to {sample_size} instances)
    {population_sample}

    ## Elite instances (most discriminatory)
    {elite_discriminatory}

    ## Elite instances (most novel)
    {elite_novel}

    ## Task
    Generate exactly {n} new instances that explore under-represented regions of
    the instance space and maximise solver discrimination.
    Remember: return ONLY a JSON array of {n} arrays of numbers. No extra text.
""").strip()


class LLMEvolutionary(Evolutionary):
    """Evolutionary instance generator that uses an LLM as the variation operator.

    Instead of classical crossover / mutation, at each generation a prompt is
    sent to an Anthropic model containing:
      - Domain description and variable bounds.
      - A sample of the current population with performance & novelty scores.
      - Elite instances (most discriminatory / most novel so far).

    The model replies with a batch of new candidate instances.  The rest of the
    loop — evaluation, novelty scoring, archive extension — mirrors the
    ``Evolutionary`` generator exactly so results are comparable.

    Parameters
    ----------
    domain : Domain
        Problem domain.  ``domain.bounds`` must expose per-variable (lo, hi)
        pairs and ``domain.describe()`` should return a human-readable string
        (falls back to ``repr`` if the method is absent).
    portfolio : Iterable[SupportsSolve[P]]
        Solvers used to evaluate instances.
    pop_size : int
        Population size and offspring size per generation.
    novelty_approach : NS
        Novelty-search strategy (k-NN sparseness, archive, etc.).
    performance_function : PerformanceFn
        Maps solver scores → scalar bias.  Defaults to ``max_gap_target``.
    generations : int
        Total number of generations.
    repetitions : int
        How many times each solver is run per instance.
    solution_set : Optional[Archive]
        Secondary archive (e.g. a MAP-Elites grid).  If *None* the primary
        novelty archive is used as the final result.
    describe_by : DESCRIPTORS
        Feature descriptor used for novelty distance computation.
    transformer : Optional[SupportsTransform]
        Optional dimensionality-reduction step applied to descriptors.
    phi : float
        Balance weight in [0, 1].  Fitness = phi * performance + (1-phi) * novelty.
    elite_size : int
        Number of elite (most discriminatory / most novel) instances sent to the
        LLM in every prompt.  Larger values give the model richer context at the
        cost of token usage.
    population_sample_size : int
        Number of population members included in the prompt as context.
    model : str
        Anthropic model string.  Defaults to ``claude-sonnet-4-20250514``.
    max_tokens : int
        Token budget for each LLM completion.
    temperature : float
        Sampling temperature passed to the model.
    api_key : Optional[str]
        Anthropic API key.  If *None*, the ``ANTHROPIC_API_KEY`` env-var is used.
    seed : int
        RNG seed for reproducibility of the evolutionary machinery.
    """

    def __init__(
        self,
        domain: Domain,
        portfolio: Iterable[SupportsSolve[P]],
        pop_size: int,
        novelty_approach: NS,
        performance_function: PerformanceFn = max_gap_target,
        generations: int = 1000,
        repetitions: int = 1,
        solution_set: Optional[Archive] = None,
        describe_by: DESCRIPTORS = "features",
        transformer: Optional[SupportsTransform] = None,
        phi: float = 0.85,
        replacement: Replacement = generational_replacement,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.9,
        seed: int = 42,
    ):
        super().__init__(
            domain,
            portfolio,
            pop_size,
            novelty_approach,
            performance_function,
            generations,
            repetitions,
            solution_set,
            describe_by,
            transformer,
            replacement=replacement,
            phi=phi,
            seed=seed,
        )

        ###
        URL = ""  # TODO
        TOKEN = ""

        self._client = ollama.Client(base_url=URL, api_key=TOKEN)
        # self.__load_model(URL, model, TOKEN)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Runtime state shared across calls
        self._elite_discriminatory: list[Instance] = []
        self._elite_novel: list[Instance] = []
        self.elite_size = 5
        self.population_sample_size = 10

    def __load_model(self, base_url: str, model_name: str, token: str) -> None:
        """Loads a model in LM Studio before inference."""
        response = httpx.post(
            f"{base_url}/models/load",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": model_name},
            timeout=120,
        )
        response.raise_for_status()
        print(f"Modelo {model_name} cargado correctamente.")

    def __call__(self, verbose: bool = False) -> GenResult:
        # ── Initialise population ─────────────────────────────────────────────
        self._population = self._domain.generate_instances(n=self._pop_size)
        perf_biases, portfolio_scores = self._evaluate_population(self._population)
        descriptors, features = describe(
            population=self._population,
            key=self._describe_by,
            scores=portfolio_scores,
            domain=self._domain,
            transformer=self._transformer,
        )

        # ── Evolutionary loop ─────────────────────────────────────────────────
        for pgen in range(self._generations):
            # 1. Update elite sets from the current population
            self._update_elites(self._population, perf_biases)

            offspring_vars = self._llm_generate(n=self._pop_size)

            offspring_instances = [Instance(variables=v) for v in offspring_vars]
            perf_biases, portfolio_scores = self._evaluate_population(
                offspring_instances
            )
            descriptors, features = describe(
                population=offspring_instances,
                key=self._describe_by,
                scores=portfolio_scores,
                domain=self._domain,
                transformer=self._transformer,
            )

            novelty_scores = self._novelty_search(instances_descriptors=descriptors)
            offspring_fitness = self.__compute_fitness(perf_biases, novelty_scores)

            offspring = [
                Instance(
                    variables=offspring_instances[i],
                    fitness=offspring_fitness[i],
                    descriptor=descriptors[i],
                    portfolio_scores=portfolio_scores[i],
                    p=perf_biases[i],
                    s=novelty_scores[i],
                    features=features[i] if features is not None else None,
                )
                for i in range(len(offspring_instances))
            ]

            feasible_idx = np.where(perf_biases > 0)[0]
            self._novelty_search.archive.extend(
                instances=[offspring[i] for i in feasible_idx],
                descriptors=descriptors[feasible_idx],
                novelty_scores=novelty_scores[feasible_idx],
            )
            if self._ns_solution_set is not None:
                ns_scores = self._ns_solution_set(instances_descriptors=descriptors)
                self._ns_solution_set.archive.extend(
                    instances=[offspring[i] for i in feasible_idx],
                    descriptors=descriptors[feasible_idx],
                    novelty_scores=ns_scores[feasible_idx],
                )

            self._population = self.replacement(self._population, offspring)
            self._logbook.update(
                generation=pgen,
                population=self._population,
                feedback=verbose,
            )

        if verbose:
            print(f"\r{' ' * 80}\r", end="")

        archive = (
            self._ns_solution_set.archive
            if self._ns_solution_set is not None
            else self._novelty_search.archive
        )
        return GenResult(
            target=self._portfolio[0].__name__,
            instances=archive,
            history=self._logbook,
        )

    def _llm_generate(self, n: int) -> list[np.ndarray]:
        """Query the LLM to produce *n* new instance variable vectors.

        The method builds a structured prompt from the current population and
        elite sets, parses the JSON reply, clips values to domain bounds and
        falls back to random generation for any instance the model fails to
        produce.

        Returns
        -------
        list[np.ndarray]
            List of *n* variable arrays, each of shape ``(num_vars,)``.
        """
        domain_desc = self._build_domain_description()
        pop_sample = self._build_population_sample()
        elite_disc = self._format_instances(self._elite_discriminatory, label="perf")
        elite_nov = self._format_instances(self._elite_novel, label="novelty")

        user_msg = _USER_TEMPLATE.format(
            domain_description=domain_desc,
            sample_size=self._pop_size,
            population_sample=pop_sample,
            elite_discriminatory=elite_disc,
            elite_novel=elite_nov,
            n=n,
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=_SYSTEM_PROMPT.format(n=n),
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            proposed = json.loads(raw)
        except Exception:
            proposed = []

        result: list[np.ndarray] = []
        bounds = self._domain.bounds  # sequence of (lo, hi) per variable
        num_vars = len(bounds)

        for raw_inst in proposed[:n]:
            try:
                vec = np.asarray(raw_inst, dtype=float)
                if vec.shape != (num_vars,):
                    raise ValueError("Shape mismatch")
                lo = np.array([b[0] for b in bounds], dtype=float)
                hi = np.array([b[1] for b in bounds], dtype=float)
                vec = np.clip(vec, lo, hi)
                result.append(vec)
            except Exception:
                result.append(self._domain.generate_instances(1))

        if len(result) < n:
            result.append(self._domain.generate_instances(n - len(result)))

        return result

    def _build_domain_description(self) -> str:
        """Return a JSON-serialisable dict describing the domain."""
        bounds = self._domain.bounds
        desc = {
            "num_variables": len(bounds),
            "bounds": [
                {"index": i, "lo": b[0], "hi": b[1]} for i, b in enumerate(bounds)
            ],
        }
        desc["description"] = "Travelling Salesman Problem"
        return json.dumps(desc, indent=2)

    def _build_population_sample(self) -> str:
        """Sample up to *population_sample_size* instances from the current
        population, keeping both high-fitness and low-fitness members so the
        model sees the full spread."""
        pop = self._population
        if not pop:
            return "[]"

        k = min(self.population_sample_size, len(pop))
        # Sort by fitness descending and interleave top / bottom halves
        sorted_pop = sorted(pop, key=lambda inst: inst.fitness, reverse=True)
        top = sorted_pop[: k // 2]
        bottom = sorted_pop[-(k - k // 2) :]
        sample = top + bottom

        out = []
        for inst in sample:
            out.append(
                {
                    "variables": [round(float(v), 6) for v in inst],
                    "performance_bias": round(float(inst.p), 6)
                    if hasattr(inst, "p")
                    else None,
                    "novelty": round(float(inst.s), 6) if hasattr(inst, "s") else None,
                    "fitness": round(float(inst.fitness), 6),
                }
            )
        return json.dumps(out, indent=2)

    @staticmethod
    def _format_instances(
        instances: Sequence[Instance],
        label: str,
    ) -> str:
        """Serialise a list of elite instances to JSON for the prompt."""
        if not instances:
            return "[]"
        out = []
        for inst in instances:
            out.append(
                {
                    "variables": [round(float(v), 6) for v in inst],
                    label: round(float(inst.p if label == "perf" else inst.s), 6),
                }
            )
        return json.dumps(out, indent=2)

    def _update_elites(
        self,
        population: list[Instance],
        perf_biases: np.ndarray,
    ) -> None:
        """Maintain two elite sets: most discriminatory and most novel."""
        k = min(self.elite_size, len(population))
        # Most discriminatory: highest performance bias
        disc_idx = np.argsort(perf_biases)[-k:][::-1]
        self._elite_discriminatory = [population[i] for i in disc_idx]
        # Most novel: highest stored novelty score (inst.s), fallback to fitness
        try:
            novelty_vals = np.array([inst.s for inst in population], dtype=float)
        except AttributeError:
            novelty_vals = np.array([inst.fitness for inst in population], dtype=float)
        nov_idx = np.argsort(novelty_vals)[-k:][::-1]
        self._elite_novel = [population[i] for i in nov_idx]
