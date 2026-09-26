"""Predeclared, matched hyperparameter candidates for the #79 action-contract experiment.

The #79 comparison varies one thing, the action contract. Running two
independent adaptive (TPE) studies per algorithm would let each contract's
results steer which configurations the other contract never sees, so the
comparison would confound action semantics with search luck. Instead the
candidate configurations are **generated once, before any training, and
frozen**; ``delta-v1`` and ``desired-replicas-v1`` then train and validate the
exact same 20 DQN and 20 PPO configurations.

Generation method ``optuna-random-sampler-v1``: for each algorithm, an
in-memory Optuna study with ``RandomSampler(seed=42)`` asks 20 trials of the
existing v1 search space (``dqn-search-v1`` / ``ppo-search-v1``, via the same
``suggest_hyperparameters`` functions as the v1 studies) and tells each a
constant. Nothing is trained or evaluated, so the candidates cannot depend on
any controller result, and their order is the generation order. The frozen
JSON (``benchmarks/v1/action-semantics-candidates-v1.json``) is the source of
truth; regenerating it with the same Optuna version reproduces it exactly.

    python -m scalerl.tuning.candidates --output benchmarks/v1/action-semantics-candidates-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from scalerl.benchmarks import load_benchmark_manifest
from scalerl.training.dqn import DQN_CONFIG_VERSION, DQNHyperparameters
from scalerl.training.ppo import PPO_CONFIG_VERSION, PPOHyperparameters
from scalerl.tuning import dqn as dqn_tuning
from scalerl.tuning import ppo as ppo_tuning

if TYPE_CHECKING:
    import optuna

CANDIDATE_SET_VERSION: Final = "matched-action-candidates-v1"
GENERATION_METHOD: Final = "optuna-random-sampler-v1"
GENERATION_SEED: Final = 42
CANDIDATES_PER_ALGORITHM: Final = 20
CandidateAlgorithm = Literal["dqn", "ppo"]
_HYPERPARAMETERS: Final[dict[str, type[DQNHyperparameters] | type[PPOHyperparameters]]] = {
    "dqn": DQNHyperparameters,
    "ppo": PPOHyperparameters,
}


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class HyperparameterCandidate(_Strict):
    """One predeclared configuration: the sampled search-space values and the full settings."""

    candidate_id: str = Field(min_length=1)
    index: int = Field(ge=0)
    sampled_params: dict[str, JsonValue]
    hyperparameters: dict[str, JsonValue]


class AlgorithmCandidates(_Strict):
    algorithm: CandidateAlgorithm
    config_version: str
    search_space_version: str
    candidates: tuple[HyperparameterCandidate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        for position, candidate in enumerate(self.candidates):
            if candidate.index != position:
                raise ValueError("candidates must be listed in generation order")
            if candidate.candidate_id != candidate_id(self.algorithm, position):
                raise ValueError(f"unexpected candidate id {candidate.candidate_id!r}")
            _HYPERPARAMETERS[self.algorithm].model_validate(candidate.hyperparameters, strict=False)
        return self

    def get(self, candidate_id: str) -> HyperparameterCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(f"unknown {self.algorithm} candidate {candidate_id!r}")


class CandidateSet(_Strict):
    """The frozen matched candidate set; ``candidate_set_id`` hashes its content."""

    candidate_set_version: Literal["matched-action-candidates-v1"] = CANDIDATE_SET_VERSION
    generation_method: Literal["optuna-random-sampler-v1"] = GENERATION_METHOD
    generation_seed: int
    optuna_version: str
    benchmark_version: str
    independent_of_results: Literal[True] = True
    dqn: AlgorithmCandidates
    ppo: AlgorithmCandidates

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.dqn.algorithm != "dqn" or self.ppo.algorithm != "ppo":
            raise ValueError("dqn/ppo candidate lists are mislabeled")
        if self.benchmark_version != load_benchmark_manifest().version:
            raise ValueError(f"candidate set benchmark {self.benchmark_version!r} is not installed")
        return self

    @property
    def candidate_set_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def for_algorithm(self, algorithm: str) -> AlgorithmCandidates:
        if algorithm == "dqn":
            return self.dqn
        if algorithm == "ppo":
            return self.ppo
        raise ValueError(f"unknown algorithm {algorithm!r}")

    def hyperparameters(
        self, algorithm: str, candidate_id: str
    ) -> DQNHyperparameters | PPOHyperparameters:
        candidate = self.for_algorithm(algorithm).get(candidate_id)
        return _HYPERPARAMETERS[algorithm].model_validate(candidate.hyperparameters, strict=False)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> CandidateSet:
        return cls.model_validate_json(Path(path).read_text())


def candidate_id(algorithm: str, index: int) -> str:
    return f"{algorithm}-c{index:02d}"


def _sample(
    algorithm: CandidateAlgorithm,
    suggest: Callable[[optuna.Trial], Any],
    *,
    count: int,
    seed: int,
) -> tuple[HyperparameterCandidate, ...]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    candidates = []
    for index in range(count):
        trial = study.ask()
        hyperparameters = suggest(trial)
        study.tell(trial, 0.0)  # constant: nothing is evaluated, so nothing can steer sampling
        candidates.append(
            HyperparameterCandidate(
                candidate_id=candidate_id(algorithm, index),
                index=index,
                sampled_params=dict(trial.params),
                hyperparameters=hyperparameters.as_params(),
            )
        )
    return tuple(candidates)


def generate_candidate_set(
    *, count: int = CANDIDATES_PER_ALGORITHM, seed: int = GENERATION_SEED
) -> CandidateSet:
    """Sample ``count`` configurations per algorithm from the v1 search spaces."""
    import optuna

    return CandidateSet(
        generation_seed=seed,
        optuna_version=optuna.__version__,
        benchmark_version=load_benchmark_manifest().version,
        dqn=AlgorithmCandidates(
            algorithm="dqn",
            config_version=DQN_CONFIG_VERSION,
            search_space_version=dqn_tuning.SEARCH_SPACE_VERSION,
            candidates=_sample("dqn", dqn_tuning.suggest_hyperparameters, count=count, seed=seed),
        ),
        ppo=AlgorithmCandidates(
            algorithm="ppo",
            config_version=PPO_CONFIG_VERSION,
            search_space_version=ppo_tuning.SEARCH_SPACE_VERSION,
            candidates=_sample("ppo", ppo_tuning.suggest_hyperparameters, count=count, seed=seed),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Generate {CANDIDATE_SET_VERSION} (#79).")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v1/action-semantics-candidates-v1.json")
    )
    args = parser.parse_args(argv)
    candidate_set = generate_candidate_set()
    candidate_set.save(args.output)
    print(f"{CANDIDATE_SET_VERSION} {candidate_set.candidate_set_id} written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
