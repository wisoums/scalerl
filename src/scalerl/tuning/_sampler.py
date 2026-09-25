"""Trial-seeded sampling so resumed studies continue exactly like uninterrupted ones.

Optuna does not persist a sampler's random state. Rebuilding a seeded sampler
on resume would restart its random sequence and repeat earlier trials. Here
each trial gets a fresh inner sampler seeded from ``(study seed, trial number)``,
so trial *k* samples identically whether or not the study was interrupted
before it (given the same completed history).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


def trial_seed(seed: int, trial_number: int) -> int:
    """Deterministic, well-mixed seed for one trial of a seeded study."""
    return int(np.random.SeedSequence([seed, trial_number]).generate_state(1)[0])


class TrialSeededSampler(BaseSampler):
    """Delegate each trial to a sampler built with that trial's own seed."""

    def __init__(self, factory: Callable[[int], BaseSampler], seed: int) -> None:
        self._factory = factory
        self._seed = seed
        self._samplers: dict[int, BaseSampler] = {}
        self._lock = threading.Lock()

    def _sampler(self, trial: FrozenTrial) -> BaseSampler:
        with self._lock:
            sampler = self._samplers.get(trial.number)
            if sampler is None:
                sampler = self._factory(trial_seed(self._seed, trial.number))
                self._samplers[trial.number] = sampler
            return sampler

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._sampler(trial).before_trial(study, trial)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self._sampler(trial).infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._sampler(trial).sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._sampler(trial).sample_independent(study, trial, param_name, param_distribution)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._sampler(trial).after_trial(study, trial, state, values)
        with self._lock:
            self._samplers.pop(trial.number, None)

    def reseed_rng(self) -> None:
        """No-op: trial seeds are already independent, including under ``n_jobs > 1``."""
