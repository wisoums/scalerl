"""Hyperparameter search with Optuna, tied to ScaleRL's MLflow and benchmark contracts.

``StudySpec`` and ``require_tuning_workloads`` work without Optuna;
``run_study`` needs the ``tuning`` extra and raises an ``ImportError`` with an
install hint otherwise.
"""

from scalerl.tuning.spec import (
    GridValue,
    PrunerName,
    SamplerName,
    StudySpec,
    require_tuning_workloads,
    safe_storage_label,
)
from scalerl.tuning.study import (
    INSTALL_HINT,
    RUN_IDS_ATTR,
    TRIAL_STATE_TAG,
    Objective,
    TrialContext,
    run_study,
)

__all__ = [
    "INSTALL_HINT",
    "RUN_IDS_ATTR",
    "TRIAL_STATE_TAG",
    "GridValue",
    "Objective",
    "PrunerName",
    "SamplerName",
    "StudySpec",
    "TrialContext",
    "require_tuning_workloads",
    "run_study",
    "safe_storage_label",
]
