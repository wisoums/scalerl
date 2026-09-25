"""Learned policies behind the common Controller interface (Stable-Baselines3).

Not imported by ``import scalerl``; importing it loads Stable-Baselines3 and
PyTorch (core dependencies) but never MLflow or Optuna.
"""

from scalerl.rl.sb3 import (
    BUNDLE_VERSION,
    COMPATIBILITY_FILE,
    METADATA_FILE,
    MODEL_FILE,
    ModelMetadata,
    SB3Controller,
    load_sb3_controller,
    read_model_bundle,
    save_model_bundle,
)

__all__ = [
    "BUNDLE_VERSION",
    "COMPATIBILITY_FILE",
    "METADATA_FILE",
    "MODEL_FILE",
    "ModelMetadata",
    "SB3Controller",
    "load_sb3_controller",
    "read_model_bundle",
    "save_model_bundle",
]
