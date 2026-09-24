"""Dedicated TransModality feature-level adaptation with joint reconstruction.

Two text-primary forward/backward translation cells supply four encoder streams;
these join the three modality context streams for sample-level classification or
regression. Four masked MAE reconstruction objectives regularize the model.

The original sequence of utterances is replaced by aligned slots within one
sample, using cached features and one sample label. Decoders read the complete
observed target context without a causal mask. This is an explicit noncausal,
observed-target adaptation, not an exact reproduction or a source-only generator
for absent modalities. Default training is joint, with no separate pretraining;
an explicitly requested warmup is an optional experiment, not a paper requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real

from multimodal_suite import (
    Config as _BaseConfig, fit_experiment as _fit_experiment,
    explain_feature_groups as _explain_feature_groups,
    load_experiment_model, predict_split,
)
from multimodal_suite.memory_translation import TransModality2020Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data


AUXILIARY_NAMES = (
    "translation_text_audio", "cycle_audio_text",
    "translation_text_vision", "cycle_vision_text",
)
_DEFAULT_AUXILIARY_WEIGHTS = {name: 0.1 for name in AUXILIARY_NAMES}
_DEFAULT_MODEL_OPTIONS = {"translation_ff_dim": 256}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="transmodality2020", init=False)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 1
    pretrain_epochs: int = 0
    auxiliary_weights: dict = field(default_factory=lambda: dict(_DEFAULT_AUXILIARY_WEIGHTS))
    model_options: dict = field(default_factory=lambda: dict(_DEFAULT_MODEL_OPTIONS))
    output_root: str = "results/second_question/transmodality2020_fusion"

    def __post_init__(self):
        self._merge_options()

    def _merge_options(self):
        if not isinstance(self.model_options, dict):
            raise ValueError("model_options must be a dictionary of recognized TransModality settings")
        unknown = set(self.model_options) - set(_DEFAULT_MODEL_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown TransModality model_options: {list(unknown)!r}")
        self.model_options = {**_DEFAULT_MODEL_OPTIONS, **self.model_options}
        for name in ("auxiliary_weights", "pretrain_auxiliary_weights"):
            weights = getattr(self, name)
            if not isinstance(weights, dict):
                raise ValueError(f"{name} must be a dictionary")
            unknown = set(weights) - set(AUXILIARY_NAMES)
            if unknown:
                raise ValueError(f"Unknown TransModality objectives in {name}: {list(unknown)!r}")
        self.auxiliary_weights = {**_DEFAULT_AUXILIARY_WEIGHTS, **self.auxiliary_weights}
        self.pretrain_auxiliary_weights = dict(self.pretrain_auxiliary_weights)

    def validate(self):
        self._merge_options()
        if self.model_name != "transmodality2020":
            raise ValueError("Use model_name='transmodality2020' for this entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs < 0):
            raise ValueError("pretrain_epochs must be a nonnegative integer; 0 selects joint training without warmup")
        self.pretrain_epochs = int(self.pretrain_epochs)
        width = self.model_options["translation_ff_dim"]
        if not isinstance(width, Integral) or isinstance(width, bool) or width <= 0:
            raise ValueError("translation_ff_dim must be a positive integer")
        self.model_options["translation_ff_dim"] = int(width)
        for name in ("auxiliary_weights", "pretrain_auxiliary_weights"):
            for objective, value in getattr(self, name).items():
                if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}[{objective!r}] must be finite and nonnegative")
                getattr(self, name)[objective] = float(value)
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    """Fit shared-runtime train/valid/test splits with model-specific validation."""
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's TransModality Config so objective and model options are validated")
    config.validate()
    experiment = _fit_experiment(data, config, mask_overrides=mask_overrides)
    for record in experiment.history:
        record["train_loss"] = record["train"]["total"]
        record["train_task_loss"] = record["train"]["supervised"]
    (experiment.run_dir / "history.json").write_text(
        json.dumps(experiment.history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return experiment


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32):
    """Cached-feature replacement through the full translation/fusion network.

    Masks and the target/reference class stay fixed. Because observed target
    context is a decoder input and cached BERT vectors contain context, these
    changes are neither isolated translation contributions nor raw-media causal
    effects. Reconstruction quality does not validate missing-modality synthesis.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)
