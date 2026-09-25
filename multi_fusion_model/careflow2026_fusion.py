"""CaReFlow released-code adaptation for cached aligned feature sequences.

Audio/vision representations are transported to language space by time-conditioned
velocity fields; fusion uses deterministic Euler integration and concatenation.
Relaxed forward matching and backward flow matching train jointly with prediction.
Backward matching uses the released code's stochastic same-pair one-step estimate;
the prediction path uses separate deterministic Euler integration.
There is no separate pretrained teacher or missing-modality feature generator.
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
from multimodal_suite.rectified_flow import CaReFlow2026Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data

_WEIGHTS = {"careflow_forward_audio": .1, "careflow_forward_vision": .1,
            "careflow_backward_audio": .1, "careflow_backward_vision": .1}
_OPTIONS = {"careflow_steps": 2, "careflow_pair_ratio": 4,
            "careflow_margin": 1e-4, "careflow_fusion_hidden": 128}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="careflow2026", init=False)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 1
    dropout: float = .2
    pretrain_epochs: int = 0
    auxiliary_weights: dict = field(default_factory=lambda: dict(_WEIGHTS))
    pretrain_auxiliary_weights: dict = field(default_factory=dict)
    model_options: dict = field(default_factory=lambda: dict(_OPTIONS))
    output_root: str = "results/second_question/careflow2026_fusion"

    def __post_init__(self):
        self._merge()

    def _merge(self):
        for name, defaults in (("model_options", _OPTIONS), ("auxiliary_weights", _WEIGHTS)):
            values = getattr(self, name)
            if not isinstance(values, dict) or set(values) - set(defaults):
                raise ValueError(f"{name} must contain only recognized CaReFlow keys: {list(defaults)}")
            setattr(self, name, {**defaults, **values})
        if not isinstance(self.pretrain_auxiliary_weights, dict) or self.pretrain_auxiliary_weights:
            raise ValueError("CaReFlow trains jointly; pretrain_auxiliary_weights must be {}")

    def validate(self):
        self._merge()
        if self.model_name != "careflow2026":
            raise ValueError("Use model_name='careflow2026' for the dedicated entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs != 0):
            raise ValueError("CaReFlow pretrain_epochs must be integer 0: training is joint")
        self.pretrain_epochs = 0
        for key in ("careflow_steps", "careflow_pair_ratio", "careflow_fusion_hidden"):
            value = self.model_options[key]
            minimum = 0 if key == "careflow_pair_ratio" else 1
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
            self.model_options[key] = int(value)
        value = self.model_options["careflow_margin"]
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("careflow_margin must be finite and nonnegative")
        self.model_options["careflow_margin"] = float(value)
        for key, value in self.auxiliary_weights.items():
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"auxiliary_weights[{key!r}] must be finite and nonnegative")
            self.auxiliary_weights[key] = float(value)
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's CaReFlow Config for model-specific validation")
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
    """Recompute encoders, forward ODEs and fusion under fixed-mask perturbations.

    Flow time is an integration parameter, not a media timestamp or attribution.
    Slot sensitivity still refers to the original cached input sequence positions.
    """
    result = _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)
    result["includes_flow_reintegration"] = True
    return result
