"""EMOE official-code adaptation: dense routing and online feature distillation.

The released regression/sum model is adapted to cached aligned features,
observation masks, smaller raw-feature router bottleneck and optional multiclass
prediction. There is no sparse top-k dispatch or frozen pretrained teacher.
Unimodal supervised learning, inverse-error router targets, entropy balance and
stop-gradient feature distillation are jointly trained from the first epoch.
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
from multimodal_suite.emotion_experts import EMOE2025Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data

_WEIGHTS = {"emoe_unimodal": 1.0, "emoe_router_fit": 0.01,
            "emoe_router_entropy": 0.1, "emoe_distillation": 0.1}
_OPTIONS = {"emoe_router_hidden": 128, "emoe_temperature": 0.1,
            "emoe_reliability_epsilon": 0.1}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="emoe2025", init=False)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    pretrain_epochs: int = 0
    auxiliary_weights: dict = field(default_factory=lambda: dict(_WEIGHTS))
    pretrain_auxiliary_weights: dict = field(default_factory=dict)
    model_options: dict = field(default_factory=lambda: dict(_OPTIONS))
    output_root: str = "results/second_question/emoe2025_fusion"

    def __post_init__(self):
        self._merge()

    def _merge(self):
        for name, defaults in (("model_options", _OPTIONS), ("auxiliary_weights", _WEIGHTS)):
            values = getattr(self, name)
            if not isinstance(values, dict) or set(values) - set(defaults):
                raise ValueError(f"{name} must contain only recognized EMOE keys: {list(defaults)}")
            setattr(self, name, {**defaults, **values})
        if not isinstance(self.pretrain_auxiliary_weights, dict) or self.pretrain_auxiliary_weights:
            raise ValueError("EMOE trains jointly; pretrain_auxiliary_weights must be {}")

    def validate(self):
        self._merge()
        if self.model_name != "emoe2025":
            raise ValueError("Use model_name='emoe2025' for the dedicated entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs != 0):
            raise ValueError("EMOE pretrain_epochs must be integer 0: online teachers are trained jointly")
        self.pretrain_epochs = 0
        hidden = self.model_options["emoe_router_hidden"]
        if not isinstance(hidden, Integral) or isinstance(hidden, bool) or hidden <= 0:
            raise ValueError("emoe_router_hidden must be a positive integer")
        self.model_options["emoe_router_hidden"] = int(hidden)
        for key in ("emoe_temperature", "emoe_reliability_epsilon"):
            value = self.model_options[key]
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            self.model_options[key] = float(value)
        for key, value in self.auxiliary_weights.items():
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"auxiliary_weights[{key!r}] must be finite and nonnegative")
            self.auxiliary_weights[key] = float(value)
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's EMOE Config for model-specific validation")
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
    """Full-model cached-feature sensitivity, including changes in router weights.

    Masks/classes stay fixed. A routing weight is not a causal contribution, and
    replacing features can change both the expert representation and its weight.
    """
    result = _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)
    result["includes_router_reweighting"] = True
    return result
