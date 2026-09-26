"""Dedicated M3ER entry for cached aligned features and sample-level targets.

CCA and proxy maps are fitted on normalized TRAIN observations only. The default
training path discards screened features; proxies are generated during evaluation.
Ridge regression is an explicit proxy adaptation, not the paper's paired-basis
coefficient-transfer algorithm and not a claim of mathematical equivalence.

The generic runtime supplies training, checkpoints and inference. Its cached-
feature ablation reruns the complete forward pass: CCA decisions can change and
proxy features can be regenerated after each perturbation. A reported score
change therefore includes changes in routing and imputation, not just removal of
one fixed contribution. It is not raw-word/audio/frame causal attribution.
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
from multimodal_suite.robust_pretraining import M3ER2020Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data


_DEFAULT_MODEL_OPTIONS = {
    "m3er_beta": 2.0,
    "m3er_cca_threshold": 0.0,
    "m3er_cca_components": 16,
    "m3er_ridge": 0.01,
    "m3er_fit_max_slots": 8192,
    "m3er_fit_seed": 1729,
    "m3er_proxy_during_training": False,
    "m3er_memory_dim": 128,
    "m3er_classifier_hidden": 64,
}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="m3er2020", init=False)
    d_model: int = 32
    pretrain_epochs: int = 0
    model_options: dict = field(default_factory=lambda: dict(_DEFAULT_MODEL_OPTIONS))
    auxiliary_weights: dict = field(default_factory=lambda: {
        "m3er_multiplicative": 1.0, "m3er_unimodal_regression": 0.1,
    })
    output_root: str = "results/second_question/m3er2020_fusion"

    def __post_init__(self):
        self._merge_model_options()

    def _merge_model_options(self):
        if not isinstance(self.model_options, dict):
            raise ValueError("model_options must be a dictionary of recognized M3ER settings")
        unknown = set(self.model_options) - set(_DEFAULT_MODEL_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown M3ER model_options: {list(unknown)!r}")
        self.model_options = {**_DEFAULT_MODEL_OPTIONS, **self.model_options}

    def validate(self):
        self._merge_model_options()
        if self.model_name != "m3er2020":
            raise ValueError("Use model_name='m3er2020' for the dedicated M3ER entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs != 0):
            raise ValueError("M3ER pretrain_epochs must be 0: CCA fitting is not unsupervised gradient pretraining")
        self.pretrain_epochs = 0
        numeric_rules = {
            "m3er_beta": (0.0, None, False),
            "m3er_cca_threshold": (-1.0, 1.0, False),
            "m3er_ridge": (0.0, None, True),
        }
        for name, (minimum, maximum, strict_minimum) in numeric_rules.items():
            value = self.model_options[name]
            valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            if valid:
                valid = value > minimum if strict_minimum else value >= minimum
                valid = valid and (maximum is None or value <= maximum)
            if not valid:
                raise ValueError(f"Invalid {name}: expected a finite value in its permitted range")
            self.model_options[name] = float(value)
        for name, minimum in (("m3er_cca_components", 2), ("m3er_fit_max_slots", 3),
                              ("m3er_fit_seed", 0), ("m3er_memory_dim", 1),
                              ("m3er_classifier_hidden", 1)):
            value = self.model_options[name]
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Invalid {name}: expected an integer >= {minimum}")
            self.model_options[name] = int(value)
        if not isinstance(self.model_options["m3er_proxy_during_training"], bool):
            raise ValueError("m3er_proxy_during_training must be boolean")
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    """Supervised fit with train-only CCA/proxy preprocessing and valid selection."""
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's M3ER Config so model-specific settings are validated")
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
    """Ablate cached groups, reevaluating CCA routing and regenerating proxies.

    Changes include downstream reliability/imputation responses; they are not
    additive contributions or interventions on raw media. Original positions and
    generated proxy provenance must remain distinguishable in any explanation.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)
