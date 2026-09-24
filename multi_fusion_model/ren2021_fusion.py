"""Ren-only notebook entry, with Poria-style history fields.

This is an IMAN-inspired within-sample, single-entity adaptation. The Ren paper's
full equations and speaker/conversation inputs are not available in this setup.
"""
import json
from dataclasses import dataclass, field
from multimodal_suite import (
    Config as _BaseConfig, fit_experiment as _fit_experiment,
    explain_feature_groups, load_experiment_model, predict_split,
)
from multimodal_suite.attention_models import Ren2021Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data

@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="ren2021", init=False)
    pretrain_epochs: int = 0
    output_root: str = "results/second_question/ren2021_fusion"

    def validate(self):
        super().validate()
        if self.model_name != "ren2021" or self.pretrain_epochs != 0:
            raise ValueError("This Ren entry uses joint training with pretrain_epochs=0")


def fit_experiment(data, config=None, mask_overrides=None):
    config = config or Config()
    if config.model_name != "ren2021":
        raise ValueError("Use a Ren Config for this model entry")
    experiment = _fit_experiment(data, config, mask_overrides=mask_overrides)
    for record in experiment.history:
        record["train_loss"] = record["train"]["total"]
    (experiment.run_dir / "history.json").write_text(
        json.dumps(experiment.history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return experiment
