# Robustness and pretraining models: implementation boundary

These are runnable, independently written **feature-level adaptations**, not
official pretrained models or numerical reproductions of published results.
Inputs are cached, aligned features and explicit observation masks. Original
modality extractors are not retrained. The target task, feature dimensions,
hidden width, data split and handling of missing observations differ from the
original experiments.

## M3ER2020Model

Verified primary source: [M3ER, sections 3.3–3.5 and 4.2](https://arxiv.org/html/1911.05659v2).
The paper-linked [author repository](https://github.com/TrishaMittal/M3ER) returned
404 during verification; no unavailable official code was claimed as inspected.

The implementation has three distinct mechanisms: fitted CCA screens modalities,
paired linear maps generate proxies, and recurrent modality states are multiplied
and combined with an updated multimodal memory. Paper Eq. (3),
`-sum_i p_i(y) ** (beta/2) * log p_i(y)`, is an auxiliary classification objective.
It is **not** a probability-product inference rule. Prediction uses the learned
classifier on hidden-state product and memory.

Explicit adaptations: regularized CCA and ridge proxy maps operate on aligned
slots; ridge replaces the paper's basis/nearest-neighbour construction. This
MFN-style memory is an independent compact implementation, not an exact MFN
checkpoint. Added unimodal heads provide Eq. (3) supervision alongside the
trainer's prediction loss. Regression uses unimodal MAE instead of pretending
Eq. (3) applies to continuous targets. Proxies are enabled during training by
default (`m3er_proxy_during_training=false` restores inference-only generation).

`fit_preprocessing(train_dataset)` must receive only the normalized training
dataset. CCA means, transforms, regression maps, fit counts and fitted flags are
registered buffers, so checkpoint reload requires no refitting. All labels are
ignored when fitting. Default options: components 16, ridge 0.01, correlation
threshold 0, maximum fitted slots per pair 8192, sampling seed 1729, beta 2.
These are implementation defaults, not reported paper hyperparameters.

Every fitting attempt first clears all prior pair statistics, transforms and fit
counts, including when a reused model is given an empty, invalid, or disjoint
dataset. An exception never restores old transforms. `preprocessing_attempted`
records that fitting was invoked; `preprocessing_fitted` means at least one pair
successfully fitted, rather than merely that fitting was called. Forward audit
includes `cca_fit_attempted`, pair-keyed `cca_pair_fitted` and
`cca_pair_fit_count`. Successful pairs from a partially failed new fitting run
contain only new data; the exception is still raised to the caller.

`proxy_sources[B,L,target_modality,source_modality]` identifies synthetic inputs;
the order is text/audio/vision. `effective_masks` identifies retained observations,
and `cca_fallback` identifies slots for which all assessed pairs failed the
threshold and one original modality was retained. The fallback is a documented
robustness policy, not proof that the retained input is reliable. A missing
modality has multiplicative identity 1 in the final product. If preprocessing
has not run, unfitted pairs do not screen or generate proxies; benchmark training
must call fitting. A pair with insufficient co-observations remains disabled.

## MEmoBERT2022Model

Verified primary source: [MEmoBERT, sections 2–3](https://arxiv.org/html/2111.00865v1);
[author repository](https://github.com/AIM3-RUC/MEmoBert).

The implementation retains a joint Transformer, modality and position embeddings,
conditional masking of one modality per example, and span reconstruction. A
learned soft prompt and a dedicated label-query position produce the prediction.
Training returns separate masked text/audio/vision MSE losses; reconstruction
targets are detached observed input features. Defaults: masking fraction 0.15,
span length 3, auxiliary weights 1, and three auxiliary-only pretraining epochs
before supervised training. Explicit `pretrain_epochs=0` skips that stage.

This is **not** pretrained MEmoBERT: it does not load a BERT checkpoint, predict
vocabulary tokens, encode the literal prompt "I am [MASK].", or reproduce whole
word masking. It has no FER teacher, so the original visual distribution KL task
is deliberately absent. A soft label verbalizer is trained from scratch. Cached
BERT features can already expose a masked word through surrounding contextual
vectors, making feature reconstruction easier than raw-input masked language
modeling. Any result must be labeled a cached-feature adaptation.

## HyCon2022Model

Verified primary source: [HyCon arXiv:2109.01797v1, equations 6–14](https://arxiv.org/html/2109.01797v1).
The implemented source version is explicitly **v1**. The final journal abstract
mentions an additional pair-selection mechanism; that mechanism is not specified
in this source and is not claimed to be reproduced.

Three modality encoders produce last-valid-slot vectors, followed by addition
and a prediction head. IAMCL compares different samples within a modality;
IEMCL compares different samples across modalities; both use label-defined
positive pairs. Their negative ratio objectives include positive-pair refinement.
SCL compares different modalities of the same sample against margin 0.8; it is
**semi-contrastive**, not semi-supervised. Inference ignores all targets.

Explicit adaptations: text uses cached BERT followed by a small Transformer,
rather than BERT finetuning; last valid positions replace literal final padding
positions. Softplus makes vectors nonnegative before L2 similarity normalization,
because normalization alone cannot ensure nonnegative dot products. Missing
modality pairs and anchors without positives are omitted safely. Classification
uses supplied class IDs; regression uses binary `target >= 0` contrastive groups.
All three auxiliary weights default to 1. Ratio losses can legitimately be
negative; replacing them with cross-entropy/SupCon would change the implemented
source objective.

## Shared interpretation and audit limits

All-invalid samples are rejected; holes and entire missing branches are supported.
Masked inputs are excluded from both prediction paths and reconstruction targets.
The extra returned tensors support audit, but attention or proxy provenance is
not an attribution guarantee. Original-input interventions must re-extract
features, especially for contextual BERT. Preserve word/time/frame mappings
outside the model and label proxies as generated rather than observed evidence.

## Local verification

Synthetic classification and regression batches passed forward/backward finite
checks for all three classes. Checks included internal holes, an entirely absent
modality, invariance to replacing masked values with 5000, evaluation independence
from supplied targets, checkpoint round-trip equivalence (including fitted CCA
and proxy buffers), and rejection of fully unobserved samples. These checks
validate implementation behavior; they are not dataset accuracy results.
