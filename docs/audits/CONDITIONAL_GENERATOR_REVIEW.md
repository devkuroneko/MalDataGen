# Conditional Generator Review

Date: 2026-07-15

## Scope

Audit and correction of the conditional generation path used by AppClassNet-style multiclass runs:

label original -> local remap -> loader -> one-hot -> encoder -> decoder/generator -> writer -> manifest -> evaluation.

## Root Cause

The main defect found was in the conditional GAN training step. Training batches already pass labels as one-hot tensors shaped `(batch, K)`, but `AdversarialAlgorithm.train_step()` expanded the label axis to `(batch, K, 1)`. That shape no longer matched the generator/discriminator declared conditional inputs and could cause incorrect broadcasting or failed conditioning.

A second defect was in split generation planning. `_generation_metadata_for_split()` rebuilt classes using `range(number_classes)` when explicit train/test synthetic quotas were used. For AppClassNet subsets this could reintroduce classes outside the effective subset.

A third audit defect was that synthetic label/quality checks treated `number_classes=200` as "all 200 classes must be generated" even when the effective run intentionally used a subset. The strict check now compares against requested classes from the generation plan.

## Label Trace

- Original labels are loaded by `NpyXYLoader` as 1D integer vectors and can be remapped explicitly from 1-based to zero-based.
- `DatasetBundle` and `SplitData` validate X/y row alignment, feature width, class range and schema hash.
- AppClassNet subset materialization writes `selected_original_classes`, `original_to_local_mapping`, `local_to_original_mapping` and `effective_num_classes`.
- Training labels are converted batch-wise or array-wise to one-hot with `K` columns.
- One-hot validation now checks shape `(n, K)` and `argmax(one_hot) == labels`.
- VAE encoder input remains `concat(features, label_embedding)`.
- VAE decoder input remains `concat(latent_vector, label_embedding)`.
- VAE/GAN generation calls decoder/generator with `(latent_vector, requested_label_one_hot)`.
- Writer manifests now record requested classes, generation plan, label mapping, batch sizes, training history, seed, code version, schema and transform metadata.
- Label audit verifies saved `y` matches requested class and requested class set.
- Quality audit verifies manifest labels and arrays against requested classes, not hidden thresholds.

## Files Changed

- `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py`
- `Engine/DataIO/LabelUtils.py`
- `Engine/DataIO/SyntheticBatchIO.py`
- `Engine/DataIO/SyntheticLabelAudit.py`
- `Engine/DataIO/SyntheticQualityAudit.py`
- `Engine/Models/GenerativeModels.py`
- `main.py`
- `tests/test_conditional_gan_labels.py`
- `tests/test_generation_strategy.py`
- `tests/test_one_hot_batch.py`
- `tests/test_synthetic_label_audit.py`
- `tests/test_synthetic_quality_audit.py`

## Before / After

Before:

- cGAN labels could be expanded from one-hot `(batch, K)` to `(batch, K, 1)`.
- Split-specific generation could request classes outside an effective subset.
- `number_classes=200` audits could fail subset runs for not generating classes intentionally absent from the subset.
- Manifests did not expose enough conditioning and training context to trace requested label to saved label.

After:

- cGAN labels are kept as rank-2 one-hot tensors; rank-1 labels are one-hot encoded only with a known class count.
- Generation plans are validated against zero-based labels and effective classes.
- Split plans preserve only requested effective classes.
- Label and quality audits compare generated classes to the requested plan.
- Decoder sensitivity probe is connected for active VAE/GAN-style generators in batch quality audit.
- Training history records configured epochs, recorded epochs, fit reached flag, LR, batch size, batches/updates, best epoch and early stopping metadata.

## Tests

Executed:

```bash
python -m pytest -q
```

Result:

```text
246 passed, 18 skipped, 9 warnings, 5 subtests passed
```

Focused tests added or extended:

- cGAN one-hot labels remain rank-2.
- one-hot argmax round-trip recovers labels.
- split generation plan preserves effective classes only.
- AppClassNet subset label audit allows requested subset and rejects missing requested classes.
- fixed-z decoder sensitivity distinguishes labels when the decoder uses labels and fails when it ignores them.
- sparse requested-class manifests pass quality validation.

## Remaining Limitations

- Full AppClassNet top-200 training was not run in this audit because it is expensive; validation used unit/integration coverage.
- Loss by class is only persisted when a model exposes it in Keras history; the current cGAN/VAE histories expose global losses.
- Existing collapse thresholds were not changed.
