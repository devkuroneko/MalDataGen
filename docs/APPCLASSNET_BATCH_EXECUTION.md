# AppClassNet Normal and Batches Execution

This guide shows the main commands for running AppClassNet top-200 experiments in compatibility mode (`normal`) and lower-memory mode (`batches`).

## Quick Commands

### A. Demo, normal mode

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode normal
```

Normal mode is the compatibility baseline. It keeps the existing CSV/full-matrix execution path and the legacy classifier configuration unless an explicit classifier override is passed.

### B. Demo, batches mode

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode batches \
  --use_mmap \
  --batch_size 8192 \
  --eval_batch_size 16384 \
  --generation_batch_size 8192 \
  --max_samples_per_class 1000
```

Batches mode reads AppClassNet `.npy` files with memory mapping and writes synthetic data incrementally. This avoids loading the full generated synthetic matrix for evaluation.

### C. Complete run, normal mode

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode normal \
  --full
```

`--full` makes the `sf` alias run the complete AppClassNet campaign set instead of the reduced demo campaigns.

### D. Complete run, batches mode

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode batches \
  --full \
  --use_mmap \
  --batch_size 8192 \
  --eval_batch_size 16384 \
  --generation_batch_size 8192 \
  --save_synthetic_format npy_batches
```

Use `npy_batches` for lower-memory synthetic persistence. Avoid `--materialize_synthetic` unless you explicitly want the old in-memory behavior.

## Compare Normal vs Batches

```bash
python3 run_appclassnet_top200.py -c sf --compare_modes
```

This launches the selected campaign twice: first with `--execution_mode normal`, then with `--execution_mode batches --use_mmap`. The runner writes separate output directories using `normal` and `batches` suffixes.

For a complete comparison:

```bash
python3 run_appclassnet_top200.py -c sf --full --compare_modes
```

## Interpreting Results

Normal mode is the baseline for backward compatibility. It is useful for checking that new code still matches the established CSV/full-matrix flow.

Batches mode is intended to reduce RAM usage. It uses memory-mapped AppClassNet arrays, incremental synthetic persistence, and batch-friendly classifiers for evaluation.

Accuracy metrics may change between normal and batches when the classifier differs. For example, normal mode may use the legacy classifier list while batches mode defaults to `decision_tree_subset` for AppClassNet. For a fairer comparison, use the same classifier family when possible, such as:

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode normal \
  --normal_classifier decision_tree_subset

python3 run_appclassnet_top200.py -c sf \
  --execution_mode batches \
  --use_mmap \
  --eval_classifier decision_tree_subset
```

If memory still runs out in batches mode, reduce batch sizes:

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode batches \
  --use_mmap \
  --batch_size 4096 \
  --eval_batch_size 4096 \
  --generation_batch_size 4096 \
  --max_samples_per_class 1000
```

Try `4096`, then `2048`, then `1024`.

## 48 GB RAM Recommendation

For a machine with 48 GB RAM, scale gradually:

1. Start with `--max_samples_per_class 500`.
2. Increase to `--max_samples_per_class 1000`.
3. Increase to `--max_samples_per_class 3000`.
4. Remove `--max_samples_per_class` only after the previous run fits comfortably.

For AppClassNet, prefer tree-based eval classifiers first:

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode batches \
  --use_mmap \
  --eval_classifier decision_tree_subset \
  --train_samples_per_class 1000 \
  --test_samples_per_class 500
```

`SGDClassifier` remains available through `--eval_classifier sgd` for very low RAM checks, but it is linear and can severely underestimate AppClassNet synthetic quality. Use `decision_tree_subset`, `extra_trees_subset`, or `random_forest_light` before drawing conclusions from synthetic-data metrics.

## Metrics Files

Each experiment writes `Results.json` and `metrics.json` under `EvaluationResults`. `metrics.json` includes:

- predictive metrics for `TR-TS` and `TS-TR`;
- classifier metadata, including whether `partial_fit` was used;
- resource usage by stage: loading, preprocessing, training, generation, evaluation, saving;
- total elapsed seconds;
- batches mode counters: largest batch processed, number of batches and effective batch size.
