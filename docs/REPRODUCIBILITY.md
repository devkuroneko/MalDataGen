# Reproducibility

Last updated: 2026-07-15

## Minimum Run Record

Each experiment should preserve:

- command line and `command_manifest.json`;
- source dataset paths;
- train/valid/test split names;
- class counts and sample plan;
- random seed;
- model type and model parameters;
- feature-transform configuration;
- synthetic manifest paths;
- evaluation protocol IDs;
- classifier configuration;
- result JSON path;
- quality-audit JSON path when generated.

## Determinism

Use explicit `--random_state` and record generated sample counts per split. For synthetic batches, keep the manifest with per-class batch paths and shapes.

## Comparison Rules

Before/after comparisons should use the same:

- dataset split;
- class subset;
- feature transform policy;
- sample plan;
- model seed;
- evaluation protocol;
- classifier configuration.

When byte-identical output comparison is not feasible, compare class counts, row counts, protocol coverage and metric keys.

## Local Artifacts

The repository ignores local generated outputs and large datasets by default. Ignoring is not deletion. Manual archival or removal must follow `docs/audits/ARTIFACT_RETENTION_PLAN.md`.
