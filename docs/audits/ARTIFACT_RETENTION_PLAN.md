# Artifact Retention Plan

Last updated: 2026-07-15

This plan classifies artifacts for cleanup. It does not authorize automatic deletion of datasets or results.

## Necessary

Keep in the repository or archive explicitly:

- source code and tests;
- `pyproject.toml`, `requirements.txt`, `Pipfile`, and other environment descriptors;
- manual documentation under `docs/`;
- audit reports under `docs/audits/`;
- small schema/example manifests required to reproduce commands;
- command manifests selected for publication;
- dataset licenses or README files when they are intentionally tracked.

## Reproducible

May be regenerated if command manifests, source data and seeds are preserved:

- timestamped `outputs/out_*` run directories;
- synthetic batch directories under `DataGenerated/synthetic_batches/`;
- plot outputs;
- generated model checkpoints for non-publication runs;
- generated quality-audit JSON files copied into output trees.

Do not remove without checking whether the exact source dataset and command manifest are still available.

## Archivable

Move to external archival storage before removal from the workspace:

- historical `results/` trees;
- large generated batches used in reports;
- run logs that explain published results;
- final result JSON files for completed campaigns;
- any artifact referenced by papers, notebooks, README examples or audit reports.

## Removable After Manual Confirmation

Safe candidates only after a human confirms no pending analysis depends on them:

- `__pycache__/` and `.pyc` files;
- `.pytest_cache/`;
- local virtual environments such as `.venv/`;
- temporary logs not referenced by reports;
- incomplete timestamped run directories;
- generated subsets under local subset caches;
- duplicate generated outputs that have a retained canonical archive.

## Never Remove Automatically

- raw datasets;
- converted datasets;
- historical results;
- user-provided logs such as `logs/real_resample_seeds/seed_0.log`;
- files with unclear provenance;
- files only suspected because of similar names.

## Manual Cleanup Checklist

1. Confirm the artifact category.
2. Confirm whether a manifest, command line and seed exist.
3. Confirm whether the artifact is referenced from docs, notebooks or reports.
4. Archive if historical value exists.
5. Remove only after review and record the action in a cleanup report.
