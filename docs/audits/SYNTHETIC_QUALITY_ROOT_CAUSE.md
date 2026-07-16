# Synthetic Quality Root Cause

- Status: `completed`
- Conclusion: `multiple causes`
- Candidates: `conditional collapse, insufficient sample count`
- Classes: `200`
- Chance accuracy: `0.005000`
- Synthetic control: `real_resample`
- Generation strategy: `single_conditional`

## Evidence

- Schema hashes match: `True`
- Transform preserve+inverse no-op: `True`
- Collapse detected: `True`
- Collapse reasons: `['train: 200 classes have near-zero variance', 'test: 200 classes have near-zero variance']`
- Training warnings: `[]`
- Errors: `[]`

## Outputs

- JSON: `/home/gatopreto/PycharmProjects/MalDataGen/results/appclassnet_top200/batches/synthetic_quality_audit.json`
