# IEEE Access Experiment Summary

- Created at: 2026-06-18T16:08:45
- Dataset: `data\synthetic\extraction_run_20260618_085129\external_test_120\mentions_external_test_120.jsonl`
- Source mentions: `data\synthetic\extraction_run_20260618_085129\mentions_extracted.jsonl`
- Threshold: 0.62
- Same-project floor: 0.96

## Main Result

| Condition | B-cubed F1 | Pairwise F1 | False Merge | False Split | Pred Clusters |
|---|---:|---:|---:|---:|---:|
| baseline_signature | 0.381458 | 0.160714 | 2 | 374 | 89 |
| human_partial | 0.610096 | 0.487085 | 0 | 278 | 60 |
| human_full | 0.705651 | 0.605442 | 0 | 232 | 45 |
| human_noisy | 0.580222 | 0.444023 | 0 | 293 | 62 |

## Recommended IEEE Access Claim

The best human-authored knowledge condition is `human_full`, improving B-cubed F1 from 0.381458 to 0.705651 while reducing false merge from 2 to 0.

## Multi-subset Robustness

| Condition | B-cubed F1 mean | B-cubed F1 std | False merge mean | False split mean |
|---|---:|---:|---:|---:|
| baseline_signature | 0.391876 | 0.010496 | 2.000 | 370.800 |
| human_full | 0.716140 | 0.018274 | 0.000 | 223.200 |
| human_noisy | 0.587139 | 0.006406 | 0.000 | 286.800 |
| human_partial | 0.616254 | 0.010167 | 0.000 | 273.800 |
