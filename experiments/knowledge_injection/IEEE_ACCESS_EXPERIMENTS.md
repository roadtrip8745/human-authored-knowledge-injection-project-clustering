# IEEE Access Experiment Runner

This runner adds the minimum robustness evidence recommended for an IEEE Access submission without changing the main proposed method.

Script:

```powershell
python experiments\knowledge_injection\run_ieee_access_experiments.py
```

Latest verified output:

```text
data\synthetic\ieee_access_runs\run_20260618_142201
```

## Included Experiments

1. Main comparison
   - `baseline_signature`
   - `human_partial`
   - `human_full`
   - `human_noisy`

2. Threshold sensitivity
   - thresholds: `0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80`
   - output: `threshold_sweep.csv`

3. Same-project floor sensitivity
   - floors: `0.90, 0.94, 0.96, 0.98`
   - output: `floor_sweep.csv`

4. Multi-subset robustness
   - source: 398 matched mentions
   - sample size: 120
   - seeds: `20260618, 20260619, 20260620, 20260621, 20260622`
   - outputs:
     - `multi_subset_results.csv`
     - `multi_subset_summary.csv`
     - `multi_subset_samples/`

5. Human-authored full ablation
   - remove cannot-merge
   - remove same-project floor
   - remove weak boost
   - remove generic mention safety
   - output: `ablation_human_full.csv`

6. Synthetic-real distribution similarity
   - compares Outlook aggregate profile bins with synthetic raw emails
   - outputs:
     - `distribution_similarity.csv`
     - `synthetic_distribution_bins.csv`

7. Classic baseline sanity checks
   - exact
   - jaccard
   - lexical
   - proposed_signature
   - output: `classic_baselines.csv`

## Main Paper Recommendation

For IEEE Access, use these in the main paper:

- Main result table: `main_comparison.csv`
- Multi-subset robustness table: `multi_subset_summary.csv`
- Ablation table: `ablation_human_full.csv`
- Threshold sensitivity figure: `threshold_sweep.csv`

Use these as appendix or supporting material:

- `floor_sweep.csv`
- `distribution_similarity.csv`
- `classic_baselines.csv`

Important caveat for `classic_baselines.csv`:

Some simple baselines such as Jaccard can show high F1 by aggressive merging, but they produce many false merges. In enterprise reporting, false merges are high-risk because they combine different projects into one report item. The proposed MD method should be presented as a precision-preserving, false-merge-controlled method rather than as a purely recall-maximizing method.

## Latest Verified Key Results

Main 120-record subset:

| Condition | B-cubed F1 | Pairwise F1 | False Merge | False Split |
|---|---:|---:|---:|---:|
| baseline_signature | 0.381458 | 0.160714 | 2 | 374 |
| human_partial | 0.610096 | 0.487085 | 0 | 278 |
| human_full | 0.705651 | 0.605442 | 0 | 232 |
| human_noisy | 0.580222 | 0.444023 | 0 | 293 |

Five balanced 120-record subsets:

| Condition | B-cubed F1 mean | B-cubed F1 std | False Merge mean | False Split mean |
|---|---:|---:|---:|---:|
| baseline_signature | 0.391876 | 0.010496 | 2.000 | 370.800 |
| human_partial | 0.616254 | 0.010167 | 0.000 | 273.800 |
| human_full | 0.716140 | 0.018274 | 0.000 | 223.200 |
| human_noisy | 0.587139 | 0.006406 | 0.000 | 286.800 |

Ablation highlight:

- Removing same-project floor drops `human_full` B-cubed F1 from `0.705651` to `0.408668`.
- Removing cannot-merge introduces 4 false merges.
- Removing generic mention safety increases B-cubed F1 to `0.884389` but creates 10 false merges, supporting the conservative design choice.
