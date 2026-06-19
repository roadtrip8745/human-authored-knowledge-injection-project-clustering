# Human-Authored Knowledge Injection for Project Mention Clustering

This repository contains a public reproducibility package for an IEEE Access
submission on project mention clustering for enterprise email report generation.
It focuses on deterministic clustering/evaluation artifacts and synthetic data.

## Included

- Synthetic email corpus and generation manifest.
- Mention-level records extracted from the synthetic emails.
- A 120-record external test split used for the main comparison.
- Human-authored knowledge files that are readable and editable as plain Markdown: full, partial, and noisy variants.
- Deterministic knowledge-injection clustering/evaluation scripts.
- IEEE Access experiment outputs: main comparison, robustness sweeps,
  ablations, multi-subset results, classic baselines, and distribution checks.
- Aggregate Outlook profile distribution bins used only for distribution
  similarity analysis.

## Not Included

- Real corporate emails, Outlook items, message bodies, addresses, identifiers,
  PST/OST files, or folder-level private data.
- Internal service source code or deployment configuration.
- API keys, LLM endpoint configuration, logs, cache files, or local paths.
- The LLM extraction client used during development. Its synthetic extraction
  outputs are included, but the secret-bearing API wrapper is excluded.
- The prior Project Signature baseline repository. The deterministic scripts
  can use it through `PROJECT_SIGNATURE_BASELINE_ROOT` when available.

## Main Results

The primary result files are under:

`data/synthetic/ieee_access_runs/run_20260618_160823/`

Key files:

- `main_comparison.csv`
- `threshold_sweep.csv`
- `floor_sweep.csv`
- `ablation_human_full.csv`
- `multi_subset_results.csv`
- `multi_subset_summary.csv`
- `classic_baselines.csv`
- `distribution_similarity.csv`
- `IEEE_ACCESS_EXPERIMENT_SUMMARY.md`

## Reproducing Deterministic Experiments

Python 3.10+ is recommended. The scripts use standard-library functionality
plus the prior Project Signature baseline implementation.

Place the baseline implementation at:

`external/mail-to-report-research/`

or set:

```powershell
$env:PROJECT_SIGNATURE_BASELINE_ROOT="<path-to-mail-to-report-research>"
```

Then run from the repository root:

```powershell
python experiments\knowledge_injection\run_ieee_access_experiments.py --out-dir data\synthetic\ieee_access_runs\reproduce
```

The LLM mention extraction stage is not rerun by this public package. The
synthetic extraction outputs used by the deterministic experiments are already
included in `data/synthetic/extraction_run_20260618_085129/`.

## Privacy Scope

All released message data are synthetic. The only information derived from the
real Outlook environment is aggregate distribution-bin statistics used to assess
how closely the synthetic corpus resembles the internal profile. No real email
content or personal/corporate identifiers are released.
