# Quality Findings

## Executive Summary

- Missing values: `240`. Cells need imputation or row-level handling.
- Duplicate rows: `0`. Rows may distort evaluation and training.
- Numeric outliers: `0`. Values may need clipping, filtering, or review.
- Label imbalance is tracked on `label` with majority share `0.020`.

## Key Metrics

- Missing values: `240`
- Duplicate rows: `0`
- Numeric outliers: `0`
- Imbalance column: `label`

## What To Read First

- Focus on the largest counts above, not the full raw issue object.
- Use the analysis report for why the issues matter for the current ML task.
- Use the comparison report to see what changed after cleaning.
