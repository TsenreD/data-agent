# Quality Findings

## Executive Summary

- Missing values: `250`. Cells need imputation or row-level handling.
- Duplicate rows: `0`. Rows may distort evaluation and training.
- Numeric outliers: `11`. Values may need clipping, filtering, or review.
- Label imbalance is tracked on `label`.

## Key Metrics

- Missing values: `250`
- Duplicate rows: `0`
- Numeric outliers: `11`
- Imbalance column: `label`

## What To Read First

- Focus on the largest counts above, not the full raw issue object.
- Use the analysis report for why the issues matter for the current ML task.
- Use the comparison report to see what changed after cleaning.
