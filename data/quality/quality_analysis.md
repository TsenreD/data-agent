# Quality Analysis

## Task Reading

- Task type: `unknown`
- Primary modality: `text`
- Label semantics: unknown

## Recommended Strategy

- Missing values: `unknown`
- Duplicates: `unknown`
- Outliers: `unknown`

## Why This Matters

This is a math reasoning dataset where text (the problem statement) is the primary modality. Audio/image columns are entirely null and should be dropped. The 50% label null rate is not a data quality problem - it's a dataset design choice where different sources contribute different proportions of labeled vs unlabeled problems. The recommended strategy preserves all 100 rows because: (1) unlabeled math problems are still useful for evaluation, (2) they could be labeled later via model generation, (3) they provide source diversity. Class imbalance checks are irrelevant because the label is a numeric answer target (chain-of-thought with final answer), not a classification category. Numeric outlier detection on text length is misleading here - longer texts are simply more complex math problems, not errors.

## Priority Actions

- drop_null_modality_columns
- preserve_unlabeled_math_problems

## Relevant Checks

- text_non_empty
- label_format_consistency
- source_distribution

## Lower-Value Checks

- class_balance
- numeric_outliers
- imbalance_ratio

## Alternative Strategies

- Option 1: missing=`unknown`, duplicates=`unknown`, outliers=`unknown`
- Option 2: missing=`unknown`, duplicates=`unknown`, outliers=`unknown`
