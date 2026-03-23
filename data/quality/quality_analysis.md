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

This is a math reasoning dataset where the primary value is the problem-solution pairs. The recommended retention-oriented strategy is better than supervised-only because: (1) all 124 rows contain valid, informative math problems regardless of label status; (2) the 24 unlabeled rows can be used for inference evaluation or future label generation; (3) dropping them would lose 19% of the data without justification. The audio/image columns are entirely null and should be dropped as they serve no purpose. The 'outliers' in text length are legitimate long problems, not errors. Class balance analysis is meaningless here since each problem has a unique solution - this is not a classification task.

## Priority Actions

- drop_unused_columns_audio_image
- preserve_all_rows_including_unlabeled
- flag_unlabeled_rows_by_source

## Relevant Checks

- text_non_empty - all 124 rows have valid text
- label_format_validity - labels follow CoT format with '#### <answer>'
- source_distribution - tracks provenance across 3 sources
- metadata_completeness - metadata present for all rows

## Lower-Value Checks

- class_balance_plots - not applicable; each label is unique (each math problem has unique solution)
- numeric_outlier_detection - 15 text length 'outliers' are legitimate long problems, not errors
- audio_image_modality_checks - these columns are entirely null and should be dropped

## Alternative Strategies

- Option 1: missing=`unknown`, duplicates=`unknown`, outliers=`unknown`
- Option 2: missing=`unknown`, duplicates=`unknown`, outliers=`unknown`
