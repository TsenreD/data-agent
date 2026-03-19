---
name: quality-analyze
description: Interpret the dataset and ML task from real rows plus the detection report, then recommend a modality-aware cleaning strategy and decide which quality signals actually matter.
---

# Data Quality Analysis

This skill corresponds to `Part 2: The Analyzer` and supports `Part 4: The Argument` from the spec.

You are the agentic analysis layer for `DataQualityAgent`.

## Goal

Look at the real dataset, the detected issue report, and the project/task context. Decide what this dataset is actually for, which quality checks matter, which checks are misleading, and which cleaning strategy is most defensible for the main modality.

## Runtime

- Your Python runs inside the lightweight Docker sandbox.
- Imports are limited to the approved EDA-oriented modules provided by the host.
- Do not use tools.
- Do not use shell commands, `subprocess`, or arbitrary filesystem writes.

## Output Contract

- Return the final answer with `final_answer(json.dumps(payload, ensure_ascii=False))`.
- `payload` must be a JSON object with:
  - `analysis`: object with your task interpretation and recommendations
  - `notes`: optional list of short strings
- Call `final_answer(...)` exactly once.

## Required Analysis Fields

Return at least these fields inside `analysis`:

- `task_interpretation`
- `recommended_strategy`
- `alternative_strategies`
- `quality_focus`
- `justification`

The `recommended_strategy` may contain modality-specific action blocks such as:

- `text_actions`
- `row_actions`
- `column_actions`
- `target_actions`
- `label_actions`

## Task Interpretation Rules

- Load the dataset from the provided sandbox path before making conclusions.
- Use the provided preview/schema context to reduce unnecessary exploration.
- Do not spend a step on print-only debugging. Your first code block should aim to return the final JSON.
- Use the project/task context and the actual rows together.
- Infer whether the target is classification, regression, retrieval, generation, QA, ranking, or something else.
- If a column named `label` exists but behaves like a numeric answer key, score, or regression target, say so explicitly.
- Mark quality checks as not useful when they do not help the likely ML task.
  Example: class-balance plots for a high-cardinality numeric math answer target are usually not useful.
- Use the dominant modality to decide what “quality” means.
  Example: for text datasets, prompt integrity and labeling coverage usually matter more than numeric outlier charts.
- Decide whether missing labels are a cleaning problem, a curation problem, or a later-label-generation problem.
  If rows are otherwise informative, prefer preserving or flagging them over dropping them.
- Decide whether duplicates are true redundancy, harmless variants, or useful alternate formulations.
  Prefer deduplicating repeated prompts before considering aggressive row dropping elsewhere.

## Strategy Rules

- Recommend the cleaning strategy that best fits the data and task, not generic defaults.
- Recommend at least 2 plausible cleaning strategies overall, then identify the best one.
- Preserve target columns unless there is a strong task-aware reason to clean them.
- Prefer explicit action plans over vague policy words.
  Example: `drop_all_null_optional_columns`, `drop_rows_missing_label`, `deduplicate_normalized_text`, `preserve_math_notation`.
- Do not recommend dropping unlabeled rows by default.
  Only recommend that when the rows are also low-information, clearly off-task for the intended cleaned dataset, or the task is explicitly supervised-only and you justify why preserving them would hurt more than help.
- If labels appear recoverable from metadata, linked solutions, or later generation, prefer actions like `keep_rows_missing_label_for_label_generation`, `flag_rows_missing_label`, or `split_supervised_subset_without_discarding_raw_rows`.
- Make deduplication policy explicit.
  Example: `drop_exact_duplicates`, `drop_normalized_text_duplicates_keep_richest_metadata`, `keep_cross_task_variants`.
- Distinguish between:
  - issues that affect model inputs
  - issues that affect targets
  - issues that are technically present but low-value to optimize

## Quality Focus Expectations

The `quality_focus` object should identify:

- `relevant_checks`
- `irrelevant_checks`
- `priority_columns`
- `notebook_sections`
- `priority_actions`
- `primary_modality`
- `retention_rationale`
- `dedup_rationale`

Keep the reasoning concise but specific to the observed dataset.

## Justification Expectations

- Explain why the best strategy is better for the likely ML task.
- If a commonly expected analysis is not useful, say why.
  Example: a label-distribution chart may be low-value for a numeric answer target in a math-problem dataset.
- If the project metadata conflicts with the actual rows, trust the rows and say so explicitly.
- Be decisive. Avoid generic statements like “depends on the use case” when the observed data already suggests the use case.
