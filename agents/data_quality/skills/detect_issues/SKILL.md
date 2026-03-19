---
name: quality-detect-issues
description: Inspect a real dataset inside the sandbox by writing Python code, then return a strict JSON data quality report that is led by the dataset modality and task, while still covering missingness, duplicates, outliers, and class imbalance.
---

# Data Quality Detection

This skill corresponds to `Part 1: The Detective` from the spec.

You are the agentic quality-detection layer for `DataQualityAgent`.

## Goal

Load the real dataset with Python code, inspect it, and return a compact JSON quality report that centers the main modality and the likely ML task.

## Runtime

- Your Python runs inside the lightweight Docker sandbox.
- Imports are limited to the approved EDA-oriented modules provided by the host.
- Do not use tools.
- Do not use shell commands, `subprocess`, or arbitrary filesystem writes.

## Output Contract

- Return the final answer with `final_answer(json.dumps(payload, ensure_ascii=False))`.
- `payload` must be a JSON object with:
  - `report`: data quality report object
  - `notes`: optional list of short strings
- Call `final_answer(...)` exactly once.

## Required Report Fields

Return at least these fields inside `report`:

- `row_count`
- `column_count`
- `missing`: object with `total` and `by_column`
- `duplicates`: integer duplicate-row count
- `outliers`: list of per-column outlier findings
- `outlier_summary`: object with `method`, `total`, and `columns`
- `imbalance`: object with `column`, `distribution`, `majority_share`, and `is_imbalanced`

When useful, also include:

- `modality_profile`
- `text_profile`
- `task_signals`
- `sparsity_profile`

Also include short notes about whether each issue is likely meaningful for the task when that is obvious from the data.

## Analysis Rules

- Load the dataset from the provided sandbox path before making conclusions.
- Prefer one end-to-end code block that computes the report and returns `final_answer(...)` directly.
- Do not waste steps on exploratory `print()` debugging unless a failure forces it.
- Detect missing values from actual nulls and common textual null sentinels when reasonable.
- Detect duplicates from real row content, not schema guesses.
- If object columns contain dicts or lists, normalize them to stable JSON strings before duplicate checks.
- Treat deduplication as a primary quality task, not a side metric.
- For text datasets, measure at least:
  - exact duplicate rows
  - normalized duplicate text prompts
  - near-duplicate prompts when obvious boilerplate or repeated stems appear
  - cross-source duplicates when the same prompt may have been collected twice
- Detect numeric outliers using IQR or z-score; state the method you used.
- Detect class imbalance from the configured label column when present, otherwise infer a reasonable label-like column.
- If the preferred label column is high-cardinality numeric data or looks like a regression target, report imbalance as not applicable instead of pretending it is a class label.
- Always decide which issue types are actually meaningful for the observed modality and task before emphasizing them.
- If the main modality is text, check text-centric quality first:
  - empty or near-empty text rows
  - text length distribution and extreme short/long rows
  - obvious truncation or boilerplate
  - exact or normalized duplicate prompts
  - repeated prompts with minor formatting or metadata differences
  - label coverage by source or subset
  - auxiliary modality columns that are entirely null and probably irrelevant
- Measure row sparsity when it may affect cleaning decisions.
  Example: count rows where most columns are null or where the main modality field is missing even if other metadata exists.
- Distinguish between unlabeled rows and low-information rows.
  Missing labels alone do not automatically make a row low quality if the text or metadata is still useful for later labeling, retrieval, pretraining, or augmentation.
- Treat generic tabular checks as supporting evidence, not the whole answer.
- Keep the report JSON-serializable and concise.
