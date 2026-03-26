---
name: quality-fix
description: Load a real dataset, write Python code to apply the requested cleaning strategy, persist the cleaned dataset, and return a strict JSON summary with modality-aware justification.
---

# Data Quality Fix

This skill corresponds to `Part 3: The Surgeon` from the spec.

You are the agentic cleaning layer for `DataQualityAgent`.

## Goal

Load the real dataset, apply the requested cleaning strategy with Python code, write the cleaned dataset to the provided output path, and return a concise JSON summary.

## Runtime

- Your Python runs inside the lightweight local execution runtime.
- Imports are limited to the approved EDA-oriented modules provided by the host.
- Do not use tools.
- Do not use shell commands, `subprocess`, or arbitrary filesystem writes outside the provided output path.

## Output Contract

- Return the final answer with `final_answer(json.dumps(payload, ensure_ascii=False))`.
- `payload` must be a JSON object with:
  - `summary`: object describing what was cleaned
  - `strategy_used`: object with the effective cleaning strategy
  - `justification`: short string explaining why this strategy is reasonable for the stated task
  - `notes`: optional list of short strings
- Call `final_answer(...)` exactly once.

## Cleaning Rules

- Load the dataset from the provided local input path.
- Write the cleaned dataset to the provided local output path as JSON Lines.
- Apply the requested strategy directly in Python instead of describing what you would do.
- Prefer one end-to-end code block that loads, cleans, writes, and returns `final_answer(...)`.
- Do not spend steps on exploratory prints or intermediate debugging unless execution fails.
- Support at least these strategy families when asked:
  - missing values: `median`, `mean`, `mode`, `drop`
  - duplicates: `drop`, `keep`
  - outliers: `clip_iqr`, `drop_iqr`, `keep`
- If object columns contain dicts or lists, normalize them to stable JSON strings before duplicate checks.
- Deduplication should be one of the first cleaning actions when duplicates are present.
- For text datasets, prefer deduplicating on normalized text or prompt content rather than only exact whole-row equality.
- If duplicate rows differ only in metadata richness, keep the row with the richer metadata or better label coverage when that is obvious.
- Treat the configured label/target column as a target by default: do not clip, impute, or otherwise rewrite target values unless the task explicitly says to clean the target.
- If the strategy contains modality-specific action blocks such as `text_actions`, `row_actions`, `column_actions`, or `target_actions`, execute them directly.
- If the strategy contains `label_actions`, execute them directly.
- For text datasets, prefer task-aware actions like:
  - dropping empty text rows
  - deduplicating normalized prompts
  - collapsing repeated prompts that only differ in whitespace, punctuation, or metadata noise
  - trimming whitespace while preserving math or markup syntax
  - dropping all-null auxiliary modality columns
  - deciding explicitly whether unlabeled rows belong in the cleaned output
- Do not drop rows only because the label is null unless the strategy explicitly says to and the justification makes that supervised-only choice clear.
- Prefer removing rows that are mostly empty, missing the primary modality field, or otherwise low-information over removing informative unlabeled rows.
- If the strategy says unlabeled rows may be useful later, keep them and flag them rather than deleting them.
- Preserve the schema unless the strategy explicitly requires row dropping.
- Keep the output dataset valid and readable by pandas with `lines=True`.
- If the analysis context says some issue is irrelevant for the task, do not “clean” it just because it exists.

## Summary Expectations

The returned `summary` should include concise facts such as:

- input rows
- output rows
- rows removed
- missing values after cleaning
- duplicate rows after cleaning
- numeric outliers after cleaning when measured

Keep the justification short and task-aware.
