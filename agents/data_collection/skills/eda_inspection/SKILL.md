---
name: eda-inspection
description: Inspect the unified dataset in local execution by running Python code first, then return a concise JSON summary of the real data characteristics.
---

# EDA Inspection

You inspect the unified dataset before notebook generation.

## Goal

Read the real dataset with Python code and return a concise JSON summary of what is actually present.

## Output Contract

- Return the final answer with `final_answer(json.dumps(payload, ensure_ascii=False))`.
- `payload` must be a JSON object with:
  - `summary`: an object describing the observed dataset
  - `notes`: optional list of short strings
- Call `final_answer(...)` exactly once.

## Inspection Rules

- Load the dataset from the local dataset path provided in the task.
- Use Python code to inspect the real dataframe before forming conclusions.
- Keep the summary compact and JSON-serializable.
- Do not write files, use shell commands, or use `subprocess`.
- Do not use tools.

## Summary Expectations

Include only useful high-signal fields such as:

- row count
- column names and dtypes
- non-null counts
- missingness
- label summary when `label` exists
- text-length summary when `text` exists
- source distribution when `source` exists
- a short sample of metadata keys or example rows when helpful

Prefer concise facts over long excerpts.
