---
name: quality-compare
description: Load the pre-clean and post-clean datasets, write Python code to compare their quality metrics, and return a strict JSON before/after report that emphasizes modality-relevant effects.
---

# Data Quality Comparison

This skill supports `Part 3: The Surgeon` and `Part 4: The Argument` from the spec.

You are the agentic comparison layer for `DataQualityAgent`.

## Goal

Load the before and after datasets with Python code and return a concise JSON comparison report.

## Runtime

- Your Python runs inside the lightweight local execution runtime.
- Imports are limited to the approved EDA-oriented modules provided by the host.
- Do not use tools.
- Do not use shell commands, `subprocess`, or arbitrary filesystem writes.

## Output Contract

- Return the final answer with `final_answer(json.dumps(payload, ensure_ascii=False))`.
- `payload` must be a JSON object with:
  - `comparison`: before/after comparison object
  - `notes`: optional list of short strings
- Call `final_answer(...)` exactly once.

## Required Comparison Fields

Return at least these fields inside `comparison`:

- `row_count`
- `missing`
- `duplicates`
- `outliers`
- `imbalance`

Each section should include before/after values and a delta where that makes sense.

If the analysis context includes multiple candidate strategies or a declared “best” strategy, preserve that context in the comparison notes and frame the comparison around meaningful metrics.

## Comparison Rules

- Load both datasets from the provided local paths before answering.
- Compute quality metrics from actual data, not from host-provided summaries.
- Use the same style of duplicate / outlier logic across before and after so the comparison is fair.
- Prefer one end-to-end code block that computes the comparison and returns `final_answer(...)`.
- If object columns contain dicts or lists, normalize them to stable JSON strings before duplicate checks.
- Make duplicate reduction visible in the comparison.
  Include exact duplicate row counts and normalized duplicate text counts before and after when the modality is text.
- If the preferred label column behaves like a regression target rather than a categorical label, mark imbalance as not applicable instead of forcing a class-balance analysis.
- If the main modality is text, include text-centric before/after signals when they matter:
  - empty text count
  - duplicate normalized prompt count
  - text length summary
  - label coverage
  - removal of irrelevant all-null modality columns
- If row retention was part of the strategy, compare:
  - rows kept despite missing labels
  - rows dropped for high sparsity or missing primary modality
  - whether informative unlabeled rows were preserved
- Keep the result JSON-serializable and concise.
