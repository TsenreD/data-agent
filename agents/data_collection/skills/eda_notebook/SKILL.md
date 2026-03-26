---
name: eda-notebook
description: Generate an executable Jupyter EDA notebook for the unified dataset using only approved base Python modules plus the approved EDA libraries.
---

# EDA Notebook

You generate a Jupyter notebook for exploratory data analysis of the unified dataset.

## Goal

Produce a valid, executable `.ipynb` notebook as JSON.

The notebook must be self-contained and runnable later by the host environment without manual edits.
Base the notebook on actual observed data characteristics, not generic assumptions.

## Allowed Notebook Imports

Notebook code may only import approved standard-library modules and these EDA libraries:

- `IPython.display`
- `numpy`
- `pandas`
- `matplotlib.pyplot`
- `seaborn`

Preferred standard-library imports:

- `collections`
- `json`
- `math`
- `pathlib`
- `re`
- `statistics`

Do not import anything else in notebook cells.

## Notebook Requirements

- Return the final answer with `final_answer(json.dumps(payload, ensure_ascii=False))`.
- `payload` must be a JSON object with:
  - `notebook`: a valid Jupyter notebook object
  - `notes`: optional list of short strings
- Call `final_answer(...)` exactly once.
- Do not print the notebook JSON instead of returning it.
- The notebook must target nbformat 4.
- The notebook must contain at most 10 cells total.
- Every code cell must have `execution_count: null` and `outputs: []`.
- The first code cell must define the standard notebook aliases: `np`, `pd`, `plt`, `sns`, `Path`, and `display`.
- Prefer this exact setup pattern in the first code cell:
  - `from pathlib import Path`
  - `import numpy as np`
  - `import pandas as pd`
  - `import matplotlib.pyplot as plt`
  - `import seaborn as sns`
  - `try: from IPython.display import display ...`
- Use the dataset path provided in the task exactly as the notebook input path.
- Do not use shell commands, `subprocess`, or arbitrary filesystem writes inside notebook code.
- Do not use shell escapes like `!pip`.
- Do not use IPython magics except `%matplotlib inline` if you really need it.

## Content Requirements

Include concise markdown plus executable code that:

- loads the unified JSONL dataset into pandas
- previews the dataframe and schema
- summarizes nulls / missingness
- analyzes label distribution when the `label` column has values
- analyzes text-length distributions when the `text` column has values
- uses defensive checks so the notebook still runs on sparse or partially empty datasets
- only includes sections that are justified by the real dataset contents

## Workflow

- First inspect the actual dataset characteristics made available in the task.
- Use those findings to decide which EDA sections deserve notebook space.
- Prefer thoughtful sections over generic filler.

## Style

- Keep cells focused and readable.
- Prefer a small number of high-value sections over many tiny cells.
- Keep the whole notebook to 10 cells or fewer.
- Use plotting code that renders inline in a standard notebook environment.
- Prefer simple pandas/seaborn/matplotlib code over clever abstractions.
