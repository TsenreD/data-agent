# Data Quality Detective Agent

## Goal
Build a detective agent that automatically detects and resolves data quality issues.

---

## Agent Architecture

### `DataQualityAgent`
- **skill:** `detect_issues(df)` → `QualityReport` (missing values, duplicates, outliers, class imbalance)
- **skill:** `fix(df, strategy: dict)` → `DataFrame`
- **skill:** `compare(df_before, df_after)` → `ComparisonReport`

---

## Technical Contract

```python
from data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df)
# → {'missing': {...}, 'duplicates': N, 'outliers': [...], 'imbalance': {...}}

df_clean = agent.fix(df, strategy={
    'missing': 'median',
    'duplicates': 'drop',
    'outliers': 'clip_iqr'
})

comparison = agent.compare(df, df_clean)
# → table: before / after for each metric
```

---

## Requirements

- **Minimum 3 problem types:** missing values, duplicates, outliers (IQR or z-score)
- **Minimum 2 cleaning strategies** to choose from — the `strategy` parameter in `fix()`
- **Before/after comparison report** for each quality metric
- **Strategy justification** — Markdown cell in the notebook

---

## Three Parts of the Assignment

### Part 1: The Detective
Detect missing values, outliers, duplicates, and class imbalance. Visualize each issue.

### Part 2: The Analyzer
Explain detected issues and recommend a cleaning strategy based on the task description.

### Part 3: The Surgeon
Apply at least 2 cleaning strategies and compare results in a table.

### Part 4: The Argument
Justify the choice of the best approach: why is this strategy better for your ML task?
