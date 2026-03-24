# Data Collection and Quality Agents

This repository contains a `DataCollectionAgent` that collects data from multiple sources, normalizes it into a fixed schema, persists the merged dataset, and generates an executable EDA notebook. A `DataAnnotationAgent` then auto-labels rows, generates an annotation spec, exports Label Studio tasks, flags low-confidence samples for review, and exposes a deterministic `process(df_path, prompt)` tool for LLM-driven row transformations. A `ActiveLearningAgent` can inspect the task prompt, choose feature and target columns, prepare train/val/pool datasets, and train a classifier via local `smolagents.CodeAgent` execution. A final `DataQualityAgent` detects common quality issues, applies configurable cleaning strategies, persists a cleaned dataset, and appends a quality review section into the notebook.

Collection, quality, notebook inspection/generation, and active-learning training are agentic and run with `smolagents.CodeAgent(executor_type="local")`.

## Architecture

The project uses a sequential pipeline:

- `PipelineRunner` orchestrates agents one at a time.
- `DataCollectionAgent` handles source planning, collection, normalization, merge, persistence, and notebook generation.
- `DataAnnotationAgent` consumes the unified dataset, infers labels with confidence scores, writes annotation artifacts, supports deterministic Label Studio exports, and can apply staged LLM row processing to a dataset on disk.
- `ActiveLearningAgent` consumes the latest dataframe, infers feature/supervision columns from a task prompt, writes runnable training code, executes training locally via CodeAgent, and writes uncertainty-ranked review queues.
- `DataQualityAgent` consumes the collected dataframe, reports missing values / duplicates / outliers / class imbalance, cleans the dataset, and updates notebook artifacts.
- `SmolagentsCollectionBackend`, `SmolagentsNotebookBackend`, and `SmolagentsQualityBackend` run agentic workflows via local execution.

## Unified schema

Collected rows are normalized to:

- `text`
- `audio`
- `image`
- `label`
- `source`
- `collected_at`
- `metadata`

## Install

Create a Python 3.11 virtual environment and install the package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

Local skill files are packaged in the repo:

- [agents/data_collection/skills/data_collection/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_collection/skills/data_collection/SKILL.md)
- [agents/data_collection/skills/eda_inspection/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_collection/skills/eda_inspection/SKILL.md)
- [agents/data_collection/skills/eda_notebook/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_collection/skills/eda_notebook/SKILL.md)
- [agents/data_quality/skills/detect_issues/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_quality/skills/detect_issues/SKILL.md)
- [agents/data_quality/skills/fix/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_quality/skills/fix/SKILL.md)
- [agents/data_quality/skills/compare/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_quality/skills/compare/SKILL.md)

## Run

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/mpl data-agent --config config.yaml
```

Optional preview:

```bash
MPLCONFIGDIR=/tmp/mpl data-agent --config config.yaml --print-head 5
```

Run from Python:

```python
from agents import DataCollectionAgent

agent = DataCollectionAgent(config="config.yaml")
df = agent.run()
print(df.head())
```

Runtime prerequisites:

- an OpenAI-compatible model endpoint must be available at `llm.base_url`
- web-enabled sources require outbound network access from the local Python process
- browser-based scraping may require local Chromium/Chrome and matching driver binaries

## Source configuration

Supported source types:

- `hf_dataset`
- `api`
- `scrape` via local `smolagents.CodeAgent` with model-generated Python scraping code
- `kaggle_dataset` using a local exported file via `file_path`

For agentic sources, attempt history is returned in `AgentResult.metadata["source_attempts"]`.
Configure per-agent defaults under `agents.collection`, `agents.eda`, `agents.annotation`, `agents.active_learning`, and `agents.quality`.

Active-learning execution now uses `agents.active_learning.max_steps` for local CodeAgent step budget.

For `scrape` sources, `allow_network: false` is best-effort instruction-level behavior in local execution (not OS-level network isolation).

## Outputs

Running the pipeline writes outputs under `data/`:

- collection artifacts to `data/collection/`
- annotation artifacts to `data/annotation/`
- active learning artifacts to `data/active_learning/`
- quality artifacts to `data/quality/`

The EDA notebook is generated at runtime. The notebook backend first inspects the real dataset via local Python execution, then notebook generation uses that summary; the quality agent appends a deterministic review section.

## Limitations

- Kaggle support is intentionally narrow in v1 and expects a local export path.
- Audio and image EDA are left as the next increment.
- Web search / page extraction quality still depends on source structure and model quality.
