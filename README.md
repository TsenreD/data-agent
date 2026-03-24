# Data Collection and Quality Agents

This repository contains a `DataCollectionAgent` that collects data from multiple sources, normalizes it into a fixed schema, persists the merged dataset, and generates an executable EDA notebook. A `DataAnnotationAgent` then auto-labels rows, generates an annotation spec, exports Label Studio tasks, flags low-confidence samples for review, and exposes a deterministic `process(df_path, prompt)` tool for LLM-driven row transformations. The annotation stage can also be driven directly from a full prompt in config, including prompts that add new columns while labeling. A `ActiveLearningAgent` can then inspect the task prompt, choose the right feature and target columns from the current dataframe, prepare train/val/pool datasets, and train a classifier by running generated Python in a Docker sandbox (agentic mode by default). A final `DataQualityAgent` detects common data quality issues, applies configurable cleaning strategies, persists a cleaned dataset, and appends a quality review section into the notebook. The collection, quality, and active-learning training stages are agentic: each skill is executed by `smolagents.CodeAgent`, which writes Python code inside the sandbox rather than relying only on hardcoded host logic. The intended downstream ML task for the sample configuration is text classification or sentiment-style labeling over heterogeneous text sources.

## Architecture

The project uses a sequential pipeline shape:

- `PipelineRunner` orchestrates agents one at a time.
- `DataCollectionAgent` handles source planning, collection, normalization, merge, persistence, and EDA notebook generation.
- `DataAnnotationAgent` consumes the unified dataset, infers labels with confidence scores, writes annotation artifacts, supports deterministic Label Studio exports, and can apply staged LLM row processing to a dataset on disk.
- `ActiveLearningAgent` consumes the latest dataframe, infers the most relevant feature and supervision columns from a user task prompt, writes runnable training code, executes training in Docker with configurable resources, and writes uncertainty-ranked review queues.
- `DataQualityAgent` consumes the collected dataframe, reports missing values / duplicates / outliers / class imbalance, cleans the dataset, and updates notebook artifacts.
- `SmolagentsCollectionBackend` runs agentic web/API collection through `smolagents.CodeAgent`.
- `SmolagentsNotebookBackend` runs agentic dataset inspection and notebook generation through a lightweight EDA sandbox.
- `SmolagentsQualityBackend` runs agentic quality detection, cleaning, and comparison through the same lightweight EDA sandbox.
- Deterministic connectors still handle Hugging Face datasets and local Kaggle exports directly.

The top-level pipeline is still plain Python rather than a graph framework, but the agentic boundary now sits on top of `smolagents.CodeAgent(executor_type="docker")` instead of project-local code generation.

## Unified schema

Collected rows are normalized to these columns:

- `text`
- `audio`
- `image`
- `label`
- `source`
- `collected_at`
- `metadata`

`metadata` stores source-specific fields that are not part of the core schema.

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

The `smolagents` Docker executor builds from [agents/data_collection/Dockerfile.parse](/Users/eadyagin/vscode/data-agent/agents/data_collection/Dockerfile.parse) for collection work and from [agents/data_collection/Dockerfile.eda](/Users/eadyagin/vscode/data-agent/agents/data_collection/Dockerfile.eda) for EDA work. The EDA image is intentionally lightweight and only includes the notebook runtime plus core EDA libraries.

The collection and notebook behaviors are defined as packaged local skills:

- [agents/data_collection/skills/data_collection/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_collection/skills/data_collection/SKILL.md)
- [agents/data_collection/skills/eda_inspection/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_collection/skills/eda_inspection/SKILL.md)
- [agents/data_collection/skills/eda_notebook/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_collection/skills/eda_notebook/SKILL.md)
- [agents/data_quality/skills/detect_issues/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_quality/skills/detect_issues/SKILL.md)
- [agents/data_quality/skills/fix/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_quality/skills/fix/SKILL.md)
- [agents/data_quality/skills/compare/SKILL.md](/Users/eadyagin/vscode/data-agent/agents/data_quality/skills/compare/SKILL.md)

## Run

Run the installed CLI:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/mpl data-agent --config config.yaml
```

Optional preview:

```bash
MPLCONFIGDIR=/tmp/mpl data-agent --config config.yaml --print-head 5
```

Run the collection agent from Python:

```python
from agents import DataCollectionAgent

agent = DataCollectionAgent(config="config.yaml")
df = agent.run()
print(df.head())
```

Runtime prerequisites outside Python packaging:

- an OpenAI-compatible model endpoint must already be available at the `llm.base_url` configured in `config.yaml`
- Docker must be available for the remote code executor used by `smolagents`
- web-enabled sources require outbound network access from the Python process and from the sandbox container

Run inside a sequential pipeline:

```python
from agents import ActiveLearningAgent, DataAnnotationAgent, DataCollectionAgent, DataQualityAgent, PipelineRunner

runner = PipelineRunner(
    [
        DataCollectionAgent("config.yaml"),
        DataAnnotationAgent("config.yaml"),
        ActiveLearningAgent("config.yaml"),
        DataQualityAgent("config.yaml"),
    ]
)
result = runner.run()
print(result.dataframe_path)
```

## Source configuration

Supported source types in v1:

- `hf_dataset`
- `api`
- `scrape` via `smolagents.CodeAgent` in Docker with model-generated Python scraping code
- `kaggle_dataset` using a local exported file via `file_path`

Scraping is always agentic: `smolagents.CodeAgent` runs generated Python inside Docker and the model writes the scraping logic directly with prebaked libraries such as `requests` and `beautifulsoup4`. `api` sources still use the deterministic connector by default, or can opt into the same Docker-backed custom-code path with `agentic: true`.

For agentic sources, attempt history is returned in `AgentResult.metadata["source_attempts"]`. Configure per-agent defaults under `agents.collection`, `agents.eda`, `agents.annotation`, `agents.active_learning`, and `agents.quality`. The annotation agent accepts `modality`, `task`, `confidence_threshold`, `label_column`, `input_path`, optional `classes` with `name`, `description`, `keywords`, and `examples`, an optional full `prompt`/`annotation_prompt` used during `auto_label`, plus `process_config` for the `process(df_path, prompt)` tool with `parallel_workers`, `timeout_per_row`, `max_retries`, `model`, `base_url`, and `api_key`. When a prompt is configured, the model may add new fields to transformed rows and those columns are merged back into the dataset. Label Studio exports are emitted in import-ready `annotations` format. The active learning agent accepts `task_prompt`/`prompt`, optional explicit `feature_columns` and `target_column`, `strategy`, `batch_size`, `test_size`, and `training` parameters (`embedding_dim`, `epochs`, `batch_size`, `learning_rate`, `max_vocab`, `min_token_freq`). It also supports `docker` settings (`enabled`, `agentic`, `max_steps`, `image_name`, `memory_limit`, `cpu_limit`, `pids_limit`, `shm_size`) for high-resource in-container training. Legacy `model.dim`, `model.epochs`, and `model.learning_rate` map to training config for backward compatibility. The quality agent accepts `imbalance_threshold`, `label_column`, `input_path`, `notebook_path`, and a default `strategy` with `missing`, `duplicates`, and `outliers` keys. Source-level `max_attempts`, `max_steps`, and `retry_on_empty` still override collection defaults when explicitly set. Docker executor settings can be supplied under `llm.sandbox`, for example `image_name`, `build_new_image`, `memory_limit`, `cpu_limit`, `pids_limit`, `shm_size`, and `port`.

The browser-enabled sandbox now defaults to `shm_size: 1g`, because Docker's default `/dev/shm` allocation is too small for Chromium and often causes Selenium or Playwright crashes such as `tab crashed`. Override it explicitly if your environment needs a different value:

```yaml
llm:
  sandbox:
    shm_size: 2g
```

For `scrape` sources, you can also add a freeform `extraction_instructions` field to tell the agent exactly what to extract, for example “one record per problem”, “follow child links”, “ignore navigation text”, or which metadata fields to preserve. This text is injected prominently into the scrape task before the model writes code.

## Outputs

Running the agent writes stage outputs into separate directories under `data/`:

- collection artifacts to `data/collection/`, including `unified_dataset.jsonl` and `eda.ipynb`
- annotation artifacts to `data/annotation/`, including `annotated_dataset.jsonl`, `annotation_spec.md`, `annotation_quality.json`, `labelstudio_import.json`, and `low_confidence_review.json`
- active learning artifacts to `data/active_learning/`, including `active_learning_dataset.jsonl`, `active_learning_queries.jsonl`, `active_learning_summary.json`, `active_learning_curve.png`, `train.jsonl`, `val.jsonl`, `pool.jsonl`, `train_active_learning.py`, `model.pth`, and `training_metrics.json`
- quality artifacts to `data/quality/`, including `cleaned_dataset.jsonl`, `quality_report.md`, `quality_analysis.md`, and `quality_comparison.md`

The EDA notebook is generated by the collection agent at runtime, not checked in as a static scaffold. Before writing the notebook, the EDA backend first inspects the real dataset via code inside the lightweight EDA sandbox and uses that summary to shape the notebook. After that, the quality agent appends a deterministic quality-review section with strategy justification and code cells for issue visualization and before/after comparison. Notebook code is constrained to approved base modules plus `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn`.

## Limitations

- Kaggle support is intentionally narrow in v1 and expects a local export path.
- Audio and image EDA are left as the next increment; the generated notebook focuses on tabular/text EDA for now.
- Any run that includes a `scrape` source requires a configured model for `smolagents`, plus a working Docker daemon.
- Web search / page extraction quality still depends on the target site structure and model quality.
