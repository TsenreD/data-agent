# Data Collection Agent

This repository contains a `DataCollectionAgent` that collects data from multiple sources, normalizes it into a fixed schema, persists the merged dataset, and produces basic EDA artifacts. The intended downstream ML task for the sample configuration is text classification or sentiment-style labeling over heterogeneous text sources.

## Architecture

The project uses a sequential pipeline shape:

- `PipelineRunner` orchestrates agents one at a time.
- `DataCollectionAgent` handles source planning, collection, normalization, merge, persistence, and EDA.
- `SmolagentsCollectionBackend` runs agentic web/API collection through `smolagents.CodeAgent`.
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

The `smolagents` Docker executor builds from [agents/data_collection/Dockerfile](/Users/eadyagin/vscode/data-agent/agents/data_collection/Dockerfile) the first time it runs, unless the configured image already exists. That image preinstalls the scraping, browser, and notebook runtime used by model-generated code, and the agent allows imports from the installed environment.

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
from agents import DataCollectionAgent, PipelineRunner

runner = PipelineRunner([DataCollectionAgent("config.yaml")])
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

For agentic sources, attempt history is returned in `AgentResult.metadata["source_attempts"]`. Use `max_attempts`, `max_steps`, and `retry_on_empty` in a source config to control behavior. Docker executor settings can be supplied under `llm.sandbox`, for example `image_name`, `build_new_image`, `memory_limit`, `cpu_limit`, `pids_limit`, and `port`.

## Outputs

Running the agent writes:

- merged dataset to `data/raw/unified_dataset.jsonl`
- EDA plots to `data/raw/eda/`

The notebook scaffold lives at `notebooks/eda.ipynb`.

## Limitations

- Kaggle support is intentionally narrow in v1 and expects a local export path.
- Audio and image EDA are left as the next increment; text EDA is implemented now.
- Any run that includes a `scrape` source requires a configured model for `smolagents`, plus a working Docker daemon.
- Web search / page extraction quality still depends on the target site structure and model quality.
