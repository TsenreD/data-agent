# Data Collection Agent

This repository contains a v1 `DataCollectionAgent` that collects data from multiple sources, normalizes it into a fixed schema, persists the merged dataset, and produces basic EDA artifacts. The intended downstream ML task for the sample configuration is text classification or sentiment-style labeling over heterogeneous text sources.

## Architecture

The project uses a sequential pipeline shape:

- `PipelineRunner` orchestrates agents one at a time.
- `DataCollectionAgent` handles source planning, collection, normalization, merge, persistence, and EDA.
- `SandboxExecutor` runs model-generated extraction code inside Docker rather than on the host interpreter.
- `OllamaAdapter` provides a thin wrapper around a local Ollama server.

For v1, the top-level pipeline is plain Python rather than a graph framework. The agentic boundary stays inside the collection agent.

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

Build the execution image used for agentic sources:

```bash
docker build -t data-agent-execution-backend:latest -f agents/data_collection/Dockerfile.sandbox agents/data_collection
```

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
- Docker must be available for sandboxed execution of generated scraping code
- per-agent runtime logs are written under `logs/` by default

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
- `scrape` via on-the-fly LLM code generation plus sandbox execution
- `kaggle_dataset` using a local exported file via `file_path`

Scraping is always agentic in v1: the model generates Python extraction code at runtime, and the sandbox executes it to produce the `DataFrame`. `api` sources can still use the deterministic connector by default, or opt into the same model-generated path with `agentic: true`.

For agentic sources, the sandbox never raises exceptions back into the agent. It returns structured execution results with `stdout`, `stderr`, exit status, and error details so the agent can retry with revised code. Use `max_attempts` and `retry_on_empty` in a source config to control that loop.
Attempt history is returned in `AgentResult.metadata["source_attempts"]`.

The executor expects a Docker image with Python plus the allowed libraries. By default it uses `data-agent-execution-backend:latest`; override this with `DATA_AGENT_SANDBOX_IMAGE` if needed. The executor starts one or more long-lived sandbox containers for the duration of an agent run and reuses them across retries instead of launching a fresh container for every attempt. Network is disabled by default at the sandbox level and only opened for sources with `allow_network: true` or for `scrape` / `api` sources by default.

## Outputs

Running the agent writes:

- merged dataset to `data/raw/unified_dataset.jsonl`
- EDA plots to `data/raw/eda/`

The notebook scaffold lives at `notebooks/eda.ipynb`.

## Limitations

- The sandbox is container-based and materially safer than host subprocess execution, but it is still not equivalent to a VM or microVM boundary.
- Kaggle support is intentionally narrow in v1 and expects a local export path.
- Audio and image EDA are left as the next increment; text EDA is implemented now.
- Any run that includes a `scrape` source requires a configured model adapter such as `OllamaAdapter`.
- Docker and a compatible execution image are required for agentic sources.
