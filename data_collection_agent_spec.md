# Data Collection Agent Specification

## GOAL

Write an agent that can collect data from multiple sources and return a
unified dataset.
Ultimate goal is to combine multiple agents in a single pipeline, run with python.

------------------------------------------------------------------------

## AGENT ARCHITECTURE

**DataCollectionAgent**

Skills: 
- `scrape(url, selector) → DataFrame` 
- `fetch_api(endpoint, params) → DataFrame`
- `load_dataset(name, source='hf'|'kaggle') → DataFrame`
- `merge(sources: list[DataFrame]) → DataFrame`

Each skill is agentic in nature, meaning LLM is responsible for on-the fly code generation and execution.


------------------------------------------------------------------------

## TECHNICAL CONTRACT

``` python
from agents import DataCollectionAgent

agent = DataCollectionAgent(config='config.yaml')

df = agent.run(
    sources=[
        {'type': 'hf_dataset', 'name': 'imdb'},
        {'type': 'scrape', 'url': '...', 'selector': '...'},
    ]
)
```

**Output:**\
`pd.DataFrame` with standard columns: - `text / audio / image` -
`label` - `source` - `collected_at`

------------------------------------------------------------------------

## REQUIREMENTS

-   At least **2 data sources**:
    -   one **open dataset** (HuggingFace or Kaggle)
    -   one **scraping or API** source
-   **Unified output dataset schema**: fixed columns for all sources
-   **EDA (Exploratory Data Analysis)**:
    -   class distribution
    -   text length / audio duration
    -   top‑20 words / spectrogram
-   `README.md` with:
    -   description of the ML task
    -   data schema
    -   instructions for running the project
-   `requirements.txt` or `pyproject.toml`

------------------------------------------------------------------------

## REPOSITORY STRUCTURE

    agents/data_collection/data_collection_agent.py   — main agent file
    config.yaml                       — data source configuration
    notebooks/eda.ipynb               — agent-generated executable EDA notebook
    data/raw/                         — collected data
    README.md
