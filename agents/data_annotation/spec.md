# AnnotationAgent — Implementation Specification

## Goal

Build an agent that automatically labels data, generates an annotation specification,
evaluates label quality, exports tasks for manual re-labeling into LabelStudio,
flags low-confidence samples for human review, and can augment any DataFrame
via LLM-driven row-level transformations.
The agent enters the pipeline as `auto_label_op`.

---

## Agent Architecture

### `AnnotationAgent`

| Skill | Signature | Description |
|---|---|---|
| `auto_label` | `(df, modality) → DataFrame` | Automatic labeling: text → spaCy / zero-shot, audio → Whisper, image → YOLO |
| `generate_spec` | `(df, task) → AnnotationSpec` | Generates a Markdown annotation spec file |
| `check_quality` | `(df_labeled) → QualityMetrics` | Evaluates Cohen's κ, label distribution, mean confidence |
| `export_to_labelstudio` | `(df) → JSON` | Exports labeled data in LabelStudio import format |
| `flag_for_review` | `(df_labeled, threshold) → JSON` | Flags low-confidence samples into a separate review file |

---

## Technical Contract

```python
from annotation_agent import AnnotationAgent

agent = AnnotationAgent(modality='text', confidence_threshold=0.75)

# Step 1 — Auto-label
df_labeled = agent.auto_label(df)

# Step 2 — Generate annotation spec
spec = agent.generate_spec(df, task='sentiment_classification')
# → annotation_spec.md: task description, classes, examples, edge cases

# Step 3 — Quality check
metrics = agent.check_quality(df_labeled)
# → {'kappa': 0.72, 'label_dist': {...}, 'confidence_mean': 0.85}

# Step 4 — Export full dataset to LabelStudio
agent.export_to_labelstudio(df_labeled)
# → labelstudio_import.json

# Step 5 — Flag low-confidence samples for human review
agent.flag_for_review(df_labeled, threshold=0.75)
# → low_confidence_review.json

```

---

## Requirements

### `auto_label`
- Must support **at least one modality**: `text`, `audio`, or `image`
- Recommended backends:
  - `text` → spaCy pipeline or zero-shot classifier (e.g., `facebook/bart-large-mnli`)
  - `audio` → OpenAI Whisper
  - `image` → YOLOv8

### `generate_spec`
The output `annotation_spec.md` must contain:
- Task description and objective
- All label classes with clear definitions
- **At least 3 labeled examples per class**
- Edge cases and ambiguous examples with guidance

### `export_to_labelstudio`
- Output must be a valid **LabelStudio import JSON**
- The file must load into LabelStudio **without errors**
- Each task entry must follow the LabelStudio data format:
  ```json
  {
    "data": {"text": "..."},
    "annotations": [{"result": [...]}]
  }
  ```

### `check_quality`
Return a `QualityMetrics` dict with:
- `kappa` — Cohen's κ (or `agreement_pct` as a fallback)
- `label_dist` — label frequency distribution
- `confidence_mean` — mean prediction confidence score

### `flag_for_review` *(mandatory)*
- Automatically flags any sample where `confidence < threshold` (default: `0.75`)
- Saves flagged samples to `low_confidence_review.json` in LabelStudio import format
- These tasks must be ready to load directly into LabelStudio for human annotation
- `confidence_threshold` must be a configurable constructor parameter

---

## `process` Tool — LLM-Driven DataFrame Augmentation

### Overview

`process(df_path, prompt) -> df_path` augments a DataFrame by applying an
LLM-generated transformation to a filtered subset of rows. The full pipeline
has three stages: **filter → transform-prompt → parallel execution**.

### Stage 1 — Row Selection (Agent Call)

The agent sends the DataFrame schema + a sample of rows to the LLM via the
OpenAI API and asks it to decide **which rows** should be transformed based on
the user's prompt.

```
Input (JSON):
{
  "prompt": "Translate all English texts to French",
  "schema": ["id", "text", "lang"],
  "sample_rows": [...]
}

Output (JSON):
{
  "filter_condition": "lang == 'en'",
  "row_indices": [0, 3, 7, ...]
}
```

- The LLM returns either a **pandas filter expression** or an **explicit list of row indices**
- The agent applies the filter deterministically to get `df_filtered`

### Stage 2 — Transformation Prompt Generation (Agent Call)

The agent makes a second OpenAI API call, passing `df_filtered` and the original
user prompt, and asks the model to produce a **row-level transformation prompt**
— a precise instruction applicable to a single JSON row.

```
Input (JSON):
{
  "user_prompt": "Translate all English texts to French",
  "sample_filtered_row": {"id": 0, "text": "Hello world", "lang": "en"}
}

Output (JSON):
{
  "row_prompt": "Translate the value of the 'text' field from English to French. Return JSON with the same keys."
}
```

### Stage 3 — Parallel Row Transformation (Deterministic Skill)

Each filtered row is serialized to JSON and sent independently to the OpenAI API
using the generated `row_prompt`. Requests run in **parallel batches** of
configurable size.

```python
# Config (annotation_agent.yaml or constructor kwargs)
process_config:
  parallel_workers: 16      # number of concurrent API requests
  model: "gpt-4o-mini"      # model used for row transformation
  timeout_per_row: 10       # seconds
  max_retries: 3
```

**Execution flow:**

```
df_filtered
    │
    ├─ serialize each row → JSON string
    │
    ├─ split into batches of `parallel_workers`
    │
    ├─ asyncio.gather / ThreadPoolExecutor:
    │       POST /v1/chat/completions
    │       body: { "messages": [system: row_prompt, user: row_json] }
    │       response: transformed row JSON
    │
    ├─ collect all transformed rows → df_transformed
    │
    └─ merge df_transformed back into original df (update filtered rows)
         → save to {original_stem}_processed.csv → return path
```

**Each API call contract:**
```
Input row (JSON):  {"id": 3, "text": "Good morning", "lang": "en"}
Output row (JSON): {"id": 3, "text": "Bonjour", "lang": "en"}
```

- The model must return **valid JSON with id, old keys and new keys**; malformed responses are retried up to `max_retries`
- Rows outside the filter are copied unchanged into the final DataFrame

