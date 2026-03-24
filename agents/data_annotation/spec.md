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

agent = AnnotationAgent(modality='text')

# Step 1 — Auto-label
df_labeled = agent.auto_label(df)

# Step 2 — Generate annotation spec
spec = agent.generate_spec(df, task='Solve given problem')
# → annotation_spec.md: task description, classes, examples, edge cases

# Step 3 — Quality check
metrics = agent.check_quality(df_labeled)
# → {'kappa': 0.72, 'label_dist': {...}, 'confidence_mean': 0.85}

# Step 4 — Export full dataset to LabelStudio
agent.export_to_labelstudio(df_labeled)
# → labelstudio_import.json

# Step 5 — Flag low-confidence samples for human review
agent.flag_for_review(df_labeled)
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
