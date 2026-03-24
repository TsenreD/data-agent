# Annotation Specification

## Task

- Task: `annotation`
- Objective: Produce `label` for the configured annotation task.
- Modality: `text`
- Target column: `label`
- Human review threshold: `0.75`

## Classes

- No explicit classes configured.

## Edge Cases

- Preserve math, code, or symbolic formatting when present.
- Leave rows unresolved when the model output is incomplete or not confidently parseable.
