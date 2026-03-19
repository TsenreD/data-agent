import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from ..base import AgentResult, BaseAgent
from .smolagents_backend import SmolagentsQualityBackend


DEFAULT_STRATEGY = {
    "missing": "median",
    "duplicates": "drop",
    "outliers": "clip_iqr",
}
DEFAULT_IMBALANCE_THRESHOLD = 0.75


@dataclass(slots=True)
class QualityArtifacts:
    report_path: Path
    analysis_path: Path
    cleaned_dataset_path: Path
    comparison_path: Path
    notebook_path: Path


class DataQualityAgent(BaseAgent):
    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        output_dir: str | Path = "data/raw",
        notebook_path: str | Path | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_config = self._resolve_quality_config()
        self.default_strategy = self._resolve_default_strategy()
        self.notebook_path = (
            Path(notebook_path)
            if notebook_path is not None
            else Path(self.quality_config.get("notebook_path", self.output_dir / "eda.ipynb"))
        )
        self.imbalance_threshold = float(
            self.quality_config.get("imbalance_threshold", DEFAULT_IMBALANCE_THRESHOLD)
        )
        self.label_column = self.quality_config.get("label_column")
        self.task_description = str(self.quality_config.get("task_description", "")).strip()
        self.backend = SmolagentsQualityBackend(
            llm_config=self.config.get("llm", {}),
            agent_config=self.quality_config,
        )

    def run(
        self,
        dataframe: pd.DataFrame,
        strategy: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        input_path = self._stage_dataframe(dataframe, self.output_dir / "_quality_input.jsonl")
        cleaned_path = self.output_dir / "_quality_output.jsonl"
        result = self.backend.fix(
            input_path,
            cleaned_path,
            strategy=self._normalize_strategy(strategy),
            task_description=self.task_description,
            label_column=self._normalized_label_column(),
            imbalance_threshold=self.imbalance_threshold,
        )
        if not result.success:
            raise RuntimeError(self._summarize_backend_failure(result))
        cleaned = self._read_dataframe(cleaned_path)
        normalized = self._normalize_cleaned_frame(cleaned, dataframe)
        normalized.to_json(cleaned_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        return normalized

    def execute(self, payload: Any | None = None) -> AgentResult:
        logs: list[str] = []
        self._record_log("Starting DataQualityAgent execution.", logs)

        upstream = payload if isinstance(payload, AgentResult) else None
        source_frame = self._resolve_dataframe(payload)
        source_dataset_path = self._resolve_input_path(payload, source_frame)
        explicit_strategy = self._resolve_strategy(payload)
        notebook_path = self._resolve_notebook_path(upstream)
        cleaned_dataset_path = self.output_dir / "cleaned_dataset.jsonl"

        detection = self.backend.detect_issues(
            source_dataset_path,
            project_context=self.config.get("project", {}),
            label_column=self._normalized_label_column(source_frame),
            imbalance_threshold=self.imbalance_threshold,
            preview_frame=source_frame,
        )
        analysis = self.backend.analyze(
            source_dataset_path,
            detection_report=detection.report or {},
            project_context=self.config.get("project", {}),
            task_description=self.task_description,
            strategy_hint=explicit_strategy,
            label_column=self._normalized_label_column(source_frame),
            preview_frame=source_frame,
        )
        effective_strategy = self._effective_strategy(explicit_strategy, analysis.analysis or {})
        fix_result = self.backend.fix(
            source_dataset_path,
            cleaned_dataset_path,
            strategy=effective_strategy,
            task_description=self.task_description,
            analysis_context=analysis.analysis,
            label_column=self._normalized_label_column(source_frame),
            imbalance_threshold=self.imbalance_threshold,
        )
        for backend_result in (detection, analysis, fix_result):
            self._record_existing_logs(backend_result.logs, logs)
            if not backend_result.success:
                raise RuntimeError(self._summarize_backend_failure(backend_result))
        comparison = self.backend.compare(
            source_dataset_path,
            cleaned_dataset_path,
            analysis_context=analysis.analysis,
            label_column=self._normalized_label_column(source_frame),
            imbalance_threshold=self.imbalance_threshold,
        )
        self._record_existing_logs(comparison.logs, logs)
        if not comparison.success:
            raise RuntimeError(self._summarize_backend_failure(comparison))

        cleaned = self._normalize_cleaned_frame(self._read_dataframe(cleaned_dataset_path), source_frame)
        artifacts = self._write_artifacts(
            report=detection.report or {},
            analysis=analysis.analysis or {},
            cleaned=cleaned,
            comparison=comparison.comparison or {},
            notebook_path=notebook_path,
        )
        self._append_notebook_section(
            report=detection.report or {},
            analysis=analysis.analysis or {},
            comparison=comparison.comparison or {},
            strategy=fix_result.strategy_used or dict(effective_strategy),
            justification=fix_result.justification,
            notebook_path=artifacts.notebook_path,
            source_dataset_path=str(source_dataset_path),
            cleaned_dataset_path=artifacts.cleaned_dataset_path,
        )
        self._record_log(f"Wrote quality report to {artifacts.report_path}.", logs)
        self._record_log(f"Wrote cleaned dataset to {artifacts.cleaned_dataset_path}.", logs)
        self._record_log(f"Wrote comparison report to {artifacts.comparison_path}.", logs)
        self._record_log(f"Updated notebook at {artifacts.notebook_path}.", logs)
        self._record_log("DataQualityAgent execution finished successfully.", logs)

        schema = {column: str(dtype) for column, dtype in cleaned.dtypes.items()}
        comparison_payload = comparison.comparison or {}
        metrics = {
            "row_count": int(len(cleaned)),
            "missing_total_after": int(comparison_payload.get("missing", {}).get("after_total", 0)),
            "duplicates_after": int(comparison_payload.get("duplicates", {}).get("after", 0)),
            "outliers_after": int(comparison_payload.get("outliers", {}).get("after_total", 0)),
        }

        upstream_artifacts = dict(upstream.artifacts) if upstream is not None else {}
        upstream_metadata = deepcopy(upstream.metadata) if upstream is not None else {}
        upstream_logs = list(upstream.logs) if upstream is not None else []
        upstream_metrics = dict(upstream.metrics) if upstream is not None else {}

        merged_artifacts = {
            **upstream_artifacts,
            "quality_report": str(artifacts.report_path),
            "quality_analysis": str(artifacts.analysis_path),
            "cleaned_dataset": str(artifacts.cleaned_dataset_path),
            "quality_comparison": str(artifacts.comparison_path),
            "eda_notebook": str(artifacts.notebook_path),
        }
        merged_metadata = {
            **upstream_metadata,
            "quality_report": detection.report or {},
            "quality_analysis": analysis.analysis or {},
            "quality_comparison": comparison_payload,
            "quality_strategy": fix_result.strategy_used or dict(effective_strategy),
            "quality_notes": {
                "detect": detection.notes,
                "analyze": analysis.notes,
                "fix": fix_result.notes,
                "compare": comparison.notes,
            },
            "quality_justification": fix_result.justification,
        }
        merged_metrics = {**upstream_metrics, **metrics}
        merged_logs = self._merge_logs(upstream_logs, logs)

        return AgentResult(
            dataframe=cleaned,
            dataframe_path=artifacts.cleaned_dataset_path,
            dataframe_schema=schema,
            metrics=merged_metrics,
            artifacts=merged_artifacts,
            logs=merged_logs,
            metadata=merged_metadata,
        )

    def detect_issues(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        input_path = self._stage_dataframe(dataframe, self.output_dir / "_quality_detect_input.jsonl")
        result = self.backend.detect_issues(
            input_path,
            project_context=self.config.get("project", {}),
            label_column=self._normalized_label_column(dataframe),
            imbalance_threshold=self.imbalance_threshold,
            preview_frame=dataframe,
        )
        if not result.success or result.report is None:
            raise RuntimeError(self._summarize_backend_failure(result))
        return result.report

    def fix(self, dataframe: pd.DataFrame, strategy: Mapping[str, Any] | None = None) -> pd.DataFrame:
        input_path = self._stage_dataframe(dataframe, self.output_dir / "_quality_fix_input.jsonl")
        output_path = self.output_dir / "_quality_fix_output.jsonl"
        result = self.backend.fix(
            input_path,
            output_path,
            strategy=self._normalize_strategy(strategy),
            task_description=self.task_description,
            label_column=self._normalized_label_column(dataframe),
            imbalance_threshold=self.imbalance_threshold,
        )
        if not result.success:
            raise RuntimeError(self._summarize_backend_failure(result))
        cleaned = self._normalize_cleaned_frame(self._read_dataframe(output_path), dataframe)
        cleaned.to_json(output_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        return cleaned

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict[str, Any]:
        before_path = self._stage_dataframe(df_before, self.output_dir / "_quality_compare_before.jsonl")
        after_path = self._stage_dataframe(df_after, self.output_dir / "_quality_compare_after.jsonl")
        result = self.backend.compare(
            before_path,
            after_path,
            label_column=self._normalized_label_column(df_before),
            imbalance_threshold=self.imbalance_threshold,
        )
        if not result.success or result.comparison is None:
            raise RuntimeError(self._summarize_backend_failure(result))
        return result.comparison

    def _resolve_dataframe(self, payload: Any | None) -> pd.DataFrame:
        if isinstance(payload, AgentResult):
            if payload.dataframe is not None:
                return payload.dataframe.copy()
            if payload.dataframe_path is not None:
                return self._read_dataframe(Path(payload.dataframe_path))
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, Mapping):
            if isinstance(payload.get("dataframe"), pd.DataFrame):
                return payload["dataframe"].copy()
            if payload.get("dataframe_path"):
                return self._read_dataframe(Path(payload["dataframe_path"]))
        input_path = self.quality_config.get("input_path")
        if input_path:
            return self._read_dataframe(Path(input_path))
        default_input_path = self.output_dir / "unified_dataset.jsonl"
        if default_input_path.exists():
            return self._read_dataframe(default_input_path)
        raise ValueError("DataQualityAgent requires a dataframe payload or quality.input_path.")

    def _resolve_input_path(self, payload: Any | None, dataframe: pd.DataFrame) -> Path:
        if isinstance(payload, AgentResult) and payload.dataframe_path is not None:
            return Path(payload.dataframe_path)
        if isinstance(payload, Mapping) and payload.get("dataframe_path"):
            return Path(payload["dataframe_path"])
        input_path = self.quality_config.get("input_path")
        if input_path:
            return Path(input_path)
        default_input_path = self.output_dir / "unified_dataset.jsonl"
        if default_input_path.exists():
            return default_input_path
        return self._stage_dataframe(dataframe, self.output_dir / "unified_dataset.jsonl")

    def _resolve_strategy(self, payload: Any | None) -> dict[str, Any]:
        if isinstance(payload, Mapping) and isinstance(payload.get("strategy"), Mapping):
            return self._normalize_strategy(payload["strategy"])
        return self.default_strategy

    def _resolve_quality_config(self) -> dict[str, Any]:
        agents_config = self.config.get("agents", {})
        if isinstance(agents_config, Mapping) and isinstance(agents_config.get("quality"), Mapping):
            return dict(agents_config["quality"])
        quality_config = self.config.get("quality", {})
        if isinstance(quality_config, Mapping):
            return dict(quality_config)
        return {}

    def _resolve_default_strategy(self) -> dict[str, Any]:
        configured = self.quality_config.get("strategy")
        if isinstance(configured, Mapping):
            return self._normalize_strategy(configured)
        return dict(DEFAULT_STRATEGY)

    def _normalize_strategy(self, strategy: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(DEFAULT_STRATEGY)
        if strategy:
            merged.update({key: value for key, value in strategy.items() if value is not None})
        return merged

    def _effective_strategy(
        self,
        explicit_strategy: Mapping[str, Any],
        analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        recommended = analysis.get("recommended_strategy")
        if isinstance(recommended, Mapping):
            return self._normalize_strategy(recommended)
        return self._normalize_strategy(explicit_strategy)

    def _normalized_label_column(self, dataframe: pd.DataFrame | None = None) -> str | None:
        if isinstance(self.label_column, str) and self.label_column.strip():
            return self.label_column
        if dataframe is not None and "label" in dataframe.columns:
            return "label"
        return None

    def _write_artifacts(
        self,
        report: Mapping[str, Any],
        analysis: Mapping[str, Any],
        cleaned: pd.DataFrame,
        comparison: Mapping[str, Any],
        *,
        notebook_path: Path,
    ) -> QualityArtifacts:
        report_path = self.output_dir / "quality_report.json"
        analysis_path = self.output_dir / "quality_analysis.json"
        comparison_path = self.output_dir / "quality_comparison.json"
        cleaned_dataset_path = self.output_dir / "cleaned_dataset.jsonl"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        cleaned.to_json(cleaned_dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        return QualityArtifacts(
            report_path=report_path,
            analysis_path=analysis_path,
            cleaned_dataset_path=cleaned_dataset_path,
            comparison_path=comparison_path,
            notebook_path=notebook_path,
        )

    def _append_notebook_section(
        self,
        *,
        report: Mapping[str, Any],
        analysis: Mapping[str, Any],
        comparison: Mapping[str, Any],
        strategy: Mapping[str, Any],
        justification: str,
        notebook_path: Path,
        source_dataset_path: str,
        cleaned_dataset_path: Path,
    ) -> None:
        notebook = self._load_notebook(notebook_path)
        notebook["cells"].extend(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._quality_markdown_lines(report, analysis, comparison, strategy, justification),
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": self._quality_visualization_lines(
                        analysis=analysis,
                        source_dataset_path=source_dataset_path,
                        cleaned_dataset_path=cleaned_dataset_path,
                    ),
                },
            ]
        )
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_notebook(self, notebook_path: Path) -> dict[str, Any]:
        if notebook_path.exists():
            return json.loads(notebook_path.read_text(encoding="utf-8"))
        return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}

    def _quality_markdown_lines(
        self,
        report: Mapping[str, Any],
        analysis: Mapping[str, Any],
        comparison: Mapping[str, Any],
        strategy: Mapping[str, Any],
        justification: str,
    ) -> list[str]:
        imbalance = report.get("imbalance", {})
        label_column = imbalance.get("column") or "label"
        task_interpretation = analysis.get("task_interpretation", {})
        quality_focus = analysis.get("quality_focus", {})
        primary_modality = (
            task_interpretation.get("primary_modality")
            or quality_focus.get("primary_modality")
            or "unknown"
        )
        relevant_checks = ", ".join(quality_focus.get("relevant_checks", [])) or "not specified"
        irrelevant_checks = ", ".join(quality_focus.get("irrelevant_checks", [])) or "none"
        priority_actions = ", ".join(quality_focus.get("priority_actions", [])) or "not specified"
        imbalance_after = comparison.get("imbalance", {})
        if isinstance(imbalance_after, Mapping):
            majority_after = imbalance_after.get("after_majority_share")
            if majority_after is None and isinstance(imbalance_after.get("after"), Mapping):
                majority_after = imbalance_after.get("after", {}).get("majority_share")
        else:
            majority_after = None
        modality_effects = comparison.get("modality_metrics", {})
        return [
            "## Data Quality Review\n",
            "\n",
            "### Analyzer View\n",
            "\n",
            f"- Task interpretation: {task_interpretation.get('task_type', 'unknown')}\n",
            f"- Primary modality: {primary_modality}\n",
            f"- Target semantics: {task_interpretation.get('label_semantics', task_interpretation.get('target_role', 'unknown'))}\n",
            f"- Relevant checks: {relevant_checks}\n",
            f"- Lower-value checks: {irrelevant_checks}\n",
            f"- Priority actions: {priority_actions}\n",
            "\n",
            "### Strategy Justification\n",
            "\n",
            f"{justification or 'The cleaning strategy was generated by the quality agent after inspecting the real dataset and applying the requested fix policy.'}\n",
            "\n",
            f"- Missing values: `{strategy.get('missing', 'unknown')}`\n",
            f"- Duplicates: `{strategy.get('duplicates', 'unknown')}`\n",
            f"- Outliers: `{strategy.get('outliers', 'unknown')}`\n",
            "\n",
            "### Findings\n",
            "\n",
            f"- Missing values before cleaning: {report.get('missing', {}).get('total', 0)}\n",
            f"- Duplicate rows before cleaning: {report.get('duplicates', 0)}\n",
            f"- Numeric outliers before cleaning: {report.get('outlier_summary', {}).get('total', 0)}\n",
            f"- Imbalance column: `{label_column}`\n",
            (
                f"- Majority class share after cleaning: {float(majority_after):.3f}\n"
                if isinstance(majority_after, (int, float))
                else "- Majority class share after cleaning: not applicable\n"
            ),
            "\n",
            "### Before / After\n",
            "\n",
            f"- Missing values: {comparison.get('missing', {}).get('before_total', 0)} -> {comparison.get('missing', {}).get('after_total', 0)}\n",
            f"- Duplicates: {comparison.get('duplicates', {}).get('before', 0)} -> {comparison.get('duplicates', {}).get('after', 0)}\n",
            f"- Outliers: {comparison.get('outliers', {}).get('before_total', 0)} -> {comparison.get('outliers', {}).get('after_total', 0)}\n",
            (
                f"- Modality effects: {json.dumps(modality_effects, ensure_ascii=False)}\n"
                if modality_effects
                else ""
            ),
        ]

    def _quality_visualization_lines(
        self,
        *,
        analysis: Mapping[str, Any],
        source_dataset_path: str,
        cleaned_dataset_path: Path,
    ) -> list[str]:
        quality_focus = analysis.get("quality_focus", {})
        task_interpretation = analysis.get("task_interpretation", {})
        notebook_sections = quality_focus.get("notebook_sections", [])
        primary_modality = (
            quality_focus.get("primary_modality")
            or task_interpretation.get("primary_modality")
            or ""
        )
        show_label = "label_distribution" in notebook_sections
        show_text_length = any(
            section in notebook_sections for section in ("text_length_distribution", "text_length_analysis")
        )
        show_source_coverage = (
            any(section in notebook_sections for section in ("source_label_coverage", "source_based_label_coverage"))
            or primary_modality == "text"
        )
        lines = [
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "\n",
            f"raw_df = pd.read_json({source_dataset_path!r}, lines=True)\n",
            f"clean_df = pd.read_json({str(cleaned_dataset_path)!r}, lines=True)\n",
            f"primary_modality = {primary_modality!r}\n",
            "\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "missing_counts = raw_df.isna().sum().sort_values(ascending=False)\n",
            "missing_counts = missing_counts[missing_counts > 0]\n",
            "if not missing_counts.empty:\n",
            "    sns.barplot(x=missing_counts.values, y=missing_counts.index, ax=axes[0, 0], color='#d97706')\n",
            "    axes[0, 0].set_title('Missing values by column')\n",
            "else:\n",
            "    axes[0, 0].text(0.5, 0.5, 'No missing values', ha='center', va='center')\n",
            "    axes[0, 0].set_axis_off()\n",
            "\n",
        ]
        if show_label:
            lines.extend(
                [
                    "if 'label' in raw_df.columns:\n",
                    "    label_counts = raw_df['label'].dropna().astype(str).value_counts().head(10)\n",
                    "    if not label_counts.empty:\n",
                    "        sns.barplot(x=label_counts.index, y=label_counts.values, ax=axes[0, 1], color='#2563eb')\n",
                    "        axes[0, 1].tick_params(axis='x', rotation=45)\n",
                    "        axes[0, 1].set_title('Label distribution')\n",
                    "    else:\n",
                    "        axes[0, 1].text(0.5, 0.5, 'No label values', ha='center', va='center')\n",
                    "        axes[0, 1].set_axis_off()\n",
                    "else:\n",
                    "    axes[0, 1].text(0.5, 0.5, 'No label column', ha='center', va='center')\n",
                    "    axes[0, 1].set_axis_off()\n",
                    "\n",
                ]
            )
        elif show_text_length:
            lines.extend(
                [
                    "if 'text' in raw_df.columns:\n",
                    "    text_lengths = raw_df['text'].fillna('').astype(str).str.split().str.len()\n",
                    "    sns.histplot(text_lengths, bins=30, ax=axes[0, 1], color='#2563eb')\n",
                    "    axes[0, 1].set_title('Text length distribution')\n",
                    "else:\n",
                    "    axes[0, 1].text(0.5, 0.5, 'No text column', ha='center', va='center')\n",
                    "    axes[0, 1].set_axis_off()\n",
                    "\n",
                ]
            )
        elif show_source_coverage:
            lines.extend(
                [
                    "if {'source', 'label'}.issubset(raw_df.columns):\n",
                    "    coverage = raw_df.assign(label_present=raw_df['label'].notna()).groupby('source', dropna=False)['label_present'].mean().sort_values(ascending=False).head(10)\n",
                    "    if not coverage.empty:\n",
                    "        sns.barplot(x=coverage.values, y=coverage.index.astype(str), ax=axes[0, 1], color='#2563eb')\n",
                    "        axes[0, 1].set_title('Label coverage by source')\n",
                    "        axes[0, 1].set_xlim(0, 1)\n",
                    "    else:\n",
                    "        axes[0, 1].text(0.5, 0.5, 'No source coverage data', ha='center', va='center')\n",
                    "        axes[0, 1].set_axis_off()\n",
                    "elif 'source' in raw_df.columns:\n",
                    "    source_counts = raw_df['source'].astype(str).value_counts().head(10)\n",
                    "    sns.barplot(x=source_counts.values, y=source_counts.index, ax=axes[0, 1], color='#2563eb')\n",
                    "    axes[0, 1].set_title('Top sources')\n",
                    "else:\n",
                    "    axes[0, 1].text(0.5, 0.5, 'No source column', ha='center', va='center')\n",
                    "    axes[0, 1].set_axis_off()\n",
                    "\n",
                ]
            )
        else:
            lines.extend(
                [
                    "axes[0, 1].text(0.5, 0.5, 'Analyzer skipped target distribution for this task', ha='center', va='center')\n",
                    "axes[0, 1].set_axis_off()\n",
                    "\n",
                ]
            )
        lines.extend(
            [
            "numeric_columns = raw_df.select_dtypes(include=['number']).columns.tolist()\n",
            "if numeric_columns and not (len(numeric_columns) == 1 and 'label' in numeric_columns and primary_modality == 'text'):\n",
            "    sns.boxplot(data=raw_df[numeric_columns], orient='h', ax=axes[1, 0], color='#f59e0b')\n",
            "    axes[1, 0].set_title('Raw numeric distributions')\n",
            "    sns.boxplot(data=clean_df[numeric_columns], orient='h', ax=axes[1, 1], color='#10b981')\n",
            "    axes[1, 1].set_title('Cleaned numeric distributions')\n",
            "else:\n",
            "    if 'text' in raw_df.columns:\n",
            "        raw_lengths = raw_df['text'].fillna('').astype(str).str.split().str.len()\n",
            "        clean_lengths = clean_df['text'].fillna('').astype(str).str.split().str.len()\n",
            "        sns.histplot(raw_lengths, bins=30, ax=axes[1, 0], color='#f59e0b')\n",
            "        axes[1, 0].set_title('Raw text length distribution')\n",
            "        sns.histplot(clean_lengths, bins=30, ax=axes[1, 1], color='#10b981')\n",
            "        axes[1, 1].set_title('Cleaned text length distribution')\n",
            "    else:\n",
            "        axes[1, 0].text(0.5, 0.5, 'No numeric columns', ha='center', va='center')\n",
            "        axes[1, 0].set_axis_off()\n",
            "        axes[1, 1].text(0.5, 0.5, 'No numeric columns', ha='center', va='center')\n",
            "        axes[1, 1].set_axis_off()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
        ])
        return lines

    def _resolve_notebook_path(self, upstream: AgentResult | None) -> Path:
        if upstream is not None:
            candidate = upstream.artifacts.get("eda_notebook")
            if candidate:
                return Path(candidate)
        return self.notebook_path

    def _stage_dataframe(self, dataframe: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_json(path, orient="records", lines=True, force_ascii=False, date_format="iso")
        return path

    def _normalize_cleaned_frame(self, cleaned: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        normalized = cleaned.copy()
        for column in normalized.columns:
            if column not in reference.columns:
                continue
            original_sample = reference[column].dropna().head(10).tolist()
            if any(isinstance(value, dict) for value in original_sample):
                normalized[column] = normalized[column].map(self._restore_json_dict)
            elif any(isinstance(value, list) for value in original_sample):
                normalized[column] = normalized[column].map(self._restore_json_list)
            if pd.api.types.is_datetime64_any_dtype(reference[column]):
                normalized[column] = pd.to_datetime(normalized[column], errors="coerce", utc=True)
        return normalized

    def _restore_json_dict(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped.startswith("{"):
            return value
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value

    def _restore_json_list(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped.startswith("["):
            return value
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, list) else value

    def _read_dataframe(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        raise ValueError(f"Unsupported input dataset format '{path.suffix}'.")

    def _load_config(self, config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        path = Path(config)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _record_existing_logs(self, entries: list[str], logs: list[str]) -> None:
        for entry in entries:
            if entry not in logs:
                logs.append(entry)

    def _merge_logs(self, existing: list[str], new: list[str]) -> list[str]:
        merged = list(existing)
        for entry in new:
            if entry not in merged:
                merged.append(entry)
        return merged

    def _summarize_backend_failure(self, result: Any) -> str:
        attempts = getattr(result, "attempts", []) or []
        if attempts:
            last_attempt = attempts[-1]
            if last_attempt.get("error_message"):
                return str(last_attempt["error_message"])
            notes = last_attempt.get("notes") or []
            if notes:
                return " | ".join(str(note) for note in notes)
        notes = getattr(result, "notes", []) or []
        if notes:
            return " | ".join(str(note) for note in notes)
        logs = getattr(result, "logs", []) or []
        if logs:
            return str(logs[-1])
        return "unknown quality backend error"

    def _record_log(self, message: str, logs: list[str] | None = None) -> str:
        entry = f"[{datetime.now(UTC).isoformat()}] {message}"
        if logs is not None:
            logs.append(entry)
        print(entry, file=sys.stdout, flush=True)
        return entry
