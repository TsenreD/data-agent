import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


TOKEN_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "being",
    "could",
    "from",
    "have",
    "into",
    "just",
    "more",
    "only",
    "other",
    "some",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "very",
    "what",
    "when",
    "which",
    "with",
    "would",
    "your",
}


def generate_eda_report(frame: pd.DataFrame, output_dir: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {"row_count": int(len(frame))}
    artifacts: dict[str, str] = {}

    if "label" in frame.columns and frame["label"].notna().any():
        label_counts = frame["label"].fillna("unknown").astype(str).value_counts().sort_values(ascending=False)
        metrics["class_distribution"] = label_counts.to_dict()

        plt.figure(figsize=(8, 4))
        label_counts.plot(kind="bar", color="#3366cc")
        plt.title("Class distribution")
        plt.xlabel("label")
        plt.ylabel("count")
        plt.tight_layout()
        class_plot = output_path / "class_distribution.png"
        plt.savefig(class_plot)
        plt.close()
        artifacts["class_distribution_plot"] = str(class_plot)

    if "text" in frame.columns and frame["text"].notna().any():
        text_series = frame["text"].fillna("").astype(str)
        lengths = text_series.str.split().str.len()
        metrics["text_length"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
        }

        plt.figure(figsize=(8, 4))
        lengths.plot(kind="hist", bins=20, color="#cc6633")
        plt.title("Text length distribution")
        plt.xlabel("tokens")
        plt.tight_layout()
        text_plot = output_path / "text_length_distribution.png"
        plt.savefig(text_plot)
        plt.close()
        artifacts["text_length_plot"] = str(text_plot)

        tokens = Counter(
            token
            for text in text_series
            for token in TOKEN_RE.findall(text.lower())
            if token not in STOPWORDS
        )
        top_words = tokens.most_common(20)
        metrics["top_words"] = [{"word": word, "count": count} for word, count in top_words]

        if top_words:
            words, counts = zip(*top_words)
            plt.figure(figsize=(10, 5))
            plt.bar(words, counts, color="#228b22")
            plt.title("Top 20 words")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            top_words_plot = output_path / "top_words.png"
            plt.savefig(top_words_plot)
            plt.close()
            artifacts["top_words_plot"] = str(top_words_plot)

    return metrics, artifacts
