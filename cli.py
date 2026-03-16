import argparse
import json
from pathlib import Path

from agents import DataCollectionAgent, PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-agent",
        description="Run the configured agents pipeline.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the pipeline configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where collected artifacts will be written.",
    )
    parser.add_argument(
        "--print-head",
        type=int,
        default=0,
        help="Print the first N dataframe rows as JSON after the pipeline run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    runner = PipelineRunner(
        [
            DataCollectionAgent(
                config=Path(args.config),
                output_dir=Path(args.output_dir),
            )
        ]
    )
    result = runner.run()

    summary = {
        "rows": 0 if result.dataframe is None else int(len(result.dataframe)),
        "dataframe_path": None if result.dataframe_path is None else str(result.dataframe_path),
        "failed_sources": result.metadata.get("failed_sources", []),
        "artifacts": result.artifacts,
    }
    print(json.dumps(summary, indent=2))

    if args.print_head and result.dataframe is not None:
        preview = result.dataframe.head(args.print_head).to_dict(orient="records")
        print(json.dumps(preview, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
