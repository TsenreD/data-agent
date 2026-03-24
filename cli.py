import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents import ActiveLearningAgent, DataAnnotationAgent, DataCollectionAgent, DataQualityAgent, PipelineRunner


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
        default="data",
        help="Base directory where per-agent artifacts will be written.",
    )
    parser.add_argument(
        "--print-head",
        type=int,
        default=0,
        help="Print the first N dataframe rows as JSON after the pipeline run.",
    )
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _agent_enabled(config: Mapping[str, Any], agent_name: str) -> bool:
    agents_config = config.get("agents", {})
    if not isinstance(agents_config, Mapping):
        return True
    agent_config = agents_config.get(agent_name)
    if not isinstance(agent_config, Mapping):
        return True
    return bool(agent_config.get("enabled", True))


def build_runner(config_path: Path, output_dir: Path) -> PipelineRunner:
    config = _load_config(config_path)
    agents = []
    if _agent_enabled(config, "collection"):
        agents.append(
            DataCollectionAgent(
                config=config_path,
                output_dir=output_dir,
            )
        )
    if _agent_enabled(config, "quality"):
        agents.append(
            DataQualityAgent(
                config=config_path,
                output_dir=output_dir,
            )
        )
    if _agent_enabled(config, "annotation"):
        agents.append(
            DataAnnotationAgent(
                config=config_path,
                output_dir=output_dir,
            )
        )
    # if _agent_enabled(config, "active_learning"):
    #     agents.append(
    #         ActiveLearningAgent(
    #             config=config_path,
    #             output_dir=output_dir,
    #         )
    #     )
    return PipelineRunner(agents)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    runner = build_runner(config_path, output_dir)
    result = runner.run()

    summary = {
        "rows": 0 if result.dataframe is None else int(len(result.dataframe)),
        "dataframe_path": None if result.dataframe_path is None else str(result.dataframe_path),
        "failed_sources": result.metadata.get("failed_sources", []),
        "artifacts": result.artifacts,
    }
    print(json.dumps(summary, indent=2))

    if args.print_head and result.dataframe is not None:
        preview_json = result.dataframe.head(args.print_head).to_json(
            orient="records",
            date_format="iso",
            force_ascii=False,
        )
        print(json.dumps(json.loads(preview_json), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
