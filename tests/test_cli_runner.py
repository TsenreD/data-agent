from pathlib import Path

import yaml

import cli


class _DummyAgent:
    stage_name = "unknown"

    def __init__(self, config, output_dir) -> None:
        self.config = config
        self.output_dir = output_dir


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _named_agent(stage_name: str):
    class _NamedDummyAgent(_DummyAgent):
        pass

    _NamedDummyAgent.stage_name = stage_name
    return _NamedDummyAgent


def test_build_runner_includes_active_learning_last(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "DataCollectionAgent", _named_agent("collection"))
    monkeypatch.setattr(cli, "DataQualityAgent", _named_agent("quality"))
    monkeypatch.setattr(cli, "DataAnnotationAgent", _named_agent("annotation"))
    monkeypatch.setattr(cli, "ActiveLearningAgent", _named_agent("active_learning"))

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "agents": {
                "collection": {"enabled": True},
                "quality": {"enabled": True},
                "annotation": {"enabled": True},
                "active_learning": {"enabled": True},
            }
        },
    )

    runner = cli.build_runner(config_path, tmp_path / "data")
    assert len(runner.agents) == 4
    assert [agent.stage_name for agent in runner.agents] == [
        "collection",
        "quality",
        "annotation",
        "active_learning",
    ]


def test_build_runner_respects_active_learning_disable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "DataCollectionAgent", _named_agent("collection"))
    monkeypatch.setattr(cli, "DataQualityAgent", _named_agent("quality"))
    monkeypatch.setattr(cli, "DataAnnotationAgent", _named_agent("annotation"))
    monkeypatch.setattr(cli, "ActiveLearningAgent", _named_agent("active_learning"))

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        {
            "agents": {
                "collection": {"enabled": True},
                "quality": {"enabled": True},
                "annotation": {"enabled": True},
                "active_learning": {"enabled": False},
            }
        },
    )

    runner = cli.build_runner(config_path, tmp_path / "data")
    assert len(runner.agents) == 3
    assert [agent.stage_name for agent in runner.agents] == [
        "collection",
        "quality",
        "annotation",
    ]
