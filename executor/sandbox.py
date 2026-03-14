import ast
import io
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    stdout: str = ""
    stderr: str = ""
    generated_code: str = ""
    error_message: str | None = None
    exception_type: str | None = None
    returncode: int | None = None


class SandboxExecutor:
    """Long-lived Docker sandbox for model-generated extraction code."""

    def __init__(
        self,
        timeout_seconds: int = 20,
        allowed_imports: set[str] | None = None,
        image: str | None = None,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 64,
        default_network_mode: str = "none",
        container_name_prefix: str = "data-agent-sandbox",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.image = image or os.environ.get("DATA_AGENT_SANDBOX_IMAGE", "data-agent-execution-backend:latest")
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.default_network_mode = default_network_mode
        self.container_name_prefix = container_name_prefix
        self.allowed_imports = allowed_imports or {
            "bs4",
            "collections",
            "datetime",
            "json",
            "math",
            "pandas",
            "re",
            "requests",
        }
        self._workspace_dir: tempfile.TemporaryDirectory[str] | None = None
        self._workspace_root: Path | None = None
        self._containers: dict[str, str] = {}

    def start(self) -> None:
        if self._workspace_dir is None:
            self._workspace_dir = tempfile.TemporaryDirectory(prefix="data-agent-sandbox-")
            self._workspace_root = Path(self._workspace_dir.name)

    def close(self) -> None:
        for container_name in list(self._containers.values()):
            try:
                subprocess.run(
                    ["docker", "stop", container_name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception:
                pass
        self._containers.clear()

        if self._workspace_dir is not None:
            self._workspace_dir.cleanup()
            self._workspace_dir = None
            self._workspace_root = None

    def __enter__(self) -> "SandboxExecutor":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def execute(self, code: str, context: dict[str, Any]) -> ExecutionResult:
        execution_dir: Path | None = None
        try:
            self._validate_imports(code)
            self.start()

            network_mode = self._resolve_network_mode(context)
            container_name = self._ensure_container(network_mode)
            execution_dir = self._create_execution_dir()

            script_path = execution_dir / "generated_runner.py"
            context_path = execution_dir / "context.json"
            output_path = execution_dir / "output.json"

            context_path.write_text(json.dumps(context), encoding="utf-8")
            script_path.write_text(self._build_runner_script(code), encoding="utf-8")

            try:
                completed = subprocess.run(
                    self._build_exec_command(container_name, execution_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return ExecutionResult(
                    success=False,
                    stdout=error.stdout or "",
                    stderr=error.stderr or "",
                    generated_code=code,
                    error_message=f"Sandbox execution timed out after {self.timeout_seconds} seconds.",
                    exception_type=type(error).__name__,
                )

            if completed.returncode != 0:
                return ExecutionResult(
                    success=False,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    generated_code=code,
                    error_message="Sandbox container execution failed.",
                    exception_type="SubprocessError",
                    returncode=completed.returncode,
                )

            if not output_path.exists():
                return ExecutionResult(
                    success=False,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    generated_code=code,
                    error_message="Sandbox completed without writing an output dataset.",
                    exception_type="MissingOutputError",
                    returncode=completed.returncode,
                )

            frame = pd.read_json(
                io.StringIO(output_path.read_text(encoding="utf-8")),
                orient="table",
            )
            return ExecutionResult(
                success=True,
                dataframe=frame,
                stdout=completed.stdout,
                stderr=completed.stderr,
                generated_code=code,
                returncode=completed.returncode,
            )
        except Exception as error:
            return ExecutionResult(
                success=False,
                generated_code=code,
                error_message=str(error),
                exception_type=type(error).__name__,
            )
        finally:
            if execution_dir is not None:
                shutil.rmtree(execution_dir, ignore_errors=True)

    def _resolve_network_mode(self, context: dict[str, Any]) -> str:
        source = context.get("source", {})
        allow_network = bool(source.get("allow_network", source.get("type") in {"scrape", "api"}))
        return "bridge" if allow_network else self.default_network_mode

    def _ensure_container(self, network_mode: str) -> str:
        existing = self._containers.get(network_mode)
        if existing:
            return existing

        if self._workspace_root is None:
            raise RuntimeError("Sandbox workspace is not initialized.")

        container_name = f"{self.container_name_prefix}-{network_mode}-{uuid.uuid4().hex[:8]}"
        completed = subprocess.run(
            self._build_run_command(container_name, network_mode),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to start sandbox container '{container_name}': {completed.stderr.strip()}"
            )

        self._containers[network_mode] = container_name
        return container_name

    def _create_execution_dir(self) -> Path:
        if self._workspace_root is None:
            raise RuntimeError("Sandbox workspace is not initialized.")

        execution_dir = self._workspace_root / uuid.uuid4().hex
        execution_dir.mkdir(parents=True, exist_ok=False)
        return execution_dir

    def _build_run_command(self, container_name: str, network_mode: str) -> list[str]:
        if self._workspace_root is None:
            raise RuntimeError("Sandbox workspace is not initialized.")

        workspace = str(self._workspace_root)
        user = f"{os.getuid()}:{os.getgid()}"
        return [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "--network",
            network_mode,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory_limit,
            "--cpus",
            self.cpu_limit,
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--user",
            user,
            "--workdir",
            "/workspace",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            self.image,
            "python3",
            "-c",
            "import time; time.sleep(10**9)",
        ]

    def _build_exec_command(self, container_name: str, execution_dir: Path) -> list[str]:
        in_container_dir = Path("/workspace") / execution_dir.name
        return [
            "docker",
            "exec",
            container_name,
            "python3",
            str(in_container_dir / "generated_runner.py"),
            str(in_container_dir / "context.json"),
            str(in_container_dir / "output.json"),
        ]

    def _validate_imports(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_module(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._check_module(node.module)

    def _check_module(self, module_name: str) -> None:
        root_module = module_name.split(".", maxsplit=1)[0]
        if root_module not in self.allowed_imports:
            raise ValueError(f"Import '{module_name}' is not allowed in sandbox execution.")

    def _build_runner_script(self, code: str) -> str:
        indented_code = textwrap.indent(code.strip(), "    ")
        return (
            "from __future__ import annotations\n\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n\n"
            "def _load_context(path: str) -> dict:\n"
            "    return json.loads(Path(path).read_text(encoding='utf-8'))\n\n"
            "def _write_output(frame: pd.DataFrame, path: str) -> None:\n"
            "    Path(path).write_text(frame.to_json(orient='table'), encoding='utf-8')\n\n"
            "def _user_run(context: dict) -> pd.DataFrame:\n"
            f"{indented_code}\n"
            "    if 'run' not in locals():\n"
            "        raise RuntimeError('Generated code must define run(context).')\n"
            "    return locals()['run'](context)\n\n"
            "def main() -> None:\n"
            "    context_path, output_path = sys.argv[1], sys.argv[2]\n"
            "    context = _load_context(context_path)\n"
            "    frame = _user_run(context)\n"
            "    if not isinstance(frame, pd.DataFrame):\n"
            "        raise TypeError('Generated run(context) must return a pandas DataFrame.')\n"
            "    _write_output(frame, output_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
