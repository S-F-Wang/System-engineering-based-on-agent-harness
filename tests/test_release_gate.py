from __future__ import annotations

from pathlib import Path

import nbformat

from course.tools.checkpoint import verify_checkpoint_package
from course.tools.release import inspect_release, main
from course.tools.notebook import execute_notebook


ROOT = Path(__file__).resolve().parents[1]


def test_published_checkpoint_can_be_verified_without_republishing() -> None:
    checkpoint = ROOT / "course/checkpoints/ch01"
    before = (checkpoint / "checkpoint.json").read_bytes()

    gates = verify_checkpoint_package(checkpoint)

    assert gates == ("compile", "install", "import", "tests")
    assert (checkpoint / "checkpoint.json").read_bytes() == before


def test_release_command_exposes_a_fast_static_diagnostic(capsys) -> None:
    assert main(["--root", str(ROOT), "--static"]) == 0

    output = capsys.readouterr().out
    assert "Release inspection passed: 9 chapters" in output
    assert "production source: course/checkpoints/ch09" in output


def test_release_ci_runs_one_offline_python_311_gate_on_all_supported_platforms() -> None:
    workflow = (ROOT / ".github/workflows/release-gate.yml").read_text(
        encoding="utf-8"
    )

    for runner in ("ubuntu-latest", "windows-latest", "macos-latest"):
        assert runner in workflow
    assert "python-version: \"3.11\"" in workflow
    assert "UV_OFFLINE: \"1\"" in workflow
    assert "PIP_NO_INDEX: \"1\"" in workflow
    assert "AGENT_HARNESS_REAL_SMOKE: \"0\"" in workflow
    assert "python -m course.tools.release" in workflow


def test_release_inspection_locks_the_single_course_spine_and_v1_scope() -> None:
    inspection = inspect_release(ROOT)

    assert tuple(chapter.notebook.name for chapter in inspection.chapters) == (
        "01_model_boundary.ipynb",
        "02_async_runtime.ipynb",
        "03_structured_tools.ipynb",
        "04_run_control.ipynb",
        "05_durable_sessions.ipynb",
        "06_context_compaction.ipynb",
        "07_extensions.ipynb",
        "08_coding_agent.ipynb",
        "09_interfaces_and_release.ipynb",
    )
    assert tuple(chapter.checkpoint.name for chapter in inspection.chapters) == tuple(
        f"ch{number:02d}" for number in range(1, 10)
    )
    assert inspection.production_source == ROOT / "course/checkpoints/ch09"
    assert inspection.runtime_dependencies == (
        "jsonschema",
        "openai",
        "platformdirs",
    )
    assert inspection.historical_evidence == (
        ROOT / "notebooks/lessons",
        ROOT / "notebooks/raw",
    )
    assert inspection.exclusions == (
        "full TUI",
        "RPC/server mode",
        "MCP",
        "subagents",
        "multimodal content",
        "optimizer",
        "remote Extension installation",
        "model catalog",
        "built-in sandbox",
    )

    assert tuple(item.requirement for item in inspection.release_tests) == (
        "classified retries",
        "partial-stream failure",
        "cancellation settlement",
        "Session Lock contention",
        "path and link escape attempts",
        "atomic text mutation",
        "settled-boundary crash recovery",
        "Compaction",
        "JSONL contracts",
        "trace redaction",
        "output-mode contracts",
        "persisted-version migration",
        "credential-gated real endpoint smoke",
        "version-one exclusions",
    )
    assert all(item.path.is_file() for item in inspection.release_tests)


def test_clean_kernel_execution_removes_credentials_and_blocks_external_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-the-kernel")
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                """\
import os
import socket

assert "OPENAI_API_KEY" not in os.environ
try:
    socket.create_connection(("example.invalid", 443), timeout=0.1)
except Exception as error:
    assert type(error).__name__ == "OfflineNetworkError", repr(error)
else:
    raise AssertionError("external network access was not blocked")
"""
            )
        ]
    )
    path = tmp_path / "offline.ipynb"
    nbformat.write(notebook, path)

    executed = execute_notebook(path, cwd=tmp_path)

    assert executed.cells[0].outputs == []
