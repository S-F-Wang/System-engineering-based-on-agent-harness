from __future__ import annotations

from pathlib import Path
import tomllib

import agent_harness


def test_chapter_09_freezes_public_factories_adapters_stores_and_policies() -> None:
    expected = {
        "AgentRuntime",
        "AgentSession",
        "CompactionPolicy",
        "JSONLRunStore",
        "JSONLSessionStore",
        "ModelSpec",
        "NonTerminalAdapter",
        "OpenAICompatibleAdapter",
        "OpenAICompatibleConfig",
        "RetryPolicy",
        "RunGuard",
        "RunResult",
        "StandardTraceSink",
        "ToolExecutor",
        "create_coding_session",
        "create_session",
    }

    assert expected <= set(agent_harness.__all__)


def test_runtime_dependencies_stay_inside_the_accepted_small_boundary() -> None:
    checkpoint = Path(__file__).parents[1]
    project = tomllib.loads((checkpoint / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    names = {dependency.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0] for dependency in dependencies}

    assert names == {"jsonschema", "openai", "platformdirs"}
    assert not any(
        excluded in dependency.casefold()
        for dependency in dependencies
        for excluded in ("jupyter", "pytest", "textual", "mcp", "rpc")
    )


def test_checkpoint_documentation_locks_privacy_output_and_v1_exclusions() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "There is no\nAPI-key flag" in readme
    assert "disables Session, Trace, Annotation, and Artifact persistence" in readme
    assert "always ends with\n`run_end`" in readme
    for exclusion in (
        "full TUI",
        "RPC/server mode",
        "MCP",
        "subagents",
        "multimodal content",
        "optimizer",
        "remote Extension installation",
        "model catalog",
        "built-in sandbox",
    ):
        assert exclusion in readme
