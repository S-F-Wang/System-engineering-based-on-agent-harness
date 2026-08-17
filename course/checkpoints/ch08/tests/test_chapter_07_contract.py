from __future__ import annotations

import pytest

from agent_harness import (
    AgentRuntime,
    AgentSession,
    ExtensionAPI,
    HookPoint,
    ModelEnd,
    ModelSpec,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
)


def test_version_one_exposes_only_the_bounded_hook_set() -> None:
    assert tuple(point.value for point in HookPoint) == (
        "before_run",
        "before_model_request",
        "before_tool_call",
        "after_tool_call",
        "before_compaction",
    )
    for excluded_surface in (
        "discover",
        "install",
        "register_provider",
        "register_cli_argument",
        "register_tui_component",
        "start_background_service",
    ):
        assert not hasattr(ExtensionAPI, excluded_surface)


def test_core_accepts_explicit_extension_values_not_paths_or_packages() -> None:
    runtime = AgentRuntime(
        ScriptedModelAdapter(
            [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
        ),
        ModelSpec("scripted/explicit-only"),
    )

    with pytest.raises(TypeError, match="Extension values"):
        AgentSession(runtime, extensions=["project_extension.py"])  # type: ignore[list-item]
