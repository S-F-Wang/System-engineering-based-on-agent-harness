from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_harness import (
    AgentRuntime,
    MemoryProjectTrust,
    ModelEnd,
    ModelSpec,
    ResourceLoader,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    ToolCallDelta,
    create_coding_agent,
)


def test_coding_agent_uses_one_session_to_read_edit_and_verify_a_real_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "AGENTS.md").write_text(
        "Keep the verification local and deterministic.", encoding="utf-8"
    )
    adapter = ScriptedModelAdapter(
        [
            [
                ToolCallDelta(0, "read-1", "read", '{"path":"app.py"}'),
                ModelEnd(StopReason.TOOL_USE),
            ],
            [
                ToolCallDelta(
                    0,
                    "edit-1",
                    "edit",
                    '{"path":"app.py","old_text":"VALUE = 1",'
                    '"new_text":"VALUE = 2"}',
                ),
                ModelEnd(StopReason.TOOL_USE),
            ],
            [
                ToolCallDelta(
                    0,
                    "bash-1",
                    "bash",
                    '{"command":"grep -q \'VALUE = 2\' app.py"}',
                ),
                ModelEnd(StopReason.TOOL_USE),
            ],
            [TextDelta("Updated and verified app.py"), ModelEnd(StopReason.COMPLETE)],
        ]
    )
    resources = ResourceLoader(
        workspace,
        trust=MemoryProjectTrust([workspace]),
    )
    agent = create_coding_agent(
        adapter,
        ModelSpec("scripted/coding-agent"),
        workspace,
        resource_loader=resources,
    )

    result = asyncio.run(agent.session.run("Update VALUE to 2 and verify it"))

    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert result.outcome.message.content[0].text == "Updated and verified app.py"
    assert [tool.name for tool in agent.preset.tools] == [
        "read",
        "write",
        "edit",
        "bash",
    ]
    assert adapter.received_requests[0].messages[0].content[0].text.startswith(
        "You are Omega"
    )
    assert dict(agent.startup_evidence)["context:0:" + str(
        (workspace / "AGENTS.md").resolve()
    )].startswith("sha256:")


def test_general_runtime_still_installs_no_tools() -> None:
    runtime = AgentRuntime(
        ScriptedModelAdapter([TextDelta("done"), ModelEnd(StopReason.COMPLETE)]),
        ModelSpec("scripted/general"),
    )

    assert runtime.tools == ()


def test_coding_agent_rejects_resource_loader_from_another_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()
    adapter = ScriptedModelAdapter(
        [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
    )

    with pytest.raises(ValueError, match="ResourceLoader.*workspace"):
        create_coding_agent(
            adapter,
            ModelSpec("scripted/mismatched-resources"),
            workspace,
            resource_loader=ResourceLoader(other),
        )
