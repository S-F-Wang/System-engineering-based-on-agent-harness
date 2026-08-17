from __future__ import annotations

from pathlib import Path

from agent_harness import (
    AgentRuntime,
    MemoryProjectTrust,
    ModelEnd,
    ModelSpec,
    ResourceLoader,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    create_coding_tool_preset,
)


def test_chapter_08_public_contract_and_optional_tool_preset(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resources = ResourceLoader(
        workspace,
        trust=MemoryProjectTrust(),
    ).load()
    preset = create_coding_tool_preset(workspace)
    general = AgentRuntime(
        ScriptedModelAdapter([TextDelta("done"), ModelEnd(StopReason.COMPLETE)]),
        ModelSpec("scripted/ch08-contract"),
    )

    assert resources.skills == {}
    assert general.tools == ()
    assert [tool.name for tool in preset.tools] == ["read", "write", "edit", "bash"]


def test_checkpoint_documents_host_authority_without_a_sandbox_claim() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "real host process" in readme
    assert "Neither Bash nor Extensions are a sandbox guarantee" in readme
