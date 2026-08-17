from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_harness import (
    AgentSession,
    LocalWorkspace,
    MemorySessionStore,
    ModelEnd,
    ModelSpec,
    OpenAICompatibleConfig,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    ToolCallDelta,
    TracePersistenceError,
    create_session,
    create_coding_session,
)


def test_general_factory_persists_session_and_sanitized_run_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    storage_root = tmp_path / "data"
    workspace.mkdir()
    local = LocalWorkspace.open(workspace, storage_root=storage_root)
    adapter = ScriptedModelAdapter(
        [TextDelta("completed"), ModelEnd(StopReason.COMPLETE)]
    )

    session = create_session(
        adapter,
        ModelSpec("scripted/public"),
        workspace=workspace,
        storage_root=storage_root,
        trace_secrets=("super-secret-value",),
    )
    result = asyncio.run(session.run("do not retain super-secret-value"))

    assert isinstance(session, AgentSession)
    assert session.session_id is not None
    assert local.sessions.read(session.session_id).history()
    summaries = local.runs.list(session_id=session.session_id)
    assert len(summaries) == 1
    assert summaries[0].status == "completed"
    assert summaries[0].evidence_complete is True
    assert result.trace_complete is True
    assert "super-secret-value" not in local.runs.export(summaries[0].run_id)


def test_no_save_disables_every_persistence_family(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage_root = tmp_path / "must-not-exist"
    workspace.mkdir()
    session = create_session(
        ScriptedModelAdapter([TextDelta("private"), ModelEnd(StopReason.COMPLETE)]),
        ModelSpec("scripted/private"),
        workspace=workspace,
        storage_root=storage_root,
        no_save=True,
    )

    result = asyncio.run(session.run("ephemeral prompt"))

    assert session.session_id is None
    assert result.outcome.message.content[0].text == "private"
    assert not storage_root.exists()


def test_trace_failure_marks_evidence_incomplete_without_changing_success(
    tmp_path: Path,
) -> None:
    class BrokenObserver:
        def start_run(self, session_id, snapshot) -> None:
            return None

        def record_event(self, event) -> None:
            raise OSError("disk unavailable")

        def finish_run(self, session_id, run_id, result) -> None:
            raise AssertionError("observer must stop after its first failure")

        def mark_incomplete(self, run_id, reason) -> None:
            return None

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(
        ScriptedModelAdapter([TextDelta("still good"), ModelEnd(StopReason.COMPLETE)]),
        ModelSpec("scripted/trace-failure"),
        workspace=workspace,
        session_store=MemorySessionStore(),
        observer=BrokenObserver(),
    )

    result = asyncio.run(session.run("continue semantically"))

    assert result.outcome.status.value == "completed"
    assert result.trace_complete is False
    assert result.trace_error == "OSError"


def test_strict_tracing_turns_evidence_failure_into_infrastructure_failure(
    tmp_path: Path,
) -> None:
    class BrokenObserver:
        def start_run(self, session_id, snapshot) -> None:
            return None

        def record_event(self, event) -> None:
            raise OSError("disk unavailable")

        def finish_run(self, session_id, run_id, result) -> None:
            return None

        def mark_incomplete(self, run_id, reason) -> None:
            return None

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = create_session(
        ScriptedModelAdapter([TextDelta("semantic output"), ModelEnd(StopReason.COMPLETE)]),
        ModelSpec("scripted/strict-trace"),
        workspace=workspace,
        session_store=MemorySessionStore(),
        observer=BrokenObserver(),
        strict_tracing=True,
    )

    with pytest.raises(TracePersistenceError, match="record_event"):
        asyncio.run(session.run("strict evidence"))


def test_coding_factory_installs_resources_and_tools_over_same_session_interface(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        session = create_coding_session(
            ScriptedModelAdapter(
                [
                    [
                        ToolCallDelta(
                            0,
                            "write-1",
                            "write",
                            '{"path":"answer.txt","content":"42"}',
                        ),
                        ModelEnd(StopReason.TOOL_USE),
                    ],
                    [TextDelta("written"), ModelEnd(StopReason.COMPLETE)],
                ]
            ),
            ModelSpec("scripted/coding"),
            workspace=workspace,
            no_save=True,
        )

        handle = session.start("write the answer")
        result = await handle.result()

        assert isinstance(session, AgentSession)
        assert [tool.name for tool in handle.snapshot.tools] == [
            "read",
            "write",
            "edit",
            "bash",
        ]
        assert result.outcome.message.content[0].text == "written"
        assert (workspace / "answer.txt").read_text() == "42"

    asyncio.run(scenario())


def test_library_transport_configuration_is_explicit_and_ignores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-must-not-be-read")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "ambient-model")

    session = create_session(
        OpenAICompatibleConfig(
            base_url="https://explicit.example/v1",
            api_key="explicit-key",
        ),
        ModelSpec("explicit-model"),
        workspace=workspace,
        no_save=True,
    )

    assert isinstance(session, AgentSession)
    assert session.session_id is None
