from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    JSONLSessionStore,
    MemorySessionStore,
    ModelEnd,
    ModelSpec,
    MigrationResult,
    RecoveryCode,
    Role,
    ScriptedModelAdapter,
    SessionBusyError,
    StopReason,
    TextDelta,
    Tool,
    ToolCallDelta,
    ToolResult,
    UnsupportedSchemaVersionError,
    create_agent_session,
    migrate_session_file,
)


def test_memory_store_recovers_a_settled_message_batch() -> None:
    store = MemorySessionStore()
    created = store.create("session-1")

    with store.writer(created.session_id) as writer:
        entry = writer.append(
            (
                AgentMessage.text(Role.USER, "hello"),
                AgentMessage.text(Role.ASSISTANT, "hi"),
            )
        )

    recovered = store.read("session-1")

    assert recovered.active_leaf_id == entry.entry_id
    assert recovered.history() == (
        AgentMessage.text(Role.USER, "hello"),
        AgentMessage.text(Role.ASSISTANT, "hi"),
    )


def test_jsonl_store_recovers_through_the_same_public_interface(tmp_path) -> None:
    store = JSONLSessionStore(tmp_path)
    state = store.create("session-on-disk")

    with store.writer(state.session_id) as writer:
        writer.append(
            (
                AgentMessage.text(Role.USER, "persist this"),
                AgentMessage.text(Role.ASSISTANT, "persisted"),
            )
        )

    reopened = JSONLSessionStore(tmp_path).read(state.session_id)

    assert reopened.history() == (
        AgentMessage.text(Role.USER, "persist this"),
        AgentMessage.text(Role.ASSISTANT, "persisted"),
    )
    assert (tmp_path / "sessions" / "session-on-disk.jsonl").is_file()


def test_model_and_tool_history_survives_reconstruction_and_continues(tmp_path) -> None:
    async def scenario() -> None:
        async def lookup(arguments: dict[str, object]) -> ToolResult:
            return ToolResult("durable tool result", metadata={"source": "fixture"})

        first_adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-1", "lookup", '{"key":"value"}'),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("first answer"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        store = JSONLSessionStore(tmp_path)
        first = create_agent_session(
            AgentRuntime(
                first_adapter,
                ModelSpec("scripted/durable"),
                tools=[
                    Tool(
                        "lookup",
                        "Look up a fixture value",
                        {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                            "additionalProperties": False,
                        },
                        lookup,
                    )
                ],
            ),
            store=store,
        )
        first_result = await first.run("first question")
        assert first_result.outcome.message == AgentMessage.text(
            Role.ASSISTANT, "first answer"
        )

        second_adapter = ScriptedModelAdapter(
            [TextDelta("continued answer"), ModelEnd(StopReason.COMPLETE)]
        )
        continued = create_agent_session(
            AgentRuntime(second_adapter, ModelSpec("scripted/durable")),
            store=JSONLSessionStore(tmp_path),
            session_id=first.session_id,
        )
        second_result = await continued.run("follow up")

        assert second_result.outcome.message == AgentMessage.text(
            Role.ASSISTANT, "continued answer"
        )
        request = second_adapter.received_requests[0]
        assert len(request.messages) == 5
        assert request.messages[-1] == AgentMessage.text(
            Role.USER, "follow up"
        ).to_model()
        assert request.messages[2].role is Role.TOOL

    asyncio.run(scenario())


def test_incomplete_final_jsonl_record_is_reported_and_not_replayed(tmp_path) -> None:
    store = JSONLSessionStore(tmp_path)
    state = store.create("interrupted")
    with store.writer(state.session_id) as writer:
        writer.append((AgentMessage.text(Role.USER, "settled"),))
    path = store.path_for(state.session_id)
    with path.open("ab") as stream:
        stream.write(b'{"record":"settlement","entry_id":"interrupted"')

    recovered = JSONLSessionStore(tmp_path).read(state.session_id)

    assert recovered.history() == (AgentMessage.text(Role.USER, "settled"),)
    assert recovered.recovery_warning is not None
    assert recovered.recovery_warning.code is RecoveryCode.INCOMPLETE_FINAL_RECORD


def test_continuation_discards_an_uncommitted_tail_before_appending(tmp_path) -> None:
    async def scenario() -> None:
        store = JSONLSessionStore(tmp_path)
        state = store.create("recover-and-continue")
        with store.writer(state.session_id) as writer:
            writer.append((AgentMessage.text(Role.USER, "settled before crash"),))
        with store.path_for(state.session_id).open("ab") as stream:
            stream.write(b'{"record":"settlement","messages":[')

        continued = create_agent_session(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("settled after restart"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/recovery"),
            ),
            store=JSONLSessionStore(tmp_path),
            session_id=state.session_id,
        )
        await continued.run("restart operation")

        recovered = JSONLSessionStore(tmp_path).read(state.session_id)
        assert recovered.recovery_warning is None
        assert recovered.history() == (
            AgentMessage.text(Role.USER, "settled before crash"),
            AgentMessage.text(Role.USER, "restart operation"),
            AgentMessage.text(Role.ASSISTANT, "settled after restart"),
        )

    asyncio.run(scenario())


def test_durable_session_has_one_writer_while_reads_and_other_sessions_continue(
    tmp_path,
) -> None:
    first_store = JSONLSessionStore(tmp_path)
    first_store.create("shared")
    first_store.create("independent")
    competing_store = JSONLSessionStore(tmp_path)

    with first_store.writer("shared"):
        with pytest.raises(SessionBusyError) as caught:
            with competing_store.writer("shared"):
                pass
        assert caught.value.code == "session_busy"
        assert caught.value.session_id == "shared"
        assert competing_store.read("shared").session_id == "shared"
        with competing_store.writer("independent") as writer:
            writer.append((AgentMessage.text(Role.USER, "unblocked"),))


def test_session_writer_lease_contends_across_processes(tmp_path) -> None:
    store = JSONLSessionStore(tmp_path)
    store.create("cross-process")
    ready = tmp_path / "child-ready"
    release = tmp_path / "child-release"
    script = """
import sys
import time
from pathlib import Path
from agent_harness import JSONLSessionStore

root, ready, release = map(Path, sys.argv[1:])
with JSONLSessionStore(root).writer("cross-process"):
    ready.write_text("ready", encoding="utf-8")
    while not release.exists():
        time.sleep(0.01)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH", "")) if item
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(ready), str(release)],
        env=environment,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "child process did not acquire the Session Lock"
        with pytest.raises(SessionBusyError):
            with store.writer("cross-process"):
                pass
    finally:
        release.write_text("release", encoding="utf-8")
        child.wait(timeout=5)
    assert child.returncode == 0


def test_session_schema_accepts_optional_fields_and_rejects_unknown_major(
    tmp_path,
) -> None:
    store = JSONLSessionStore(tmp_path)
    state = store.create("versioned")
    with store.writer(state.session_id) as writer:
        writer.append((AgentMessage.text(Role.USER, "compatible"),))
    path = store.path_for(state.session_id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["future_header_field"] = {"safe": True}
    records[1]["future_entry_field"] = [1, 2, 3]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    assert JSONLSessionStore(tmp_path).read(state.session_id).history() == (
        AgentMessage.text(Role.USER, "compatible"),
    )

    records[0]["schema_version"] = {"major": 2, "minor": 0}
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(
        UnsupportedSchemaVersionError,
        match=r"Session schema major 2.*migrate",
    ):
        JSONLSessionStore(tmp_path).read(state.session_id)


def test_complete_jsonl_record_cannot_reference_an_unknown_parent(tmp_path) -> None:
    store = JSONLSessionStore(tmp_path)
    state = store.create("invalid-tree")
    with store.writer(state.session_id) as writer:
        writer.append((AgentMessage.text(Role.USER, "root"),))
    path = store.path_for(state.session_id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["parent_id"] = "missing-parent"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="parent.*earlier Session entry"):
        JSONLSessionStore(tmp_path).read(state.session_id)


def test_migration_validates_a_new_file_and_preserves_the_original(tmp_path) -> None:
    source_store = JSONLSessionStore(tmp_path / "source")
    state = source_store.create("migration-fixture")
    with source_store.writer(state.session_id) as writer:
        writer.append((AgentMessage.text(Role.USER, "keep the original"),))
    source = source_store.path_for(state.session_id)
    original = source.read_bytes()
    destination = (
        tmp_path / "migrated" / "sessions" / "migration-fixture.jsonl"
    )

    result = migrate_session_file(source, destination)

    assert result == MigrationResult(
        source=source.resolve(),
        destination=destination.resolve(),
        session_id="migration-fixture",
        entries=1,
    )
    assert source.read_bytes() == original
    assert destination.read_bytes() != b""
    assert JSONLSessionStore(tmp_path / "migrated").read(state.session_id).history() == (
        AgentMessage.text(Role.USER, "keep the original"),
    )


def test_both_store_adapters_preserve_history_when_a_historical_entry_forks(
    tmp_path,
) -> None:
    for store in (MemorySessionStore(), JSONLSessionStore(tmp_path)):
        state = store.create(f"tree-{type(store).__name__}")
        with store.writer(state.session_id) as writer:
            root = writer.append((AgentMessage.text(Role.USER, "root"),))
            main = writer.append((AgentMessage.text(Role.ASSISTANT, "main"),))
            branch = writer.append(
                (AgentMessage.text(Role.ASSISTANT, "branch"),),
                parent_id=root.entry_id,
            )

        recovered = store.read(state.session_id)

        assert recovered.history(main.entry_id) == (
            AgentMessage.text(Role.USER, "root"),
            AgentMessage.text(Role.ASSISTANT, "main"),
        )
        assert recovered.history() == (
            AgentMessage.text(Role.USER, "root"),
            AgentMessage.text(Role.ASSISTANT, "branch"),
        )
        assert branch.parent_id == root.entry_id
        assert {entry.entry_id for entry in recovered.entries} == {
            root.entry_id,
            main.entry_id,
            branch.entry_id,
        }


def test_session_factory_forks_from_an_explicit_historical_entry() -> None:
    async def scenario() -> None:
        store = MemorySessionStore()
        state = store.create("factory-fork")
        with store.writer(state.session_id) as writer:
            root = writer.append((AgentMessage.text(Role.USER, "root"),))
            main = writer.append((AgentMessage.text(Role.ASSISTANT, "main"),))
        session = create_agent_session(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("branch answer"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/fork"),
            ),
            store=store,
            session_id=state.session_id,
            fork_from=root.entry_id,
        )

        await session.run("branch question")

        recovered = store.read(state.session_id)
        assert recovered.history(main.entry_id)[-1] == AgentMessage.text(
            Role.ASSISTANT, "main"
        )
        assert recovered.history() == (
            AgentMessage.text(Role.USER, "root"),
            AgentMessage.text(Role.USER, "branch question"),
            AgentMessage.text(Role.ASSISTANT, "branch answer"),
        )

    asyncio.run(scenario())


def test_session_factory_creates_new_durable_state_and_no_save_rejects_continuation(
    tmp_path,
) -> None:
    def runtime(answer: str) -> AgentRuntime:
        return AgentRuntime(
            ScriptedModelAdapter([TextDelta(answer), ModelEnd(StopReason.COMPLETE)]),
            ModelSpec("scripted/factory"),
        )

    store = JSONLSessionStore(tmp_path)
    first = create_agent_session(runtime("one"), store=store)
    second = create_agent_session(runtime("two"), store=store)

    assert first.session_id is not None
    assert second.session_id is not None
    assert first.session_id != second.session_id
    with pytest.raises(ValueError, match="no_save.*continuation"):
        create_agent_session(
            runtime("never"),
            store=store,
            session_id=first.session_id,
            no_save=True,
        )
    ephemeral = create_agent_session(runtime("private"), no_save=True)
    assert ephemeral.session_id is None
