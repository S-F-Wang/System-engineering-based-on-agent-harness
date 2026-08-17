from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_harness import (
    LocalWorkspace,
    ModelAdapterError,
    ModelEnd,
    ModelError,
    ModelErrorCode,
    ModelSpec,
    RetryPolicy,
    RunArtifactStore,
    ScriptedModelAdapter,
    StopReason,
    StandardTraceRedactor,
    TextDelta,
    Tool,
    ToolCallDelta,
    ToolResult,
    UnsupportedRunSchemaVersionError,
    Usage,
    UsageUpdate,
    create_session,
    migrate_run_trace,
)


def test_standard_trace_records_retries_tools_latency_usage_and_snapshot(
    tmp_path: Path,
) -> None:
    class RetryingToolAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request):
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError(
                    ModelError(ModelErrorCode.SERVER, "temporary", True, status_code=503)
                )
            if self.calls == 2:
                yield ToolCallDelta(0, "call-1", "echo", '{"text":"hello"}')
                yield ModelEnd(StopReason.TOOL_USE)
                return
            yield TextDelta("finished")
            yield UsageUpdate(Usage(7, 2, 9))
            yield ModelEnd(StopReason.COMPLETE)

    async def echo(arguments: dict[str, object]) -> ToolResult:
        return ToolResult(str(arguments["text"]))

    async def no_sleep(delay: float) -> None:
        return None

    workspace = tmp_path / "workspace"
    storage = tmp_path / "data"
    workspace.mkdir()
    session = create_session(
        RetryingToolAdapter(),
        ModelSpec("scripted/observed"),
        workspace=workspace,
        storage_root=storage,
        tools=(
            Tool(
                "echo",
                "echo text",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                echo,
            ),
        ),
        retry_policy=RetryPolicy(delays=(0,)),
        sleeper=no_sleep,
    )

    asyncio.run(session.run("exercise trace"))

    local = LocalWorkspace.open(workspace, storage_root=storage)
    run = local.runs.list()[0]
    records = local.runs.records(run.run_id)
    header = records[0]
    event_types = {
        record.get("type") for record in records if record.get("record") == "event"
    }
    ending = records[-1]
    assert header["snapshot"]["fingerprint"]
    assert header["snapshot"]["platform"]["python"].startswith("3.11")
    assert {"model_attempt_failed", "retry_scheduled", "tool_call_start", "tool_call_end"} <= event_types
    assert any(record.get("tool_arguments") == {"text": "hello"} for record in records)
    assert any(record.get("record") == "first_token" for record in records)
    assert ending["record"] == "run_end"
    assert ending["attempts"] == 3
    assert ending["usage_provenance"] == "provider"


def test_run_store_keeps_append_only_annotations_and_content_addressed_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "data"
    workspace.mkdir()
    session = create_session(
        ScriptedModelAdapter(
            [TextDelta("done"), ModelEnd(StopReason.COMPLETE)]
        ),
        ModelSpec("scripted/layout"),
        workspace=workspace,
        storage_root=storage,
    )
    asyncio.run(session.run("persist layout"))
    local = LocalWorkspace.open(workspace, storage_root=storage)
    run_id = local.runs.list()[0].run_id

    first = local.runs.put_artifact(run_id, b"same content")
    second = local.runs.put_artifact(run_id, b"same content")
    local.runs.annotate(run_id, "review.score", {"score": 1})
    local.runs.annotate(run_id, "review.note", {"note": "kept"})

    run_directory = local.runs.directory_for(run_id)
    assert first == second
    assert len(list((run_directory / "artifacts" / "sha256").iterdir())) == 1
    assert len(local.runs.annotations(run_id)) == 2
    assert local.sessions.path_for(session.session_id).parent == local.root / "sessions"
    assert local.runs.trace_path(run_id) == run_directory / "trace.jsonl"
    assert local.runs.annotation_path(run_id) == run_directory / "annotations.jsonl"


def test_run_trace_recovery_and_versions_are_actionable_and_nondestructive(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "data"
    workspace.mkdir()
    session = create_session(
        ScriptedModelAdapter(
            [TextDelta("done"), ModelEnd(StopReason.COMPLETE)]
        ),
        ModelSpec("scripted/versioned"),
        workspace=workspace,
        storage_root=storage,
    )
    asyncio.run(session.run("version me"))
    local = LocalWorkspace.open(workspace, storage_root=storage)
    run_id = local.runs.list()[0].run_id
    source = local.runs.trace_path(run_id)
    original = source.read_bytes()
    migrated = tmp_path / "migrated" / "trace.jsonl"

    assert migrate_run_trace(source, migrated) == migrated.resolve()
    assert source.read_bytes() == original
    assert migrated.read_bytes() == original

    source.write_bytes(original + b'{"incomplete"')
    assert local.runs.summary(run_id).evidence_complete is False
    source.write_bytes(original.replace(b'"major":1', b'"major":99', 1))
    with pytest.raises(UnsupportedRunSchemaVersionError, match="migrate_run_trace"):
        local.runs.records(run_id)


def test_artifact_redaction_preserves_full_sanitized_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    local = LocalWorkspace.open(
        workspace,
        storage_root=tmp_path / "data",
        redactor=StandardTraceRedactor(secrets=("secret",), preview_chars=8),
    )
    local.runs.start("run-1", session_id=None, snapshot={})
    artifacts = RunArtifactStore(local.runs)
    artifacts.activate("run-1")
    content = "prefix-secret-" + ("x" * 5000)

    reference = artifacts.put_text(content)

    digest = reference.removeprefix("sha256:")
    retained = (
        local.runs.directory_for("run-1") / "artifacts" / "sha256" / digest
    ).read_text(encoding="utf-8")
    assert "secret" not in retained
    assert "[REDACTED]" in retained
    assert retained.endswith("x" * 5000)
