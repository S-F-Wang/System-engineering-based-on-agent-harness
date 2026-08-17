from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from agent_harness import (
    LocalWorkspace,
    ModelEnd,
    ModelSpec,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    create_session,
)
from agent_harness.cli import build_parser, main


def test_exec_json_uses_flag_precedence_and_persists_by_default(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "data"
    workspace.mkdir()
    stdout = io.StringIO()
    stderr = io.StringIO()
    captured = []

    def adapter_factory(config):
        captured.append(config)
        return ScriptedModelAdapter(
            [TextDelta("machine result"), ModelEnd(StopReason.COMPLETE)]
        )

    code = main(
        [
            "exec",
            "finish it",
            "--format",
            "json",
            "--workspace",
            str(workspace),
            "--storage-root",
            str(storage),
            "--model",
            "flag-model",
            "--base-url",
            "https://flag.example/v1",
        ],
        environ={
            "OPENAI_API_KEY": "cli-secret",
            "OPENAI_MODEL": "env-model",
            "OPENAI_BASE_URL": "https://env.example/v1",
        },
        stdout=stdout,
        stderr=stderr,
        adapter_factory=adapter_factory,
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert stderr.getvalue() == ""
    assert payload["schema"] == "agent_harness.run_result"
    assert payload["schema_version"] == {"major": 1, "minor": 0}
    assert payload["status"] == "completed"
    assert payload["final_text"] == "machine result"
    assert payload["session_id"]
    assert payload["run_id"]
    assert captured[0].base_url == "https://flag.example/v1"
    assert captured[0].api_key == "cli-secret"
    assert list(storage.rglob("trace.jsonl"))
    assert "cli-secret" not in list(storage.rglob("trace.jsonl"))[0].read_text()


def test_exec_jsonl_emits_ordered_events_and_mandatory_run_end(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stdout = io.StringIO()

    code = main(
        [
            "exec",
            "stream",
            "--format",
            "jsonl",
            "--workspace",
            str(workspace),
            "--no-save",
        ],
        environ={"OPENAI_API_KEY": "x", "OPENAI_MODEL": "scripted/jsonl"},
        stdout=stdout,
        stderr=io.StringIO(),
        adapter_factory=lambda config: ScriptedModelAdapter(
            [TextDelta("one"), TextDelta(" two"), ModelEnd(StopReason.COMPLETE)]
        ),
    )

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 0
    assert lines[-1]["type"] == "run_end"
    assert lines[-1]["payload"]["final_text"] == "one two"
    assert [line["sequence"] for line in lines] == sorted(
        line["sequence"] for line in lines
    )
    assert all(line["schema"] == "agent_harness.event" for line in lines)


def test_chat_runs_a_scripted_multi_turn_terminal_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stdout = io.StringIO()
    stderr = io.StringIO()
    adapter = ScriptedModelAdapter(
        [
            [TextDelta("first answer"), ModelEnd(StopReason.COMPLETE)],
            [TextDelta("second answer"), ModelEnd(StopReason.COMPLETE)],
        ]
    )

    code = main(
        ["chat", "--workspace", str(workspace), "--no-save"],
        environ={"OPENAI_API_KEY": "x", "OPENAI_MODEL": "scripted/chat"},
        stdout=stdout,
        stderr=stderr,
        input_stream=io.StringIO(
            "first question\n/wait\nsecond question\n/wait\n/quit\n"
        ),
        adapter_factory=lambda config: adapter,
    )

    assert code == 0
    assert stdout.getvalue().splitlines() == [
        "assistant: first answer",
        "assistant: second answer",
    ]
    assert stderr.getvalue() == ""
    assert len(adapter.received_requests) == 2


def test_session_and_run_commands_manage_local_evidence_without_credentials(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "data"
    workspace.mkdir()
    session = create_session(
        ScriptedModelAdapter([TextDelta("kept"), ModelEnd(StopReason.COMPLETE)]),
        ModelSpec("scripted/catalog"),
        workspace=workspace,
        storage_root=storage,
    )
    asyncio_result = asyncio.run(session.run("persist"))
    assert asyncio_result.outcome.message.content[0].text == "kept"
    local = LocalWorkspace.open(workspace, storage_root=storage)
    run_id = local.runs.list()[0].run_id

    def invoke(arguments: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = main(arguments, environ={}, stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    common = ["--workspace", str(workspace), "--storage-root", str(storage)]
    code, output, error = invoke(["sessions", "list", *common])
    assert code == 0 and error == ""
    assert json.loads(output)[0]["session_id"] == session.session_id

    code, output, error = invoke(
        ["runs", "annotate", run_id, "--namespace", "review.outcome", "--payload", '{"score":1}', *common]
    )
    assert code == 0 and error == ""
    assert local.runs.annotations(run_id)[0]["payload"] == {"score": 1}

    code, output, error = invoke(["runs", "prune", "--status", "completed", *common])
    preview = json.loads(output)
    assert code == 0 and error == ""
    assert preview["applied"] is False
    assert preview["run_ids"] == [run_id]
    assert local.runs.summary(run_id).status == "completed"

    code, output, error = invoke(["sessions", "fork", session.session_id, *common])
    forked = json.loads(output)["session_id"]
    assert code == 0 and error == ""
    assert forked != session.session_id
    assert local.sessions.read(forked).history() == local.sessions.read(session.session_id).history()


def test_cli_has_no_api_key_flag_and_distinguishes_configuration_and_model_errors(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    help_text = build_parser().format_help()
    assert "--api-key" not in help_text

    missing_error = io.StringIO()
    assert main(
        ["exec", "x", "--workspace", str(workspace), "--no-save"],
        environ={},
        stdout=io.StringIO(),
        stderr=missing_error,
    ) == 2
    assert "OPENAI_API_KEY" in missing_error.getvalue()

    incompatible_error = io.StringIO()
    assert main(
        [
            "exec",
            "x",
            "--workspace",
            str(workspace),
            "--no-save",
            "--session",
            "existing",
        ],
        environ={"OPENAI_API_KEY": "x", "OPENAI_MODEL": "scripted/error"},
        stdout=io.StringIO(),
        stderr=incompatible_error,
        adapter_factory=lambda config: ScriptedModelAdapter([]),
    ) == 2
    assert "cannot continue" in incompatible_error.getvalue()

    output = io.StringIO()
    assert main(
        ["exec", "x", "--format", "json", "--workspace", str(workspace), "--no-save"],
        environ={"OPENAI_API_KEY": "x", "OPENAI_MODEL": "scripted/error"},
        stdout=output,
        stderr=io.StringIO(),
        adapter_factory=lambda config: ScriptedModelAdapter([TextDelta("partial")]),
    ) == 3
    assert json.loads(output.getvalue())["status"] == "model_error"


def test_exec_continues_only_the_explicit_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "data"
    workspace.mkdir()
    environment = {"OPENAI_API_KEY": "x", "OPENAI_MODEL": "scripted/continue"}

    first_output = io.StringIO()
    assert main(
        ["exec", "first", "--format", "json", "--workspace", str(workspace), "--storage-root", str(storage)],
        environ=environment,
        stdout=first_output,
        stderr=io.StringIO(),
        adapter_factory=lambda config: ScriptedModelAdapter(
            [TextDelta("first answer"), ModelEnd(StopReason.COMPLETE)]
        ),
    ) == 0
    session_id = json.loads(first_output.getvalue())["session_id"]

    second_output = io.StringIO()
    assert main(
        [
            "exec",
            "second",
            "--format",
            "json",
            "--workspace",
            str(workspace),
            "--storage-root",
            str(storage),
            "--session",
            session_id,
        ],
        environ=environment,
        stdout=second_output,
        stderr=io.StringIO(),
        adapter_factory=lambda config: ScriptedModelAdapter(
            [TextDelta("second answer"), ModelEnd(StopReason.COMPLETE)]
        ),
    ) == 0

    assert json.loads(second_output.getvalue())["session_id"] == session_id
    local = LocalWorkspace.open(workspace, storage_root=storage)
    assert len(local.runs.list(session_id=session_id)) == 2
    assert len(local.sessions.read(session_id).history()) == 5


def test_chat_applies_live_steering_cancellation_and_restored_follow_up(
    tmp_path: Path,
) -> None:
    class ControllableAdapter:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            turn = len(self.requests)
            if turn == 1:
                await asyncio.Event().wait()
            responses = {2: "retry", 3: "steered", 4: "restored"}
            yield TextDelta(responses[turn])
            yield ModelEnd(StopReason.COMPLETE)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = ControllableAdapter()
    stdout, stderr = io.StringIO(), io.StringIO()

    code = main(
        ["chat", "--workspace", str(workspace), "--no-save"],
        environ={"OPENAI_API_KEY": "x", "OPENAI_MODEL": "scripted/live-chat"},
        stdout=stdout,
        stderr=stderr,
        input_stream=io.StringIO(
            "begin\n/steer adjust\n/follow-up later\n/cancel\n/wait\n"
            "retry prompt\n/wait\n/quit\n"
        ),
        adapter_factory=lambda config: adapter,
    )

    assert code == 0
    assert stdout.getvalue().splitlines() == ["assistant: restored"]
    assert "[run_cancelled]" in stderr.getvalue()
    assert len(adapter.requests) == 4
    requests = repr(adapter.requests)
    assert "adjust" in requests
    assert "later" in requests
