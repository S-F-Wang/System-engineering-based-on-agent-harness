"""Omega command-line Interface Adapters over public AgentSession contracts."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import sys
import threading
from typing import TextIO

from .api import create_coding_session
from .interfaces import EventEnvelope, NonTerminalAdapter, RunResult, exit_code
from .model import ModelAdapter, ModelSpec, OpenAICompatibleAdapter, OpenAICompatibleConfig
from .observability import LocalWorkspace
from .runtime import RunGuard


AdapterFactory = Callable[[OpenAICompatibleConfig], ModelAdapter]


class CLIConfigurationError(ValueError):
    """Command-line and environment configuration is incomplete or inconsistent."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omega")
    subcommands = parser.add_subparsers(dest="command", required=True)
    execute = subcommands.add_parser("exec", help="run one prompt")
    execute.add_argument("prompt")
    _add_runtime_arguments(execute)
    execute.add_argument("--format", choices=("text", "json", "jsonl"), default="text")
    execute.add_argument("--quiet", action="store_true")

    chat = subcommands.add_parser("chat", help="run a line-oriented conversation")
    _add_runtime_arguments(chat)
    chat.add_argument("--continue", dest="continue_latest", action="store_true")

    sessions = subcommands.add_parser("sessions", help="inspect durable Sessions")
    session_commands = sessions.add_subparsers(dest="sessions_command", required=True)
    for name in ("list", "show", "export", "fork"):
        command = session_commands.add_parser(name)
        if name != "list":
            command.add_argument("session_id")
        if name == "fork":
            command.add_argument("--entry")
        _add_storage_arguments(command)

    runs = subcommands.add_parser("runs", help="inspect durable Runs")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_commands.add_parser("list")
    run_list.add_argument("--session")
    run_list.add_argument("--status")
    _add_storage_arguments(run_list)
    for name in ("show", "export"):
        command = run_commands.add_parser(name)
        command.add_argument("run_id")
        _add_storage_arguments(command)
    annotate = run_commands.add_parser("annotate")
    annotate.add_argument("run_id")
    annotate.add_argument("--namespace", required=True)
    annotate.add_argument("--payload", required=True)
    _add_storage_arguments(annotate)
    prune = run_commands.add_parser("prune")
    prune.add_argument("--session")
    prune.add_argument("--status")
    prune.add_argument("--apply", action="store_true")
    _add_storage_arguments(prune)
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--storage-root")
    parser.add_argument("--session")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-total-tokens", type=int)


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--storage-root")


def _configuration(
    arguments: argparse.Namespace,
    environ: Mapping[str, str],
) -> tuple[OpenAICompatibleConfig, ModelSpec, RunGuard | None]:
    api_key = environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise CLIConfigurationError("OPENAI_API_KEY is required")
    model_id = arguments.model or environ.get("OPENAI_MODEL", "")
    if not model_id:
        raise CLIConfigurationError("provide --model or OPENAI_MODEL")
    base_url = (
        arguments.base_url
        or environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    config = OpenAICompatibleConfig(base_url=base_url, api_key=api_key)
    model = ModelSpec(
        model_id,
        context_window=arguments.context_window,
        max_output_tokens=arguments.max_output_tokens,
    )
    guard_values = (
        arguments.max_turns,
        arguments.max_tool_calls,
        arguments.timeout,
        arguments.max_total_tokens,
    )
    guard = None
    if any(value is not None for value in guard_values):
        guard = RunGuard(
            max_turns=arguments.max_turns,
            max_tool_calls=arguments.max_tool_calls,
            timeout_seconds=arguments.timeout,
            max_total_tokens=arguments.max_total_tokens,
        )
    return config, model, guard


async def _exec(
    arguments: argparse.Namespace,
    *,
    config: OpenAICompatibleConfig,
    model: ModelSpec,
    guard: RunGuard | None,
    stdout: TextIO,
    stderr: TextIO,
    adapter_factory: AdapterFactory,
) -> int:
    workspace = Path(arguments.workspace).resolve()
    if arguments.no_save and arguments.session is not None:
        raise CLIConfigurationError("--no-save cannot continue a durable Session")
    adapter = adapter_factory(config)
    session = create_coding_session(
        adapter,
        model,
        workspace=workspace,
        storage_root=arguments.storage_root,
        session_id=arguments.session,
        no_save=arguments.no_save,
        trace_secrets=(config.api_key, *config.headers.values()),
        run_guard=guard,
    )
    handle = session.start(arguments.prompt)
    envelopes: list[EventEnvelope] = []
    async for event in handle.events():
        envelope = EventEnvelope.from_runtime(event, session_id=session.session_id)
        envelopes.append(envelope)
        if (
            arguments.format == "text"
            and not arguments.quiet
            and event.type.value in {"retry_scheduled", "tool_call_start", "run_cancelled"}
        ):
            print(event.type.value, file=stderr)
    session_result = await handle.result()
    result = RunResult.from_session(handle, session.session_id, session_result)
    if arguments.format == "text":
        stdout.write(result.final_text)
        if result.final_text and not result.final_text.endswith("\n"):
            stdout.write("\n")
    elif arguments.format == "json":
        stdout.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    else:
        for envelope in envelopes:
            stdout.write(
                json.dumps(envelope.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
        ending = EventEnvelope.run_end(
            result,
            max((item.sequence for item in envelopes), default=0) + 1,
        )
        stdout.write(json.dumps(ending.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return exit_code(result.status)


def _latest_session_id(
    workspace: Path,
    storage_root: str | Path | None,
) -> str:
    local = LocalWorkspace.open(workspace, storage_root=storage_root)
    directory = local.sessions.sessions_directory
    candidates = tuple(directory.glob("*.jsonl")) if directory.is_dir() else ()
    if not candidates:
        raise CLIConfigurationError("no durable Session exists for this workspace")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).stem


async def _chat(
    arguments: argparse.Namespace,
    *,
    config: OpenAICompatibleConfig,
    model: ModelSpec,
    guard: RunGuard | None,
    stdout: TextIO,
    stderr: TextIO,
    input_stream: TextIO,
    adapter_factory: AdapterFactory,
) -> int:
    workspace = Path(arguments.workspace).resolve()
    if arguments.continue_latest and arguments.session is not None:
        raise CLIConfigurationError("--continue and --session are mutually exclusive")
    if arguments.no_save and (arguments.continue_latest or arguments.session is not None):
        raise CLIConfigurationError("--no-save cannot continue a durable Session")
    session_id = arguments.session
    if arguments.continue_latest:
        session_id = _latest_session_id(workspace, arguments.storage_root)
    session = create_coding_session(
        adapter_factory(config),
        model,
        workspace=workspace,
        storage_root=arguments.storage_root,
        session_id=session_id,
        no_save=arguments.no_save,
        trace_secrets=(config.api_key, *config.headers.values()),
        run_guard=guard,
    )
    controller = NonTerminalAdapter(session)
    lines: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def read_lines() -> None:
        while True:
            line = input_stream.readline()
            if line == "":
                try:
                    loop.call_soon_threadsafe(lines.put_nowait, None)
                except RuntimeError:
                    pass
                return
            try:
                loop.call_soon_threadsafe(lines.put_nowait, line)
            except RuntimeError:
                return

    threading.Thread(target=read_lines, daemon=True).start()

    async def settle_active() -> RunResult:
        async for event in controller.events():
            if event.type in {
                "retry_scheduled",
                "tool_call_start",
                "tool_call_end",
                "run_cancelled",
            }:
                tool_name = event.payload.get("tool_name")
                detail = f" {tool_name}" if tool_name else ""
                print(f"[{event.type}]{detail}", file=stderr)
        return await controller.result()

    active: asyncio.Task[RunResult] | None = None
    pending_line: str | None = None
    wait_for_result = False
    quit_after_run = False
    while True:
        if active is not None:
            input_task: asyncio.Task[str | None] | None = None
            waiters: set[asyncio.Task[object]] = {active}
            if not wait_for_result:
                input_task = asyncio.create_task(lines.get())
                waiters.add(input_task)
            completed, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED
            )
            if active in completed:
                result = active.result()
                if result.final_text:
                    print(f"assistant: {result.final_text}", file=stdout)
                active = None
                wait_for_result = False
                if input_task is not None:
                    if input_task.done():
                        consumed = input_task.result()
                        if consumed is None:
                            quit_after_run = True
                        else:
                            pending_line = consumed
                    else:
                        input_task.cancel()
                if quit_after_run:
                    return 0
                if result.status == "cancelled":
                    continue
                if result.status != "completed":
                    return exit_code(result.status)
                continue
            assert input_task is not None
            incoming = input_task.result()
            if incoming is None:
                wait_for_result = True
                quit_after_run = True
                continue
            control = incoming.strip()
            if not control:
                continue
            if control == "/wait":
                wait_for_result = True
            elif control == "/quit":
                controller.cancel()
                wait_for_result = True
                quit_after_run = True
            elif control == "/cancel":
                controller.cancel()
                wait_for_result = True
            elif control.startswith("/steer "):
                controller.steer(control.removeprefix("/steer "))
            elif control.startswith("/follow-up "):
                controller.follow_up(control.removeprefix("/follow-up "))
            elif control.startswith("/"):
                print("chat commands require an idle Session", file=stderr)
            else:
                controller.follow_up(control)
            continue

        line = pending_line
        pending_line = None
        if line is None:
            line = await lines.get()
        if line is None:
            return 0
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            return 0
        if text == "/wait":
            continue
        if text == "/cancel" or text.startswith(("/steer ", "/follow-up ")):
            print("no active Run", file=stderr)
            continue
        if text.startswith("/compact"):
            focus = text.removeprefix("/compact").strip() or None
            await session.compact(focus)
            continue
        if text.startswith("/"):
            command, _, command_arguments = text[1:].partition(" ")
            rendered = await controller.command(command, command_arguments)
            if rendered is not None:
                print(rendered, file=stdout)
            continue
        controller.start(text)
        active = asyncio.create_task(settle_active())
    return 0


def _session_summaries(local: LocalWorkspace) -> list[dict[str, object]]:
    directory = local.sessions.sessions_directory
    if not directory.is_dir():
        return []
    summaries: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        state = local.sessions.read(path.stem)
        summaries.append(
            {
                "session_id": state.session_id,
                "entries": len(state.entries),
                "active_leaf_id": state.active_leaf_id,
                "recovery_warning": (
                    None
                    if state.recovery_warning is None
                    else state.recovery_warning.code.value
                ),
                "modified_ns": path.stat().st_mtime_ns,
            }
        )
    def modified(item: Mapping[str, object]) -> int:
        value = item["modified_ns"]
        assert isinstance(value, int)
        return value

    return sorted(summaries, key=modified, reverse=True)


def _storage_command(arguments: argparse.Namespace, stdout: TextIO) -> int:
    local = LocalWorkspace.open(
        Path(arguments.workspace).resolve(), storage_root=arguments.storage_root
    )
    if arguments.command == "sessions":
        if arguments.sessions_command == "list":
            payload: object = _session_summaries(local)
        elif arguments.sessions_command == "show":
            state = local.sessions.read(arguments.session_id)
            payload = {
                "session_id": state.session_id,
                "entries": len(state.entries),
                "active_leaf_id": state.active_leaf_id,
                "messages": len(state.history()),
                "compactions": len(state.compactions()),
                "custom_entries": len(state.custom_entries()),
            }
        elif arguments.sessions_command == "export":
            stdout.write(local.sessions.path_for(arguments.session_id).read_text(encoding="utf-8"))
            return 0
        else:
            source = local.sessions.read(arguments.session_id)
            leaf = arguments.entry or source.active_leaf_id
            history = source.history(leaf)
            forked = local.sessions.create()
            if history:
                with local.sessions.writer(forked.session_id) as writer:
                    writer.append(history)
            payload = {
                "session_id": forked.session_id,
                "forked_from": source.session_id,
                "entry_id": leaf,
            }
    else:
        if arguments.runs_command == "list":
            payload = [
                {
                    "run_id": item.run_id,
                    "session_id": item.session_id,
                    "status": item.status,
                    "started_at": item.started_at,
                    "finished_at": item.finished_at,
                    "evidence_complete": item.evidence_complete,
                }
                for item in local.runs.list(
                    session_id=arguments.session, status=arguments.status
                )
            ]
        elif arguments.runs_command == "show":
            summary = local.runs.summary(arguments.run_id)
            payload = {
                "run_id": summary.run_id,
                "session_id": summary.session_id,
                "status": summary.status,
                "started_at": summary.started_at,
                "finished_at": summary.finished_at,
                "evidence_complete": summary.evidence_complete,
                "records": list(local.runs.records(arguments.run_id)),
                "annotations": list(local.runs.annotations(arguments.run_id)),
            }
        elif arguments.runs_command == "export":
            stdout.write(local.runs.export(arguments.run_id))
            return 0
        elif arguments.runs_command == "annotate":
            try:
                annotation = json.loads(arguments.payload)
            except json.JSONDecodeError as error:
                raise CLIConfigurationError("--payload must be valid JSON") from error
            if not isinstance(annotation, dict):
                raise CLIConfigurationError("--payload must be a JSON object")
            local.runs.annotate(arguments.run_id, arguments.namespace, annotation)
            payload = {"run_id": arguments.run_id, "annotated": True}
        else:
            preview = local.runs.prune_preview(
                session_id=arguments.session, status=arguments.status
            )
            if arguments.apply:
                local.runs.prune(preview)
            payload = {
                "run_ids": list(preview.run_ids),
                "bytes": preview.bytes,
                "applied": bool(arguments.apply),
            }
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    adapter_factory: AdapterFactory = OpenAICompatibleAdapter,
    input_stream: TextIO | None = None,
) -> int:
    """Run Omega without hiding terminal or environment state in the library."""

    accepted_environ = os.environ if environ is None else environ
    output = sys.stdout if stdout is None else stdout
    diagnostics = sys.stderr if stderr is None else stderr
    terminal_input = sys.stdin if input_stream is None else input_stream
    try:
        arguments = build_parser().parse_args(list(argv) if argv is not None else None)
        if arguments.command in {"sessions", "runs"}:
            return _storage_command(arguments, output)
        config, model, guard = _configuration(arguments, accepted_environ)
        if arguments.command == "exec":
            operation = _exec(
                    arguments,
                    config=config,
                    model=model,
                    guard=guard,
                    stdout=output,
                    stderr=diagnostics,
                    adapter_factory=adapter_factory,
                )
        elif arguments.command == "chat":
            operation = _chat(
                    arguments,
                    config=config,
                    model=model,
                    guard=guard,
                    stdout=output,
                    stderr=diagnostics,
                    input_stream=terminal_input,
                    adapter_factory=adapter_factory,
                )
        else:
            raise CLIConfigurationError(
                f"{arguments.command} command is not implemented by this checkpoint"
            )
        return asyncio.run(operation)
    except CLIConfigurationError as error:
        print(f"omega: configuration error: {error}", file=diagnostics)
        return 2
    except KeyboardInterrupt:
        print("omega: cancelled", file=diagnostics)
        return 130
    except Exception as error:
        print(f"omega: infrastructure error: {type(error).__name__}: {error}", file=diagnostics)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
