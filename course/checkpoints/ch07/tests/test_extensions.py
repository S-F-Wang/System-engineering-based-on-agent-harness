from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_harness import (
    AgentRuntime,
    AgentSession,
    Extension,
    ModelEnd,
    ModelSpec,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    Tool,
    ToolCallDelta,
    ToolResult,
    ToolErrorCode,
    EventType,
    MemorySessionStore,
    AgentMessage,
    HookPoint,
    ModelRequest,
    Role,
    HookExecutionError,
    SessionBusyError,
    ExtensionInitializationError,
    JSONLSessionStore,
    CompactionPolicy,
    RetryPolicy,
    RunGuard,
    CommandExecutionError,
    CompactionWarningCode,
    ModelAdapterError,
    ModelError,
    ModelErrorCode,
)


def test_explicit_extension_registers_a_tool_for_the_session() -> None:
    async def scenario() -> None:
        calls: list[dict[str, object]] = []

        async def inspect(arguments: dict[str, object]) -> ToolResult:
            calls.append(arguments)
            return ToolResult("extension result")

        def configure(api) -> None:
            api.register_tool(
                Tool(
                    "inspect",
                    "Inspect one value",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    inspect,
                )
            )

        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-1", "inspect", '{"value":"ready"}'),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("done"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/extensions")),
            extensions=[Extension("example", "1.0.0", configure)],
        )

        result = await session.run("use the extension")

        assert result.outcome.message.content[0].text == "done"
        assert calls == [{"value": "ready"}]

    asyncio.run(scenario())


def test_high_level_event_subscribers_are_ordered_lifecycle_barriers() -> None:
    async def scenario() -> None:
        observed: list[str] = []
        store = MemorySessionStore(id_factory=iter(("session-1", "entry-1", "entry-2")).__next__)
        state = store.create()
        session: AgentSession

        def configure_first(api) -> None:
            async def observe(event) -> None:
                if event.type is EventType.MESSAGE_END:
                    assert len(store.read(state.session_id).history()) == 2
                    observed.append("first:message_end")
                if event.type is EventType.AGENT_END:
                    assert session.busy is True
                    observed.append("first:agent_end")

            api.subscribe(observe)

        def configure_second(api) -> None:
            async def observe(event) -> None:
                if event.type in {EventType.MESSAGE_END, EventType.AGENT_END}:
                    await asyncio.sleep(0)
                    observed.append(f"second:{event.type.value}")

            api.subscribe(observe)

        adapter = ScriptedModelAdapter(
            [TextDelta("settled"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/event-barriers")),
            store=store,
            session_id=state.session_id,
            extensions=[
                Extension("first", "1", configure_first),
                Extension("second", "1", configure_second),
            ],
        )

        await session.run("persist before observing")

        assert observed == [
            "first:message_end",
            "second:message_end",
            "first:agent_end",
            "second:agent_end",
        ]
        assert session.busy is False

    asyncio.run(scenario())


def test_low_level_event_iteration_is_ordered_but_not_a_runtime_barrier() -> None:
    async def scenario() -> None:
        session = AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("complete"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/observational-stream"),
            )
        )

        handle = session.start("do not wait for my consumer")
        result = await asyncio.wait_for(handle.result(), timeout=1)
        events = [event async for event in handle.events()]

        assert result.outcome.stop_reason is StopReason.COMPLETE
        assert [event.sequence for event in events] == list(
            range(1, len(events) + 1)
        )
        assert events[-1].type is EventType.AGENT_END

    asyncio.run(scenario())


def test_before_model_request_hook_can_transform_the_pending_request() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def add_context(context):
                request = context.value
                assert isinstance(request, ModelRequest)
                return replace(
                    request,
                    messages=(
                        *request.messages,
                        AgentMessage.text(Role.SYSTEM, "extension context").to_model(),
                    ),
                )

            api.register_hook(HookPoint.BEFORE_MODEL_REQUEST, add_context)

        adapter = ScriptedModelAdapter(
            [TextDelta("hooked"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/before-model-hook")),
            extensions=[Extension("request-context", "1", configure)],
        )

        await session.run("start")

        assert adapter.received_requests[0].messages[-1].content[0].text == (
            "extension context"
        )

    asyncio.run(scenario())


def test_before_run_hook_failure_terminates_without_model_work() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def reject(context):
                raise RuntimeError("policy unavailable")

            api.register_hook(HookPoint.BEFORE_RUN, reject)

        adapter = ScriptedModelAdapter(
            [TextDelta("must not run"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/before-run-failure")),
            extensions=[Extension("guard", "1", configure)],
        )

        result = await session.run("blocked")

        assert result.outcome.stop_reason is StopReason.ERROR
        assert result.outcome.error is not None
        assert result.outcome.error.code.value == "hook_failed"
        assert "guard" in result.outcome.error.message
        assert adapter.received_requests == ()
        assert session.busy is False

    asyncio.run(scenario())


def test_before_tool_call_hook_failure_blocks_tool_execution_safely() -> None:
    async def scenario() -> None:
        executed = False

        async def dangerous(arguments: dict[str, object]) -> ToolResult:
            nonlocal executed
            executed = True
            return ToolResult("unsafe")

        def configure(api) -> None:
            async def block(context):
                raise PermissionError("approval unavailable")

            api.register_hook(HookPoint.BEFORE_TOOL_CALL, block)

        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-blocked", "dangerous", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("handled safely"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        session = AgentSession(
            AgentRuntime(
                adapter,
                ModelSpec("scripted/before-tool-failure"),
                tools=[
                    Tool(
                        "dangerous",
                        "A guarded operation",
                        {"type": "object"},
                        dangerous,
                    )
                ],
            ),
            extensions=[Extension("approval", "1", configure)],
        )

        result = await session.run("try it")

        assert executed is False
        assert result.outcome.tool_results[0].error_code is ToolErrorCode.HOOK_FAILED
        assert result.outcome.message.content[0].text == "handled safely"

    asyncio.run(scenario())


def test_after_tool_call_hook_failure_withholds_the_unreviewed_result() -> None:
    async def scenario() -> None:
        async def reveal(arguments: dict[str, object]) -> ToolResult:
            return ToolResult("unreviewed secret")

        def configure(api) -> None:
            async def fail_review(context):
                raise RuntimeError("redactor failed")

            api.register_hook(HookPoint.AFTER_TOOL_CALL, fail_review)

        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-review", "reveal", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("safe continuation"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        session = AgentSession(
            AgentRuntime(
                adapter,
                ModelSpec("scripted/after-tool-failure"),
                tools=[
                    Tool("reveal", "Return sensitive data", {"type": "object"}, reveal)
                ],
            ),
            extensions=[Extension("redactor", "1", configure)],
        )

        result = await session.run("inspect")

        tool_result = result.outcome.tool_results[0]
        assert tool_result.error_code is ToolErrorCode.HOOK_FAILED
        assert "unreviewed secret" not in adapter.received_requests[1].messages[-1].content

    asyncio.run(scenario())


def test_before_compaction_hook_failure_cancels_manual_compaction() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def reject(context):
                raise RuntimeError("summary policy unavailable")

            api.register_hook(HookPoint.BEFORE_COMPACTION, reject)

        adapter = ScriptedModelAdapter(
            [TextDelta("must not summarize"), ModelEnd(StopReason.COMPLETE)]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/compaction-hook"),
            history=(
                AgentMessage.text(Role.USER, "old question"),
                AgentMessage.text(Role.ASSISTANT, "old answer"),
            ),
        )
        session = AgentSession(
            runtime,
            extensions=[Extension("summary-policy", "1", configure)],
        )

        try:
            await session.compact()
        except HookExecutionError as error:
            assert error.point is HookPoint.BEFORE_COMPACTION
            assert error.extension_name == "summary-policy"
        else:
            raise AssertionError("manual Compaction must fail safe")

        assert adapter.received_requests == ()
        assert len(runtime.history) == 2
        assert session.busy is False

    asyncio.run(scenario())


def test_subscriber_failure_is_traced_and_does_not_change_the_run() -> None:
    async def scenario() -> None:
        reached_second: list[EventType] = []

        def configure_failing(api) -> None:
            async def fail(event) -> None:
                if event.type is EventType.AGENT_START:
                    raise RuntimeError("observer broke")

            api.subscribe(fail)

        def configure_healthy(api) -> None:
            async def observe(event) -> None:
                reached_second.append(event.type)

            api.subscribe(observe)

        session = AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("still succeeds"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/subscriber-failure"),
            ),
            extensions=[
                Extension("broken-observer", "1", configure_failing),
                Extension("healthy-observer", "1", configure_healthy),
            ],
        )
        handle = session.start("observe")
        result = await handle.result()
        events = [event async for event in handle.events()]

        failures = [
            event for event in events if event.type is EventType.SUBSCRIBER_FAILED
        ]
        assert result.outcome.stop_reason is StopReason.COMPLETE
        assert EventType.AGENT_START in reached_second
        assert len(failures) == 1
        assert failures[0].extension_name == "broken-observer"
        assert failures[0].diagnostic == "RuntimeError"

    asyncio.run(scenario())


def test_extension_registers_a_chat_command_without_starting_a_run() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def greeting(arguments: str) -> str:
                return f"hello {arguments}"

            api.register_command("greet", greeting)

        adapter = ScriptedModelAdapter(
            [TextDelta("must not run"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/chat-command")),
            extensions=[Extension("greeter", "1", configure)],
        )

        result = await session.execute_command("greet", "Omega")

        assert result == "hello Omega"
        assert adapter.received_requests == ()
        assert session.busy is False

    asyncio.run(scenario())


def test_extension_appends_a_versioned_namespaced_custom_entry() -> None:
    store = MemorySessionStore(
        id_factory=iter(("session-custom", "entry-custom")).__next__
    )
    state = store.create()

    def configure(api) -> None:
        api.append_custom_entry(
            "stateful.cursor",
            1,
            {"position": 3},
        )

    AgentSession(
        AgentRuntime(
            ScriptedModelAdapter(
                [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
            ),
            ModelSpec("scripted/custom-entry"),
        ),
        store=store,
        session_id=state.session_id,
        extensions=[Extension("stateful", "1", configure)],
    )

    restored = store.read(state.session_id)
    assert restored.history() == ()
    assert len(restored.custom_entries()) == 1
    entry = restored.custom_entries()[0]
    assert entry.namespace == "stateful.cursor"
    assert entry.version == 1
    assert dict(entry.payload) == {"position": 3}


def test_run_snapshot_freezes_tools_until_the_next_run() -> None:
    async def scenario() -> None:
        old_calls = 0
        new_calls = 0

        async def old_handler(arguments: dict[str, object]) -> ToolResult:
            nonlocal old_calls
            old_calls += 1
            return ToolResult("old")

        async def new_handler(arguments: dict[str, object]) -> ToolResult:
            nonlocal new_calls
            new_calls += 1
            return ToolResult("new")

        class PausingAdapter:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.turn = 0

            async def stream(self, request):
                self.turn += 1
                if self.turn == 1:
                    self.started.set()
                    await self.release.wait()
                if self.turn in {1, 3}:
                    yield ToolCallDelta(0, f"call-{self.turn}", "versioned", "{}")
                    yield ModelEnd(StopReason.TOOL_USE)
                    return
                yield TextDelta(f"complete-{self.turn}")
                yield ModelEnd(StopReason.COMPLETE)

        schema = {"type": "object"}
        adapter = PausingAdapter()
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/snapshot"),
            tools=[Tool("versioned", "old behavior", schema, old_handler)],
        )
        session = AgentSession(runtime)

        first = session.start("first")
        await adapter.started.wait()
        runtime.configure_tools(
            [Tool("versioned", "new behavior", schema, new_handler)]
        )
        adapter.release.set()
        first_result = await first.result()

        second_result = await session.run("second")

        assert old_calls == 1
        assert new_calls == 1
        assert first.snapshot.fingerprint == first_result.outcome.snapshot_fingerprint
        assert first.snapshot.fingerprint != second_result.outcome.snapshot_fingerprint

    asyncio.run(scenario())


def test_extension_adds_a_versioned_run_annotation() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def annotate(context):
                api.add_run_annotation(
                    context.snapshot.run_id,
                    "reviewer.outcome",
                    1,
                    {"label": "accepted"},
                )

            api.register_hook(HookPoint.BEFORE_RUN, annotate)

        session = AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("annotated"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/annotation"),
            ),
            extensions=[Extension("reviewer", "1", configure)],
        )

        handle = session.start("review")
        await handle.result()

        annotations = session.run_annotations(handle.run_id)
        assert len(annotations) == 1
        assert annotations[0].namespace == "reviewer.outcome"
        assert annotations[0].version == 1
        assert dict(annotations[0].payload) == {"label": "accepted"}

    asyncio.run(scenario())


def test_idle_reload_tears_down_before_constructing_named_replacement() -> None:
    async def scenario() -> None:
        torn_down = False
        new_calls = 0

        async def teardown_old() -> None:
            nonlocal torn_down
            torn_down = True

        def configure_old(api) -> None:
            return None

        async def new_tool(arguments: dict[str, object]) -> ToolResult:
            nonlocal new_calls
            new_calls += 1
            return ToolResult("new")

        def configure_new(api) -> None:
            assert torn_down is True
            api.register_tool(
                Tool("reloaded", "Reloaded Tool", {"type": "object"}, new_tool)
            )

        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-reload", "reloaded", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("reloaded"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/reload")),
            extensions=[
                Extension(
                    "switcher",
                    "1",
                    configure_old,
                    teardown=teardown_old,
                )
            ],
        )

        reload_result = await session.reload_extensions(
            [Extension("switcher", "2", configure_new)]
        )
        run_result = await session.run("after reload")

        assert reload_result.replaced_extensions == ("switcher",)
        assert new_calls == 1
        assert run_result.outcome.message.content[0].text == "reloaded"

    asyncio.run(scenario())


def test_teardown_failure_is_a_warning_and_never_leaves_session_busy() -> None:
    async def scenario() -> None:
        async def broken_teardown() -> None:
            raise RuntimeError("cleanup failed")

        session = AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/teardown-warning"),
            ),
            extensions=[
                Extension(
                    "fragile",
                    "1",
                    lambda api: None,
                    source="explicit:fragile",
                    teardown=broken_teardown,
                )
            ],
        )

        result = await session.reload_extensions([])

        assert session.busy is False
        assert len(result.warnings) == 1
        assert result.warnings[0].source == "explicit:fragile"
        assert result.warnings[0].phase == "teardown"
        assert result.warnings[0].diagnostic == "RuntimeError"

    asyncio.run(scenario())


def test_reload_is_rejected_while_a_run_is_active() -> None:
    async def scenario() -> None:
        class BlockingAdapter:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def stream(self, request):
                self.started.set()
                await self.release.wait()
                yield TextDelta("done")
                yield ModelEnd(StopReason.COMPLETE)

        adapter = BlockingAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/busy-reload"))
        )
        handle = session.start("active")
        await adapter.started.wait()

        with pytest.raises(SessionBusyError, match="idle Session"):
            await session.reload_extensions([])

        adapter.release.set()
        await handle.result()

    asyncio.run(scenario())


def test_extension_startup_is_awaited_once_before_run_hooks() -> None:
    async def scenario() -> None:
        order: list[str] = []

        async def startup() -> None:
            await asyncio.sleep(0)
            order.append("startup")

        def configure(api) -> None:
            async def before_run(context):
                order.append("before_run")

            api.register_hook(HookPoint.BEFORE_RUN, before_run)

        session = AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [
                        [TextDelta("first"), ModelEnd(StopReason.COMPLETE)],
                        [TextDelta("second"), ModelEnd(StopReason.COMPLETE)],
                    ]
                ),
                ModelSpec("scripted/startup"),
            ),
            extensions=[
                Extension("lifecycle", "1", configure, startup=startup)
            ],
        )

        await session.run("one")
        await session.run("two")

        assert order == ["startup", "before_run", "before_run"]

    asyncio.run(scenario())


def test_tool_and_command_name_replacement_must_be_explicit_and_is_recorded() -> None:
    async def base(arguments: dict[str, object]) -> ToolResult:
        return ToolResult("base")

    async def replacement(arguments: dict[str, object]) -> ToolResult:
        return ToolResult("replacement")

    async def first_command(arguments: str) -> str:
        return "first"

    async def second_command(arguments: str) -> str:
        return "second"

    def configure_first(api) -> None:
        api.register_command("status", first_command)

    def configure_second(api) -> None:
        api.register_tool(
            Tool("shared", "Replacement", {"type": "object"}, replacement),
            replace=True,
        )
        api.register_command("status", second_command, replace=True)

    session = AgentSession(
        AgentRuntime(
            ScriptedModelAdapter(
                [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
            ),
            ModelSpec("scripted/replacements"),
            tools=[Tool("shared", "Base", {"type": "object"}, base)],
        ),
        extensions=[
            Extension("first", "1", configure_first),
            Extension("second", "1", configure_second),
        ],
    )

    assert [
        (record.kind, record.name, record.previous_owner, record.replacement_owner)
        for record in session.extension_replacements
    ] == [
        ("tool", "shared", "runtime", "second"),
        ("command", "status", "first", "second"),
    ]


def test_extension_initialization_failure_prevents_session_construction() -> None:
    def broken(api) -> None:
        raise RuntimeError("bad configuration")

    with pytest.raises(ExtensionInitializationError) as captured:
        AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/init-failure"),
            ),
            extensions=[
                Extension(
                    "broken",
                    "1",
                    broken,
                    source="explicit:/trusted/broken.py",
                )
            ],
        )

    assert captured.value.source == "explicit:/trusted/broken.py"
    assert "RuntimeError" in str(captured.value)


def test_custom_entries_round_trip_through_jsonl_without_entering_model_history(
    tmp_path,
) -> None:
    store = JSONLSessionStore(
        tmp_path,
        id_factory=iter(("session-jsonl-custom", "entry-jsonl-custom")).__next__,
    )
    state = store.create()

    def configure(api) -> None:
        api.append_custom_entry("durable.state", 2, {"enabled": True})

    AgentSession(
        AgentRuntime(
            ScriptedModelAdapter(
                [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
            ),
            ModelSpec("scripted/jsonl-custom"),
        ),
        store=store,
        session_id=state.session_id,
        extensions=[Extension("durable", "1", configure)],
    )

    restored = JSONLSessionStore(tmp_path).read(state.session_id)
    assert restored.history() == ()
    assert restored.effective_history() == ()
    assert restored.custom_entries()[0].version == 2
    assert dict(restored.custom_entries()[0].payload) == {"enabled": True}


def test_run_snapshot_and_events_retain_effective_configuration_evidence() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def inspect(context):
                return None

            api.register_hook(HookPoint.BEFORE_RUN, inspect)

        retry = RetryPolicy(delays=(0.25,))
        guard = RunGuard(max_turns=3)
        compaction = CompactionPolicy(
            reserve_tokens=32,
            keep_recent_tokens=16,
        )
        session = AgentSession(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("evidence"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec(
                    "scripted/evidence",
                    context_window=1024,
                    max_output_tokens=128,
                ),
                retry_policy=retry,
                run_guard=guard,
                generation_settings={"temperature": 0},
            ),
            compaction_policy=compaction,
            extensions=[Extension("evidence", "2", configure)],
            prompt_hashes={"system": "sha256:prompt"},
            resource_hashes={"skill:review": "sha256:skill"},
        )

        handle = session.start("capture")
        await handle.result()
        events = [event async for event in handle.events()]
        snapshot = handle.snapshot

        assert snapshot.model.model_id == "scripted/evidence"
        assert snapshot.extension_identities == ("evidence@2",)
        assert snapshot.retry_policy is retry
        assert snapshot.run_guard is guard
        assert snapshot.compaction_policy is compaction
        assert dict(snapshot.generation_settings) == {"temperature": 0}
        assert snapshot.prompt_hashes == (("system", "sha256:prompt"),)
        assert snapshot.resource_hashes == (
            ("skill:review", "sha256:skill"),
        )
        assert all(event.run_id == handle.run_id for event in events)
        assert all(
            event.snapshot_fingerprint == snapshot.fingerprint for event in events
        )

    asyncio.run(scenario())


def test_before_run_context_replacement_is_local_to_each_run() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def add_transient_context(context):
                return (
                    *context.value,
                    AgentMessage.text(Role.SYSTEM, "transient extension context"),
                )

            api.register_hook(HookPoint.BEFORE_RUN, add_transient_context)

        adapter = ScriptedModelAdapter(
            [
                [TextDelta("first"), ModelEnd(StopReason.COMPLETE)],
                [TextDelta("second"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(adapter, ModelSpec("scripted/run-local-context"))
        session = AgentSession(
            runtime,
            extensions=[Extension("transient", "1", configure)],
        )

        await session.run("one")
        await session.run("two")

        for request in adapter.received_requests:
            transient = [
                message
                for message in request.messages
                if message.role is Role.SYSTEM
                and message.content[0].text == "transient extension context"
            ]
            assert len(transient) == 1
        assert all(
            message.role is not Role.SYSTEM
            for message in runtime.history
            if isinstance(message, AgentMessage)
        )

    asyncio.run(scenario())


def test_before_model_hook_failure_terminates_before_provider_io() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def fail(context):
                raise RuntimeError("request review failed")

            api.register_hook(HookPoint.BEFORE_MODEL_REQUEST, fail)

        adapter = ScriptedModelAdapter(
            [TextDelta("must not run"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/model-hook-failure")),
            extensions=[Extension("request-review", "1", configure)],
        )

        result = await session.run("review")

        assert result.outcome.stop_reason is StopReason.ERROR
        assert result.outcome.error is not None
        assert result.outcome.error.code.value == "hook_failed"
        assert adapter.received_requests == ()

    asyncio.run(scenario())


def test_chat_command_failure_is_isolated_without_starting_a_run() -> None:
    async def scenario() -> None:
        def configure(api) -> None:
            async def broken(arguments: str) -> object:
                raise RuntimeError("command failed")

            api.register_command("broken", broken)

        adapter = ScriptedModelAdapter(
            [TextDelta("must not run"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/command-failure")),
            extensions=[Extension("commands", "1", configure)],
        )

        with pytest.raises(CommandExecutionError, match="/broken"):
            await session.execute_command("broken")

        assert adapter.received_requests == ()
        assert session.busy is False

    asyncio.run(scenario())


def test_compaction_hook_failure_warns_at_threshold_but_is_terminal_for_overflow() -> None:
    async def threshold_scenario() -> None:
        def configure(api) -> None:
            async def reject(context):
                raise RuntimeError("compaction review failed")

            api.register_hook(HookPoint.BEFORE_COMPACTION, reject)

        adapter = ScriptedModelAdapter(
            [TextDelta("settled"), ModelEnd(StopReason.COMPLETE)]
        )
        session = AgentSession(
            AgentRuntime(
                adapter,
                ModelSpec(
                    "scripted/threshold-hook-failure",
                    context_window=8,
                    max_output_tokens=1,
                ),
            ),
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            extensions=[Extension("compaction-review", "1", configure)],
        )

        result = await session.run("a prompt long enough for threshold compaction")

        assert result.outcome.stop_reason is StopReason.COMPLETE
        assert session.warnings[-1].code is CompactionWarningCode.THRESHOLD_FAILED
        assert len(adapter.received_requests) == 1

    async def overflow_scenario() -> None:
        class OverflowAdapter:
            async def stream(self, request):
                raise ModelAdapterError(
                    ModelError(
                        ModelErrorCode.CONTEXT_OVERFLOW,
                        "too large",
                        False,
                    )
                )
                yield  # pragma: no cover

        def configure(api) -> None:
            async def reject(context):
                raise RuntimeError("compaction review failed")

            api.register_hook(HookPoint.BEFORE_COMPACTION, reject)

        runtime = AgentRuntime(
            OverflowAdapter(),
            ModelSpec("scripted/overflow-hook-failure"),
            history=(
                AgentMessage.text(Role.USER, "old question"),
                AgentMessage.text(Role.ASSISTANT, "old answer"),
            ),
        )
        session = AgentSession(
            runtime,
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            extensions=[Extension("compaction-review", "1", configure)],
        )

        result = await session.run("continue")

        assert result.outcome.stop_reason is StopReason.ERROR
        assert result.outcome.error is not None
        assert result.outcome.error.code is ModelErrorCode.COMPACTION_FAILED

    asyncio.run(threshold_scenario())
    asyncio.run(overflow_scenario())
