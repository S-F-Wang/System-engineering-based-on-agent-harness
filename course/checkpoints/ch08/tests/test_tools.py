from __future__ import annotations

import asyncio

import pytest
from jsonschema.exceptions import SchemaError

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    CompleteOutputReference,
    EventType,
    ModelEnd,
    ModelSpec,
    PreparedToolCall,
    Role,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    Tool,
    ToolCallDelta,
    ToolErrorCode,
    ToolExecutor,
    ToolOutputBudget,
    ToolResult,
    ToolResultMessage,
)


def test_model_can_call_one_typed_tool_and_continue() -> None:
    async def scenario() -> None:
        received: list[dict[str, object]] = []

        async def add(arguments: dict[str, object]) -> ToolResult:
            received.append(arguments)
            return ToolResult(content=str(arguments["left"] + arguments["right"]))

        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-add", "add", '{"left": 2, "right": 3}'),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("The sum is 5."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/tools"),
            tools=[
                Tool(
                    name="add",
                    description="Add two integers",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "left": {"type": "integer"},
                            "right": {"type": "integer"},
                        },
                        "required": ["left", "right"],
                        "additionalProperties": False,
                    },
                    execute=add,
                )
            ],
        )

        outcome = await runtime.run([AgentMessage.text(Role.USER, "add 2 and 3")])

        assert outcome.stop_reason is StopReason.COMPLETE
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "The sum is 5.")
        assert received == [{"left": 2, "right": 3}]
        result_messages = [
            message for message in runtime.history if isinstance(message, ToolResultMessage)
        ]
        assert len(result_messages) == 1
        assert result_messages[0].tool_call_id == "call-add"
        assert result_messages[0].result.content == "5"
        assert adapter.received_requests[1].messages[-1] == result_messages[0].to_model()

    asyncio.run(scenario())


def test_tool_declarations_are_explicit_and_runtime_registers_nothing_implicitly() -> None:
    async def execute(arguments: dict[str, object]) -> ToolResult:
        return ToolResult("ok")

    with pytest.raises(ValueError, match="name"):
        Tool("", "description", {"type": "object"}, execute)
    with pytest.raises(ValueError, match="description"):
        Tool("named", "", {"type": "object"}, execute)
    with pytest.raises(SchemaError):
        Tool("broken", "Bad schema", {"type": "not-a-json-schema-type"}, execute)
    with pytest.raises(ValueError, match="cannot terminate"):
        ToolResult(
            "failed",
            terminate=True,
            is_error=True,
            error_code=ToolErrorCode.EXECUTION_FAILED,
        )

    async def scenario() -> None:
        adapter = ScriptedModelAdapter(
            [TextDelta("No Tools installed."), ModelEnd(StopReason.COMPLETE)]
        )
        runtime = AgentRuntime(adapter, ModelSpec("scripted/no-tools"))

        await runtime.run([AgentMessage.text(Role.USER, "plain completion")])

        assert adapter.received_requests[0].tools == ()

    asyncio.run(scenario())


def test_oversized_tool_output_is_bounded_with_notice_and_complete_reference() -> None:
    async def scenario() -> None:
        async def verbose(arguments: dict[str, object]) -> ToolResult:
            return ToolResult(
                "abcdefghijk",
                complete_output=CompleteOutputReference.artifact("sha256:full-output"),
            )

        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "verbose-call", "verbose", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("Output was bounded."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/output-budget"),
            tools=[
                Tool(
                    "verbose",
                    "Return verbose output",
                    {"type": "object"},
                    verbose,
                )
            ],
            tool_output_budget=ToolOutputBudget(max_bytes=5, max_lines=10),
        )

        await runtime.run([AgentMessage.text(Role.USER, "be verbose")])

        message = next(
            item for item in runtime.history if isinstance(item, ToolResultMessage)
        )
        assert message.result.truncation is not None
        assert message.result.truncation.original_bytes == 11
        assert message.result.truncation.retained_start_byte == 0
        assert message.result.truncation.retained_end_byte == 5
        assert message.result.complete_output == CompleteOutputReference.artifact(
            "sha256:full-output"
        )
        assert message.result.content.startswith("abcde\n\n[tool output truncated:")
        assert "complete output: artifact sha256:full-output" in message.result.content
        assert adapter.received_requests[1].messages[-1].content == message.result.content

    asyncio.run(scenario())


def test_read_only_batch_finishes_in_parallel_but_history_keeps_call_order() -> None:
    async def scenario() -> None:
        second_started = asyncio.Event()

        async def first(arguments: dict[str, object]) -> ToolResult:
            await second_started.wait()
            await asyncio.sleep(0)
            return ToolResult("first result")

        async def second(arguments: dict[str, object]) -> ToolResult:
            second_started.set()
            return ToolResult("second result")

        schema = {"type": "object", "additionalProperties": False}
        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "first-call", "first", "{}"),
                    ToolCallDelta(1, "second-call", "second", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("Both complete."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/parallel"),
            tools=[
                Tool("first", "First read", schema, first),
                Tool("second", "Second read", schema, second),
            ],
        )

        handle = runtime.start([AgentMessage.text(Role.USER, "read both")])
        events_task = asyncio.create_task(
            _collect_events(handle)
        )
        outcome = await asyncio.wait_for(handle.result(), timeout=1)
        events = await events_task

        assert outcome.stop_reason is StopReason.COMPLETE
        completed_ids = [
            event.tool_call_id
            for event in events
            if event.type is EventType.TOOL_CALL_END
        ]
        assert completed_ids == ["second-call", "first-call"]
        persisted_ids = [
            message.tool_call_id
            for message in runtime.history
            if isinstance(message, ToolResultMessage)
        ]
        assert persisted_ids == ["first-call", "second-call"]

    async def _collect_events(handle):
        return [event async for event in handle.events()]

    asyncio.run(scenario())


def test_one_sequential_tool_forces_the_whole_batch_into_source_order() -> None:
    async def scenario() -> None:
        active = 0
        maximum_active = 0
        activity: list[str] = []

        def handler(name: str):
            async def execute(arguments: dict[str, object]) -> ToolResult:
                nonlocal active, maximum_active
                active += 1
                maximum_active = max(maximum_active, active)
                activity.append(f"start:{name}")
                await asyncio.sleep(0)
                activity.append(f"end:{name}")
                active -= 1
                return ToolResult(name)

            return execute

        schema = {"type": "object"}
        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "one", "one", "{}"),
                    ToolCallDelta(1, "write", "write", "{}"),
                    ToolCallDelta(2, "three", "three", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("Sequential batch complete."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/sequential"),
            tools=[
                Tool("one", "Read one", schema, handler("one")),
                Tool("write", "Mutate state", schema, handler("write"), sequential=True),
                Tool("three", "Read three", schema, handler("three")),
            ],
        )

        await runtime.run([AgentMessage.text(Role.USER, "run the batch")])

        assert maximum_active == 1
        assert activity == [
            "start:one",
            "end:one",
            "start:write",
            "end:write",
            "start:three",
            "end:three",
        ]

    asyncio.run(scenario())


def test_model_continuation_is_skipped_only_for_unanimous_termination() -> None:
    async def scenario() -> None:
        async def stop(arguments: dict[str, object]) -> ToolResult:
            return ToolResult("stop", terminate=True)

        async def keep_going(arguments: dict[str, object]) -> ToolResult:
            return ToolResult("continue", terminate=False)

        schema = {"type": "object"}
        tools = [
            Tool("stop", "Terminate", schema, stop),
            Tool("keep_going", "Continue", schema, keep_going),
        ]
        unanimous_adapter = ScriptedModelAdapter(
            [
                ToolCallDelta(0, "stop-one", "stop", "{}"),
                ToolCallDelta(1, "stop-two", "stop", "{}"),
                ModelEnd(StopReason.TOOL_USE),
            ]
        )
        unanimous_runtime = AgentRuntime(
            unanimous_adapter,
            ModelSpec("scripted/unanimous"),
            tools=tools,
        )

        unanimous = await unanimous_runtime.run(
            [AgentMessage.text(Role.USER, "both stop")]
        )

        assert len(unanimous_adapter.received_requests) == 1
        assert unanimous.stop_reason is StopReason.TOOL_USE
        assert [result.terminate for result in unanimous.tool_results] == [True, True]

        mixed_adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "mixed-stop", "stop", "{}"),
                    ToolCallDelta(1, "mixed-go", "keep_going", "{}"),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("Mixed batch continued."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        mixed_runtime = AgentRuntime(
            mixed_adapter,
            ModelSpec("scripted/mixed"),
            tools=tools,
        )

        mixed = await mixed_runtime.run([AgentMessage.text(Role.USER, "mixed batch")])

        assert len(mixed_adapter.received_requests) == 2
        assert mixed.stop_reason is StopReason.COMPLETE
        assert [result.terminate for result in mixed.tool_results] == [True, False]

    asyncio.run(scenario())


def test_replacing_tool_executor_changes_backend_without_changing_loop() -> None:
    async def scenario() -> None:
        local_handler_called = False

        async def local_handler(arguments: dict[str, object]) -> ToolResult:
            nonlocal local_handler_called
            local_handler_called = True
            return ToolResult("local")

        class RemoteExecutor:
            def __init__(self) -> None:
                self.calls: list[PreparedToolCall] = []

            async def execute(self, call: PreparedToolCall) -> ToolResult:
                self.calls.append(call)
                return ToolResult("remote:ok", metadata={"backend": "remote"})

        executor: ToolExecutor = RemoteExecutor()
        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "call-remote", "lookup", '{"key": "answer"}'),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("Remote result accepted."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/remote-executor"),
            tools=[
                Tool(
                    "lookup",
                    "Look up a key",
                    {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    local_handler,
                )
            ],
            tool_executor=executor,
        )

        outcome = await runtime.run([AgentMessage.text(Role.USER, "look it up")])

        assert outcome.stop_reason is StopReason.COMPLETE
        assert not local_handler_called
        assert len(executor.calls) == 1  # type: ignore[attr-defined]
        assert executor.calls[0].arguments == {"key": "answer"}  # type: ignore[attr-defined]
        result = next(
            message.result
            for message in runtime.history
            if isinstance(message, ToolResultMessage)
        )
        assert result.content == "remote:ok"
        assert result.metadata == {"backend": "remote"}

    asyncio.run(scenario())


def test_bad_calls_and_ordinary_failures_become_safe_actionable_results() -> None:
    async def scenario() -> None:
        async def explode(arguments: dict[str, object]) -> ToolResult:
            raise RuntimeError("SECRET backend path /private/executor.py")

        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        adapter = ScriptedModelAdapter(
            [
                [
                    ToolCallDelta(0, "bad-json", "explode", "{"),
                    ToolCallDelta(1, "bad-schema", "explode", '{"value": "wrong"}'),
                    ToolCallDelta(2, "unknown", "missing", "{}"),
                    ToolCallDelta(3, "raised", "explode", '{"value": 1}'),
                    ModelEnd(StopReason.TOOL_USE),
                ],
                [TextDelta("I handled the tool errors."), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/tool-errors"),
            tools=[Tool("explode", "Fail safely", schema, explode)],
        )

        outcome = await runtime.run([AgentMessage.text(Role.USER, "trigger errors")])

        results = [
            message.result
            for message in runtime.history
            if isinstance(message, ToolResultMessage)
        ]
        assert [result.error_code for result in results] == [
            ToolErrorCode.INVALID_JSON,
            ToolErrorCode.INVALID_ARGUMENTS,
            ToolErrorCode.UNKNOWN_TOOL,
            ToolErrorCode.EXECUTION_FAILED,
        ]
        assert all(result.is_error and not result.terminate for result in results)
        assert all("Tool error" in result.content for result in results)
        assert "SECRET" not in "\n".join(result.content for result in results)
        assert outcome.stop_reason is StopReason.COMPLETE

    asyncio.run(scenario())
