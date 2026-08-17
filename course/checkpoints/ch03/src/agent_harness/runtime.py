"""Async Agent Runtime with structured Tool batches."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import TypeAlias, cast

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    ValidationError,
)

from .model import (
    AgentMessage,
    ContentBlock,
    ModelAdapter,
    ModelAdapterError,
    ModelEnd,
    ModelError,
    ModelErrorCode,
    ModelEvent,
    ModelRequest,
    ModelSpec,
    Role,
    StopReason,
    TextContent,
    TextDelta,
    ToolCallContent,
    ToolCallDelta,
    Usage,
    UsageUpdate,
    to_model_messages,
)
from .tools import (
    LocalToolExecutor,
    PreparedToolCall,
    Tool,
    ToolErrorCode,
    ToolExecutor,
    ToolOutputBudget,
    ToolResult,
    ToolResultMessage,
    bound_tool_result,
)


class EventType(str, Enum):
    AGENT_START = "agent_start"
    MODEL_ATTEMPT_START = "model_attempt_start"
    MODEL_EVENT = "model_event"
    MODEL_ATTEMPT_FAILED = "model_attempt_failed"
    RETRY_SCHEDULED = "retry_scheduled"
    TOOL_BATCH_START = "tool_batch_start"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_BATCH_END = "tool_batch_end"
    RUN_CANCELLED = "run_cancelled"
    MESSAGE_END = "message_end"
    AGENT_END = "agent_end"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    type: EventType
    attempt: int | None = None
    model_event: ModelEvent | None = None
    error: ModelError | None = None
    retry_delay_seconds: float | None = None
    partial_text: str = ""
    partial_usage: Usage | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class AssistantOutcome:
    message: AgentMessage
    stop_reason: StopReason
    usage: Usage | None = None
    error: ModelError | None = None
    attempts: int = 1
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    delays: tuple[float, ...] = (2.0, 4.0, 8.0)
    max_retry_after_seconds: float = 60.0

    def __post_init__(self) -> None:
        if any(delay < 0 for delay in self.delays):
            raise ValueError("retry delays cannot be negative")
        if self.max_retry_after_seconds < 0:
            raise ValueError("max_retry_after_seconds cannot be negative")

    def delay_for(
        self,
        error: ModelError,
        failed_attempt: int,
        *,
        retry_after_seconds: float | None = None,
    ) -> float | None:
        retryable_codes = {
            ModelErrorCode.RATE_LIMIT,
            ModelErrorCode.TIMEOUT,
            ModelErrorCode.CONNECTION,
            ModelErrorCode.SERVER,
        }
        retryable_status = error.status_code in {408, 429} or (
            error.status_code is not None and error.status_code >= 500
        )
        if (
            error.code not in retryable_codes and not retryable_status
        ) or failed_attempt > len(self.delays):
            return None
        if retry_after_seconds is not None:
            if not 0 <= retry_after_seconds <= self.max_retry_after_seconds:
                return None
            return retry_after_seconds
        return self.delays[failed_attempt - 1]


Sleeper: TypeAlias = Callable[[float], Awaitable[None]]
ConversationMessage: TypeAlias = AgentMessage | ToolResultMessage
_EVENTS_DONE = object()


def _add_usage(left: Usage | None, right: Usage | None) -> Usage | None:
    if left is None:
        return right
    if right is None:
        return left
    return Usage(
        left.input_tokens + right.input_tokens,
        left.output_tokens + right.output_tokens,
        left.total_tokens + right.total_tokens,
        left.estimated or right.estimated,
    )


class AgentRunHandle:
    """One accepted run's observations, cancellation, and eventual outcome."""

    def __init__(
        self,
        task: asyncio.Task[AssistantOutcome],
        events: asyncio.Queue[RuntimeEvent | object],
    ) -> None:
        self._task = task
        self._events = events
        self._cancel_requested = False

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        while True:
            event = await self._events.get()
            if event is _EVENTS_DONE:
                break
            yield cast(RuntimeEvent, event)

    async def result(self) -> AssistantOutcome:
        return await self._task

    def cancel(self) -> None:
        if not self._cancel_requested and not self._task.done():
            self._cancel_requested = True
            self._task.get_loop().call_soon(self._task.cancel)


class AgentRuntime:
    """Advance typed conversation state through model and Tool turns."""

    def __init__(
        self,
        adapter: ModelAdapter,
        model: ModelSpec,
        *,
        tools: Sequence[Tool] = (),
        tool_executor: ToolExecutor | None = None,
        tool_output_budget: ToolOutputBudget | None = None,
        retry_policy: RetryPolicy | None = None,
        sleeper: Sleeper = asyncio.sleep,
        run_guard: object | None = None,
    ) -> None:
        if not isinstance(model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if not callable(getattr(adapter, "stream", None)):
            raise TypeError("adapter must implement ModelAdapter.stream")
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy")
        if not callable(sleeper):
            raise TypeError("sleeper must be an async callable")
        if tool_executor is not None and not callable(
            getattr(tool_executor, "execute", None)
        ):
            raise TypeError("tool_executor must implement ToolExecutor.execute")
        if tool_output_budget is not None and not isinstance(
            tool_output_budget, ToolOutputBudget
        ):
            raise TypeError("tool_output_budget must be a ToolOutputBudget")
        registered: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("tools must contain Tool values")
            if tool.name in registered:
                raise ValueError(f"duplicate Tool name: {tool.name!r}")
            registered[tool.name] = tool
        if registered and not model.supports_tools:
            raise ValueError("configured ModelSpec does not support Tools")
        self._adapter = adapter
        self._model = model
        self._tools = registered
        self._tool_executor = tool_executor or LocalToolExecutor()
        self._tool_output_budget = tool_output_budget or ToolOutputBudget()
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper
        self._run_guard = run_guard
        self._history: list[ConversationMessage] = []

    @property
    def history(self) -> tuple[ConversationMessage, ...]:
        return tuple(self._history)

    @property
    def run_guard(self) -> object | None:
        return self._run_guard

    def start(self, messages: Sequence[AgentMessage]) -> AgentRunHandle:
        accepted = tuple(messages)
        to_model_messages((*self._history, *accepted))
        self._history.extend(accepted)
        events: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue()
        task = asyncio.get_running_loop().create_task(self._execute(events))
        return AgentRunHandle(task, events)

    async def run(self, messages: Sequence[AgentMessage]) -> AssistantOutcome:
        return await self.start(messages).result()

    async def _execute(
        self,
        events: asyncio.Queue[RuntimeEvent | object],
    ) -> AssistantOutcome:
        sequence = 0
        total_attempts = 0
        text_parts: list[str] = []
        current_usage: Usage | None = None
        run_usage: Usage | None = None
        run_tool_results: list[ToolResult] = []

        async def emit(
            type_: EventType,
            *,
            attempt: int | None = None,
            model_event: ModelEvent | None = None,
            error: ModelError | None = None,
            retry_delay_seconds: float | None = None,
            partial_text: str = "",
            partial_usage: Usage | None = None,
            tool_call_id: str | None = None,
            tool_name: str | None = None,
            tool_result: ToolResult | None = None,
        ) -> None:
            nonlocal sequence
            sequence += 1
            await events.put(
                RuntimeEvent(
                    sequence=sequence,
                    type=type_,
                    attempt=attempt,
                    model_event=model_event,
                    error=error,
                    retry_delay_seconds=retry_delay_seconds,
                    partial_text=partial_text,
                    partial_usage=partial_usage,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_result=tool_result,
                )
            )

        async def finish(outcome: AssistantOutcome) -> AssistantOutcome:
            self._history.append(outcome.message)
            await emit(EventType.MESSAGE_END, attempt=max(total_attempts, 1))
            await emit(EventType.AGENT_END, attempt=max(total_attempts, 1))
            return outcome

        try:
            await emit(EventType.AGENT_START)
            while True:
                turn_attempt = 0
                while True:
                    turn_attempt += 1
                    total_attempts += 1
                    attempt = total_attempts
                    request = ModelRequest(
                        to_model_messages(self._history),
                        self._model,
                        tuple(tool.definition() for tool in self._tools.values()),
                    )
                    await emit(EventType.MODEL_ATTEMPT_START, attempt=attempt)
                    text_parts = []
                    tool_drafts: dict[int, dict[str, str]] = {}
                    current_usage = None
                    end: ModelEnd | None = None
                    schema_error: ModelError | None = None
                    try:
                        async for event in self._adapter.stream(request):
                            await emit(
                                EventType.MODEL_EVENT,
                                attempt=attempt,
                                model_event=event,
                            )
                            if end is not None:
                                schema_error = ModelError(
                                    ModelErrorCode.SCHEMA,
                                    "model stream emitted data after ModelEnd",
                                    False,
                                )
                                break
                            if isinstance(event, TextDelta):
                                text_parts.append(event.text)
                            elif isinstance(event, ToolCallDelta):
                                if event.index < 0:
                                    schema_error = ModelError(
                                        ModelErrorCode.SCHEMA,
                                        "model stream emitted an invalid Tool Call index",
                                        False,
                                    )
                                    break
                                draft = tool_drafts.setdefault(
                                    event.index,
                                    {"id": "", "name": "", "arguments": ""},
                                )
                                draft["id"] += event.id
                                draft["name"] += event.name
                                draft["arguments"] += event.arguments_delta
                            elif isinstance(event, UsageUpdate):
                                current_usage = event.usage
                            elif isinstance(event, ModelEnd):
                                end = event
                            else:
                                schema_error = ModelError(
                                    ModelErrorCode.SCHEMA,
                                    "model stream emitted an unsupported event",
                                    False,
                                )
                                break
                    except ModelAdapterError as failure:
                        partial_text = "".join(text_parts)
                        await emit(
                            EventType.MODEL_ATTEMPT_FAILED,
                            attempt=attempt,
                            error=failure.error,
                            partial_text=partial_text,
                            partial_usage=current_usage,
                        )
                        delay = self._retry_policy.delay_for(
                            failure.error,
                            turn_attempt,
                            retry_after_seconds=failure.error.retry_after_seconds,
                        )
                        if delay is not None:
                            await emit(
                                EventType.RETRY_SCHEDULED,
                                attempt=attempt,
                                error=failure.error,
                                retry_delay_seconds=delay,
                                partial_text=partial_text,
                                partial_usage=current_usage,
                            )
                            text_parts = []
                            current_usage = None
                            await self._sleeper(delay)
                            continue
                        terminal_usage = _add_usage(run_usage, current_usage)
                        return await finish(
                            AssistantOutcome(
                                AgentMessage.text(Role.ASSISTANT, partial_text),
                                StopReason.ERROR,
                                terminal_usage,
                                failure.error,
                                total_attempts,
                                tuple(run_tool_results),
                            )
                        )

                    error = schema_error
                    if error is None and end is None:
                        error = ModelError(
                            ModelErrorCode.SCHEMA,
                            "model stream violated the provider-neutral event contract",
                            False,
                        )
                    blocks: list[ContentBlock] = []
                    if error is None:
                        try:
                            if text_parts:
                                blocks.append(TextContent("".join(text_parts)))
                            for index in sorted(tool_drafts):
                                blocks.append(ToolCallContent(**tool_drafts[index]))
                        except (TypeError, ValueError):
                            error = ModelError(
                                ModelErrorCode.SCHEMA,
                                "model stream emitted an incomplete Tool Call",
                                False,
                            )
                    if error is not None:
                        partial_text = "".join(text_parts)
                        await emit(
                            EventType.MODEL_ATTEMPT_FAILED,
                            attempt=attempt,
                            error=error,
                            partial_text=partial_text,
                            partial_usage=current_usage,
                        )
                        terminal_usage = _add_usage(run_usage, current_usage)
                        return await finish(
                            AssistantOutcome(
                                AgentMessage.text(Role.ASSISTANT, partial_text),
                                StopReason.ERROR,
                                terminal_usage,
                                error,
                                total_attempts,
                                tuple(run_tool_results),
                            )
                        )
                    assert end is not None
                    break

                run_usage = _add_usage(run_usage, current_usage)
                assistant = AgentMessage(Role.ASSISTANT, tuple(blocks))
                self._history.append(assistant)
                await emit(EventType.MESSAGE_END, attempt=total_attempts)
                calls = tuple(
                    block
                    for block in assistant.content
                    if isinstance(block, ToolCallContent)
                )
                if not calls:
                    await emit(EventType.AGENT_END, attempt=total_attempts)
                    return AssistantOutcome(
                        assistant,
                        end.stop_reason,
                        run_usage,
                        attempts=total_attempts,
                        tool_results=tuple(run_tool_results),
                    )

                await emit(EventType.TOOL_BATCH_START, attempt=total_attempts)
                prepared: dict[int, PreparedToolCall] = {}
                results: dict[int, ToolResult] = {}
                for index, call in enumerate(calls):
                    tool = self._tools.get(call.name)
                    if tool is None:
                        results[index] = ToolResult.error(
                            ToolErrorCode.UNKNOWN_TOOL, call.name
                        )
                        continue
                    try:
                        parsed = json.loads(call.arguments)
                    except (json.JSONDecodeError, TypeError):
                        results[index] = ToolResult.error(
                            ToolErrorCode.INVALID_JSON, call.name
                        )
                        continue
                    try:
                        Draft202012Validator(tool.input_schema).validate(parsed)
                    except ValidationError:
                        results[index] = ToolResult.error(
                            ToolErrorCode.INVALID_ARGUMENTS, call.name
                        )
                        continue
                    if not isinstance(parsed, dict):
                        results[index] = ToolResult.error(
                            ToolErrorCode.INVALID_ARGUMENTS, call.name
                        )
                        continue
                    prepared[index] = PreparedToolCall(call, tool, parsed)

                async def execute_one(index: int, call: PreparedToolCall) -> None:
                    await emit(
                        EventType.TOOL_CALL_START,
                        attempt=total_attempts,
                        tool_call_id=call.call.id,
                        tool_name=call.call.name,
                    )
                    try:
                        result = await self._tool_executor.execute(call)
                        if not isinstance(result, ToolResult):
                            raise TypeError("ToolExecutor returned an invalid result")
                    except Exception:
                        result = ToolResult.error(
                            ToolErrorCode.EXECUTION_FAILED, call.call.name
                        )
                    result = bound_tool_result(
                        result,
                        self._tool_output_budget,
                        call.tool.output_direction,
                    )
                    results[index] = result
                    await emit(
                        EventType.TOOL_CALL_END,
                        attempt=total_attempts,
                        tool_call_id=call.call.id,
                        tool_name=call.call.name,
                        tool_result=result,
                    )

                if any(call.tool.sequential for call in prepared.values()):
                    for index, prepared_call in prepared.items():
                        await execute_one(index, prepared_call)
                else:
                    await asyncio.gather(
                        *(
                            execute_one(index, prepared_call)
                            for index, prepared_call in prepared.items()
                        )
                    )

                batch_results: list[ToolResult] = []
                for index, call in enumerate(calls):
                    result = results[index]
                    batch_results.append(result)
                    run_tool_results.append(result)
                    self._history.append(
                        ToolResultMessage(call.id, call.name, result)
                    )
                await emit(EventType.TOOL_BATCH_END, attempt=total_attempts)
                if batch_results and all(result.terminate for result in batch_results):
                    await emit(EventType.AGENT_END, attempt=total_attempts)
                    return AssistantOutcome(
                        assistant,
                        end.stop_reason,
                        run_usage,
                        attempts=total_attempts,
                        tool_results=tuple(run_tool_results),
                    )
        except asyncio.CancelledError:
            message = AgentMessage.text(Role.ASSISTANT, "".join(text_parts))
            outcome = AssistantOutcome(
                message,
                StopReason.ABORTED,
                _add_usage(run_usage, current_usage),
                attempts=max(total_attempts, 1),
                tool_results=tuple(run_tool_results),
            )
            self._history.append(message)
            await emit(
                EventType.RUN_CANCELLED,
                attempt=max(total_attempts, 1),
                partial_text="".join(text_parts),
                partial_usage=current_usage,
            )
            await emit(EventType.MESSAGE_END, attempt=max(total_attempts, 1))
            await emit(EventType.AGENT_END, attempt=max(total_attempts, 1))
            return outcome
        finally:
            await events.put(_EVENTS_DONE)
