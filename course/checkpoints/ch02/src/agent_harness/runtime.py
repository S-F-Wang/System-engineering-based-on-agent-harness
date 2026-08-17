"""Async-first agent execution with ordered passive observations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, cast

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


class EventType(str, Enum):
    AGENT_START = "agent_start"
    MODEL_ATTEMPT_START = "model_attempt_start"
    MODEL_EVENT = "model_event"
    MODEL_ATTEMPT_FAILED = "model_attempt_failed"
    RETRY_SCHEDULED = "retry_scheduled"
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


@dataclass(frozen=True, slots=True)
class AssistantOutcome:
    message: AgentMessage
    stop_reason: StopReason
    usage: Usage | None = None
    error: ModelError | None = None
    attempts: int = 1


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
            error.code not in retryable_codes
            and not retryable_status
        ) or failed_attempt > len(self.delays):
            return None
        if retry_after_seconds is not None:
            if not 0 <= retry_after_seconds <= self.max_retry_after_seconds:
                return None
            return retry_after_seconds
        return self.delays[failed_attempt - 1]


Sleeper: TypeAlias = Callable[[float], Awaitable[None]]
_EVENTS_DONE = object()


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
    """Advance typed conversation state through one async model runtime."""

    def __init__(
        self,
        adapter: ModelAdapter,
        model: ModelSpec,
        *,
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
        self._adapter = adapter
        self._model = model
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper
        self._run_guard = run_guard
        self._history: list[AgentMessage] = []

    @property
    def history(self) -> tuple[AgentMessage, ...]:
        return tuple(self._history)

    @property
    def run_guard(self) -> object | None:
        return self._run_guard

    def start(self, messages: Sequence[AgentMessage]) -> AgentRunHandle:
        loop = asyncio.get_running_loop()
        accepted = tuple(messages)
        request = ModelRequest(
            to_model_messages((*self._history, *accepted)), self._model
        )
        self._history.extend(accepted)
        events: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue()
        task = loop.create_task(self._execute(request, events))
        return AgentRunHandle(task, events)

    async def run(self, messages: Sequence[AgentMessage]) -> AssistantOutcome:
        return await self.start(messages).result()

    async def _execute(
        self,
        request: ModelRequest,
        events: asyncio.Queue[RuntimeEvent | object],
    ) -> AssistantOutcome:
        sequence = 0
        attempt = 0
        text_parts: list[str] = []
        usage: Usage | None = None

        async def emit(
            type_: EventType,
            *,
            attempt: int | None = None,
            model_event: ModelEvent | None = None,
            error: ModelError | None = None,
            retry_delay_seconds: float | None = None,
            partial_text: str = "",
            partial_usage: Usage | None = None,
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
                )
            )

        try:
            await emit(EventType.AGENT_START)
            while True:
                attempt += 1
                await emit(EventType.MODEL_ATTEMPT_START, attempt=attempt)
                text_parts = []
                tool_drafts: dict[int, dict[str, str]] = {}
                usage = None
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
                            usage = event.usage
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
                        partial_usage=usage,
                    )
                    delay = self._retry_policy.delay_for(
                        failure.error,
                        attempt,
                        retry_after_seconds=failure.error.retry_after_seconds,
                    )
                    if delay is not None:
                        await emit(
                            EventType.RETRY_SCHEDULED,
                            attempt=attempt,
                            error=failure.error,
                            retry_delay_seconds=delay,
                            partial_text=partial_text,
                            partial_usage=usage,
                        )
                        text_parts = []
                        usage = None
                        await self._sleeper(delay)
                        continue
                    message = AgentMessage.text(Role.ASSISTANT, partial_text)
                    outcome = AssistantOutcome(
                        message=message,
                        stop_reason=StopReason.ERROR,
                        usage=usage,
                        error=failure.error,
                        attempts=attempt,
                    )
                else:
                    error = schema_error
                    if error is None and end is None:
                        error = ModelError(
                            ModelErrorCode.SCHEMA,
                            "model stream violated the provider-neutral event contract",
                            False,
                        )
                    if error is None:
                        try:
                            blocks: list[ContentBlock] = []
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
                            partial_usage=usage,
                        )
                        outcome = AssistantOutcome(
                            message=AgentMessage.text(Role.ASSISTANT, partial_text),
                            stop_reason=StopReason.ERROR,
                            usage=usage,
                            error=error,
                            attempts=attempt,
                        )
                    else:
                        assert end is not None
                        message = AgentMessage(Role.ASSISTANT, tuple(blocks))
                        outcome = AssistantOutcome(
                            message, end.stop_reason, usage, attempts=attempt
                        )
                self._history.append(outcome.message)
                await emit(EventType.MESSAGE_END, attempt=attempt)
                await emit(EventType.AGENT_END, attempt=attempt)
                return outcome
        except asyncio.CancelledError:
            message = AgentMessage.text(Role.ASSISTANT, "".join(text_parts))
            outcome = AssistantOutcome(
                message=message,
                stop_reason=StopReason.ABORTED,
                usage=usage,
                attempts=max(attempt, 1),
            )
            self._history.append(message)
            await emit(
                EventType.RUN_CANCELLED,
                attempt=max(attempt, 1),
                partial_text="".join(text_parts),
                partial_usage=usage,
            )
            await emit(EventType.MESSAGE_END, attempt=max(attempt, 1))
            await emit(EventType.AGENT_END, attempt=max(attempt, 1))
            return outcome
        finally:
            await events.put(_EVENTS_DONE)
