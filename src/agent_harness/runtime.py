"""Async Agent Runtime with structured Tool batches."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast
from uuid import uuid4

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
    ModelOperation,
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
from .compaction import CompactionCheckpoint
from .extensions import (
    HookContext,
    HookExecutionError,
    HookPoint,
    HookRegistration,
    SubscriberRegistration,
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
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"
    COMPACTION_FAILED = "compaction_failed"
    TOOL_BATCH_START = "tool_batch_start"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_BATCH_END = "tool_batch_end"
    RUN_CANCELLED = "run_cancelled"
    MESSAGE_END = "message_end"
    AGENT_END = "agent_end"
    SUBSCRIBER_FAILED = "subscriber_failed"


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    MODEL_ERROR = "model_error"
    CANCELLED = "cancelled"
    MAX_TURNS = "max_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    TIMEOUT = "timeout"
    MAX_TOTAL_TOKENS = "max_total_tokens"


@dataclass(frozen=True, slots=True)
class RunGuard:
    max_turns: int | None = None
    max_tool_calls: int | None = None
    timeout_seconds: float | None = None
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        integer_limits = {
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_total_tokens": self.max_total_tokens,
        }
        for name, value in integer_limits.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when supplied")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when supplied")

    def reached(
        self,
        *,
        turns: int,
        tool_calls: int,
        usage: Usage | None,
        elapsed_seconds: float,
    ) -> TerminalStatus | None:
        if (
            self.timeout_seconds is not None
            and elapsed_seconds >= self.timeout_seconds
        ):
            return TerminalStatus.TIMEOUT
        if self.max_turns is not None and turns >= self.max_turns:
            return TerminalStatus.MAX_TURNS
        if (
            self.max_tool_calls is not None
            and tool_calls >= self.max_tool_calls
        ):
            return TerminalStatus.MAX_TOOL_CALLS
        if (
            self.max_total_tokens is not None
            and usage is not None
            and usage.total_tokens >= self.max_total_tokens
        ):
            return TerminalStatus.MAX_TOTAL_TOKENS
        return None


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
    tool_arguments: Mapping[str, object] | None = None
    tool_result: ToolResult | None = None
    operation: ModelOperation = ModelOperation.RUN
    extension_name: str | None = None
    diagnostic: str | None = None
    run_id: str | None = None
    snapshot_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class AssistantOutcome:
    message: AgentMessage
    stop_reason: StopReason
    usage: Usage | None = None
    error: ModelError | None = None
    attempts: int = 1
    tool_results: tuple[ToolResult, ...] = ()
    status: TerminalStatus = TerminalStatus.COMPLETED
    snapshot_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class SummaryGeneration:
    text: str
    usage: Usage | None
    attempts: int


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


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Immutable effective behavior captured when a Run is accepted."""

    run_id: str
    adapter: ModelAdapter
    model: ModelSpec
    tools: tuple[Tool, ...]
    tool_executor: ToolExecutor
    tool_output_budget: ToolOutputBudget
    retry_policy: RetryPolicy
    sleeper: Sleeper
    run_guard: object | None
    extension_identities: tuple[str, ...]
    subscribers: tuple[SubscriberRegistration, ...]
    hooks: tuple[HookRegistration, ...]
    generation_settings: Mapping[str, object]
    compaction_policy: object | None
    prompt_hashes: tuple[tuple[str, str], ...]
    resource_hashes: tuple[tuple[str, str], ...]
    fingerprint: str

    @classmethod
    def capture(
        cls,
        *,
        adapter: ModelAdapter,
        model: ModelSpec,
        tools: Sequence[Tool],
        tool_executor: ToolExecutor,
        tool_output_budget: ToolOutputBudget,
        retry_policy: RetryPolicy,
        sleeper: Sleeper,
        run_guard: object | None,
        extension_identities: Sequence[str] = (),
        subscribers: Sequence[SubscriberRegistration] = (),
        hooks: Sequence[HookRegistration] = (),
        generation_settings: Mapping[str, object] | None = None,
        compaction_policy: object | None = None,
        prompt_hashes: Mapping[str, str] | None = None,
        resource_hashes: Mapping[str, str] | None = None,
    ) -> "RunSnapshot":
        accepted_tools = tuple(tools)
        accepted_extensions = tuple(extension_identities)
        accepted_subscribers = tuple(subscribers)
        accepted_hooks = tuple(hooks)
        accepted_generation = dict(generation_settings or {})
        accepted_prompt_hashes = tuple(sorted((prompt_hashes or {}).items()))
        accepted_resource_hashes = tuple(sorted((resource_hashes or {}).items()))
        payload = {
            "model": {
                "id": model.model_id,
                "context_window": model.context_window,
                "max_output_tokens": model.max_output_tokens,
                "supports_tools": model.supports_tools,
            },
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": dict(tool.input_schema),
                    "sequential": tool.sequential,
                    "output_direction": tool.output_direction.value,
                }
                for tool in accepted_tools
            ],
            "extensions": list(accepted_extensions),
            "hooks": [
                [registration.extension_name, registration.point.value]
                for registration in accepted_hooks
            ],
            "generation": accepted_generation,
            "retry": {
                "delays": retry_policy.delays,
                "max_retry_after_seconds": retry_policy.max_retry_after_seconds,
            },
            "compaction": repr(compaction_policy),
            "guard": repr(run_guard),
            "prompt_hashes": accepted_prompt_hashes,
            "resource_hashes": accepted_resource_hashes,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            uuid4().hex,
            adapter,
            model,
            accepted_tools,
            tool_executor,
            tool_output_budget,
            retry_policy,
            sleeper,
            run_guard,
            accepted_extensions,
            accepted_subscribers,
            accepted_hooks,
            MappingProxyType(accepted_generation),
            compaction_policy,
            accepted_prompt_hashes,
            accepted_resource_hashes,
            hashlib.sha256(canonical).hexdigest(),
        )


Sleeper: TypeAlias = Callable[[float], Awaitable[None]]
ConversationMessage: TypeAlias = AgentMessage | ToolResultMessage
SettlementSink: TypeAlias = Callable[[Sequence[ConversationMessage]], None]
ContextOverflowRecovery: TypeAlias = Callable[[], Awaitable[bool]]
SettledTurnHandler: TypeAlias = Callable[[], Awaitable[object]]
_EVENTS_DONE = object()


class TurnInput(Protocol):
    """Runtime-facing view of Steering Messages waiting at a turn boundary."""

    def pending(self) -> bool: ...

    def take(self) -> Sequence[AgentMessage]: ...


@dataclass(slots=True)
class _CancellationState:
    status: TerminalStatus = TerminalStatus.CANCELLED
    requested: bool = False
    started: bool = False

    def request(self, status: TerminalStatus) -> bool:
        if self.requested:
            return False
        self.status = status
        self.requested = True
        return True


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
        cancellation: _CancellationState,
        snapshot: RunSnapshot,
    ) -> None:
        self._task = task
        self._events = events
        self._cancellation = cancellation
        self._snapshot = snapshot

    @property
    def run_id(self) -> str:
        return self._snapshot.run_id

    @property
    def snapshot(self) -> RunSnapshot:
        return self._snapshot

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        while True:
            event = await self._events.get()
            if event is _EVENTS_DONE:
                break
            yield cast(RuntimeEvent, event)

    async def result(self) -> AssistantOutcome:
        outcome = await self._task
        if outcome.snapshot_fingerprint == self._snapshot.fingerprint:
            return outcome
        return replace(
            outcome,
            snapshot_fingerprint=self._snapshot.fingerprint,
        )

    def cancel(self) -> None:
        self._cancel_with(TerminalStatus.CANCELLED)

    def _cancel_with(self, status: TerminalStatus) -> None:
        if not self._task.done() and self._cancellation.request(status):
            if self._cancellation.started:
                self._task.cancel()


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
        generation_settings: Mapping[str, object] | None = None,
        history: Sequence[ConversationMessage] = (),
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
        self._generation_settings = MappingProxyType(dict(generation_settings or {}))
        accepted_history = tuple(history)
        to_model_messages(accepted_history)
        self._history: list[ConversationMessage] = list(accepted_history)
        self._effective_history: list[ConversationMessage] | None = None
        self._settlement_sink: SettlementSink | None = None
        self._context_overflow_recovery: ContextOverflowRecovery | None = None
        self._settled_turn_handler: SettledTurnHandler | None = None
        self._subscribers: tuple[SubscriberRegistration, ...] = ()
        self._hooks: tuple[HookRegistration, ...] = ()

    @property
    def history(self) -> tuple[ConversationMessage, ...]:
        return tuple(self._history)

    @property
    def effective_history(self) -> tuple[ConversationMessage, ...]:
        return tuple(
            self._history
            if self._effective_history is None
            else self._effective_history
        )

    @property
    def model(self) -> ModelSpec:
        return self._model

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def configure_tools(self, tools: Sequence[Tool]) -> None:
        """Replace the effective Tool set before a Session accepts work."""

        registered: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("tools must contain Tool values")
            if tool.name in registered:
                raise ValueError(f"duplicate Tool name: {tool.name!r}")
            registered[tool.name] = tool
        if registered and not self._model.supports_tools:
            raise ValueError("configured ModelSpec does not support Tools")
        self._tools = registered

    def configure_subscribers(
        self,
        subscribers: Sequence[SubscriberRegistration],
    ) -> None:
        """Install high-level passive observers before accepting a Run."""

        accepted = tuple(subscribers)
        if any(not isinstance(item, SubscriberRegistration) for item in accepted):
            raise TypeError("subscribers must contain SubscriberRegistration values")
        self._subscribers = accepted

    def configure_hooks(self, hooks: Sequence[HookRegistration]) -> None:
        """Install the bounded ordered Hook set before accepting a Run."""

        accepted = tuple(hooks)
        if any(not isinstance(item, HookRegistration) for item in accepted):
            raise TypeError("hooks must contain HookRegistration values")
        self._hooks = accepted

    async def _apply_hooks(
        self,
        point: HookPoint,
        value: object,
        snapshot: RunSnapshot | None = None,
    ) -> object:
        current = value
        registrations = self._hooks if snapshot is None else snapshot.hooks
        for registration in registrations:
            if registration.point is not point:
                continue
            try:
                replacement = await registration.callback(
                    HookContext(point, current, snapshot)
                )
            except Exception as error:
                raise HookExecutionError(
                    registration.extension_name,
                    point,
                    type(error).__name__,
                ) from error
            if replacement is not None:
                current = replacement
        return current

    async def apply_hooks(
        self,
        point: HookPoint,
        value: object,
        *,
        snapshot: RunSnapshot | None = None,
    ) -> object:
        """Invoke one frozen-order Hook point for an AgentSession operation."""

        return await self._apply_hooks(point, value, snapshot)

    def capture_snapshot(
        self,
        *,
        extension_identities: Sequence[str] = (),
        compaction_policy: object | None = None,
        prompt_hashes: Mapping[str, str] | None = None,
        resource_hashes: Mapping[str, str] | None = None,
    ) -> RunSnapshot:
        return RunSnapshot.capture(
            adapter=self._adapter,
            model=self._model,
            tools=self.tools,
            tool_executor=self._tool_executor,
            tool_output_budget=self._tool_output_budget,
            retry_policy=self._retry_policy,
            sleeper=self._sleeper,
            run_guard=self._run_guard,
            extension_identities=extension_identities,
            subscribers=self._subscribers,
            hooks=self._hooks,
            generation_settings=self._generation_settings,
            compaction_policy=compaction_policy,
            prompt_hashes=prompt_hashes,
            resource_hashes=resource_hashes,
        )

    @property
    def run_guard(self) -> object | None:
        return self._run_guard

    def restore_history(
        self,
        history: Sequence[ConversationMessage],
        *,
        effective_history: Sequence[ConversationMessage] | None = None,
    ) -> None:
        """Seed a newly constructed Runtime from one settled Session branch."""

        if self._history:
            raise RuntimeError("Runtime history must be empty before restoration")
        accepted = tuple(history)
        to_model_messages(accepted)
        self._history.extend(accepted)
        if effective_history is not None:
            effective = tuple(effective_history)
            to_model_messages(effective)
            self._effective_history = list(effective)

    def install_compaction(self, checkpoint: CompactionCheckpoint) -> None:
        """Replace only the model-facing prefix at a Settled Boundary."""

        self._effective_history = [
            checkpoint.summary.as_message(),
            *checkpoint.retained_tail,
        ]

    def set_settlement_sink(self, sink: SettlementSink | None) -> None:
        """Install the AgentSession-owned persistence barrier for the next Run."""

        if sink is not None and not callable(sink):
            raise TypeError("settlement sink must be callable")
        self._settlement_sink = sink

    def set_context_overflow_recovery(
        self,
        recovery: ContextOverflowRecovery | None,
    ) -> None:
        if recovery is not None and not callable(recovery):
            raise TypeError("context overflow recovery must be callable")
        self._context_overflow_recovery = recovery

    def set_settled_turn_handler(
        self,
        handler: SettledTurnHandler | None,
    ) -> None:
        if handler is not None and not callable(handler):
            raise TypeError("settled turn handler must be callable")
        self._settled_turn_handler = handler

    def _append_settled(self, messages: Sequence[ConversationMessage]) -> None:
        accepted = tuple(messages)
        self._history.extend(accepted)
        if self._effective_history is not None:
            self._effective_history.extend(accepted)
        if accepted and self._settlement_sink is not None:
            self._settlement_sink(accepted)

    def _append_unsettled(self, message: ConversationMessage) -> None:
        self._history.append(message)
        if self._effective_history is not None:
            self._effective_history.append(message)

    def _mark_settled(self, messages: Sequence[ConversationMessage]) -> None:
        accepted = tuple(messages)
        if accepted and self._settlement_sink is not None:
            self._settlement_sink(accepted)

    def start(
        self,
        messages: Sequence[AgentMessage],
        *,
        turn_input: TurnInput | None = None,
        snapshot: RunSnapshot | None = None,
    ) -> AgentRunHandle:
        accepted = tuple(messages)
        to_model_messages((*self.effective_history, *accepted))
        self._append_settled(accepted)
        events: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue()
        cancellation = _CancellationState()
        accepted_snapshot = snapshot or self.capture_snapshot()
        if not isinstance(accepted_snapshot, RunSnapshot):
            raise TypeError("snapshot must be a RunSnapshot")
        task = asyncio.get_running_loop().create_task(
            self._execute(events, cancellation, turn_input, accepted_snapshot)
        )
        handle = AgentRunHandle(task, events, cancellation, accepted_snapshot)
        if (
            isinstance(accepted_snapshot.run_guard, RunGuard)
            and accepted_snapshot.run_guard.timeout_seconds is not None
        ):
            timer = asyncio.get_running_loop().call_later(
                accepted_snapshot.run_guard.timeout_seconds,
                handle._cancel_with,
                TerminalStatus.TIMEOUT,
            )
            task.add_done_callback(lambda completed: timer.cancel())
        return handle

    async def run(self, messages: Sequence[AgentMessage]) -> AssistantOutcome:
        return await self.start(messages).result()

    async def generate_summary(
        self,
        messages: Sequence[ConversationMessage],
        *,
        focus: str | None = None,
        snapshot: RunSnapshot | None = None,
    ) -> SummaryGeneration:
        """Run a retryable model operation without mutating conversation history."""

        accepted = tuple(messages)
        if not accepted:
            raise ValueError("summary generation requires source history")
        effective = snapshot or self.capture_snapshot()
        instruction = (
            "Summarize the following settled conversation as a durable context "
            "checkpoint. Preserve decisions, constraints, unresolved work, and "
            "facts needed to continue. Return summary text only."
        )
        if focus is not None:
            instruction += f"\nFocus requested by the caller: {focus}"
        summary_prompt = AgentMessage.text(Role.SYSTEM, instruction)
        attempt = 0
        while True:
            attempt += 1
            request = ModelRequest(
                messages=to_model_messages((summary_prompt, *accepted)),
                model=effective.model,
                operation=ModelOperation.COMPACTION,
            )
            text_parts: list[str] = []
            usage: Usage | None = None
            end: ModelEnd | None = None
            schema_error: ModelError | None = None
            try:
                async for event in effective.adapter.stream(request):
                    if end is not None:
                        schema_error = ModelError(
                            ModelErrorCode.SCHEMA,
                            "Compaction stream emitted data after ModelEnd",
                            False,
                        )
                        break
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                    elif isinstance(event, UsageUpdate):
                        usage = event.usage
                    elif isinstance(event, ModelEnd):
                        end = event
                    else:
                        schema_error = ModelError(
                            ModelErrorCode.SCHEMA,
                            "Compaction summary must contain text only",
                            False,
                        )
                        break
            except ModelAdapterError as failure:
                delay = effective.retry_policy.delay_for(
                    failure.error,
                    attempt,
                    retry_after_seconds=failure.error.retry_after_seconds,
                )
                if delay is None:
                    raise
                await effective.sleeper(delay)
                continue
            if schema_error is None and end is None:
                schema_error = ModelError(
                    ModelErrorCode.SCHEMA,
                    "Compaction stream did not end with ModelEnd",
                    False,
                )
            if (
                schema_error is None
                and end is not None
                and end.stop_reason is not StopReason.COMPLETE
            ):
                schema_error = ModelError(
                    ModelErrorCode.SCHEMA,
                    "Compaction summary did not complete successfully",
                    False,
                )
            text = "".join(text_parts)
            if schema_error is None and not text.strip():
                schema_error = ModelError(
                    ModelErrorCode.SCHEMA,
                    "Compaction summary cannot be empty",
                    False,
                )
            if schema_error is not None:
                raise ModelAdapterError(schema_error)
            return SummaryGeneration(text, usage, attempt)

    async def _execute(
        self,
        events: asyncio.Queue[RuntimeEvent | object],
        cancellation: _CancellationState,
        turn_input: TurnInput | None = None,
        snapshot: RunSnapshot | None = None,
    ) -> AssistantOutcome:
        cancellation.started = True
        if snapshot is None:
            snapshot = self.capture_snapshot()
        tools = {tool.name: tool for tool in snapshot.tools}
        request_history: list[ConversationMessage] = list(self.effective_history)
        sequence = 0
        total_attempts = 0
        text_parts: list[str] = []
        current_usage: Usage | None = None
        run_usage: Usage | None = None
        run_tool_results: list[ToolResult] = []
        turns = 0
        tool_calls = 0
        started_at = asyncio.get_running_loop().time()
        overflow_recovery_attempted = False

        def append_settled(messages: Sequence[ConversationMessage]) -> None:
            accepted = tuple(messages)
            self._append_settled(accepted)
            request_history.extend(accepted)

        def append_unsettled(message: ConversationMessage) -> None:
            self._append_unsettled(message)
            request_history.append(message)

        def guard_status() -> TerminalStatus | None:
            if not isinstance(snapshot.run_guard, RunGuard):
                return None
            return snapshot.run_guard.reached(
                turns=turns,
                tool_calls=tool_calls,
                usage=run_usage,
                elapsed_seconds=asyncio.get_running_loop().time() - started_at,
            )

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
            tool_arguments: Mapping[str, object] | None = None,
            tool_result: ToolResult | None = None,
            operation: ModelOperation = ModelOperation.RUN,
        ) -> None:
            nonlocal sequence
            sequence += 1
            event = RuntimeEvent(
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
                tool_arguments=tool_arguments,
                tool_result=tool_result,
                operation=operation,
                run_id=snapshot.run_id,
                snapshot_fingerprint=snapshot.fingerprint,
            )
            await events.put(event)
            for registration in snapshot.subscribers:
                try:
                    await registration.callback(event)
                except Exception as subscriber_error:
                    sequence += 1
                    await events.put(
                        RuntimeEvent(
                            sequence=sequence,
                            type=EventType.SUBSCRIBER_FAILED,
                            attempt=attempt,
                            operation=operation,
                            extension_name=registration.extension_name,
                            diagnostic=type(subscriber_error).__name__,
                            run_id=snapshot.run_id,
                            snapshot_fingerprint=snapshot.fingerprint,
                        )
                    )

        async def finish(outcome: AssistantOutcome) -> AssistantOutcome:
            append_settled((outcome.message,))
            await emit(EventType.MESSAGE_END, attempt=max(total_attempts, 1))
            await emit(EventType.AGENT_END, attempt=max(total_attempts, 1))
            return outcome

        try:
            if cancellation.requested:
                raise asyncio.CancelledError
            await emit(EventType.AGENT_START)
            try:
                hooked_history = await self._apply_hooks(
                    HookPoint.BEFORE_RUN,
                    tuple(request_history),
                    snapshot,
                )
                if isinstance(hooked_history, (str, bytes)) or not isinstance(
                    hooked_history, Sequence
                ):
                    raise TypeError(
                        "before_run Hook must return a message sequence or None"
                    )
                accepted_hooked_history = tuple(hooked_history)
                to_model_messages(accepted_hooked_history)
                request_history[:] = accepted_hooked_history
            except (HookExecutionError, TypeError, ValueError) as hook_error:
                error = ModelError(
                    ModelErrorCode.HOOK_FAILED,
                    str(hook_error),
                    False,
                )
                return await finish(
                    AssistantOutcome(
                        AgentMessage.text(Role.ASSISTANT, ""),
                        StopReason.ERROR,
                        run_usage,
                        error,
                        1,
                        tuple(run_tool_results),
                        TerminalStatus.MODEL_ERROR,
                    )
                )
            while True:
                turn_attempt = 0
                while True:
                    turn_attempt += 1
                    total_attempts += 1
                    attempt = total_attempts
                    request = ModelRequest(
                        to_model_messages(request_history),
                        snapshot.model,
                        tuple(tool.definition() for tool in tools.values()),
                    )
                    try:
                        hooked_request = await self._apply_hooks(
                            HookPoint.BEFORE_MODEL_REQUEST, request, snapshot
                        )
                        if not isinstance(hooked_request, ModelRequest):
                            raise TypeError(
                                "before_model_request Hook must return ModelRequest or None"
                            )
                        request = hooked_request
                    except (HookExecutionError, TypeError) as hook_error:
                        error = ModelError(
                            ModelErrorCode.HOOK_FAILED,
                            str(hook_error),
                            False,
                        )
                        return await finish(
                            AssistantOutcome(
                                AgentMessage.text(Role.ASSISTANT, ""),
                                StopReason.ERROR,
                                run_usage,
                                error,
                                max(total_attempts, 1),
                                tuple(run_tool_results),
                                TerminalStatus.MODEL_ERROR,
                            )
                        )
                    await emit(EventType.MODEL_ATTEMPT_START, attempt=attempt)
                    text_parts = []
                    tool_drafts: dict[int, dict[str, str]] = {}
                    current_usage = None
                    end: ModelEnd | None = None
                    schema_error: ModelError | None = None
                    try:
                        async for event in snapshot.adapter.stream(request):
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
                        if (
                            failure.error.code is ModelErrorCode.CONTEXT_OVERFLOW
                            and not overflow_recovery_attempted
                            and self._context_overflow_recovery is not None
                        ):
                            overflow_recovery_attempted = True
                            await emit(
                                EventType.COMPACTION_START,
                                attempt=attempt,
                                error=failure.error,
                                operation=ModelOperation.COMPACTION,
                            )
                            try:
                                recovered = await self._context_overflow_recovery()
                            except Exception as recovery_error:
                                recovered = False
                                failure = ModelAdapterError(
                                    ModelError(
                                        ModelErrorCode.COMPACTION_FAILED,
                                        "context overflow recovery Compaction failed: "
                                        f"{type(recovery_error).__name__}",
                                        False,
                                    )
                                )
                            await emit(
                                (
                                    EventType.COMPACTION_END
                                    if recovered
                                    else EventType.COMPACTION_FAILED
                                ),
                                attempt=attempt,
                                error=None if recovered else failure.error,
                                operation=ModelOperation.COMPACTION,
                            )
                            if recovered:
                                request_history[:] = self.effective_history
                                text_parts = []
                                current_usage = None
                                continue
                        delay = snapshot.retry_policy.delay_for(
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
                            await snapshot.sleeper(delay)
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
                                TerminalStatus.MODEL_ERROR,
                            )
                        )

                    stream_error = schema_error
                    if stream_error is None and end is None:
                        stream_error = ModelError(
                            ModelErrorCode.SCHEMA,
                            "model stream violated the provider-neutral event contract",
                            False,
                        )
                    blocks: list[ContentBlock] = []
                    if stream_error is None:
                        try:
                            if text_parts:
                                blocks.append(TextContent("".join(text_parts)))
                            for index in sorted(tool_drafts):
                                blocks.append(ToolCallContent(**tool_drafts[index]))
                        except (TypeError, ValueError):
                            stream_error = ModelError(
                                ModelErrorCode.SCHEMA,
                                "model stream emitted an incomplete Tool Call",
                                False,
                            )
                    if stream_error is not None:
                        partial_text = "".join(text_parts)
                        await emit(
                            EventType.MODEL_ATTEMPT_FAILED,
                            attempt=attempt,
                            error=stream_error,
                            partial_text=partial_text,
                            partial_usage=current_usage,
                        )
                        terminal_usage = _add_usage(run_usage, current_usage)
                        return await finish(
                            AssistantOutcome(
                                AgentMessage.text(Role.ASSISTANT, partial_text),
                                StopReason.ERROR,
                                terminal_usage,
                                stream_error,
                                total_attempts,
                                tuple(run_tool_results),
                                TerminalStatus.MODEL_ERROR,
                            )
                        )
                    assert end is not None
                    break

                run_usage = _add_usage(run_usage, current_usage)
                turns += 1
                assistant = AgentMessage(Role.ASSISTANT, tuple(blocks))
                calls = tuple(
                    block
                    for block in assistant.content
                    if isinstance(block, ToolCallContent)
                )
                if calls:
                    append_unsettled(assistant)
                else:
                    append_settled((assistant,))
                await emit(EventType.MESSAGE_END, attempt=total_attempts)
                if not calls:
                    if self._settled_turn_handler is not None:
                        previous_effective = self.effective_history
                        await self._settled_turn_handler()
                        if self.effective_history != previous_effective:
                            request_history[:] = self.effective_history
                    if turn_input is not None and turn_input.pending():
                        reached = guard_status()
                        if reached is not None:
                            await emit(EventType.AGENT_END, attempt=total_attempts)
                            return AssistantOutcome(
                                assistant,
                                StopReason.ABORTED,
                                run_usage,
                                attempts=total_attempts,
                                tool_results=tuple(run_tool_results),
                                status=reached,
                            )
                        steering = tuple(turn_input.take())
                        to_model_messages(steering)
                        append_settled(steering)
                        continue
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
                    tool = tools.get(call.name)
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
                    candidate = PreparedToolCall(call, tool, parsed)
                    try:
                        hooked_call = await self._apply_hooks(
                            HookPoint.BEFORE_TOOL_CALL,
                            candidate,
                            snapshot,
                        )
                        if not isinstance(hooked_call, PreparedToolCall):
                            raise TypeError(
                                "before_tool_call Hook must return PreparedToolCall or None"
                            )
                    except (HookExecutionError, TypeError):
                        results[index] = ToolResult.error(
                            ToolErrorCode.HOOK_FAILED,
                            call.name,
                        )
                        continue
                    prepared[index] = hooked_call

                async def execute_one(index: int, call: PreparedToolCall) -> None:
                    await emit(
                        EventType.TOOL_CALL_START,
                        attempt=total_attempts,
                        tool_call_id=call.call.id,
                        tool_name=call.call.name,
                        tool_arguments=call.arguments,
                    )
                    try:
                        result = await snapshot.tool_executor.execute(call)
                        if not isinstance(result, ToolResult):
                            raise TypeError("ToolExecutor returned an invalid result")
                    except Exception:
                        result = ToolResult.error(
                            ToolErrorCode.EXECUTION_FAILED, call.call.name
                        )
                    try:
                        hooked_result = await self._apply_hooks(
                            HookPoint.AFTER_TOOL_CALL,
                            result,
                            snapshot,
                        )
                        if not isinstance(hooked_result, ToolResult):
                            raise TypeError(
                                "after_tool_call Hook must return ToolResult or None"
                            )
                        result = hooked_result
                    except (HookExecutionError, TypeError):
                        result = ToolResult.error(
                            ToolErrorCode.HOOK_FAILED,
                            call.call.name,
                        )
                    result = bound_tool_result(
                        result,
                        snapshot.tool_output_budget,
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

                try:
                    if any(call.tool.sequential for call in prepared.values()):
                        for index, prepared_call in prepared.items():
                            await execute_one(index, prepared_call)
                    else:
                        tasks = {
                            index: asyncio.create_task(
                                execute_one(index, prepared_call)
                            )
                            for index, prepared_call in prepared.items()
                        }
                        try:
                            await asyncio.gather(*tasks.values())
                        except asyncio.CancelledError:
                            for task in tasks.values():
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(
                                *tasks.values(), return_exceptions=True
                            )
                            raise
                except asyncio.CancelledError:
                    for index, prepared_call in prepared.items():
                        if index not in results:
                            cancelled_result = ToolResult.error(
                                ToolErrorCode.CANCELLED,
                                prepared_call.call.name,
                            )
                            results[index] = cancelled_result
                            await emit(
                                EventType.TOOL_CALL_END,
                                attempt=total_attempts,
                                tool_call_id=prepared_call.call.id,
                                tool_name=prepared_call.call.name,
                                tool_result=cancelled_result,
                            )
                    for index, call in enumerate(calls):
                        result = results[index]
                        run_tool_results.append(result)
                        append_unsettled(
                            ToolResultMessage(call.id, call.name, result)
                        )
                    self._mark_settled(self._history[-(len(calls) + 1) :])
                    await emit(EventType.TOOL_BATCH_END, attempt=total_attempts)
                    raise

                batch_results: list[ToolResult] = []
                for index, call in enumerate(calls):
                    result = results[index]
                    batch_results.append(result)
                    run_tool_results.append(result)
                    append_unsettled(
                        ToolResultMessage(call.id, call.name, result)
                    )
                self._mark_settled(self._history[-(len(calls) + 1) :])
                tool_calls += len(calls)
                await emit(EventType.TOOL_BATCH_END, attempt=total_attempts)
                if self._settled_turn_handler is not None:
                    previous_effective = self.effective_history
                    await self._settled_turn_handler()
                    if self.effective_history != previous_effective:
                        request_history[:] = self.effective_history
                if batch_results and all(result.terminate for result in batch_results):
                    await emit(EventType.AGENT_END, attempt=total_attempts)
                    return AssistantOutcome(
                        assistant,
                        end.stop_reason,
                        run_usage,
                        attempts=total_attempts,
                        tool_results=tuple(run_tool_results),
                    )
                reached = guard_status()
                if reached is not None:
                    await emit(EventType.AGENT_END, attempt=total_attempts)
                    return AssistantOutcome(
                        assistant,
                        StopReason.ABORTED,
                        run_usage,
                        attempts=total_attempts,
                        tool_results=tuple(run_tool_results),
                        status=reached,
                    )
                if turn_input is not None and turn_input.pending():
                    steering = tuple(turn_input.take())
                    to_model_messages(steering)
                    append_settled(steering)
        except asyncio.CancelledError:
            message = AgentMessage.text(Role.ASSISTANT, "".join(text_parts))
            outcome = AssistantOutcome(
                message,
                StopReason.ABORTED,
                _add_usage(run_usage, current_usage),
                attempts=max(total_attempts, 1),
                tool_results=tuple(run_tool_results),
                status=cancellation.status,
            )
            append_settled((message,))
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
