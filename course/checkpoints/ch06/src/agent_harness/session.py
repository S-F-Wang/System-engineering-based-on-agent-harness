"""Application-facing control for one in-memory agent conversation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from collections.abc import AsyncIterator, Sequence
from enum import Enum
from typing import cast

from .compaction import (
    CharacterTokenEstimator,
    CompactionCheckpoint,
    CompactionPolicy,
    CompactionStrategy,
    CompactionTrigger,
    CompactionWarning,
    CompactionWarningCode,
    StructuredSummary,
    TokenEstimator,
)
from .model import AgentMessage, Role, StopReason
from .persistence import SessionBusyError, SessionStore, SessionWriter
from .runtime import AgentRunHandle, AgentRuntime, AssistantOutcome, RuntimeEvent


_SESSION_EVENTS_DONE = object()


class InputKind(str, Enum):
    STEERING = "steering"
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True, slots=True)
class PendingInput:
    kind: InputKind
    message: AgentMessage


@dataclass(frozen=True, slots=True)
class SessionRunResult:
    outcome: AssistantOutcome
    outcomes: tuple[AssistantOutcome, ...] = ()
    pending_inputs: tuple[PendingInput, ...] = ()


@dataclass(frozen=True, slots=True)
class CompactionResult:
    checkpoint: CompactionCheckpoint


class SessionRunHandle:
    def __init__(
        self,
        task: asyncio.Task[SessionRunResult],
        session: "AgentSession",
        events: asyncio.Queue[RuntimeEvent | object],
    ) -> None:
        self._task = task
        self._session = session
        self._events = events

    async def result(self) -> SessionRunResult:
        return await self._task

    def cancel(self) -> None:
        if not self._task.done():
            self._session.cancel()

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        while True:
            event = await self._events.get()
            if event is _SESSION_EVENTS_DONE:
                break
            yield cast(RuntimeEvent, event)


class _SteeringQueue:
    def __init__(self) -> None:
        self._messages: deque[PendingInput] = deque()

    def append(self, message: AgentMessage) -> None:
        self._messages.append(PendingInput(InputKind.STEERING, message))

    def pending(self) -> bool:
        return bool(self._messages)

    def take(self) -> Sequence[AgentMessage]:
        return (self._messages.popleft().message,)

    def drain(self) -> tuple[PendingInput, ...]:
        drained = tuple(self._messages)
        self._messages.clear()
        return drained


class AgentSession:
    """Coordinate one active Run with optional durable Session state."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        store: SessionStore | None = None,
        session_id: str | None = None,
        parent_entry_id: str | None = None,
        compaction_policy: CompactionPolicy | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        if not isinstance(runtime, AgentRuntime):
            raise TypeError("runtime must be an AgentRuntime")
        if (store is None) != (session_id is None):
            raise ValueError("durable Sessions require both store and session_id")
        if compaction_policy is not None and not isinstance(
            compaction_policy, CompactionPolicy
        ):
            raise TypeError("compaction_policy must be a CompactionPolicy")
        if compaction_strategy is not None and not callable(
            getattr(compaction_strategy, "plan", None)
        ):
            raise TypeError("compaction_strategy must provide plan()")
        if token_estimator is not None and not callable(
            getattr(token_estimator, "estimate", None)
        ):
            raise TypeError("token_estimator must provide estimate()")
        self._runtime = runtime
        self._store = store
        self._session_id = session_id
        self._parent_entry_id = parent_entry_id
        self._compaction_policy = compaction_policy or CompactionPolicy()
        self._compaction_strategy = compaction_strategy or CompactionStrategy()
        self._token_estimator = token_estimator or CharacterTokenEstimator()
        self._warnings: list[CompactionWarning] = []
        self._busy = False
        self._active: AgentRunHandle | None = None
        self._compaction_task: asyncio.Task[object] | None = None
        self._cancel_requested = False
        self._steering = _SteeringQueue()
        self._follow_ups: deque[PendingInput] = deque()

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def warnings(self) -> tuple[CompactionWarning, ...]:
        return tuple(self._warnings)

    def start(self, prompt: str | AgentMessage) -> SessionRunHandle:
        if self._busy:
            raise SessionBusyError(
                self._session_id or "ephemeral",
                "Session already has an active Run",
            )
        message = (
            AgentMessage.text(Role.USER, prompt) if isinstance(prompt, str) else prompt
        )
        if not isinstance(message, AgentMessage) or message.role is not Role.USER:
            raise TypeError("prompt must be text or a user AgentMessage")
        writer: SessionWriter | None = None
        if self._store is not None:
            assert self._session_id is not None
            writer = self._store.writer(self._session_id)
            writer = writer.__enter__()
        self._busy = True
        self._cancel_requested = False
        events: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue()
        try:
            task = asyncio.get_running_loop().create_task(
                self._drive(message, events, writer)
            )
        except BaseException:
            self._busy = False
            if writer is not None:
                writer.__exit__(None, None, None)
            raise
        return SessionRunHandle(task, self, events)

    async def run(self, prompt: str | AgentMessage) -> SessionRunResult:
        return await self.start(prompt).result()

    async def prompt(self, prompt: str | AgentMessage) -> SessionRunResult:
        return await self.run(prompt)

    async def compact(self, focus: str | None = None) -> CompactionResult:
        """Create and install a context checkpoint at a Settled Boundary."""

        if focus is not None and not focus.strip():
            raise ValueError("Compaction focus cannot be empty")
        if self._busy:
            raise SessionBusyError(
                self._session_id or "ephemeral",
                "Compaction requires an idle Session at a Settled Boundary",
            )
        writer: SessionWriter | None = None
        if self._store is not None:
            assert self._session_id is not None
            writer = self._store.writer(self._session_id).__enter__()
        self._busy = True
        try:
            return await self._compact(
                CompactionTrigger.MANUAL,
                focus=focus,
                writer=writer,
            )
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)
            self._busy = False

    async def _compact(
        self,
        trigger: CompactionTrigger,
        *,
        focus: str | None,
        writer: SessionWriter | None,
    ) -> CompactionResult:
        current = asyncio.current_task()
        previous = self._compaction_task
        self._compaction_task = cast(asyncio.Task[object] | None, current)
        try:
            policy = self._compaction_policy.resolve(self._runtime.model)
            plan = self._compaction_strategy.plan(
                self._runtime.effective_history,
                keep_recent_tokens=policy.keep_recent_tokens,
                estimator=self._token_estimator,
            )
            generated = await self._runtime.generate_summary(plan.source, focus=focus)
            checkpoint = CompactionCheckpoint(
                trigger=trigger,
                summary=StructuredSummary(generated.text, focus=focus),
                tokens_before=plan.tokens_before,
                summary_usage=generated.usage,
                retained_tail=plan.retained_tail,
            )
            if writer is not None:
                entry = writer.append_compaction(
                    checkpoint,
                    parent_id=self._parent_entry_id,
                )
                self._parent_entry_id = entry.entry_id
            self._runtime.install_compaction(checkpoint)
            return CompactionResult(checkpoint)
        finally:
            self._compaction_task = previous

    async def _maybe_compact_after_settlement(
        self,
        writer: SessionWriter | None,
    ) -> CompactionResult | None:
        resolved = self._compaction_policy.resolve(self._runtime.model)
        threshold = resolved.threshold_tokens
        if threshold is None:
            if (
                resolved.context_window is None
                and not any(
                    warning.code is CompactionWarningCode.CONTEXT_WINDOW_UNKNOWN
                    for warning in self._warnings
                )
            ):
                self._warnings.append(
                    CompactionWarning(
                        CompactionWarningCode.CONTEXT_WINDOW_UNKNOWN,
                        "threshold Compaction disabled because ModelSpec has no "
                        "context_window; normalized overflow recovery remains enabled",
                    )
                )
            return None
        current_tokens = self._token_estimator.estimate(
            self._runtime.effective_history
        )
        if current_tokens <= threshold:
            return None
        try:
            return await self._compact(
                CompactionTrigger.THRESHOLD,
                focus=None,
                writer=writer,
            )
        except Exception as error:
            self._warnings.append(
                CompactionWarning(
                    CompactionWarningCode.THRESHOLD_FAILED,
                    "threshold Compaction failed; Session history unchanged: "
                    f"{type(error).__name__}",
                )
            )
            return None

    def steer(self, message: str | AgentMessage) -> None:
        if not self._busy:
            raise RuntimeError("Steering requires an active Run")
        self._steering.append(self._user_message(message))

    def follow_up(self, message: str | AgentMessage) -> None:
        if not self._busy:
            raise RuntimeError("Follow-up requires an active Run")
        self._follow_ups.append(
            PendingInput(InputKind.FOLLOW_UP, self._user_message(message))
        )

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._compaction_task is not None and not self._compaction_task.done():
            self._compaction_task.cancel()
        if self._active is not None:
            self._active.cancel()

    async def _drive(
        self,
        message: AgentMessage,
        events: asyncio.Queue[RuntimeEvent | object],
        writer: SessionWriter | None,
    ) -> SessionRunResult:
        outcomes: list[AssistantOutcome] = []

        if writer is not None:
            def settle(messages) -> None:
                entry = writer.append(messages, parent_id=self._parent_entry_id)
                self._parent_entry_id = entry.entry_id

            self._runtime.set_settlement_sink(settle)

        async def recover_context_overflow() -> bool:
            await self._compact(
                CompactionTrigger.OVERFLOW,
                focus=None,
                writer=writer,
            )
            return True

        self._runtime.set_context_overflow_recovery(recover_context_overflow)
        self._runtime.set_settled_turn_handler(
            lambda: self._maybe_compact_after_settlement(writer)
        )

        async def forward(handle: AgentRunHandle) -> None:
            async for event in handle.events():
                await events.put(event)

        try:
            next_message = message
            while True:
                self._active = self._runtime.start(
                    [next_message], turn_input=self._steering
                )
                if self._cancel_requested:
                    self._active.cancel()
                forwarding = asyncio.create_task(forward(self._active))
                outcome = await self._active.result()
                await forwarding
                outcomes.append(outcome)
                if outcome.stop_reason in {StopReason.ABORTED, StopReason.ERROR}:
                    break
                if not self._follow_ups:
                    break
                next_message = self._follow_ups.popleft().message
            pending = self._steering.drain() + tuple(self._follow_ups)
            self._follow_ups.clear()
            return SessionRunResult(outcome, tuple(outcomes), pending)
        finally:
            self._runtime.set_settlement_sink(None)
            self._runtime.set_context_overflow_recovery(None)
            self._runtime.set_settled_turn_handler(None)
            if writer is not None:
                writer.__exit__(None, None, None)
            self._active = None
            self._cancel_requested = False
            self._busy = False
            await events.put(_SESSION_EVENTS_DONE)

    @staticmethod
    def _user_message(message: str | AgentMessage) -> AgentMessage:
        accepted = (
            AgentMessage.text(Role.USER, message)
            if isinstance(message, str)
            else message
        )
        if not isinstance(accepted, AgentMessage) or accepted.role is not Role.USER:
            raise TypeError("Session input must be text or a user AgentMessage")
        return accepted


def create_agent_session(
    runtime: AgentRuntime,
    *,
    store: SessionStore | None = None,
    session_id: str | None = None,
    fork_from: str | None = None,
    no_save: bool = False,
    compaction_policy: CompactionPolicy | None = None,
    compaction_strategy: CompactionStrategy | None = None,
    token_estimator: TokenEstimator | None = None,
) -> AgentSession:
    """Create a new durable Session, explicitly continue/fork one, or opt out."""

    if no_save:
        if store is not None or session_id is not None or fork_from is not None:
            raise ValueError("no_save cannot be combined with persistence or continuation")
        return AgentSession(
            runtime,
            compaction_policy=compaction_policy,
            compaction_strategy=compaction_strategy,
            token_estimator=token_estimator,
        )
    if store is None:
        if session_id is not None or fork_from is not None:
            raise ValueError("continuation requires a SessionStore")
        return AgentSession(
            runtime,
            compaction_policy=compaction_policy,
            compaction_strategy=compaction_strategy,
            token_estimator=token_estimator,
        )
    if runtime.history:
        raise ValueError("a durable AgentSession requires a fresh AgentRuntime")
    if session_id is None:
        if fork_from is not None:
            raise ValueError("fork_from requires an existing session_id")
        state = store.create()
        leaf = None
    else:
        state = store.read(session_id)
        leaf = fork_from if fork_from is not None else state.active_leaf_id
        runtime.restore_history(
            state.history(leaf),
            effective_history=state.effective_history(leaf),
        )
    return AgentSession(
        runtime,
        store=store,
        session_id=state.session_id,
        parent_entry_id=leaf,
        compaction_policy=compaction_policy,
        compaction_strategy=compaction_strategy,
        token_estimator=token_estimator,
    )
