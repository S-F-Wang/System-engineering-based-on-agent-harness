"""Application-facing control for one in-memory agent conversation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from collections.abc import AsyncIterator, Sequence
from enum import Enum
from typing import cast

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
    ) -> None:
        if not isinstance(runtime, AgentRuntime):
            raise TypeError("runtime must be an AgentRuntime")
        if (store is None) != (session_id is None):
            raise ValueError("durable Sessions require both store and session_id")
        self._runtime = runtime
        self._store = store
        self._session_id = session_id
        self._parent_entry_id = parent_entry_id
        self._busy = False
        self._active: AgentRunHandle | None = None
        self._cancel_requested = False
        self._steering = _SteeringQueue()
        self._follow_ups: deque[PendingInput] = deque()

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def session_id(self) -> str | None:
        return self._session_id

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
) -> AgentSession:
    """Create a new durable Session, explicitly continue/fork one, or opt out."""

    if no_save:
        if store is not None or session_id is not None or fork_from is not None:
            raise ValueError("no_save cannot be combined with persistence or continuation")
        return AgentSession(runtime)
    if store is None:
        if session_id is not None or fork_from is not None:
            raise ValueError("continuation requires a SessionStore")
        return AgentSession(runtime)
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
        runtime.restore_history(state.history(leaf))
    return AgentSession(
        runtime,
        store=store,
        session_id=state.session_id,
        parent_entry_id=leaf,
    )
