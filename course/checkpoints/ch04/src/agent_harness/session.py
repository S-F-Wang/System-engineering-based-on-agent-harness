"""Application-facing control for one in-memory agent conversation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from collections.abc import AsyncIterator, Sequence
from enum import Enum
from typing import cast

from .model import AgentMessage, Role, StopReason
from .runtime import AgentRunHandle, AgentRuntime, AssistantOutcome, RuntimeEvent


_SESSION_EVENTS_DONE = object()


class SessionBusyError(RuntimeError):
    """Raised when a Session already owns an active Agent Run."""


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
    """Keep one active Run and its lifecycle behind a small interface."""

    def __init__(self, runtime: AgentRuntime) -> None:
        if not isinstance(runtime, AgentRuntime):
            raise TypeError("runtime must be an AgentRuntime")
        self._runtime = runtime
        self._busy = False
        self._active: AgentRunHandle | None = None
        self._cancel_requested = False
        self._steering = _SteeringQueue()
        self._follow_ups: deque[PendingInput] = deque()

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, prompt: str | AgentMessage) -> SessionRunHandle:
        if self._busy:
            raise SessionBusyError("Session already has an active Run")
        message = (
            AgentMessage.text(Role.USER, prompt) if isinstance(prompt, str) else prompt
        )
        if not isinstance(message, AgentMessage) or message.role is not Role.USER:
            raise TypeError("prompt must be text or a user AgentMessage")
        self._busy = True
        self._cancel_requested = False
        events: asyncio.Queue[RuntimeEvent | object] = asyncio.Queue()
        task = asyncio.get_running_loop().create_task(self._drive(message, events))
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
    ) -> SessionRunResult:
        outcomes: list[AssistantOutcome] = []

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
