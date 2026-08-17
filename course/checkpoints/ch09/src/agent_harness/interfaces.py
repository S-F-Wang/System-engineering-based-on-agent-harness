"""Terminal-neutral result, event, and live Session adapter contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Literal

from .model import TextContent
from .runtime import RuntimeEvent, TerminalStatus
from .session import (
    AgentSession,
    InputKind,
    PendingInput,
    SessionRunHandle,
    SessionRunResult,
)


RUN_RESULT_SCHEMA_VERSION = {"major": 1, "minor": 0}
EVENT_ENVELOPE_SCHEMA_VERSION = {"major": 1, "minor": 0}


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    session_id: str | None
    status: str
    stop_reason: str
    final_text: str
    usage: object | None
    attempts: int
    trace_complete: bool
    trace_error: str | None = None
    schema: Literal["agent_harness.run_result"] = "agent_harness.run_result"

    @classmethod
    def from_session(
        cls,
        handle: SessionRunHandle,
        session_id: str | None,
        result: SessionRunResult,
    ) -> "RunResult":
        outcome = result.outcome
        final_text = "".join(
            block.text for block in outcome.message.content if isinstance(block, TextContent)
        )
        return cls(
            handle.run_id,
            session_id,
            outcome.status.value,
            outcome.stop_reason.value,
            final_text,
            _json_value(outcome.usage),
            outcome.attempts,
            result.trace_complete,
            result.trace_error,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": dict(RUN_RESULT_SCHEMA_VERSION),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "final_text": self.final_text,
            "usage": self.usage,
            "attempts": self.attempts,
            "trace_complete": self.trace_complete,
            "trace_error": self.trace_error,
        }


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    sequence: int
    type: str
    session_id: str | None
    run_id: str
    payload: Mapping[str, object]
    schema: Literal["agent_harness.event"] = "agent_harness.event"

    @classmethod
    def from_runtime(
        cls, event: RuntimeEvent, *, session_id: str | None
    ) -> "EventEnvelope":
        if event.run_id is None:
            raise ValueError("Runtime Event is missing its Run id")
        return cls(
            event.sequence,
            event.type.value,
            session_id,
            event.run_id,
            {
                "attempt": event.attempt,
                "operation": event.operation.value,
                "model_event": _json_value(event.model_event),
                "error": _json_value(event.error),
                "retry_delay_seconds": event.retry_delay_seconds,
                "partial_text": event.partial_text,
                "partial_usage": _json_value(event.partial_usage),
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "tool_arguments": _json_value(event.tool_arguments),
                "tool_result": _json_value(event.tool_result),
                "extension_name": event.extension_name,
                "diagnostic": event.diagnostic,
                "snapshot_fingerprint": event.snapshot_fingerprint,
            },
        )

    @classmethod
    def run_end(cls, result: RunResult, sequence: int) -> "EventEnvelope":
        return cls(
            sequence,
            "run_end",
            result.session_id,
            result.run_id,
            result.to_dict(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": dict(EVENT_ENVELOPE_SCHEMA_VERSION),
            "sequence": self.sequence,
            "type": self.type,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "payload": dict(self.payload),
        }


class NonTerminalAdapter:
    """A UI-neutral controller proving the live Session seam is reusable."""

    def __init__(self, session: AgentSession) -> None:
        if not isinstance(session, AgentSession):
            raise TypeError("session must be an AgentSession")
        self.session = session
        self._handle: SessionRunHandle | None = None
        self._restored_inputs: tuple[PendingInput, ...] = ()

    @property
    def snapshot(self):
        if self._handle is None:
            raise RuntimeError("no Run has started")
        return self._handle.snapshot

    @property
    def restored_inputs(self) -> tuple[PendingInput, ...]:
        return self._restored_inputs

    def start(self, prompt: str) -> None:
        if self._handle is not None:
            raise RuntimeError("the adapter already controls a Run")
        self._handle = self.session.start(prompt)
        restored = self._restored_inputs
        self._restored_inputs = ()
        for pending in restored:
            text = "".join(
                block.text
                for block in pending.message.content
                if isinstance(block, TextContent)
            )
            if pending.kind is InputKind.STEERING:
                self.session.steer(text)
            else:
                self.session.follow_up(text)

    async def events(self) -> AsyncIterator[EventEnvelope]:
        if self._handle is None:
            raise RuntimeError("no Run has started")
        async for event in self._handle.events():
            yield EventEnvelope.from_runtime(event, session_id=self.session.session_id)

    def steer(self, message: str) -> None:
        self.session.steer(message)

    def follow_up(self, message: str) -> None:
        self.session.follow_up(message)

    def cancel(self) -> None:
        if self._handle is None:
            raise RuntimeError("no Run has started")
        self._handle.cancel()

    async def result(self) -> RunResult:
        if self._handle is None:
            raise RuntimeError("no Run has started")
        result = await self._handle.result()
        converted = RunResult.from_session(
            self._handle, self.session.session_id, result
        )
        self._restored_inputs = result.pending_inputs
        self._handle = None
        return converted

    async def command(self, name: str, arguments: str = "") -> object:
        return await self.session.execute_command(name, arguments)


def exit_code(status: str | TerminalStatus) -> int:
    accepted = status.value if isinstance(status, TerminalStatus) else status
    if accepted == TerminalStatus.COMPLETED.value:
        return 0
    if accepted == TerminalStatus.MODEL_ERROR.value:
        return 3
    if accepted == TerminalStatus.CANCELLED.value:
        return 130
    if accepted in {
        TerminalStatus.MAX_TURNS.value,
        TerminalStatus.MAX_TOOL_CALLS.value,
        TerminalStatus.TIMEOUT.value,
        TerminalStatus.MAX_TOTAL_TOKENS.value,
    }:
        return 5
    return 4
