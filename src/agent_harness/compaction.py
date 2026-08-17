"""Session Compaction policy, planning, and durable checkpoint contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from .model import (
    AgentMessage,
    ModelContextMessage,
    ModelSpec,
    Role,
    TextContent,
    ToolCallContent,
    Usage,
    to_model_messages,
)
from .tools import ToolResultMessage


ConversationMessage: TypeAlias = AgentMessage | ToolResultMessage


class CompactionTrigger(str, Enum):
    MANUAL = "manual"
    THRESHOLD = "threshold"
    OVERFLOW = "overflow"


class CompactionWarningCode(str, Enum):
    THRESHOLD_FAILED = "threshold_failed"
    CONTEXT_WINDOW_UNKNOWN = "context_window_unknown"


@dataclass(frozen=True, slots=True)
class CompactionWarning:
    code: CompactionWarningCode
    message: str


@dataclass(frozen=True, slots=True)
class StructuredSummary:
    text: str
    focus: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a Compaction summary cannot be empty")
        if self.focus is not None and not self.focus.strip():
            raise ValueError("Compaction focus cannot be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported Compaction summary schema version")

    def as_message(self) -> AgentMessage:
        focus = f"Focus: {self.focus}\n" if self.focus is not None else ""
        return AgentMessage.text(
            Role.SYSTEM,
            "Compaction checkpoint (schema 1)\n"
            f"{focus}Summary:\n{self.text}",
        )


@dataclass(frozen=True, slots=True)
class CompactionCheckpoint:
    trigger: CompactionTrigger
    summary: StructuredSummary
    tokens_before: int
    summary_usage: Usage | None
    retained_tail: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        if self.tokens_before < 0:
            raise ValueError("tokens_before cannot be negative")
        if not self.retained_tail:
            raise ValueError("a Compaction checkpoint requires a Retained Tail")
        to_model_messages(self.retained_tail)

    def model_context(self) -> tuple[ModelContextMessage, ...]:
        return to_model_messages((self.summary.as_message(), *self.retained_tail))


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    source: tuple[ConversationMessage, ...]
    retained_tail: tuple[ConversationMessage, ...]
    tokens_before: int


class TokenEstimator(Protocol):
    def estimate(self, messages: Sequence[ConversationMessage]) -> int: ...


class CharacterTokenEstimator:
    """Deterministic offline estimate with explicit estimated provenance."""

    def estimate(self, messages: Sequence[ConversationMessage]) -> int:
        characters = 0
        for message in messages:
            if isinstance(message, ToolResultMessage):
                characters += len(message.result.content) + len(message.tool_name)
                continue
            for block in message.content:
                if isinstance(block, TextContent):
                    characters += len(block.text)
                elif isinstance(block, ToolCallContent):
                    characters += len(block.name) + len(block.arguments)
            characters += 8
        return 0 if not messages else max(1, (characters + 3) // 4)


@dataclass(frozen=True, slots=True)
class ResolvedCompactionPolicy:
    context_window: int | None
    reserve_tokens: int | None
    keep_recent_tokens: int

    @property
    def threshold_tokens(self) -> int | None:
        if self.context_window is None or self.reserve_tokens is None:
            return None
        return max(0, self.context_window - self.reserve_tokens)


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    reserve_tokens: int | None = None
    keep_recent_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("reserve_tokens", self.reserve_tokens),
            ("keep_recent_tokens", self.keep_recent_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when supplied")

    def resolve(self, model: ModelSpec) -> ResolvedCompactionPolicy:
        window = model.context_window
        reserve = self.reserve_tokens
        keep = self.keep_recent_tokens
        if window is not None:
            if reserve is None:
                reserve = min(
                    16_384,
                    max(model.max_output_tokens, window // 8),
                )
            if keep is None:
                keep = min(20_000, window // 4)
        return ResolvedCompactionPolicy(
            window,
            reserve,
            keep if keep is not None else 20_000,
        )


class CompactionStrategy:
    """Choose a Retained Tail without splitting a Tool Call batch."""

    def plan(
        self,
        messages: Sequence[ConversationMessage],
        *,
        keep_recent_tokens: int,
        estimator: TokenEstimator,
    ) -> CompactionPlan:
        if (
            isinstance(keep_recent_tokens, bool)
            or not isinstance(keep_recent_tokens, int)
            or keep_recent_tokens < 0
        ):
            raise ValueError("keep_recent_tokens must be a non-negative integer")
        if not callable(getattr(estimator, "estimate", None)):
            raise TypeError("estimator must implement TokenEstimator.estimate")
        accepted = tuple(messages)
        if len(accepted) < 2:
            raise ValueError("Compaction requires history before the Retained Tail")
        groups = self._structural_groups(accepted)
        if len(groups) < 2:
            raise ValueError("Compaction requires history before the Retained Tail")
        retained_groups: list[tuple[ConversationMessage, ...]] = []
        retained_tokens = 0
        for group in reversed(groups):
            group_tokens = estimator.estimate(group)
            if group_tokens < 0:
                raise ValueError("TokenEstimator cannot return negative values")
            if retained_groups and retained_tokens + group_tokens > keep_recent_tokens:
                break
            retained_groups.append(group)
            retained_tokens += group_tokens
            if retained_tokens >= keep_recent_tokens:
                break
        tail = tuple(
            message
            for group in reversed(retained_groups)
            for message in group
        )
        if len(tail) >= len(accepted):
            first = groups[0]
            tail = accepted[len(first) :]
        if not tail:
            tail = groups[-1]
        tokens_before = estimator.estimate(accepted)
        if tokens_before < 0:
            raise ValueError("TokenEstimator cannot return negative values")
        return CompactionPlan(
            source=accepted[: -len(tail)],
            retained_tail=tail,
            tokens_before=tokens_before,
        )

    @staticmethod
    def _structural_groups(
        messages: tuple[ConversationMessage, ...],
    ) -> tuple[tuple[ConversationMessage, ...], ...]:
        groups: list[tuple[ConversationMessage, ...]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if isinstance(message, ToolResultMessage):
                raise ValueError("a settled history cannot contain an orphan ToolResult")
            calls = (
                tuple(
                    block
                    for block in message.content
                    if isinstance(block, ToolCallContent)
                )
                if isinstance(message, AgentMessage)
                else ()
            )
            if not calls:
                groups.append((message,))
                index += 1
                continue
            expected = {call.id for call in calls}
            end = index + 1
            while end < len(messages):
                result = messages[end]
                if not isinstance(result, ToolResultMessage):
                    break
                if result.tool_call_id not in expected:
                    raise ValueError(
                        "a settled history cannot contain an orphan ToolResult"
                    )
                expected.remove(result.tool_call_id)
                end += 1
            if expected:
                raise ValueError("a settled history cannot contain an orphan Tool Call")
            groups.append(messages[index:end])
            index = end
        return tuple(groups)
