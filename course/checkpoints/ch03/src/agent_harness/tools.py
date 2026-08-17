"""Explicit Tools and their replaceable execution seam."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from .model import ModelToolResultMessage, ToolCallContent, ToolDefinition


ToolHandler = Callable[[dict[str, object]], Awaitable["ToolResult"]]


class ToolErrorCode(str, Enum):
    INVALID_JSON = "invalid_json"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    EXECUTION_FAILED = "execution_failed"


class TruncationDirection(str, Enum):
    HEAD = "head"
    TAIL = "tail"


class CompleteOutputKind(str, Enum):
    ARTIFACT = "artifact"
    EXTERNAL = "external"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CompleteOutputReference:
    kind: CompleteOutputKind
    reference: str | None = None
    reason: str | None = None

    @classmethod
    def artifact(cls, reference: str) -> "CompleteOutputReference":
        return cls(CompleteOutputKind.ARTIFACT, reference=reference)

    @classmethod
    def unavailable(cls) -> "CompleteOutputReference":
        return cls(
            CompleteOutputKind.UNAVAILABLE,
            reason="complete output was not retained",
        )

    def render(self) -> str:
        if self.reference is not None:
            return f"{self.kind.value} {self.reference}"
        return f"{self.kind.value} ({self.reason})"


@dataclass(frozen=True, slots=True)
class TruncationNotice:
    original_bytes: int
    original_lines: int
    retained_start_byte: int
    retained_end_byte: int
    retained_start_line: int
    retained_end_line: int
    direction: TruncationDirection


@dataclass(frozen=True, slots=True)
class ToolOutputBudget:
    max_bytes: int = 50 * 1024
    max_lines: int = 2000

    def __post_init__(self) -> None:
        if self.max_bytes <= 0 or self.max_lines <= 0:
            raise ValueError("Tool output limits must be positive")


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    terminate: bool = False
    is_error: bool = False
    error_code: ToolErrorCode | None = None
    truncation: TruncationNotice | None = None
    complete_output: CompleteOutputReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.is_error != (self.error_code is not None):
            raise ValueError("error ToolResult values require exactly one error_code")
        if self.is_error and self.terminate:
            raise ValueError("error ToolResult values cannot terminate a Tool Batch")

    @classmethod
    def error(cls, code: ToolErrorCode, tool_name: str) -> "ToolResult":
        guidance = {
            ToolErrorCode.INVALID_JSON: "provide one valid JSON object",
            ToolErrorCode.INVALID_ARGUMENTS: "match the Tool's declared input schema",
            ToolErrorCode.UNKNOWN_TOOL: "choose one of the advertised Tools",
            ToolErrorCode.EXECUTION_FAILED: "revise the call or choose another Tool",
        }[code]
        return cls(
            content=f"Tool error [{code.value}] for '{tool_name}': {guidance}.",
            is_error=True,
            error_code=code,
        )


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: Mapping[str, object]
    execute: ToolHandler
    sequential: bool = False
    output_direction: TruncationDirection = TruncationDirection.HEAD

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name cannot be empty")
        if not self.description.strip():
            raise ValueError("Tool description cannot be empty")
        if not callable(self.execute):
            raise TypeError("Tool execute must be an async callable")
        schema = dict(self.input_schema)
        Draft202012Validator.check_schema(schema)
        if schema.get("type") != "object":
            raise ValueError("Tool input_schema must describe a JSON object")
        object.__setattr__(self, "input_schema", MappingProxyType(schema))

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, self.input_schema)


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    call: ToolCallContent
    tool: Tool
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    result: ToolResult

    def to_model(self) -> ModelToolResultMessage:
        return ModelToolResultMessage(
            self.tool_call_id,
            self.tool_name,
            self.result.content,
            self.result.is_error,
        )


class ToolExecutor(Protocol):
    async def execute(self, call: PreparedToolCall) -> ToolResult: ...


class LocalToolExecutor:
    async def execute(self, call: PreparedToolCall) -> ToolResult:
        return await call.tool.execute(call.arguments)


def bound_tool_result(
    result: ToolResult,
    budget: ToolOutputBudget,
    direction: TruncationDirection,
) -> ToolResult:
    """Bound model-facing text and attach explicit recovery provenance."""

    content = result.content
    encoded = content.encode("utf-8")
    lines = content.splitlines(keepends=True)
    original_lines = len(content.splitlines())
    if len(encoded) <= budget.max_bytes and original_lines <= budget.max_lines:
        return result

    if direction is TruncationDirection.HEAD:
        line_limited = "".join(lines[: budget.max_lines])
        retained_bytes = line_limited.encode("utf-8")[: budget.max_bytes]
        retained = retained_bytes.decode("utf-8", errors="ignore")
        retained_start_byte = 0
        retained_end_byte = len(retained.encode("utf-8"))
        retained_start_line = 1 if retained else 0
        retained_end_line = len(retained.splitlines())
    else:
        line_limited = "".join(lines[-budget.max_lines :])
        retained_bytes = line_limited.encode("utf-8")[-budget.max_bytes :]
        retained = retained_bytes.decode("utf-8", errors="ignore")
        retained_end_byte = len(encoded)
        retained_start_byte = retained_end_byte - len(retained.encode("utf-8"))
        retained_end_line = original_lines
        retained_line_count = len(retained.splitlines())
        retained_start_line = max(1, original_lines - retained_line_count + 1)

    reference = result.complete_output or CompleteOutputReference.unavailable()
    notice = TruncationNotice(
        original_bytes=len(encoded),
        original_lines=original_lines,
        retained_start_byte=retained_start_byte,
        retained_end_byte=retained_end_byte,
        retained_start_line=retained_start_line,
        retained_end_line=retained_end_line,
        direction=direction,
    )
    model_notice = (
        "[tool output truncated: "
        f"retained {direction.value} bytes {retained_start_byte}-{retained_end_byte} "
        f"of {len(encoded)}, lines {retained_start_line}-{retained_end_line} "
        f"of {original_lines}; complete output: {reference.render()}]"
    )
    return ToolResult(
        content=f"{retained}\n\n{model_notice}",
        metadata=result.metadata,
        terminate=result.terminate,
        is_error=result.is_error,
        error_code=result.error_code,
        truncation=notice,
        complete_output=reference,
    )
