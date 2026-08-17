"""Tree-structured Session persistence at settled boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO, Protocol, TypeAlias
from uuid import uuid4

from .compaction import (
    CompactionCheckpoint,
    CompactionTrigger,
    StructuredSummary,
)
from .extensions import CustomEntry
from .model import AgentMessage, Role, TextContent, ToolCallContent, Usage
from .tools import (
    CompleteOutputKind,
    CompleteOutputReference,
    ToolErrorCode,
    ToolResult,
    ToolResultMessage,
    TruncationDirection,
    TruncationNotice,
)


ConversationMessage: TypeAlias = AgentMessage | ToolResultMessage
IdFactory: TypeAlias = Callable[[], str]


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    major: int
    minor: int = 0


SESSION_SCHEMA_VERSION = SchemaVersion(1, 2)
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RecoveryCode(str, Enum):
    INCOMPLETE_FINAL_RECORD = "incomplete_final_record"


@dataclass(frozen=True, slots=True)
class RecoveryWarning:
    code: RecoveryCode
    message: str
    line_number: int


class SessionBusyError(RuntimeError):
    """Structured failure raised when a Session already owns its writer lease."""

    code = "session_busy"

    def __init__(self, session_id: str, message: str | None = None) -> None:
        self.session_id = session_id
        super().__init__(
            message or f"Session {session_id!r} already has an active writer"
        )


class UnsupportedSchemaVersionError(ValueError):
    def __init__(self, found_major: int) -> None:
        self.artifact = "Session"
        self.found_major = found_major
        self.supported_major = SESSION_SCHEMA_VERSION.major
        super().__init__(
            f"Session schema major {found_major} is unsupported; "
            f"this reader supports major {self.supported_major}. "
            "Preserve the original and migrate it with migrate_session_file()."
        )


def _validate_record_version(record: object, line_number: int) -> Mapping[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"Session record at line {line_number} must be an object")
    if record.get("schema") != "agent_harness.session":
        raise ValueError(f"invalid Session schema at line {line_number}")
    raw_version = record.get("schema_version")
    if not isinstance(raw_version, dict):
        raise ValueError(f"invalid Session schema version at line {line_number}")
    major = raw_version.get("major")
    minor = raw_version.get("minor")
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or major < 1
        or minor < 0
    ):
        raise ValueError(f"invalid Session schema version at line {line_number}")
    if major != SESSION_SCHEMA_VERSION.major:
        raise UnsupportedSchemaVersionError(major)
    return record


def _session_id(value: str) -> str:
    if not _SESSION_ID.fullmatch(value):
        raise ValueError("Session id must be a safe local identifier")
    return value


def _version_record() -> dict[str, int]:
    return {
        "major": SESSION_SCHEMA_VERSION.major,
        "minor": SESSION_SCHEMA_VERSION.minor,
    }


def _encode_message(message: ConversationMessage) -> dict[str, object]:
    if isinstance(message, AgentMessage):
        content: list[dict[str, object]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCallContent):
                content.append(
                    {
                        "type": "tool_call",
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.arguments,
                    }
                )
            else:
                raise TypeError(f"unsupported Content Block: {type(block).__name__}")
        return {
            "kind": "agent_message",
            "role": message.role.value,
            "content": content,
        }
    if isinstance(message, ToolResultMessage):
        result = message.result
        truncation = (
            None
            if result.truncation is None
            else {
                "original_bytes": result.truncation.original_bytes,
                "original_lines": result.truncation.original_lines,
                "retained_start_byte": result.truncation.retained_start_byte,
                "retained_end_byte": result.truncation.retained_end_byte,
                "retained_start_line": result.truncation.retained_start_line,
                "retained_end_line": result.truncation.retained_end_line,
                "direction": result.truncation.direction.value,
            }
        )
        complete_output = (
            None
            if result.complete_output is None
            else {
                "kind": result.complete_output.kind.value,
                "reference": result.complete_output.reference,
                "reason": result.complete_output.reason,
            }
        )
        return {
            "kind": "tool_result",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "result": {
                "content": result.content,
                "metadata": dict(result.metadata),
                "terminate": result.terminate,
                "is_error": result.is_error,
                "error_code": (
                    None if result.error_code is None else result.error_code.value
                ),
                "truncation": truncation,
                "complete_output": complete_output,
            },
        }
    raise TypeError(f"unsupported Session message: {type(message).__name__}")


def _decode_message(record: Mapping[str, object]) -> ConversationMessage:
    if record.get("kind") == "agent_message":
        role = Role(str(record["role"]))
        raw_content = record.get("content")
        if not isinstance(raw_content, list):
            raise ValueError("Session AgentMessage content must be a list")
        blocks: list[TextContent | ToolCallContent] = []
        for raw_block in raw_content:
            if not isinstance(raw_block, dict):
                raise ValueError("Session Content Block must be an object")
            if raw_block.get("type") == "text":
                blocks.append(TextContent(str(raw_block["text"])))
            elif raw_block.get("type") == "tool_call":
                blocks.append(
                    ToolCallContent(
                        str(raw_block["id"]),
                        str(raw_block["name"]),
                        str(raw_block["arguments"]),
                    )
                )
            else:
                raise ValueError("unsupported Session Content Block type")
        return AgentMessage(role, tuple(blocks))
    if record.get("kind") == "tool_result":
        raw_result = record.get("result")
        if not isinstance(raw_result, dict):
            raise ValueError("Session ToolResult must be an object")
        raw_truncation = raw_result.get("truncation")
        truncation = None
        if isinstance(raw_truncation, dict):
            truncation = TruncationNotice(
                int(raw_truncation["original_bytes"]),
                int(raw_truncation["original_lines"]),
                int(raw_truncation["retained_start_byte"]),
                int(raw_truncation["retained_end_byte"]),
                int(raw_truncation["retained_start_line"]),
                int(raw_truncation["retained_end_line"]),
                TruncationDirection(str(raw_truncation["direction"])),
            )
        raw_complete = raw_result.get("complete_output")
        complete_output = None
        if isinstance(raw_complete, dict):
            complete_output = CompleteOutputReference(
                CompleteOutputKind(str(raw_complete["kind"])),
                None
                if raw_complete.get("reference") is None
                else str(raw_complete["reference"]),
                None
                if raw_complete.get("reason") is None
                else str(raw_complete["reason"]),
            )
        raw_metadata = raw_result.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("Session ToolResult metadata must be an object")
        raw_error_code = raw_result.get("error_code")
        result = ToolResult(
            str(raw_result["content"]),
            metadata=raw_metadata,
            terminate=bool(raw_result.get("terminate", False)),
            is_error=bool(raw_result.get("is_error", False)),
            error_code=(
                None
                if raw_error_code is None
                else ToolErrorCode(str(raw_error_code))
            ),
            truncation=truncation,
            complete_output=complete_output,
        )
        return ToolResultMessage(
            str(record["tool_call_id"]), str(record["tool_name"]), result
        )
    raise ValueError("unsupported Session message kind")


def _encode_compaction(checkpoint: CompactionCheckpoint) -> dict[str, object]:
    usage = checkpoint.summary_usage
    return {
        "trigger": checkpoint.trigger.value,
        "summary": {
            "schema_version": checkpoint.summary.schema_version,
            "text": checkpoint.summary.text,
            "focus": checkpoint.summary.focus,
        },
        "tokens_before": checkpoint.tokens_before,
        "summary_usage": (
            None
            if usage is None
            else {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "estimated": usage.estimated,
            }
        ),
        "retained_tail": [
            _encode_message(message) for message in checkpoint.retained_tail
        ],
    }


def _decode_compaction(record: Mapping[str, object]) -> CompactionCheckpoint:
    raw_summary = record.get("summary")
    raw_tail = record.get("retained_tail")
    if not isinstance(raw_summary, dict) or not isinstance(raw_tail, list):
        raise ValueError("invalid Compaction checkpoint")
    raw_usage = record.get("summary_usage")
    usage = None
    if isinstance(raw_usage, dict):
        usage = Usage(
            int(raw_usage["input_tokens"]),
            int(raw_usage["output_tokens"]),
            int(raw_usage["total_tokens"]),
            bool(raw_usage.get("estimated", False)),
        )
    return CompactionCheckpoint(
        trigger=CompactionTrigger(str(record["trigger"])),
        summary=StructuredSummary(
            text=str(raw_summary["text"]),
            focus=(
                None
                if raw_summary.get("focus") is None
                else str(raw_summary["focus"])
            ),
            schema_version=int(raw_summary.get("schema_version", 1)),
        ),
        tokens_before=int(str(record["tokens_before"])),
        summary_usage=usage,
        retained_tail=tuple(_decode_message(message) for message in raw_tail),
    )


def _encode_custom_entry(entry: CustomEntry) -> dict[str, object]:
    return {
        "namespace": entry.namespace,
        "version": entry.version,
        "payload": dict(entry.payload),
    }


def _decode_custom_entry(record: Mapping[str, object]) -> CustomEntry:
    payload = record.get("payload")
    version = record.get("version")
    if not isinstance(payload, dict):
        raise ValueError("Custom Entry payload must be an object")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Custom Entry version must be an integer")
    return CustomEntry(
        str(record["namespace"]),
        version,
        payload,
    )


class SessionWriter(Protocol):
    session_id: str

    def __enter__(self) -> "SessionWriter": ...

    def __exit__(self, *exc_info: object) -> None: ...

    def append(
        self,
        messages: Sequence[ConversationMessage],
        *,
        parent_id: str | None = None,
    ) -> "SessionEntry": ...

    def append_compaction(
        self,
        checkpoint: CompactionCheckpoint,
        *,
        parent_id: str | None = None,
    ) -> "SessionEntry": ...

    def append_custom(
        self,
        custom: CustomEntry,
        *,
        parent_id: str | None = None,
    ) -> "SessionEntry": ...


class SessionStore(Protocol):
    def create(self, session_id: str | None = None) -> "SessionState": ...

    def read(self, session_id: str) -> "SessionState": ...

    def writer(self, session_id: str) -> SessionWriter: ...


@dataclass(frozen=True, slots=True)
class SessionEntry:
    entry_id: str
    parent_id: str | None
    messages: tuple[ConversationMessage, ...] = ()
    compaction: CompactionCheckpoint | None = None
    custom: CustomEntry | None = None


@dataclass(frozen=True, slots=True)
class SessionState:
    session_id: str
    entries: tuple[SessionEntry, ...] = ()
    recovery_warning: RecoveryWarning | None = None

    def __post_init__(self) -> None:
        _session_id(self.session_id)
        earlier: set[str] = set()
        for entry in self.entries:
            if not _SESSION_ID.fullmatch(entry.entry_id):
                raise ValueError("Session entry id must be a safe local identifier")
            if entry.entry_id in earlier:
                raise ValueError(f"duplicate Session entry id: {entry.entry_id}")
            if entry.parent_id is not None and entry.parent_id not in earlier:
                raise ValueError(
                    "Session entry parent must reference an earlier Session entry"
                )
            kinds = (
                int(bool(entry.messages))
                + int(entry.compaction is not None)
                + int(entry.custom is not None)
            )
            if kinds != 1:
                raise ValueError(
                    "a Session entry requires messages, Compaction, or Custom Entry"
                )
            earlier.add(entry.entry_id)

    @property
    def active_leaf_id(self) -> str | None:
        return self.entries[-1].entry_id if self.entries else None

    def history(self, leaf_id: str | None = None) -> tuple[ConversationMessage, ...]:
        if not self.entries:
            if leaf_id is not None:
                raise KeyError(f"unknown Session entry: {leaf_id}")
            return ()
        by_id = {entry.entry_id: entry for entry in self.entries}
        cursor = self.active_leaf_id if leaf_id is None else leaf_id
        path: list[SessionEntry] = []
        while cursor is not None:
            try:
                entry = by_id[cursor]
            except KeyError:
                raise KeyError(f"unknown Session entry: {cursor}") from None
            path.append(entry)
            cursor = entry.parent_id
        return tuple(
            message for entry in reversed(path) for message in entry.messages
        )

    def compactions(
        self, leaf_id: str | None = None
    ) -> tuple[CompactionCheckpoint, ...]:
        return tuple(
            entry.compaction
            for entry in self._path(leaf_id)
            if entry.compaction is not None
        )

    def custom_entries(self, leaf_id: str | None = None) -> tuple[CustomEntry, ...]:
        return tuple(
            entry.custom
            for entry in self._path(leaf_id)
            if entry.custom is not None
        )

    def effective_history(
        self, leaf_id: str | None = None
    ) -> tuple[ConversationMessage, ...]:
        path = self._path(leaf_id)
        latest = next(
            (
                index
                for index in range(len(path) - 1, -1, -1)
                if path[index].compaction is not None
            ),
            None,
        )
        if latest is None:
            return tuple(message for entry in path for message in entry.messages)
        checkpoint = path[latest].compaction
        assert checkpoint is not None
        return (
            checkpoint.summary.as_message(),
            *checkpoint.retained_tail,
            *(
                message
                for entry in path[latest + 1 :]
                for message in entry.messages
            ),
        )

    def _path(self, leaf_id: str | None = None) -> tuple[SessionEntry, ...]:
        if not self.entries:
            if leaf_id is not None:
                raise KeyError(f"unknown Session entry: {leaf_id}")
            return ()
        by_id = {entry.entry_id: entry for entry in self.entries}
        cursor = self.active_leaf_id if leaf_id is None else leaf_id
        path: list[SessionEntry] = []
        while cursor is not None:
            try:
                entry = by_id[cursor]
            except KeyError:
                raise KeyError(f"unknown Session entry: {cursor}") from None
            path.append(entry)
            cursor = entry.parent_id
        return tuple(reversed(path))


@dataclass(frozen=True, slots=True)
class MigrationResult:
    source: Path
    destination: Path
    session_id: str
    entries: int


class MemorySessionWriter:
    def __init__(self, store: "MemorySessionStore", session_id: str) -> None:
        self._store = store
        self.session_id = session_id

    def __enter__(self) -> "MemorySessionWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def append(
        self,
        messages: Sequence[ConversationMessage],
        *,
        parent_id: str | None = None,
    ) -> SessionEntry:
        state = self._store.read(self.session_id)
        parent = state.active_leaf_id if parent_id is None else parent_id
        if parent is not None and parent not in {
            entry.entry_id for entry in state.entries
        }:
            raise KeyError(f"unknown Session entry: {parent}")
        accepted = tuple(messages)
        if not accepted:
            raise ValueError("a settled Session entry requires messages")
        entry = SessionEntry(self._store._id_factory(), parent, accepted)
        self._store._sessions[self.session_id] = SessionState(
            self.session_id, (*state.entries, entry)
        )
        return entry

    def append_compaction(
        self,
        checkpoint: CompactionCheckpoint,
        *,
        parent_id: str | None = None,
    ) -> SessionEntry:
        state = self._store.read(self.session_id)
        parent = state.active_leaf_id if parent_id is None else parent_id
        if parent is not None and parent not in {
            entry.entry_id for entry in state.entries
        }:
            raise KeyError(f"unknown Session entry: {parent}")
        entry = SessionEntry(
            self._store._id_factory(),
            parent,
            compaction=checkpoint,
        )
        self._store._sessions[self.session_id] = SessionState(
            self.session_id, (*state.entries, entry)
        )
        return entry

    def append_custom(
        self,
        custom: CustomEntry,
        *,
        parent_id: str | None = None,
    ) -> SessionEntry:
        if not isinstance(custom, CustomEntry):
            raise TypeError("custom must be a CustomEntry")
        state = self._store.read(self.session_id)
        parent = state.active_leaf_id if parent_id is None else parent_id
        if parent is not None and parent not in {
            entry.entry_id for entry in state.entries
        }:
            raise KeyError(f"unknown Session entry: {parent}")
        entry = SessionEntry(
            self._store._id_factory(),
            parent,
            custom=custom,
        )
        self._store._sessions[self.session_id] = SessionState(
            self.session_id, (*state.entries, entry)
        )
        return entry


class MemorySessionStore:
    """In-process SessionStore adapter with the durable tree contract."""

    def __init__(self, *, id_factory: IdFactory | None = None) -> None:
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str | None = None) -> SessionState:
        accepted = _session_id(session_id or self._id_factory())
        if accepted in self._sessions:
            raise ValueError(f"Session already exists: {accepted}")
        state = SessionState(accepted)
        self._sessions[accepted] = state
        return state

    def read(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown Session: {session_id}") from None

    def writer(self, session_id: str) -> MemorySessionWriter:
        self.read(session_id)
        return MemorySessionWriter(self, session_id)


class JSONLSessionWriter:
    def __init__(self, store: "JSONLSessionStore", session_id: str) -> None:
        self._store = store
        self.session_id = session_id
        self._lock_stream: BinaryIO | None = None

    def __enter__(self) -> "JSONLSessionWriter":
        if self._lock_stream is not None:
            raise RuntimeError("Session writer lease is already active")
        self._lock_stream = self._store._acquire_lock(self.session_id)
        try:
            state = self._store.read(self.session_id)
            if state.recovery_warning is not None:
                self._store._discard_uncommitted_tail(self.session_id)
        except BaseException:
            self._store._release_lock(self._lock_stream)
            self._lock_stream = None
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._lock_stream is not None:
            self._store._release_lock(self._lock_stream)
            self._lock_stream = None

    def append(
        self,
        messages: Sequence[ConversationMessage],
        *,
        parent_id: str | None = None,
    ) -> SessionEntry:
        if self._lock_stream is None:
            raise RuntimeError("Session writer lease is not active")
        state = self._store.read(self.session_id)
        parent = state.active_leaf_id if parent_id is None else parent_id
        if parent is not None and parent not in {
            entry.entry_id for entry in state.entries
        }:
            raise KeyError(f"unknown Session entry: {parent}")
        accepted = tuple(messages)
        if not accepted:
            raise ValueError("a settled Session entry requires messages")
        entry = SessionEntry(self._store._id_factory(), parent, accepted)
        record = {
            "schema": "agent_harness.session",
            "schema_version": _version_record(),
            "record": "settlement",
            "session_id": self.session_id,
            "entry_id": entry.entry_id,
            "parent_id": entry.parent_id,
            "messages": [_encode_message(message) for message in accepted],
        }
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        with self._store.path_for(self.session_id).open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def append_compaction(
        self,
        checkpoint: CompactionCheckpoint,
        *,
        parent_id: str | None = None,
    ) -> SessionEntry:
        if self._lock_stream is None:
            raise RuntimeError("Session writer lease is not active")
        state = self._store.read(self.session_id)
        parent = state.active_leaf_id if parent_id is None else parent_id
        if parent is not None and parent not in {
            entry.entry_id for entry in state.entries
        }:
            raise KeyError(f"unknown Session entry: {parent}")
        entry = SessionEntry(
            self._store._id_factory(),
            parent,
            compaction=checkpoint,
        )
        record = {
            "schema": "agent_harness.session",
            "schema_version": _version_record(),
            "record": "compaction",
            "session_id": self.session_id,
            "entry_id": entry.entry_id,
            "parent_id": entry.parent_id,
            "checkpoint": _encode_compaction(checkpoint),
        }
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        with self._store.path_for(self.session_id).open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def append_custom(
        self,
        custom: CustomEntry,
        *,
        parent_id: str | None = None,
    ) -> SessionEntry:
        if self._lock_stream is None:
            raise RuntimeError("Session writer lease is not active")
        if not isinstance(custom, CustomEntry):
            raise TypeError("custom must be a CustomEntry")
        state = self._store.read(self.session_id)
        parent = state.active_leaf_id if parent_id is None else parent_id
        if parent is not None and parent not in {
            entry.entry_id for entry in state.entries
        }:
            raise KeyError(f"unknown Session entry: {parent}")
        entry = SessionEntry(
            self._store._id_factory(),
            parent,
            custom=custom,
        )
        record = {
            "schema": "agent_harness.session",
            "schema_version": _version_record(),
            "record": "custom",
            "session_id": self.session_id,
            "entry_id": entry.entry_id,
            "parent_id": entry.parent_id,
            "custom": _encode_custom_entry(custom),
        }
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        with self._store.path_for(self.session_id).open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return entry


class JSONLSessionStore:
    """Transparent file-per-Session JSONL adapter."""

    def __init__(self, root: str | Path, *, id_factory: IdFactory | None = None) -> None:
        self.root = Path(root)
        self.sessions_directory = self.root / "sessions"
        self._id_factory = id_factory or (lambda: uuid4().hex)

    def path_for(self, session_id: str) -> Path:
        return self.sessions_directory / f"{_session_id(session_id)}.jsonl"

    def _lock_path(self, session_id: str) -> Path:
        return self.sessions_directory / ".locks" / f"{_session_id(session_id)}.lock"

    def _discard_uncommitted_tail(self, session_id: str) -> None:
        path = self.path_for(session_id)
        content = path.read_bytes()
        committed_end = content.rfind(b"\n")
        if committed_end < 0:
            raise ValueError("Session file has no committed header")
        with path.open("r+b") as stream:
            stream.truncate(committed_end + 1)
            stream.flush()
            os.fsync(stream.fileno())

    def _acquire_lock(self, session_id: str) -> BinaryIO:
        path = self._lock_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    stream.fileno(), msvcrt.LK_NBLCK, 1  # type: ignore[attr-defined]
                )
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise SessionBusyError(session_id) from None
        return stream

    @staticmethod
    def _release_lock(stream: BinaryIO) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    stream.fileno(), msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
                )
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def create(self, session_id: str | None = None) -> SessionState:
        accepted = _session_id(session_id or self._id_factory())
        path = self.path_for(accepted)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "agent_harness.session",
            "schema_version": _version_record(),
            "record": "session",
            "session_id": accepted,
        }
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise ValueError(f"Session already exists: {accepted}") from None
        return SessionState(accepted)

    def read(self, session_id: str) -> SessionState:
        accepted = _session_id(session_id)
        path = self.path_for(accepted)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            raise KeyError(f"unknown Session: {accepted}") from None
        raw_lines = content.splitlines(keepends=True)
        warning = None
        if raw_lines and not raw_lines[-1].endswith(b"\n"):
            warning = RecoveryWarning(
                RecoveryCode.INCOMPLETE_FINAL_RECORD,
                "ignored an incomplete final Session record; restart the operation",
                len(raw_lines),
            )
            raw_lines = raw_lines[:-1]
        try:
            lines = [line.decode("utf-8").rstrip("\r\n") for line in raw_lines]
        except UnicodeDecodeError as error:
            raise ValueError("Session JSONL must be UTF-8 text") from error
        if not lines:
            raise ValueError("Session file has no committed header")
        entries: list[SessionEntry] = []
        for line_number, line in enumerate(lines, 1):
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid Session JSONL record at line {line_number}"
                ) from error
            record = _validate_record_version(decoded, line_number)
            if line_number == 1:
                if record.get("record") != "session" or record.get("session_id") != accepted:
                    raise ValueError("invalid Session header")
                continue
            if record.get("session_id") != accepted:
                raise ValueError(f"invalid Session record at line {line_number}")
            if record.get("record") == "custom":
                raw_custom = record.get("custom")
                if not isinstance(raw_custom, dict):
                    raise ValueError("invalid Custom Entry record")
                entries.append(
                    SessionEntry(
                        str(record["entry_id"]),
                        (
                            None
                            if record.get("parent_id") is None
                            else str(record["parent_id"])
                        ),
                        custom=_decode_custom_entry(raw_custom),
                    )
                )
                continue
            if record.get("record") == "compaction":
                raw_checkpoint = record.get("checkpoint")
                if not isinstance(raw_checkpoint, dict):
                    raise ValueError("invalid Compaction checkpoint record")
                entries.append(
                    SessionEntry(
                        str(record["entry_id"]),
                        (
                            None
                            if record.get("parent_id") is None
                            else str(record["parent_id"])
                        ),
                        compaction=_decode_compaction(raw_checkpoint),
                    )
                )
                continue
            if record.get("record") != "settlement":
                raise ValueError(f"invalid Session record at line {line_number}")
            raw_messages = record.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise ValueError("a settled Session entry requires messages")
            entries.append(
                SessionEntry(
                    str(record["entry_id"]),
                    None if record.get("parent_id") is None else str(record["parent_id"]),
                    tuple(_decode_message(message) for message in raw_messages),
                )
            )
        state = SessionState(accepted, tuple(entries), warning)
        state.history()
        return state

    def writer(self, session_id: str) -> JSONLSessionWriter:
        self.read(session_id)
        return JSONLSessionWriter(self, session_id)


def migrate_session_file(
    source: str | Path,
    destination: str | Path,
) -> MigrationResult:
    """Write and validate a current Session file without changing its source."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path.parent.name != "sessions":
        raise ValueError("source must be a file from a sessions directory")
    session_id = _session_id(source_path.stem)
    if destination_path.name != f"{session_id}.jsonl":
        raise ValueError("migration destination must retain the Session filename")
    if destination_path.exists():
        raise FileExistsError(f"migration destination exists: {destination_path}")
    state = JSONLSessionStore(source_path.parent.parent).read(session_id)
    records: list[dict[str, object]] = [
        {
            "schema": "agent_harness.session",
            "schema_version": _version_record(),
            "record": "session",
            "session_id": session_id,
        }
    ]
    for entry in state.entries:
        common: dict[str, object] = {
            "schema": "agent_harness.session",
            "schema_version": _version_record(),
            "session_id": session_id,
            "entry_id": entry.entry_id,
            "parent_id": entry.parent_id,
        }
        if entry.custom is not None:
            records.append(
                {
                    **common,
                    "record": "custom",
                    "custom": _encode_custom_entry(entry.custom),
                }
            )
        elif entry.compaction is not None:
            records.append(
                {
                    **common,
                    "record": "compaction",
                    "checkpoint": _encode_compaction(entry.compaction),
                }
            )
        else:
            records.append(
                {
                    **common,
                    "record": "settlement",
                    "messages": [
                        _encode_message(message) for message in entry.messages
                    ],
                }
            )
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".session-migration-", dir=destination_path.parent.parent
    ) as temporary:
        staging_root = Path(temporary)
        staging = staging_root / "sessions" / f"{session_id}.jsonl"
        staging.parent.mkdir(parents=True)
        with staging.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        validated = JSONLSessionStore(staging_root).read(session_id)
        if (
            validated.history() != state.history()
            or validated.compactions() != state.compactions()
            or validated.custom_entries() != state.custom_entries()
        ):
            raise ValueError("migrated Session failed history validation")
        staging.replace(destination_path)
    return MigrationResult(
        source_path,
        destination_path,
        session_id,
        len(state.entries),
    )
