"""Local, versioned Run evidence for reusable AgentSession execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Protocol

from platformdirs import user_data_path

from .coding import ArtifactStore
from .model import ModelEnd, TextDelta, ToolCallDelta, UsageUpdate
from .persistence import JSONLSessionStore, SchemaVersion
from .runtime import EventType, RunSnapshot, RuntimeEvent
from .session import SessionRunResult


RUN_TRACE_SCHEMA_VERSION = SchemaVersion(1, 0)
RUN_ANNOTATION_SCHEMA_VERSION = SchemaVersion(1, 0)
ARTIFACT_SCHEMA_VERSION = SchemaVersion(1, 0)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|passwd|secret|token|"
    r"private[_-]?key)",
    re.IGNORECASE,
)
_BOUNDED_PREVIEW_KEY = re.compile(
    r"(?:text|content|message|arguments|diagnostic|reason|preview|output)",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _version(version: SchemaVersion) -> dict[str, int]:
    return {"major": version.major, "minor": version.minor}


def _safe_id(value: str, kind: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{kind} id must be a safe local identifier")
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class WorkspaceIdentity:
    canonical_path: str
    value: str

    @classmethod
    def from_path(cls, workspace: str | Path) -> "WorkspaceIdentity":
        canonical = Path(workspace).resolve()
        if not canonical.is_dir():
            raise ValueError("workspace must be an existing directory")
        normalized = os.path.normcase(str(canonical))
        digest = sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return cls(str(canonical), f"workspace-{digest}")


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    session_id: str | None
    status: str | None
    started_at: str
    finished_at: str | None
    evidence_complete: bool


@dataclass(frozen=True, slots=True)
class PrunePreview:
    run_ids: tuple[str, ...]
    bytes: int


class UnsupportedRunSchemaVersionError(ValueError):
    def __init__(self, artifact: str, found_major: int, supported_major: int) -> None:
        self.artifact = artifact
        self.found_major = found_major
        self.supported_major = supported_major
        super().__init__(
            f"{artifact} schema major {found_major} is unsupported; this reader "
            f"supports major {supported_major}. Preserve the original and migrate "
            "it with migrate_run_trace()."
        )


class TraceRedactor(Protocol):
    def redact(self, value: object) -> object: ...


class StandardTraceRedactor:
    """Remove credential-shaped fields and bound persisted string previews."""

    def __init__(
        self,
        *,
        secrets: Sequence[str] = (),
        preview_chars: int = 4096,
    ) -> None:
        if preview_chars <= 0:
            raise ValueError("preview_chars must be positive")
        self._secrets = tuple(secret for secret in secrets if secret)
        self._preview_chars = preview_chars

    def redact(self, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(str(key))
                    else self._redact_text(
                        item,
                        bound=_BOUNDED_PREVIEW_KEY.search(str(key)) is not None,
                    )
                    if isinstance(item, str)
                    else self.redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value, bound=True)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)

    def redact_artifact_text(self, value: str) -> str:
        """Redact secrets without applying the bounded Trace preview limit."""

        return self._redact_text(value, bound=False)

    def _redact_text(self, value: str, *, bound: bool) -> str:
        sanitized = value
        for secret in self._secrets:
            sanitized = sanitized.replace(secret, "[REDACTED]")
        if bound and len(sanitized) > self._preview_chars:
            omitted = len(sanitized) - self._preview_chars
            sanitized = f"{sanitized[: self._preview_chars]}…[{omitted} chars omitted]"
        return sanitized


class RunStore(Protocol):
    def list(
        self, *, session_id: str | None = None, status: str | None = None
    ) -> tuple[RunSummary, ...]: ...

    def export(self, run_id: str) -> str: ...

    def annotate(
        self,
        run_id: str,
        namespace: str,
        payload: Mapping[str, object],
    ) -> None: ...


class JSONLRunStore:
    """File-per-Run JSONL store with append-only annotations and artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        workspace_identity: WorkspaceIdentity | None = None,
        redactor: TraceRedactor | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.runs_directory = self.root / "runs"
        self.workspace_identity = workspace_identity
        self.redactor = redactor or StandardTraceRedactor()

    def directory_for(self, run_id: str) -> Path:
        return self.runs_directory / _safe_id(run_id, "Run")

    def trace_path(self, run_id: str) -> Path:
        return self.directory_for(run_id) / "trace.jsonl"

    def annotation_path(self, run_id: str) -> Path:
        return self.directory_for(run_id) / "annotations.jsonl"

    def start(
        self,
        run_id: str,
        *,
        session_id: str | None,
        snapshot: Mapping[str, object],
    ) -> None:
        accepted = _safe_id(run_id, "Run")
        directory = self.directory_for(accepted)
        directory.mkdir(parents=True, exist_ok=False)
        record = {
            "schema": "agent_harness.run_trace",
            "schema_version": _version(RUN_TRACE_SCHEMA_VERSION),
            "record": "run_start",
            "run_id": accepted,
            "session_id": session_id,
            "workspace_identity": (
                None if self.workspace_identity is None else self.workspace_identity.value
            ),
            "timestamp": _utc_now(),
            "snapshot": dict(snapshot),
        }
        _atomic_text(self.trace_path(accepted), self._encode(record))

    def append(self, run_id: str, record: str, payload: Mapping[str, object]) -> None:
        accepted = _safe_id(run_id, "Run")
        path = self.trace_path(accepted)
        if not path.is_file():
            raise KeyError(f"unknown Run: {accepted}")
        value = {
            "schema": "agent_harness.run_trace",
            "schema_version": _version(RUN_TRACE_SCHEMA_VERSION),
            "record": record,
            "run_id": accepted,
            "timestamp": _utc_now(),
            **dict(payload),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(self._encode(value))
            stream.flush()
            os.fsync(stream.fileno())

    def mark_incomplete(self, run_id: str, reason: str) -> None:
        if self.trace_path(run_id).is_file():
            self.append(run_id, "trace_incomplete", {"reason": reason})

    def records(self, run_id: str) -> tuple[Mapping[str, object], ...]:
        records, _ = self._read_jsonl(
            self.trace_path(run_id),
            schema="agent_harness.run_trace",
            version=RUN_TRACE_SCHEMA_VERSION,
            artifact="Run Trace",
        )
        return records

    def list(
        self, *, session_id: str | None = None, status: str | None = None
    ) -> tuple[RunSummary, ...]:
        if not self.runs_directory.is_dir():
            return ()
        summaries: list[RunSummary] = []
        for path in sorted(self.runs_directory.iterdir()):
            if not path.is_dir() or not self.trace_path(path.name).is_file():
                continue
            summary = self.summary(path.name)
            if session_id is not None and summary.session_id != session_id:
                continue
            if status is not None and summary.status != status:
                continue
            summaries.append(summary)
        return tuple(sorted(summaries, key=lambda item: item.started_at, reverse=True))

    def summary(self, run_id: str) -> RunSummary:
        records, final_line_complete = self._read_jsonl(
            self.trace_path(run_id),
            schema="agent_harness.run_trace",
            version=RUN_TRACE_SCHEMA_VERSION,
            artifact="Run Trace",
        )
        if not records:
            raise ValueError("Run Trace has no committed header")
        header = records[0]
        end = next(
            (record for record in reversed(records) if record.get("record") == "run_end"),
            None,
        )
        explicitly_incomplete = any(
            record.get("record") == "trace_incomplete" for record in records
        )
        return RunSummary(
            str(header["run_id"]),
            None if header.get("session_id") is None else str(header["session_id"]),
            None if end is None else str(end.get("status")),
            str(header["timestamp"]),
            None if end is None else str(end.get("timestamp")),
            final_line_complete and end is not None and not explicitly_incomplete,
        )

    def export(self, run_id: str) -> str:
        records = self.records(run_id)
        return "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        )

    def annotate(
        self,
        run_id: str,
        namespace: str,
        payload: Mapping[str, object],
    ) -> None:
        accepted = _safe_id(run_id, "Run")
        self.summary(accepted)
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", namespace):
            raise ValueError("annotation namespace must be a safe dotted name")
        path = self.annotation_path(accepted)
        record = {
            "schema": "agent_harness.run_annotation",
            "schema_version": _version(RUN_ANNOTATION_SCHEMA_VERSION),
            "record": "annotation",
            "run_id": accepted,
            "timestamp": _utc_now(),
            "namespace": namespace,
            "payload": dict(payload),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(self._encode(record))
            stream.flush()
            os.fsync(stream.fileno())

    def annotations(self, run_id: str) -> tuple[Mapping[str, object], ...]:
        path = self.annotation_path(run_id)
        if not path.exists():
            return ()
        records, _ = self._read_jsonl(
            path,
            schema="agent_harness.run_annotation",
            version=RUN_ANNOTATION_SCHEMA_VERSION,
            artifact="Run Annotation",
        )
        return records

    def put_artifact(
        self,
        run_id: str,
        content: bytes,
        *,
        media_type: str = "text/plain; charset=utf-8",
    ) -> str:
        accepted = _safe_id(run_id, "Run")
        self.summary(accepted)
        digest = sha256(content).hexdigest()
        path = self.directory_for(accepted) / "artifacts" / "sha256" / digest
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".artifact.", dir=path.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        reference = f"sha256:{digest}"
        self.append(
            accepted,
            "artifact",
            {
                "reference": reference,
                "schema_version": _version(ARTIFACT_SCHEMA_VERSION),
                "media_type": media_type,
                "bytes": len(content),
            },
        )
        return reference

    def prune_preview(
        self, *, status: str | None = None, session_id: str | None = None
    ) -> PrunePreview:
        run_ids = tuple(
            summary.run_id for summary in self.list(status=status, session_id=session_id)
        )
        total = sum(
            path.stat().st_size
            for run_id in run_ids
            for path in self.directory_for(run_id).rglob("*")
            if path.is_file()
        )
        return PrunePreview(run_ids, total)

    def prune(self, preview: PrunePreview) -> None:
        for run_id in preview.run_ids:
            directory = self.directory_for(run_id)
            if directory.parent != self.runs_directory:
                raise ValueError("Run prune target escaped the Runs directory")
            shutil.rmtree(directory)

    def _encode(self, value: Mapping[str, object]) -> str:
        redacted = self.redactor.redact(value)
        return (
            json.dumps(
                redacted,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    @staticmethod
    def _read_jsonl(
        path: Path,
        *,
        schema: str,
        version: SchemaVersion,
        artifact: str,
    ) -> tuple[tuple[Mapping[str, object], ...], bool]:
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            raise KeyError(f"unknown {artifact}: {path.parent.name}") from None
        lines = content.splitlines(keepends=True)
        final_line_complete = not lines or lines[-1].endswith(b"\n")
        if not final_line_complete:
            lines = lines[:-1]
        decoded: list[Mapping[str, object]] = []
        for line_number, raw in enumerate(lines, 1):
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid {artifact} JSONL at line {line_number}"
                ) from error
            if not isinstance(value, dict) or value.get("schema") != schema:
                raise ValueError(f"invalid {artifact} schema at line {line_number}")
            raw_version = value.get("schema_version")
            if not isinstance(raw_version, dict):
                raise ValueError(f"invalid {artifact} version at line {line_number}")
            major = raw_version.get("major")
            if isinstance(major, bool) or not isinstance(major, int):
                raise ValueError(f"invalid {artifact} version at line {line_number}")
            if major != version.major:
                raise UnsupportedRunSchemaVersionError(artifact, major, version.major)
            decoded.append(value)
        return tuple(decoded), final_line_complete


@dataclass(frozen=True, slots=True)
class LocalWorkspace:
    identity: WorkspaceIdentity
    root: Path
    sessions: JSONLSessionStore
    runs: JSONLRunStore

    @classmethod
    def open(
        cls,
        workspace: str | Path,
        *,
        storage_root: str | Path | None = None,
        redactor: TraceRedactor | None = None,
    ) -> "LocalWorkspace":
        identity = WorkspaceIdentity.from_path(workspace)
        base = (
            Path(storage_root).resolve()
            if storage_root is not None
            else user_data_path("omega", "agent-harness").resolve()
        )
        root = base / "workspaces" / identity.value
        root.mkdir(parents=True, exist_ok=True)
        metadata = root / "workspace.json"
        if not metadata.exists():
            _atomic_text(
                metadata,
                json.dumps(
                    {
                        "schema": "agent_harness.workspace",
                        "schema_version": {"major": 1, "minor": 0},
                        "identity": identity.value,
                        "canonical_path": identity.canonical_path,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        return cls(
            identity,
            root,
            JSONLSessionStore(root),
            JSONLRunStore(root, workspace_identity=identity, redactor=redactor),
        )


class RunArtifactStore(ArtifactStore):
    """Route Coding Tool full output to the currently active durable Run."""

    def __init__(self, store: JSONLRunStore) -> None:
        self._store = store
        self._run_id: str | None = None

    def activate(self, run_id: str) -> None:
        self._run_id = run_id

    def deactivate(self, run_id: str) -> None:
        if self._run_id == run_id:
            self._run_id = None

    def put_text(self, content: str) -> str:
        if self._run_id is None:
            raise RuntimeError("Run Artifact storage requires an active Run")
        artifact_redactor = getattr(self._store.redactor, "redact_artifact_text", None)
        sanitized = (
            artifact_redactor(content)
            if callable(artifact_redactor)
            else self._store.redactor.redact(content)
        )
        if not isinstance(sanitized, str):
            raise TypeError("Trace redactor must return text for text artifacts")
        return self._store.put_artifact(
            self._run_id, sanitized.encode("utf-8"), media_type="text/plain; charset=utf-8"
        )


class StandardTraceSink:
    """Translate Runtime Events into bounded, sanitized, queryable Run records."""

    def __init__(
        self,
        store: JSONLRunStore,
        *,
        workspace: str | Path,
        artifacts: RunArtifactStore | None = None,
    ) -> None:
        self.store = store
        self.workspace = Path(workspace).resolve()
        self.artifacts = artifacts
        self._started: dict[str, float] = {}
        self._first_token: set[str] = set()

    def start_run(self, session_id: str | None, snapshot: RunSnapshot) -> None:
        snapshot_record: dict[str, object] = {
            "fingerprint": snapshot.fingerprint,
            "model": {
                "id": snapshot.model.model_id,
                "context_window": snapshot.model.context_window,
                "max_output_tokens": snapshot.model.max_output_tokens,
                "supports_tools": snapshot.model.supports_tools,
            },
            "tools": [tool.name for tool in snapshot.tools],
            "extensions": list(snapshot.extension_identities),
            "prompt_hashes": dict(snapshot.prompt_hashes),
            "resource_hashes": dict(snapshot.resource_hashes),
            "retry_policy": {
                "delays": list(snapshot.retry_policy.delays),
                "max_retry_after_seconds": snapshot.retry_policy.max_retry_after_seconds,
            },
            "platform": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "git": self._git_evidence(),
        }
        self.store.start(
            snapshot.run_id,
            session_id=session_id,
            snapshot=snapshot_record,
        )
        self._started[snapshot.run_id] = time.perf_counter()
        if self.artifacts is not None:
            self.artifacts.activate(snapshot.run_id)

    def record_event(self, event: RuntimeEvent) -> None:
        if event.run_id is None:
            raise ValueError("Runtime Event is missing its Run id")
        elapsed = time.perf_counter() - self._started[event.run_id]
        if (
            event.type is EventType.MODEL_EVENT
            and isinstance(event.model_event, TextDelta)
            and event.model_event.text
            and event.run_id not in self._first_token
        ):
            self._first_token.add(event.run_id)
            self.store.append(
                event.run_id,
                "first_token",
                {"latency_seconds": round(elapsed, 6)},
            )
        self.store.append(
            event.run_id,
            "event",
            {
                "sequence": event.sequence,
                "type": event.type.value,
                "attempt": event.attempt,
                "elapsed_seconds": round(elapsed, 6),
                "operation": event.operation.value,
                "retry_delay_seconds": event.retry_delay_seconds,
                "partial_text": event.partial_text,
                "error": self._json_value(event.error),
                "model_event": self._model_event(event.model_event),
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "tool_arguments": event.tool_arguments,
                "tool_result": self._json_value(event.tool_result),
                "snapshot_fingerprint": event.snapshot_fingerprint,
            },
        )

    def finish_run(
        self,
        session_id: str | None,
        run_id: str,
        result: SessionRunResult,
    ) -> None:
        outcome = result.outcome
        usage = outcome.usage
        self.store.append(
            run_id,
            "run_end",
            {
                "session_id": session_id,
                "status": outcome.status.value,
                "stop_reason": outcome.stop_reason.value,
                "attempts": outcome.attempts,
                "duration_seconds": round(
                    time.perf_counter() - self._started[run_id], 6
                ),
                "usage": self._json_value(usage),
                "usage_provenance": (
                    None if usage is None else "estimated" if usage.estimated else "provider"
                ),
                "error": self._json_value(outcome.error),
                "pending_inputs": len(result.pending_inputs),
            },
        )
        if self.artifacts is not None:
            self.artifacts.deactivate(run_id)
        self._started.pop(run_id, None)

    def mark_incomplete(self, run_id: str, reason: str) -> None:
        self.store.mark_incomplete(run_id, reason)
        if self.artifacts is not None:
            self.artifacts.deactivate(run_id)
        self._started.pop(run_id, None)

    @staticmethod
    def _json_value(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {
                field.name: StandardTraceSink._json_value(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {
                str(key): StandardTraceSink._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [StandardTraceSink._json_value(item) for item in value]
        return value

    @classmethod
    def _model_event(cls, event: object) -> object:
        if event is None:
            return None
        value = cls._json_value(event)
        if isinstance(event, TextDelta):
            kind = "text_delta"
        elif isinstance(event, ToolCallDelta):
            kind = "tool_call_delta"
        elif isinstance(event, UsageUpdate):
            kind = "usage"
        elif isinstance(event, ModelEnd):
            kind = "model_end"
        else:
            kind = type(event).__name__
        return {"type": kind, "value": value}

    def _git_evidence(self) -> Mapping[str, object] | None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=self.workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return {"commit": commit, "dirty": dirty}


def migrate_run_trace(source: str | Path, destination: str | Path) -> Path:
    """Copy a supported Run Trace to a new validated file, preserving source."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"migration destination exists: {destination_path}")
    records, complete = JSONLRunStore._read_jsonl(
        source_path,
        schema="agent_harness.run_trace",
        version=RUN_TRACE_SCHEMA_VERSION,
        artifact="Run Trace",
    )
    if not complete:
        raise ValueError("cannot migrate an incomplete final Run Trace record")
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    _atomic_text(destination_path, encoded)
    return destination_path
