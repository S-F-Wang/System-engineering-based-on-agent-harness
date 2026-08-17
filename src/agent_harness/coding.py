"""Optional host-authority Coding Tool Preset."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
from typing import Protocol

from .tools import CompleteOutputReference, Tool, ToolErrorCode, ToolResult
from .tools import ProcessResult, TruncationDirection
from .model import AgentMessage, ModelAdapter, ModelSpec, Role
from .resources import LoadedResources, PromptAssembler, PromptAssembly, ResourceLoader
from .runtime import AgentRuntime
from .session import AgentSession


_SENSITIVE_ENVIRONMENT = re.compile(
    r"(?:api[_-]?key|access[_-]?key|token|secret|password|passwd|"
    r"authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)


class WorkspaceBoundaryError(ValueError):
    """A file Tool path could not be confined to its Workspace Boundary."""


class ArtifactStore(Protocol):
    def put_text(self, content: str) -> str: ...


class FileArtifactStore:
    """Retain sanitized text under a content-addressed local reference."""

    def __init__(
        self,
        root: str | Path,
        *,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._redact = redact or (lambda text: text)

    def put_text(self, content: str) -> str:
        sanitized = self._redact(content)
        if not isinstance(sanitized, str):
            raise TypeError("Artifact redactor must return text")
        digest = sha256(sanitized.encode("utf-8")).hexdigest()
        path = self._root / "sha256" / f"{digest}.txt"
        if not path.exists():
            _atomic_write(path, sanitized)
        return f"sha256:{digest}"

    def read_text(self, reference: str) -> str:
        prefix, separator, digest = reference.partition(":")
        if prefix != "sha256" or not separator or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("invalid Artifact reference")
        return (self._root / "sha256" / f"{digest}.txt").read_text(
            encoding="utf-8"
        )


class WorkspaceBoundary:
    """Resolve file Tool paths through canonical parents inside one workspace."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError("Workspace Boundary must be an existing directory")

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, supplied: object) -> Path:
        if not isinstance(supplied, str) or not supplied:
            raise WorkspaceBoundaryError("file path must be a non-empty string")
        path = Path(supplied)
        candidate = path if path.is_absolute() else self._root / path
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise WorkspaceBoundaryError(
                "path cannot be resolved inside the Workspace Boundary"
            ) from error
        if not resolved.is_relative_to(self._root):
            raise WorkspaceBoundaryError(
                "path resolves outside the Workspace Boundary"
            )
        return resolved


def filter_sensitive_environment(
    environment: Mapping[str, str],
    *,
    allow_sensitive: Sequence[str] = (),
) -> dict[str, str]:
    """Remove credential-shaped names unless the caller deliberately allows one."""

    allowed = {name.casefold() for name in allow_sensitive}
    return {
        name: value
        for name, value in environment.items()
        if name.casefold() in allowed or _SENSITIVE_ENVIRONMENT.search(name) is None
    }


def resolve_bash(
    shell_path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one real Bash executable without translating shell commands."""

    if shell_path is not None:
        candidate = Path(shell_path).expanduser().resolve()
        if candidate.is_file() and (
            os.name == "nt" or os.access(candidate, os.X_OK)
        ):
            return candidate
        raise FileNotFoundError(
            f"configured Bash executable does not exist or is not executable: {candidate}"
        )
    source = dict(os.environ if environment is None else environment)
    candidates: list[Path] = []
    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
            base = source.get(variable)
            if base:
                candidates.append(Path(base) / "Git" / "bin" / "bash.exe")
                if variable == "LocalAppData":
                    candidates.append(
                        Path(base) / "Programs" / "Git" / "bin" / "bash.exe"
                    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which("bash", path=source.get("PATH"))
    if found is not None:
        return Path(found).resolve()
    raise FileNotFoundError(
        "A real Bash executable is required; install Bash (Git Bash on Windows) "
        "or configure shell_path"
    )


class BashOperations:
    """Execute real Bash commands with fixed cwd and cancellable host authority."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        shell_path: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        allow_sensitive: Sequence[str] = (),
        default_timeout_seconds: float = 60.0,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        if not self._workspace.is_dir():
            raise ValueError("Bash workspace must be an existing directory")
        if default_timeout_seconds <= 0:
            raise ValueError("default Bash timeout must be positive")
        source = dict(os.environ if environment is None else environment)
        self._shell = resolve_bash(shell_path, environment=source)
        self._environment = filter_sensitive_environment(
            source, allow_sensitive=allow_sensitive
        )
        self._default_timeout_seconds = default_timeout_seconds

    @property
    def shell_path(self) -> Path:
        return self._shell

    @property
    def workspace(self) -> Path:
        return self._workspace

    async def run(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash command must be non-empty text")
        timeout = (
            self._default_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout <= 0:
            raise ValueError("Bash timeout must be positive")
        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                str(self._shell),
                "--noprofile",
                "--norc",
                "-lc",
                command,
                cwd=self._workspace,
                env=self._environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            process = await asyncio.create_subprocess_exec(
                str(self._shell),
                "--noprofile",
                "--norc",
                "-lc",
                command,
                cwd=self._workspace,
                env=self._environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.CancelledError:
            await self._terminate(process)
            raise
        except TimeoutError:
            await self._terminate(process)
            raise TimeoutError(f"Bash command exceeded {timeout:g} seconds") from None
        assert process.returncode is not None
        return ProcessResult(
            process.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            process.terminate()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except TimeoutError:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()


def _tool_error(tool_name: str, error: Exception) -> ToolResult:
    return ToolResult(
        f"Tool error [execution_failed] for '{tool_name}': {error}",
        is_error=True,
        error_code=ToolErrorCode.EXECUTION_FAILED,
    )


def _atomic_write(path: Path, content: str) -> None:
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".omega-tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_text_for_mutation(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError("binary files cannot be mutated by text Tools")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("binary or non-UTF-8 files cannot be mutated") from error


@dataclass(frozen=True, slots=True)
class CodingToolPreset:
    workspace: WorkspaceBoundary
    bash_operations: BashOperations
    tools: tuple[Tool, ...]

    def tool(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"unknown Coding Tool: {name}")


@dataclass(frozen=True, slots=True)
class CodingAgent:
    """The assembled Coding Agent product around one reusable AgentSession."""

    session: AgentSession
    preset: CodingToolPreset
    resources: LoadedResources
    prompt: PromptAssembly

    @property
    def startup_evidence(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.prompt.resource_hashes.items()))


def create_coding_tool_preset(
    workspace: str | Path,
    *,
    bash_operations: BashOperations | None = None,
    shell_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    allow_sensitive_environment: Sequence[str] = (),
    artifact_store: ArtifactStore | None = None,
    artifact_threshold_bytes: int = 50 * 1024,
) -> CodingToolPreset:
    """Create explicitly installed Coding Tools; no sandbox is implied."""

    boundary = WorkspaceBoundary(workspace)
    if artifact_threshold_bytes <= 0:
        raise ValueError("artifact_threshold_bytes must be positive")
    bash_backend = bash_operations or BashOperations(
        boundary.root,
        shell_path=shell_path,
        environment=environment,
        allow_sensitive=allow_sensitive_environment,
    )
    if bash_backend.workspace != boundary.root:
        raise ValueError("BashOperations must use the Coding Tool Preset workspace")

    def complete_output(content: str) -> CompleteOutputReference | None:
        if (
            artifact_store is None
            or len(content.encode("utf-8")) <= artifact_threshold_bytes
        ):
            return None
        return CompleteOutputReference.artifact(artifact_store.put_text(content))

    async def read(arguments: dict[str, object]) -> ToolResult:
        try:
            path = boundary.resolve(arguments.get("path"))
            if not path.is_file():
                raise ValueError("read path must be an existing text file")
            raw = path.read_bytes()
            if b"\x00" in raw:
                raise ValueError("binary files cannot be read as text")
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "binary or non-UTF-8 files cannot be read as text"
                ) from error
            offset = arguments.get("offset", 1)
            limit = arguments.get("limit")
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
                raise ValueError("read offset must be a positive line number")
            if limit is not None and (
                not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
            ):
                raise ValueError("read limit must be a positive line count")
            lines = content.splitlines(keepends=True)
            selected = (
                lines[offset - 1 :]
                if limit is None
                else lines[offset - 1 : offset - 1 + limit]
            )
            return ToolResult(
                "".join(selected),
                metadata={
                    "path": path.relative_to(boundary.root).as_posix(),
                    "offset": offset,
                    "lines": len(selected),
                },
                complete_output=complete_output(content),
            )
        except (OSError, ValueError) as error:
            return _tool_error("read", error)

    async def write(arguments: dict[str, object]) -> ToolResult:
        try:
            path = boundary.resolve(arguments.get("path"))
            content = arguments.get("content")
            if not isinstance(content, str):
                raise ValueError("write content must be text")
            if path.exists():
                if not path.is_file():
                    raise ValueError("write path must be a text file")
                _read_text_for_mutation(path)
            _atomic_write(path, content)
            return ToolResult(
                f"Wrote {len(content.encode('utf-8'))} bytes to "
                f"{path.relative_to(boundary.root).as_posix()}",
                metadata={"path": path.relative_to(boundary.root).as_posix()},
            )
        except (OSError, ValueError) as error:
            return _tool_error("write", error)

    async def edit(arguments: dict[str, object]) -> ToolResult:
        try:
            path = boundary.resolve(arguments.get("path"))
            if not path.is_file():
                raise ValueError("edit path must be an existing text file")
            old_text = arguments.get("old_text")
            new_text = arguments.get("new_text")
            if not isinstance(old_text, str) or not old_text:
                raise ValueError("edit old_text must be non-empty text")
            if not isinstance(new_text, str):
                raise ValueError("edit new_text must be text")
            content = _read_text_for_mutation(path)
            occurrences = content.count(old_text)
            if occurrences != 1:
                raise ValueError(
                    f"edit old_text must match exactly once; found {occurrences} matches"
                )
            updated = content.replace(old_text, new_text, 1)
            _atomic_write(path, updated)
            relative = path.relative_to(boundary.root).as_posix()
            return ToolResult(
                f"Edited {relative}",
                metadata={"path": relative, "replacements": 1},
            )
        except (OSError, ValueError) as error:
            return _tool_error("edit", error)

    async def bash(arguments: dict[str, object]) -> ToolResult:
        try:
            command = arguments.get("command")
            timeout = arguments.get("timeout_seconds")
            if not isinstance(command, str):
                raise ValueError("Bash command must be text")
            if timeout is not None and (
                not isinstance(timeout, (int, float))
                or isinstance(timeout, bool)
                or timeout <= 0
            ):
                raise ValueError("Bash timeout_seconds must be positive")
            result = await bash_backend.run(command, timeout_seconds=timeout)
            content = result.stdout
            if result.stderr:
                content += ("\n" if content else "") + result.stderr
            return ToolResult(
                content,
                metadata={"returncode": result.returncode},
                is_error=result.returncode != 0,
                error_code=(
                    ToolErrorCode.EXECUTION_FAILED if result.returncode != 0 else None
                ),
                complete_output=complete_output(content),
            )
        except (OSError, TimeoutError, ValueError) as error:
            return _tool_error("bash", error)

    read_tool = Tool(
        "read",
        "Read UTF-8 text by line range inside the Workspace Boundary",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        read,
    )
    write_tool = Tool(
        "write",
        "Atomically write one UTF-8 text file inside the Workspace Boundary",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        write,
        sequential=True,
    )
    edit_tool = Tool(
        "edit",
        "Atomically replace one exact text occurrence inside the Workspace Boundary",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        edit,
        sequential=True,
    )
    bash_tool = Tool(
        "bash",
        "Run a real Bash command in the fixed workspace with host-process authority",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        bash,
        sequential=True,
        output_direction=TruncationDirection.TAIL,
    )
    return CodingToolPreset(
        boundary,
        bash_backend,
        (read_tool, write_tool, edit_tool, bash_tool),
    )


def create_coding_agent(
    adapter: ModelAdapter,
    model: ModelSpec,
    workspace: str | Path,
    *,
    resource_loader: ResourceLoader | None = None,
    prompt_assembler: PromptAssembler | None = None,
    active_skills: Sequence[str] = (),
    preset: CodingToolPreset | None = None,
) -> CodingAgent:
    """Assemble local resources and explicitly install the Coding Tool Preset."""

    boundary = Path(workspace).resolve()
    effective_preset = preset or create_coding_tool_preset(boundary)
    if effective_preset.workspace.root != boundary:
        raise ValueError("Coding Tool Preset must use the Coding Agent workspace")
    effective_loader = resource_loader or ResourceLoader(boundary)
    if effective_loader.workspace != boundary:
        raise ValueError("ResourceLoader must use the Coding Agent workspace")
    resources = effective_loader.load()
    assembler = prompt_assembler or PromptAssembler(
        "You are Omega, a coding agent. Work carefully inside the supplied "
        "workspace, use Tools for evidence, and verify changes before finishing."
    )
    prompt = assembler.assemble(
        tools=effective_preset.tools,
        resources=resources,
        active_skills=active_skills,
    )
    runtime = AgentRuntime(
        adapter,
        model,
        tools=effective_preset.tools,
        history=(AgentMessage.text(Role.SYSTEM, prompt.text),),
    )
    session = AgentSession(
        runtime,
        prompt_hashes=prompt.prompt_hashes,
        resource_hashes=prompt.resource_hashes,
    )
    return CodingAgent(session, effective_preset, resources, prompt)
