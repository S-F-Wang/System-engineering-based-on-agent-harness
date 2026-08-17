"""Build a Checkpoint Package from tagged cells in a Chapter Notebook."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import shutil
import subprocess
import sys
import tempfile


@dataclass(frozen=True, slots=True)
class ExportResult:
    """The stable, caller-visible result of a successful checkpoint export."""

    destination: Path
    files: tuple[str, ...]
    gates: tuple[str, ...] = ()
    digest: str | None = None


class CheckpointVerificationError(RuntimeError):
    """Raised when a staged Checkpoint fails an offline gate."""

    def __init__(self, gate: str, detail: str) -> None:
        self.gate = gate
        super().__init__(f"checkpoint {gate} gate failed: {detail}")


def _export_spec(cell: dict[str, object]) -> tuple[str, str] | None:
    metadata = cell.get("metadata")
    if not isinstance(metadata, dict):
        return None
    marker = metadata.get("agent_harness_export")
    if not isinstance(marker, dict):
        return None
    path = marker.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("an Export Cell requires a non-empty string path")
    mode = marker.get("mode", "code")
    if mode not in {"assignment", "code", "literal"}:
        raise ValueError(f"unsupported Export Cell mode: {mode!r}")
    return path, mode


def _safe_relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe checkpoint export path: {raw!r}")
    return path


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise ValueError("notebook cell source must be a string or a list of strings")


def _collect_exports(notebook: Path) -> dict[str, str]:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook must contain a cells list")

    exports: dict[str, list[str]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("notebook cells must be objects")
        spec = _export_spec(cell)
        if spec is None:
            continue
        path, mode = spec
        if cell.get("cell_type") != "code":
            raise ValueError(f"Export Cell {path!r} must be a code cell")
        safe_path = _safe_relative_path(path).as_posix()
        source = _cell_source(cell)
        if mode in {"assignment", "literal"}:
            try:
                if mode == "assignment":
                    parsed = ast.parse(source)
                    if len(parsed.body) != 1 or not isinstance(
                        parsed.body[0], (ast.Assign, ast.AnnAssign)
                    ):
                        raise ValueError
                    value = parsed.body[0].value
                    if value is None:
                        raise ValueError
                    source = ast.literal_eval(value)
                else:
                    source = ast.literal_eval(source)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"{mode} Export Cell {path!r} must contain one string value"
                ) from error
            if not isinstance(source, str):
                raise ValueError(
                    f"{mode} Export Cell {path!r} must contain one string value"
                )
        exports.setdefault(safe_path, []).append(source)

    if not exports:
        raise ValueError("notebook contains no Export Cells")
    return {path: "\n".join(parts) for path, parts in exports.items()}


def _base_checkpoint(notebook: Path) -> Path | None:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("notebook metadata must be an object")
    marker = metadata.get("agent_harness_base_checkpoint")
    if marker is None:
        return None
    if not isinstance(marker, str) or not marker:
        raise ValueError("base Checkpoint must be a non-empty directory name")
    relative = _safe_relative_path(marker)
    if len(relative.parts) != 1:
        raise ValueError("base Checkpoint must name one sibling directory")
    checkpoints = notebook.parent.parent / "checkpoints"
    base = (checkpoints / marker).resolve()
    try:
        base.relative_to(checkpoints.resolve())
    except ValueError as error:
        raise ValueError("base Checkpoint escapes course/checkpoints") from error
    if not base.is_dir():
        raise ValueError(f"base Checkpoint does not exist: {marker!r}")
    return base


def _checkpoint_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "checkpoint.json"
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        )
    )


def _assemble_checkpoint(
    notebook: Path,
    staging: Path,
    exports: dict[str, str],
) -> tuple[str, ...]:
    base = _base_checkpoint(notebook)
    if base is not None:
        shutil.copytree(
            base,
            staging,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "checkpoint.json", "__pycache__", ".pytest_cache", "*.pyc"
            ),
        )
    for relative, source in exports.items():
        target = staging.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return _checkpoint_files(staging)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_digest(notebook: Path) -> str:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    stable_cells = []
    for cell in document.get("cells", []):
        stable_cells.append(
            {
                "cell_type": cell.get("cell_type"),
                "metadata": cell.get("metadata", {}),
                "source": _cell_source(cell),
            }
        )
    stable_document = {
        "cells": stable_cells,
        "metadata": document.get("metadata", {}),
        "nbformat": document.get("nbformat"),
        "nbformat_minor": document.get("nbformat_minor"),
    }
    encoded = json.dumps(
        stable_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def _write_manifest(
    staging: Path,
    *,
    course_root: Path,
    notebook: Path,
    exported_files: tuple[str, ...],
) -> str:
    files = [
        {
            "path": relative,
            "sha256": _sha256((staging / relative).read_bytes()),
        }
        for relative in exported_files
    ]
    payload = {
        "schema_version": 1,
        "source_notebook": notebook.relative_to(course_root).as_posix(),
        "source_sha256": _source_digest(notebook),
        "files": files,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = _sha256(canonical)
    manifest = {**payload, "digest": digest}
    (staging / "checkpoint.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def _publish(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        staging.replace(destination)
        return

    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    destination.replace(backup)
    try:
        staging.replace(destination)
    except BaseException:
        backup.replace(destination)
        raise
    else:
        shutil.rmtree(backup)


def _run_gate(
    gate: str,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise CheckpointVerificationError(gate, detail or "command failed")


def _importable_packages(staging: Path) -> tuple[str, ...]:
    source_root = staging / "src"
    if not source_root.is_dir():
        raise CheckpointVerificationError("import", "checkpoint has no src directory")
    packages = tuple(
        child.name
        for child in sorted(source_root.iterdir())
        if child.is_dir() and (child / "__init__.py").is_file()
    )
    if not packages:
        raise CheckpointVerificationError("import", "checkpoint exports no packages")
    return packages


def _verify_candidate(staging: Path) -> tuple[str, ...]:
    _run_gate(
        "compile",
        [sys.executable, "-m", "compileall", "-q", str(staging)],
    )
    if not (staging / "pyproject.toml").is_file():
        raise CheckpointVerificationError("install", "pyproject.toml is missing")

    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-install-", dir=staging.parent
    ) as temporary:
        installed = Path(temporary) / "site-packages"
        python = sys.executable
        offline_env = dict(os.environ)
        offline_env.update(
            {
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "PYTHONPATH": str(installed),
            }
        )
        _run_gate(
            "install",
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--no-input",
                "--target",
                str(installed),
                str(staging),
            ],
            cwd=staging,
            env=offline_env,
        )
        packages = _importable_packages(staging)
        _run_gate(
            "import",
            [python, "-c", "; ".join(f"import {name}" for name in packages)],
            cwd=staging,
            env=offline_env,
        )
        tests = staging / "tests"
        if not tests.is_dir():
            raise CheckpointVerificationError("tests", "checkpoint has no tests directory")
        _run_gate(
            "tests",
            [python, "-m", "pytest", "-q", str(tests)],
            cwd=staging,
            env=offline_env,
        )
    return ("compile", "install", "import", "tests")


def _verify_checkpoint(staging: Path) -> tuple[str, ...]:
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-verify-", dir=staging.parent
    ) as temporary:
        candidate = Path(temporary) / "candidate"
        shutil.copytree(staging, candidate)
        return _verify_candidate(candidate)


def verify_checkpoint_package(checkpoint: str | Path) -> tuple[str, ...]:
    """Verify a published Checkpoint copy without modifying the source tree."""

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    if not (checkpoint_path / "checkpoint.json").is_file():
        raise CheckpointVerificationError(
            "manifest", "checkpoint.json is missing"
        )
    return _verify_checkpoint(checkpoint_path)


def export_checkpoint(
    notebook: str | Path,
    destination: str | Path,
    *,
    verify: bool = True,
) -> ExportResult:
    """Export tagged notebook code into a complete checkpoint directory.

    Files are assembled away from ``destination``. A failed assembly therefore
    cannot expose a partially written Checkpoint.
    """

    notebook_path = Path(notebook).resolve()
    destination_path = Path(destination).resolve()
    if notebook_path.parent.name != "notebooks":
        raise ValueError("a Chapter Notebook must be inside course/notebooks")
    course_root = notebook_path.parent.parent
    try:
        destination_path.relative_to(course_root / "checkpoints")
    except ValueError as error:
        raise ValueError("a Checkpoint must be inside course/checkpoints") from error
    exports = _collect_exports(notebook_path)
    exported_files = tuple(sorted(exports))
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination_path.name}-", dir=destination_path.parent
    ) as temporary:
        staging = Path(temporary) / destination_path.name
        exported_files = _assemble_checkpoint(notebook_path, staging, exports)

        digest = _write_manifest(
            staging,
            course_root=course_root,
            notebook=notebook_path,
            exported_files=exported_files,
        )
        gates = _verify_checkpoint(staging) if verify else ()
        _publish(staging, destination_path)

    return ExportResult(
        destination=destination_path,
        files=exported_files,
        gates=gates,
        digest=digest,
    )


def checkpoint_drift(
    notebook: str | Path,
    destination: str | Path,
) -> tuple[str, ...]:
    """Return changed, missing, or unexpected Checkpoint paths without publishing."""

    notebook_path = Path(notebook).resolve()
    destination_path = Path(destination).resolve()
    if notebook_path.parent.name != "notebooks":
        raise ValueError("a Chapter Notebook must be inside course/notebooks")
    course_root = notebook_path.parent.parent
    try:
        destination_path.relative_to(course_root / "checkpoints")
    except ValueError as error:
        raise ValueError("a Checkpoint must be inside course/checkpoints") from error

    exports = _collect_exports(notebook_path)
    with tempfile.TemporaryDirectory(prefix="checkpoint-drift-") as temporary:
        expected_root = Path(temporary)
        exported_files = _assemble_checkpoint(notebook_path, expected_root, exports)
        _write_manifest(
            expected_root,
            course_root=course_root,
            notebook=notebook_path,
            exported_files=exported_files,
        )
        expected_paths = {
            path.relative_to(expected_root).as_posix()
            for path in expected_root.rglob("*")
            if path.is_file()
        }
        actual_paths = (
            {
                path.relative_to(destination_path).as_posix()
                for path in destination_path.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".pytest_cache" not in path.parts
                and path.suffix != ".pyc"
            }
            if destination_path.is_dir()
            else set()
        )
        drift = expected_paths ^ actual_paths
        for relative in expected_paths & actual_paths:
            if (expected_root / relative).read_bytes() != (
                destination_path / relative
            ).read_bytes():
                drift.add(relative)
    return tuple(sorted(drift))


def _production_files(root: Path) -> tuple[str, ...]:
    package = root / "src" / "agent_harness"
    if not package.is_dir():
        return ()
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in package.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    )


def export_production_package(
    checkpoint: str | Path,
    repository: str | Path,
) -> tuple[str, ...]:
    """Atomically generate ``src/agent_harness`` only from a final Checkpoint."""

    checkpoint_path = Path(checkpoint).resolve()
    source = checkpoint_path / "src" / "agent_harness"
    if not source.is_dir() or not (source / "__init__.py").is_file():
        raise ValueError("final Checkpoint has no importable agent_harness package")
    repository_path = Path(repository).resolve()
    source_root = repository_path / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    target = source_root / "agent_harness"
    with tempfile.TemporaryDirectory(
        prefix=".agent-harness-production-", dir=source_root
    ) as temporary:
        staging = Path(temporary) / "agent_harness"
        shutil.copytree(
            source,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        _publish(staging, target)
    return _production_files(repository_path)


def production_drift(
    checkpoint: str | Path,
    repository: str | Path,
) -> tuple[str, ...]:
    """Return production paths that differ from ``Checkpoint/src``."""

    checkpoint_path = Path(checkpoint).resolve()
    repository_path = Path(repository).resolve()
    expected_root = checkpoint_path / "src" / "agent_harness"
    if not expected_root.is_dir():
        raise ValueError("final Checkpoint has no agent_harness package")
    actual_root = repository_path / "src" / "agent_harness"
    expected = {
        path.relative_to(expected_root).as_posix(): path
        for path in expected_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    actual = (
        {
            path.relative_to(actual_root).as_posix(): path
            for path in actual_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        if actual_root.is_dir()
        else {}
    )
    drift = []
    for relative in sorted(expected.keys() | actual.keys()):
        expected_path = expected.get(relative)
        actual_path = actual.get(relative)
        if (
            expected_path is None
            or actual_path is None
            or expected_path.read_bytes() != actual_path.read_bytes()
        ):
            drift.append(f"src/agent_harness/{relative}")
    return tuple(drift)
