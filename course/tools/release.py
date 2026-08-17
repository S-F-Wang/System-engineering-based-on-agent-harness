"""Version-one Release Gate for the authoritative nine-chapter course."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sys
import tempfile
import tomllib

from .checkpoint import (
    checkpoint_drift,
    export_production_package,
    production_drift,
    verify_checkpoint_package,
)
from .notebook import execute_notebook, validate_chapter_notebook


CHAPTER_NOTEBOOKS = (
    "01_model_boundary.ipynb",
    "02_async_runtime.ipynb",
    "03_structured_tools.ipynb",
    "04_run_control.ipynb",
    "05_durable_sessions.ipynb",
    "06_context_compaction.ipynb",
    "07_extensions.ipynb",
    "08_coding_agent.ipynb",
    "09_interfaces_and_release.ipynb",
)
RUNTIME_DEPENDENCIES = ("jsonschema", "openai", "platformdirs")
V1_EXCLUSIONS = (
    "full TUI",
    "RPC/server mode",
    "MCP",
    "subagents",
    "multimodal content",
    "optimizer",
    "remote Extension installation",
    "model catalog",
    "built-in sandbox",
)
_RELEASE_TEST_CATALOG = (
    (
        "classified retries",
        "test_runtime.py",
        (
            "test_transient_failures_retry_on_the_default_two_four_eight_schedule",
            "test_runtime_classifies_retryability_from_normalized_error_codes",
        ),
    ),
    (
        "partial-stream failure",
        "test_runtime.py",
        (
            "test_provider_failure_after_acceptance_settles_partial_assistant_outcome",
        ),
    ),
    (
        "cancellation settlement",
        "test_run_control.py",
        ("test_cancellation_settles_parallel_tools_and_returns_unconsumed_input",),
    ),
    (
        "Session Lock contention",
        "test_durable_sessions.py",
        ("test_session_writer_lease_contends_across_processes",),
    ),
    (
        "path and link escape attempts",
        "test_coding_tools.py",
        (
            "test_file_tools_reject_parent_traversal_outside_the_workspace",
            "test_file_tools_reject_symlink_escape_and_unsafe_nonexistent_parent",
        ),
    ),
    (
        "atomic text mutation",
        "test_coding_tools.py",
        ("test_text_edit_commits_atomically_without_losing_file_permissions",),
    ),
    (
        "settled-boundary crash recovery",
        "test_durable_sessions.py",
        (
            "test_incomplete_final_jsonl_record_is_reported_and_not_replayed",
            "test_continuation_discards_an_uncommitted_tail_before_appending",
        ),
    ),
    (
        "Compaction",
        "test_compaction.py",
        (
            "test_manual_compaction_persists_a_checkpoint_and_changes_only_model_context",
        ),
    ),
    (
        "JSONL contracts",
        "test_cli.py",
        ("test_exec_jsonl_emits_ordered_events_and_mandatory_run_end",),
    ),
    (
        "trace redaction",
        "test_observability.py",
        ("test_artifact_redaction_preserves_full_sanitized_output",),
    ),
    (
        "output-mode contracts",
        "test_cli.py",
        (
            "test_exec_json_uses_flag_precedence_and_persists_by_default",
            "test_exec_jsonl_emits_ordered_events_and_mandatory_run_end",
            "test_chat_runs_a_scripted_multi_turn_terminal_session",
        ),
    ),
    (
        "persisted-version migration",
        "test_durable_sessions.py",
        (
            "test_session_schema_accepts_optional_fields_and_rejects_unknown_major",
            "test_migration_validates_a_new_file_and_preserves_the_original",
        ),
    ),
    (
        "credential-gated real endpoint smoke",
        "test_real_endpoint_smoke.py",
        ("test_explicit_real_openai_compatible_endpoint",),
    ),
    (
        "version-one exclusions",
        "test_chapter_09_contract.py",
        ("test_checkpoint_documentation_locks_privacy_output_and_v1_exclusions",),
    ),
)


class ReleaseGateError(RuntimeError):
    """Raised when repository evidence cannot satisfy the Release Gate."""


@dataclass(frozen=True, slots=True)
class ChapterArtifact:
    """One Chapter Notebook and its cumulative Checkpoint Package."""

    number: int
    notebook: Path
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class ReleaseTestEvidence:
    """Chapter 9 regression evidence for one required release behavior."""

    requirement: str
    path: Path
    tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseInspection:
    """Stable description of the inputs accepted by the version-one gate."""

    root: Path
    chapters: tuple[ChapterArtifact, ...]
    production_source: Path
    runtime_dependencies: tuple[str, ...]
    historical_evidence: tuple[Path, ...]
    exclusions: tuple[str, ...]
    release_tests: tuple[ReleaseTestEvidence, ...]


@dataclass(frozen=True, slots=True)
class ChapterVerification:
    """Completed offline gates for one Chapter artifact pair."""

    number: int
    notebook: str
    checkpoint_gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseVerification:
    """Evidence returned only after the complete Release Gate passes."""

    inspection: ReleaseInspection
    chapters: tuple[ChapterVerification, ...]
    production_digest: str


def _dependency_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    if match is None:
        raise ReleaseGateError(f"invalid runtime dependency: {requirement!r}")
    return match.group(0).lower().replace("_", "-")


def _project_contract(root: Path) -> tuple[str, ...]:
    pyproject_path = root / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ReleaseGateError(f"cannot read project metadata: {error}") from error
    if project.get("requires-python") != ">=3.11":
        raise ReleaseGateError("version one must require Python >=3.11")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise ReleaseGateError("project.dependencies must be a string list")
    names = tuple(sorted(_dependency_name(item) for item in dependencies))
    if names != RUNTIME_DEPENDENCIES:
        raise ReleaseGateError(
            "runtime dependency boundary changed: "
            f"expected {RUNTIME_DEPENDENCIES!r}, found {names!r}"
        )
    return names


def _scope_contract(root: Path) -> None:
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseGateError(f"cannot read README.md: {error}") from error
    missing = tuple(item for item in V1_EXCLUSIONS if item not in readme)
    if missing:
        raise ReleaseGateError(f"README does not lock v1 exclusions: {missing!r}")
    if (
        "historical evidence" not in readme
        or "not production or export inputs" not in readme
    ):
        raise ReleaseGateError(
            "README must mark historical notebooks as non-authoritative evidence"
        )


def _release_test_evidence(root: Path) -> tuple[ReleaseTestEvidence, ...]:
    tests_root = root / "course" / "checkpoints" / "ch09" / "tests"
    evidence = tuple(
        ReleaseTestEvidence(
            requirement=requirement,
            path=tests_root / filename,
            tests=tests,
        )
        for requirement, filename, tests in _RELEASE_TEST_CATALOG
    )
    for item in evidence:
        try:
            source = item.path.read_text(encoding="utf-8")
        except OSError as error:
            raise ReleaseGateError(
                f"release evidence is missing for {item.requirement}: {error}"
            ) from error
        missing = tuple(name for name in item.tests if f"def {name}(" not in source)
        if missing:
            raise ReleaseGateError(
                f"release evidence changed for {item.requirement}: {missing!r}"
            )
    smoke_evidence = next(
        item
        for item in evidence
        if item.requirement == "credential-gated real endpoint smoke"
    )
    smoke = smoke_evidence.path.read_text(encoding="utf-8")
    for marker in (
        "AGENT_HARNESS_REAL_SMOKE",
        "AGENT_HARNESS_BASE_URL",
        "AGENT_HARNESS_API_KEY",
        "AGENT_HARNESS_MODEL",
        "pytest.mark.skipif",
    ):
        if marker not in smoke:
            raise ReleaseGateError(
                f"real endpoint smoke lost its explicit gate: {marker}"
            )
    return evidence


def inspect_release(repository: str | Path) -> ReleaseInspection:
    """Inspect and validate the one authoritative version-one artifact chain."""

    root = Path(repository).resolve()
    notebook_root = root / "course" / "notebooks"
    checkpoint_root = root / "course" / "checkpoints"
    actual_notebooks = tuple(path.name for path in sorted(notebook_root.glob("*.ipynb")))
    if actual_notebooks != CHAPTER_NOTEBOOKS:
        raise ReleaseGateError(
            f"authoritative Chapter Notebook set changed: {actual_notebooks!r}"
        )
    actual_checkpoints = (
        tuple(
            path.name for path in sorted(checkpoint_root.iterdir()) if path.is_dir()
        )
        if checkpoint_root.is_dir()
        else ()
    )
    expected_checkpoints = tuple(f"ch{number:02d}" for number in range(1, 10))
    if actual_checkpoints != expected_checkpoints:
        raise ReleaseGateError(
            f"cumulative Checkpoint set changed: {actual_checkpoints!r}"
        )

    chapters = tuple(
        ChapterArtifact(
            number=number,
            notebook=notebook_root / notebook_name,
            checkpoint=checkpoint_root / f"ch{number:02d}",
        )
        for number, notebook_name in enumerate(CHAPTER_NOTEBOOKS, start=1)
    )
    for chapter in chapters:
        try:
            validate_chapter_notebook(chapter.notebook)
        except (OSError, ValueError) as error:
            raise ReleaseGateError(
                f"Chapter {chapter.number} structure failed: {error}"
            ) from error
        drift = checkpoint_drift(chapter.notebook, chapter.checkpoint)
        if drift:
            raise ReleaseGateError(
                f"Chapter {chapter.number} Checkpoint drift: {', '.join(drift)}"
            )

    production_source = checkpoint_root / "ch09"
    drift = production_drift(production_source, root)
    if drift:
        raise ReleaseGateError(f"production package drift: {', '.join(drift)}")

    dependencies = _project_contract(root)
    _scope_contract(root)
    historical_evidence = (
        root / "notebooks" / "lessons",
        root / "notebooks" / "raw",
    )
    missing_evidence = tuple(path for path in historical_evidence if not path.is_dir())
    if missing_evidence:
        raise ReleaseGateError(
            f"historical evidence is missing: {', '.join(map(str, missing_evidence))}"
        )

    return ReleaseInspection(
        root=root,
        chapters=chapters,
        production_source=production_source,
        runtime_dependencies=dependencies,
        historical_evidence=historical_evidence,
        exclusions=V1_EXCLUSIONS,
        release_tests=_release_test_evidence(root),
    )


def _production_digest(root: Path, files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_deterministic_production(inspection: ReleaseInspection) -> str:
    with tempfile.TemporaryDirectory(prefix="release-production-") as temporary:
        first_root = Path(temporary) / "first"
        second_root = Path(temporary) / "second"
        first_files = export_production_package(
            inspection.production_source,
            first_root,
        )
        second_files = export_production_package(
            inspection.production_source,
            second_root,
        )
        if first_files != second_files:
            raise ReleaseGateError("production generation returned unstable file sets")
        first_digest = _production_digest(first_root, first_files)
        second_digest = _production_digest(second_root, second_files)
        if first_digest != second_digest:
            raise ReleaseGateError("production generation is not deterministic")
        return first_digest


def verify_release(
    repository: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> ReleaseVerification:
    """Run the complete offline Release Gate in narrative Chapter order."""

    if sys.version_info < (3, 11):
        raise ReleaseGateError("the version-one Release Gate requires Python 3.11+")
    notify = progress or (lambda _message: None)
    inspection = inspect_release(repository)
    notify("static artifact, drift, dependency, scope, and evidence checks passed")

    verified: list[ChapterVerification] = []
    for chapter in inspection.chapters:
        notify(f"Chapter {chapter.number}: executing clean offline kernel")
        execute_notebook(chapter.notebook, cwd=inspection.root, timeout=600)
        notify(f"Chapter {chapter.number}: verifying cumulative Checkpoint")
        gates = verify_checkpoint_package(chapter.checkpoint)
        verified.append(
            ChapterVerification(
                number=chapter.number,
                notebook=chapter.notebook.name,
                checkpoint_gates=gates,
            )
        )

    notify("regenerating the production package twice")
    production_digest = _verify_deterministic_production(inspection)
    notify("complete version-one Release Gate passed")
    return ReleaseVerification(
        inspection=inspection,
        chapters=tuple(verified),
        production_digest=production_digest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Release Gate command."""

    parser = argparse.ArgumentParser(
        description="verify the nine-chapter version-one artifact chain"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="run only artifact, drift, dependency, scope, and evidence checks",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.static:
            inspection = inspect_release(arguments.root)
            print(f"Release inspection passed: {len(inspection.chapters)} chapters")
            print(
                "production source: "
                f"{inspection.production_source.relative_to(inspection.root).as_posix()}"
            )
            return 0
        verification = verify_release(
            arguments.root,
            progress=lambda line: print(line, flush=True),
        )
    except Exception as error:
        print(f"Release Gate failed: {error}", file=sys.stderr)
        return 1

    print(f"verified chapters: {len(verification.chapters)}")
    print(f"production digest: {verification.production_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
