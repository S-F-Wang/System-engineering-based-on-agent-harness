from __future__ import annotations

import json
from pathlib import Path

import pytest

from course.tools.checkpoint import CheckpointVerificationError, export_checkpoint


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.11"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _export_cell(path: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"agent_harness_export": {"path": path}},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _literal_export_cell(path: str, content: str) -> dict[str, object]:
    cell = _export_cell(path, repr(content))
    cell["metadata"] = {
        "agent_harness_export": {"path": path, "mode": "literal"}
    }
    return cell


def _assignment_export_cell(path: str, content: str) -> dict[str, object]:
    cell = _export_cell(path, f"CHECKPOINT_SOURCE = {content!r}\n")
    cell["metadata"] = {
        "agent_harness_export": {"path": path, "mode": "assignment"}
    }
    return cell


def test_export_publishes_only_tagged_code(tmp_path: Path) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    _write_notebook(
        notebook,
        [
            _markdown("NARRATIVE_SENTINEL must never be exported."),
            _export_cell("src/example/__init__.py", 'VALUE = "from notebook"\n'),
        ],
    )

    result = export_checkpoint(notebook, destination, verify=False)

    exported = destination / "src" / "example" / "__init__.py"
    assert exported.read_text(encoding="utf-8") == 'VALUE = "from notebook"\n'
    assert "NARRATIVE_SENTINEL" not in exported.read_text(encoding="utf-8")
    assert result.files == ("src/example/__init__.py",)


def test_failed_compile_preserves_published_checkpoint(tmp_path: Path) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    destination.mkdir(parents=True)
    existing = destination / "published.txt"
    existing.write_bytes(b"previous checkpoint\n")
    _write_notebook(
        notebook,
        [_export_cell("src/example/__init__.py", "this is not valid Python !\n")],
    )

    with pytest.raises(CheckpointVerificationError, match="compile"):
        export_checkpoint(notebook, destination)

    assert existing.read_bytes() == b"previous checkpoint\n"
    assert list(destination.iterdir()) == [existing]


def test_literal_export_cell_writes_non_python_content(tmp_path: Path) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    pyproject = '[project]\nname = "example"\nversion = "1.0.0"\n'
    _write_notebook(notebook, [_literal_export_cell("pyproject.toml", pyproject)])

    export_checkpoint(notebook, destination, verify=False)

    assert (destination / "pyproject.toml").read_text(encoding="utf-8") == pyproject


def test_assignment_export_cell_is_executable_and_exports_its_string(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    source = 'VALUE = "teachable source"\n'
    cell = _assignment_export_cell("src/example/__init__.py", source)
    _write_notebook(notebook, [cell])

    compile("".join(cell["source"]), str(notebook), "exec")
    export_checkpoint(notebook, destination, verify=False)

    assert (destination / "src/example/__init__.py").read_text() == source


def test_verification_installs_imports_and_runs_checkpoint_tests(
    tmp_path: Path,
) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    pyproject = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "example-checkpoint"
version = "1.0.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
"""
    checkpoint_test = """\
import importlib.metadata

import example


def test_installed_distribution_is_importable():
    assert example.VALUE == "verified"
    assert importlib.metadata.version("example-checkpoint") == "1.0.0"
"""
    _write_notebook(
        notebook,
        [
            _literal_export_cell("pyproject.toml", pyproject),
            _export_cell("src/example/__init__.py", 'VALUE = "verified"\n'),
            _export_cell("tests/test_checkpoint.py", checkpoint_test),
        ],
    )

    result = export_checkpoint(notebook, destination)

    assert result.gates == ("compile", "install", "import", "tests")
    assert (destination / "tests" / "test_checkpoint.py").is_file()
    published_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert published_files == {
        "checkpoint.json",
        "pyproject.toml",
        "src/example/__init__.py",
        "tests/test_checkpoint.py",
    }


def test_repeated_export_has_a_deterministic_source_manifest(tmp_path: Path) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    _write_notebook(
        notebook,
        [
            _markdown("A stable narrative."),
            _export_cell("src/example/__init__.py", "VALUE = 1\n"),
        ],
    )

    first = export_checkpoint(notebook, destination, verify=False)
    first_manifest = (destination / "checkpoint.json").read_bytes()
    second = export_checkpoint(notebook, destination, verify=False)
    second_manifest = (destination / "checkpoint.json").read_bytes()

    manifest = json.loads(first_manifest)
    assert first_manifest == second_manifest
    assert first.digest == second.digest == manifest["digest"]
    assert manifest["source_notebook"] == "notebooks/01.ipynb"
    assert str(tmp_path) not in first_manifest.decode()
    assert manifest["files"][0]["path"] == "src/example/__init__.py"


def test_failed_import_preserves_published_checkpoint(tmp_path: Path) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    destination.mkdir(parents=True)
    existing = destination / "published.txt"
    existing.write_bytes(b"previous checkpoint\n")
    pyproject = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "broken-import"
version = "1.0.0"
[tool.setuptools.packages.find]
where = ["src"]
"""
    _write_notebook(
        notebook,
        [
            _literal_export_cell("pyproject.toml", pyproject),
            _export_cell(
                "src/broken_import/__init__.py",
                'raise RuntimeError("cannot import")\n',
            ),
            _export_cell("tests/test_placeholder.py", "def test_placeholder(): pass\n"),
        ],
    )

    with pytest.raises(CheckpointVerificationError) as captured:
        export_checkpoint(notebook, destination)

    assert captured.value.gate == "import"
    assert existing.read_bytes() == b"previous checkpoint\n"


def test_failed_checkpoint_test_preserves_published_checkpoint(tmp_path: Path) -> None:
    course = tmp_path / "course"
    notebook = course / "notebooks" / "01.ipynb"
    destination = course / "checkpoints" / "ch01"
    destination.mkdir(parents=True)
    existing = destination / "published.txt"
    existing.write_bytes(b"previous checkpoint\n")
    pyproject = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "failing-tests"
version = "1.0.0"
[tool.setuptools.packages.find]
where = ["src"]
"""
    _write_notebook(
        notebook,
        [
            _literal_export_cell("pyproject.toml", pyproject),
            _export_cell("src/failing_tests/__init__.py", "VALUE = 1\n"),
            _export_cell(
                "tests/test_failure.py",
                'def test_failure():\n    assert False, "expected failure"\n',
            ),
        ],
    )

    with pytest.raises(CheckpointVerificationError) as captured:
        export_checkpoint(notebook, destination)

    assert captured.value.gate == "tests"
    assert existing.read_bytes() == b"previous checkpoint\n"
