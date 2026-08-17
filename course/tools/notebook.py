"""Structural and clean-kernel gates for Chapter Notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from nbclient import NotebookClient
import nbformat


REQUIRED_SECTIONS = (
    "Goal and Previous Limitation",
    "Conceptual Model",
    "Minimal Execution",
    "Staged Construction",
    "Observable Trace",
    "Failure Boundaries and Trade-offs",
    "Checkpoint Export and Verification",
    "Public API Summary",
)


@dataclass(frozen=True, slots=True)
class NotebookReport:
    sections: tuple[str, ...]
    export_cells: int


def validate_chapter_notebook(path: str | Path) -> NotebookReport:
    """Validate the stable Chapter Template without executing the notebook."""

    notebook_path = Path(path)
    notebook = nbformat.read(notebook_path, as_version=4)
    headings: list[str] = []
    export_cells = 0
    all_source: list[str] = []

    for cell in notebook.cells:
        source = str(cell.source)
        all_source.append(source)
        if cell.cell_type == "markdown":
            headings.extend(
                match.group(1).strip()
                for match in re.finditer(r"^##\s+(.+?)\s*$", source, re.MULTILINE)
            )
        marker = cell.metadata.get("agent_harness_export")
        if marker is not None:
            if cell.cell_type != "code":
                raise ValueError("Export Cells must be code cells")
            export_cells += 1

    positions: list[int] = []
    for required in REQUIRED_SECTIONS:
        try:
            positions.append(headings.index(required))
        except ValueError as error:
            raise ValueError(f"missing Chapter Template section: {required}") from error
    if positions != sorted(positions):
        raise ValueError("Chapter Template sections are out of narrative order")

    combined = "\n".join(all_source)
    has_todo = re.search(r"\bTODOs?\b", combined, re.IGNORECASE)
    has_exercise_section = any(
        re.search(r"\bexercises?\b", heading, re.IGNORECASE) for heading in headings
    )
    if has_todo or has_exercise_section:
        raise ValueError("Chapter Notebooks cannot contain TODOs or exercises")
    historical_inputs = ("notebooks/raw", "notebooks/lessons", "mini_harness.ipynb")
    if any(token in combined for token in historical_inputs):
        raise ValueError("Chapter Notebook refers to a historical course input")

    return NotebookReport(sections=REQUIRED_SECTIONS, export_cells=export_cells)


def execute_notebook(
    path: str | Path,
    *,
    cwd: str | Path,
    timeout: int = 300,
) -> Any:
    """Execute a Chapter Notebook in a new kernel and return the in-memory copy."""

    notebook = nbformat.read(Path(path), as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
    )
    return client.execute(cwd=str(Path(cwd).resolve()))
