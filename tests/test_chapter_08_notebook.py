from __future__ import annotations

from pathlib import Path

from course.tools.checkpoint import checkpoint_drift
from course.tools.notebook import execute_notebook, validate_chapter_notebook


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "course" / "notebooks" / "08_coding_agent.ipynb"
CHECKPOINT = ROOT / "course" / "checkpoints" / "ch08"


def test_chapter_08_is_structured_and_runs_in_a_clean_kernel() -> None:
    report = validate_chapter_notebook(CHAPTER)

    assert report.sections == (
        "Goal and Previous Limitation",
        "Conceptual Model",
        "Minimal Execution",
        "Staged Construction",
        "Observable Trace",
        "Failure Boundaries and Trade-offs",
        "Checkpoint Export and Verification",
        "Public API Summary",
    )
    assert report.export_cells >= 9

    executed = execute_notebook(CHAPTER, cwd=ROOT)
    assert all(
        output.get("output_type") != "error"
        for cell in executed["cells"]
        for output in cell.get("outputs", [])
    )


def test_chapter_08_checkpoint_has_no_export_drift() -> None:
    assert checkpoint_drift(CHAPTER, CHECKPOINT) == ()
