from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest

from agent_harness import (
    BashOperations,
    CompleteOutputKind,
    FileArtifactStore,
    ToolErrorCode,
    create_coding_tool_preset,
)


def test_file_tools_reject_parent_traversal_outside_the_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preset = create_coding_tool_preset(workspace)

    result = asyncio.run(
        preset.tool("write").execute(
            {"path": "../escaped.txt", "content": "must not escape"}
        )
    )

    assert result.is_error is True
    assert result.error_code is ToolErrorCode.EXECUTION_FAILED
    assert "Workspace Boundary" in result.content
    assert not (tmp_path / "escaped.txt").exists()


def test_file_tools_reject_symlink_escape_and_unsafe_nonexistent_parent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    try:
        (workspace / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    preset = create_coding_tool_preset(workspace)

    read_result = asyncio.run(
        preset.tool("read").execute({"path": "linked/secret.txt"})
    )
    write_result = asyncio.run(
        preset.tool("write").execute(
            {"path": "linked/new/deep.txt", "content": "escape"}
        )
    )

    assert read_result.error_code is ToolErrorCode.EXECUTION_FAILED
    assert write_result.error_code is ToolErrorCode.EXECUTION_FAILED
    assert not (outside / "new" / "deep.txt").exists()


def test_write_and_edit_reject_binary_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binary = workspace / "image.bin"
    original = b"\x00\xff\x10binary"
    binary.write_bytes(original)
    preset = create_coding_tool_preset(workspace)

    write_result = asyncio.run(
        preset.tool("write").execute({"path": "image.bin", "content": "text"})
    )
    edit_result = asyncio.run(
        preset.tool("edit").execute(
            {"path": "image.bin", "old_text": "binary", "new_text": "text"}
        )
    )

    assert write_result.error_code is ToolErrorCode.EXECUTION_FAILED
    assert edit_result.error_code is ToolErrorCode.EXECUTION_FAILED
    assert binary.read_bytes() == original


def test_text_edit_commits_atomically_without_losing_file_permissions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("VALUE = 'old'\n", encoding="utf-8")
    source.chmod(0o640)
    original_mode = stat.S_IMODE(source.stat().st_mode)
    preset = create_coding_tool_preset(workspace)

    result = asyncio.run(
        preset.tool("edit").execute(
            {"path": "app.py", "old_text": "old", "new_text": "new"}
        )
    )

    assert result.is_error is False
    assert source.read_text(encoding="utf-8") == "VALUE = 'new'\n"
    assert stat.S_IMODE(source.stat().st_mode) == original_mode
    assert list(workspace.glob(".*.omega-tmp")) == []


def test_bash_uses_fixed_workspace_and_filters_sensitive_environment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    operations = BashOperations(
        workspace,
        environment={
            "PATH": os.environ["PATH"],
            "SAFE_VALUE": "visible",
            "API_KEY": "hidden",
            "AWS_ACCESS_KEY_ID": "also-hidden",
            "ALLOWED_TOKEN": "deliberate",
        },
        allow_sensitive=["ALLOWED_TOKEN"],
    )

    result = asyncio.run(
        operations.run(
            "printf '%s|%s|%s|%s\\n' \"$SAFE_VALUE\" "
            "\"${API_KEY-unset}\" \"${AWS_ACCESS_KEY_ID-unset}\" "
            "\"$ALLOWED_TOKEN\"; pwd"
        )
    )

    lines = result.stdout.splitlines()
    assert lines[0] == "visible|unset|unset|deliberate"
    assert Path(lines[1]).resolve() == workspace.resolve()
    assert result.returncode == 0


def test_bash_timeout_and_cancellation_settle_the_host_process_promptly(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    operations = BashOperations(workspace)
    preset = create_coding_tool_preset(workspace, bash_operations=operations)

    timed_out = asyncio.run(
        preset.tool("bash").execute(
            {"command": "sleep 30", "timeout_seconds": 0.05}
        )
    )

    async def cancel_running_command() -> None:
        task = asyncio.create_task(operations.run("sleep 30"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(cancel_running_command())
    assert timed_out.error_code is ToolErrorCode.EXECUTION_FAILED
    assert "exceeded" in timed_out.content


def test_large_bash_output_can_be_retained_as_a_sanitized_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = FileArtifactStore(
        tmp_path / "artifacts",
        redact=lambda text: text.replace("secret", "[REDACTED]"),
    )
    preset = create_coding_tool_preset(
        workspace,
        artifact_store=artifacts,
        artifact_threshold_bytes=8,
    )

    result = asyncio.run(
        preset.tool("bash").execute({"command": "printf 'secret-value'"})
    )

    assert result.complete_output is not None
    assert result.complete_output.kind is CompleteOutputKind.ARTIFACT
    reference = result.complete_output.reference
    assert reference is not None and reference.startswith("sha256:")
    assert artifacts.read_text(reference) == "[REDACTED]-value"


def test_replaceable_bash_backend_must_keep_the_preset_fixed_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()

    with pytest.raises(ValueError, match="BashOperations.*workspace"):
        create_coding_tool_preset(
            workspace,
            bash_operations=BashOperations(other),
        )

def test_bash_uses_fixed_workspace_and_filters_sensitive_environment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    operations = BashOperations(
        workspace,
        environment={
            "PATH": os.environ["PATH"],
            "SAFE_VALUE": "visible",
            "API_KEY": "hidden",
            "AWS_ACCESS_KEY_ID": "also-hidden",
            "ALLOWED_TOKEN": "deliberate",
        },
        allow_sensitive=["ALLOWED_TOKEN"],
    )

    result = asyncio.run(
        operations.run(
            "printf '%s|%s|%s|%s' \"$SAFE_VALUE\" "
            "\"${API_KEY-unset}\" \"${AWS_ACCESS_KEY_ID-unset}\" "
            "\"$ALLOWED_TOKEN\"; printf 'fixed' > cwd-marker.txt"
        )
    )

    assert result.stdout == "visible|unset|unset|deliberate"
    assert (workspace / "cwd-marker.txt").read_text(encoding="utf-8") == "fixed"
    assert result.returncode == 0
