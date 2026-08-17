from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness import (
    JSONProjectTrustStore,
    MemoryProjectTrust,
    PromptAssembler,
    Resource,
    ResourceLoader,
    ResourceScope,
    Tool,
    ToolResult,
)


def _write_skill(root: Path, name: str, description: str, instructions: str) -> Path:
    path = root / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n",
        encoding="utf-8",
    )
    return path


def test_resource_loader_resolves_skills_by_explicit_project_user_builtin_precedence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    user_root = tmp_path / "user-resources"
    _write_skill(user_root, "review", "user review", "USER")
    _write_skill(workspace / ".omega", "review", "project review", "PROJECT")
    trust = MemoryProjectTrust([workspace])
    explicit = Resource.skill(
        "review",
        "explicit review",
        "EXPLICIT",
        source="command-line:review",
    )
    builtin = Resource.skill(
        "review",
        "built-in review",
        "BUILTIN",
        source="builtin:review",
    )

    loaded = ResourceLoader(
        workspace,
        user_root=user_root,
        trust=trust,
        explicit=[explicit],
        builtins=[builtin],
    ).load()

    assert loaded.skills["review"].scope is ResourceScope.EXPLICIT
    assert loaded.skills["review"].description == "explicit review"
    assert loaded.skills["review"].content == "EXPLICIT"
    assert loaded.skills["review"].source == "command-line:review"
    assert loaded.evidence["skill:review"].sha256.startswith("sha256:")


def test_untrusted_project_resources_are_skipped_but_context_files_remain_independent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    working_directory = workspace / "packages" / "app"
    working_directory.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("ROOT CONTEXT", encoding="utf-8")
    (workspace / "packages" / "AGENTS.md").write_text(
        "PACKAGE CONTEXT", encoding="utf-8"
    )
    _write_skill(workspace / ".omega", "deploy", "deploy safely", "PROTECTED")
    (workspace / ".omega" / "APPEND_SYSTEM.md").write_text(
        "PROTECTED APPEND", encoding="utf-8"
    )
    (workspace / ".omega" / "extensions").mkdir()
    (workspace / ".omega" / "extensions" / "audit.py").write_text(
        "PROTECTED_EXTENSION = True\n", encoding="utf-8"
    )
    (workspace / ".omega" / "settings.json").write_text(
        '{"temperature": 0}', encoding="utf-8"
    )

    loaded = ResourceLoader(
        workspace,
        working_directory=working_directory,
        trust=MemoryProjectTrust(),
    ).load()

    assert [context.content for context in loaded.context_files] == [
        "ROOT CONTEXT",
        "PACKAGE CONTEXT",
    ]
    assert "deploy" not in loaded.skills
    assert "audit" not in loaded.extensions
    assert loaded.settings is None
    assert loaded.append_system is None
    assert any(path.endswith("SKILL.md") for path in loaded.skipped_protected)
    assert any(path.endswith("APPEND_SYSTEM.md") for path in loaded.skipped_protected)
    assert any(path.endswith("audit.py") for path in loaded.skipped_protected)
    assert any(path.endswith("settings.json") for path in loaded.skipped_protected)

    without_context = ResourceLoader(
        workspace,
        working_directory=working_directory,
        trust=MemoryProjectTrust(),
        context_enabled=False,
    ).load()
    assert without_context.context_files == ()


def test_prompt_assembly_is_ordered_and_discloses_only_active_skill_instructions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("CONTEXT BODY", encoding="utf-8")
    omega = workspace / ".omega"
    omega.mkdir()
    (omega / "APPEND_SYSTEM.md").write_text("APPENDED BODY", encoding="utf-8")
    _write_skill(omega, "review", "review changes", "ACTIVE BODY")
    _write_skill(omega, "deploy", "deploy changes", "INACTIVE BODY")
    resources = ResourceLoader(
        workspace,
        trust=MemoryProjectTrust([workspace]),
    ).load()

    async def read_file(arguments: dict[str, object]) -> ToolResult:
        return ToolResult("unused")

    tool = Tool(
        "read",
        "Read a workspace text file",
        {"type": "object", "properties": {"path": {"type": "string"}}},
        read_file,
    )
    assembled = PromptAssembler("BUILT-IN BODY").assemble(
        tools=[tool], resources=resources, active_skills=["review"]
    )

    positions = [
        assembled.text.index(marker)
        for marker in (
            "BUILT-IN BODY",
            "## Active Tools",
            "## Available Skills",
            "## Active Skill: review",
            "## Context File:",
            "## Appended System Instructions",
        )
    ]
    assert positions == sorted(positions)
    assert "review changes" in assembled.text
    assert "deploy changes" in assembled.text
    assert "ACTIVE BODY" in assembled.text
    assert "INACTIVE BODY" not in assembled.text
    assert assembled.prompt_hashes["effective_system"].startswith("sha256:")
    assert set(assembled.resource_hashes) == set(resources.evidence)


def test_trusted_system_replacement_replaces_the_entire_default_assembly(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    omega = workspace / ".omega"
    omega.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("CONTEXT", encoding="utf-8")
    (omega / "SYSTEM.md").write_text("REPLACEMENT BODY", encoding="utf-8")
    (omega / "APPEND_SYSTEM.md").write_text("MUST NOT APPEND", encoding="utf-8")
    _write_skill(omega, "review", "review changes", "SKILL BODY")
    resources = ResourceLoader(
        workspace,
        trust=MemoryProjectTrust([workspace]),
    ).load()

    assembled = PromptAssembler("BUILT-IN BODY").assemble(
        tools=[], resources=resources
    )

    assert assembled.text == "REPLACEMENT BODY\n"
    assert set(assembled.resource_hashes) == {"system:replacement"}


def test_loader_discovers_only_local_prompt_and_extension_sources_with_precedence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    user_root = tmp_path / "user-resources"
    for root, prompt, extension in (
        (user_root, "USER {target}", "USER_EXTENSION = True\n"),
        (workspace / ".omega", "PROJECT {target}", "PROJECT_EXTENSION = True\n"),
    ):
        (root / "prompts").mkdir(parents=True)
        (root / "prompts" / "review.md").write_text(prompt, encoding="utf-8")
        (root / "extensions").mkdir()
        (root / "extensions" / "audit.py").write_text(extension, encoding="utf-8")
    (workspace / ".omega" / "settings.json").write_text(
        '{"temperature": 0}', encoding="utf-8"
    )

    loaded = ResourceLoader(
        workspace,
        user_root=user_root,
        trust=MemoryProjectTrust([workspace]),
    ).load()

    assert loaded.prompt_templates["review"].scope is ResourceScope.PROJECT
    assert loaded.expand_prompt("review", {"target": "app.py"}) == "PROJECT app.py"
    assert loaded.extensions["audit"].scope is ResourceScope.PROJECT
    assert loaded.settings is not None
    assert set(loaded.evidence) >= {
        "prompt:review",
        "extension:audit",
        "settings:project",
    }


def test_project_trust_is_persisted_against_the_canonical_workspace_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    trust_file = tmp_path / "omega-data" / "trusted-projects.json"
    store = JSONProjectTrustStore(trust_file)

    store.approve(workspace / ".." / "project")

    reloaded = JSONProjectTrustStore(trust_file)
    assert reloaded.is_trusted(workspace.resolve()) is True
    assert str(workspace.resolve()) in trust_file.read_text(encoding="utf-8")


def test_related_skill_resources_enter_context_only_after_activation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    skill = _write_skill(
        workspace / ".omega", "review", "review changes", "Read the checklist."
    )
    reference = skill.parent / "references" / "checklist.md"
    reference.parent.mkdir()
    reference.write_text("CHECK GENERATED FILES", encoding="utf-8")
    resources = ResourceLoader(
        workspace,
        trust=MemoryProjectTrust([workspace]),
    ).load()
    assembler = PromptAssembler("BUILT IN")

    inactive = assembler.assemble(tools=[], resources=resources)
    active = assembler.assemble(
        tools=[], resources=resources, active_skills=["review"]
    )

    assert "CHECK GENERATED FILES" not in inactive.text
    assert "CHECK GENERATED FILES" in active.text
    assert "references/checklist.md" in active.text


def test_local_resource_discovery_rejects_scoped_symlink_escape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    prompts = workspace / ".omega" / "prompts"
    prompts.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("OUTSIDE", encoding="utf-8")
    try:
        (prompts / "escaped.md").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="resource escapes its scope"):
        ResourceLoader(
            workspace,
            trust=MemoryProjectTrust([workspace]),
        ).load()

def test_project_trust_is_persisted_against_the_canonical_workspace_path(
    tmp_path: Path,
) -> None:
    import json

    workspace = tmp_path / "project"
    workspace.mkdir()
    trust_file = tmp_path / "omega-data" / "trusted-projects.json"
    store = JSONProjectTrustStore(trust_file)

    store.approve(workspace / ".." / "project")

    reloaded = JSONProjectTrustStore(trust_file)
    payload = json.loads(trust_file.read_text(encoding="utf-8"))
    assert reloaded.is_trusted(workspace.resolve()) is True
    assert payload["trusted_paths"] == [str(workspace.resolve())]
