"""Trusted, deterministic local resources for the optional Coding Agent layer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Protocol

from .tools import Tool


_RESOURCE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


class ResourceScope(str, Enum):
    EXPLICIT = "explicit"
    PROJECT = "project"
    USER = "user"
    BUILTIN = "builtin"


class ResourceKind(str, Enum):
    SKILL = "skill"
    CONTEXT = "context"
    SYSTEM = "system"
    APPEND_SYSTEM = "append_system"
    PROMPT_TEMPLATE = "prompt_template"
    EXTENSION = "extension"
    SETTINGS = "settings"


@dataclass(frozen=True, slots=True)
class Resource:
    kind: ResourceKind
    name: str
    description: str
    content: str
    source: str
    scope: ResourceScope = ResourceScope.EXPLICIT
    related_resources: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def skill(
        cls,
        name: str,
        description: str,
        content: str,
        *,
        source: str,
        related_resources: Mapping[str, str] | None = None,
    ) -> "Resource":
        return cls(
            ResourceKind.SKILL,
            name,
            description,
            content,
            source,
            related_resources=related_resources or {},
        )

    @classmethod
    def prompt_template(
        cls, name: str, content: str, *, source: str
    ) -> "Resource":
        return cls(
            ResourceKind.PROMPT_TEMPLATE,
            name,
            f"Local Prompt Template {name}",
            content,
            source,
        )

    @classmethod
    def extension_source(
        cls, name: str, content: str, *, source: str
    ) -> "Resource":
        return cls(
            ResourceKind.EXTENSION,
            name,
            f"Local Extension source {name}",
            content,
            source,
        )

    def __post_init__(self) -> None:
        if self.kind in {
            ResourceKind.SKILL,
            ResourceKind.PROMPT_TEMPLATE,
            ResourceKind.EXTENSION,
        } and not _RESOURCE_NAME.fullmatch(self.name):
            raise ValueError("resource names must be safe lowercase identifiers")
        if not self.description.strip():
            raise ValueError("resources require a description")
        if not isinstance(self.content, str):
            raise TypeError("resource content must be text")
        if not self.source.strip():
            raise ValueError("resources require a source")
        if any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(content, str)
            for name, content in self.related_resources.items()
        ):
            raise ValueError("related Skill resources require safe relative text paths")
        object.__setattr__(
            self,
            "related_resources",
            MappingProxyType(dict(sorted(self.related_resources.items()))),
        )


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    kind: ResourceKind
    scope: ResourceScope
    source: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedResources:
    skills: Mapping[str, Resource]
    evidence: Mapping[str, ResourceEvidence]
    context_files: tuple[Resource, ...] = ()
    system_replacement: Resource | None = None
    append_system: Resource | None = None
    prompt_templates: Mapping[str, Resource] = field(default_factory=dict)
    extensions: Mapping[str, Resource] = field(default_factory=dict)
    settings: Resource | None = None
    skipped_protected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "skills", MappingProxyType(dict(self.skills)))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(
            self, "prompt_templates", MappingProxyType(dict(self.prompt_templates))
        )
        object.__setattr__(self, "extensions", MappingProxyType(dict(self.extensions)))

    def expand_prompt(self, name: str, arguments: Mapping[str, str]) -> str:
        try:
            template = self.prompt_templates[name]
        except KeyError:
            raise KeyError(f"unknown Prompt Template: {name}") from None
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in arguments.items()
        ):
            raise TypeError("Prompt Template arguments must map text names to text values")
        try:
            return template.content.format_map(dict(arguments))
        except KeyError as error:
            raise ValueError(
                f"Prompt Template {name!r} requires argument {error.args[0]!r}"
            ) from None


@dataclass(frozen=True, slots=True)
class PromptAssembly:
    text: str
    prompt_hashes: Mapping[str, str]
    resource_hashes: Mapping[str, str]
    active_skills: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prompt_hashes", MappingProxyType(dict(self.prompt_hashes))
        )
        object.__setattr__(
            self, "resource_hashes", MappingProxyType(dict(self.resource_hashes))
        )


class PromptAssembler:
    """Assemble bounded system context in one documented deterministic order."""

    def __init__(self, builtin_prompt: str) -> None:
        if not builtin_prompt.strip():
            raise ValueError("the built-in prompt cannot be empty")
        self._builtin_prompt = builtin_prompt.strip()

    def assemble(
        self,
        *,
        tools: Sequence[Tool],
        resources: LoadedResources,
        active_skills: Sequence[str] = (),
    ) -> PromptAssembly:
        if resources.system_replacement is not None:
            text = resources.system_replacement.content.strip() + "\n"
            digest = f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"
            evidence = resources.evidence["system:replacement"]
            return PromptAssembly(
                text,
                {"effective_system": digest},
                {"system:replacement": evidence.sha256},
                (),
            )
        accepted_skills = tuple(sorted(set(active_skills)))
        unknown = [name for name in accepted_skills if name not in resources.skills]
        if unknown:
            raise KeyError(f"unknown Skill: {unknown[0]}")
        sections = [self._builtin_prompt]
        accepted_tools = sorted(tools, key=lambda tool: tool.name)
        if accepted_tools:
            sections.append(
                "## Active Tools\n"
                + "\n".join(
                    f"- {tool.name}: {tool.description}" for tool in accepted_tools
                )
            )
        if resources.skills:
            sections.append(
                "## Available Skills\n"
                + "\n".join(
                    f"- {skill.name}: {skill.description} ({skill.source})"
                    for skill in sorted(
                        resources.skills.values(), key=lambda resource: resource.name
                    )
                )
            )
        for name in accepted_skills:
            skill = resources.skills[name]
            sections.append(f"## Active Skill: {name}\n{skill.content}")
            sections.extend(
                f"### Related Skill Resource: {relative}\n{content}"
                for relative, content in skill.related_resources.items()
            )
        sections.extend(
            f"## Context File: {context.source}\n{context.content}"
            for context in resources.context_files
        )
        if resources.append_system is not None:
            sections.append(
                "## Appended System Instructions\n"
                + resources.append_system.content
            )
        text = "\n\n".join(section.strip() for section in sections) + "\n"
        digest = f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"
        resource_hashes = {
            name: evidence.sha256
            for name, evidence in resources.evidence.items()
            if ":related:" not in name
        }
        for name in accepted_skills:
            for relative in resources.skills[name].related_resources:
                key = f"skill:{name}:related:{relative}"
                resource_hashes[key] = resources.evidence[key].sha256
        return PromptAssembly(
            text,
            {"effective_system": digest},
            resource_hashes,
            accepted_skills,
        )


class ProjectTrust(Protocol):
    def is_trusted(self, canonical_workspace: Path) -> bool: ...


class MemoryProjectTrust:
    """An explicit trust adapter useful to applications and deterministic tests."""

    def __init__(self, trusted: Iterable[str | Path] = ()) -> None:
        self._trusted = {Path(path).resolve() for path in trusted}

    def is_trusted(self, canonical_workspace: Path) -> bool:
        return canonical_workspace.resolve() in self._trusted

    def approve(self, workspace: str | Path) -> None:
        self._trusted.add(Path(workspace).resolve())


class JSONProjectTrustStore:
    """Persist canonical Project Trust decisions in one transparent JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()

    def is_trusted(self, canonical_workspace: Path) -> bool:
        return str(canonical_workspace.resolve()) in self._read()

    def approve(self, workspace: str | Path) -> None:
        trusted = self._read()
        trusted.add(str(Path(workspace).resolve()))
        self._write(trusted)

    def revoke(self, workspace: str | Path) -> None:
        trusted = self._read()
        trusted.discard(str(Path(workspace).resolve()))
        self._write(trusted)

    def _read(self) -> set[str]:
        if not self._path.exists():
            return set()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(
                "Unsupported Project Trust schema; migrate trusted-projects.json"
            )
        paths = payload.get("trusted_paths")
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not Path(path).is_absolute()
            for path in paths
        ):
            raise ValueError("Project Trust file contains invalid canonical paths")
        return set(paths)

    def _write(self, trusted: set[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"schema_version": 1, "trusted_paths": sorted(trusted)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".omega-tmp",
            dir=self._path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class ResourceLoader:
    """Resolve local resource scopes before constructing reusable harness values."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        working_directory: str | Path | None = None,
        user_root: str | Path | None = None,
        trust: ProjectTrust | None = None,
        approve_project: bool = False,
        context_enabled: bool = True,
        explicit: Sequence[Resource] = (),
        builtins: Sequence[Resource] = (),
    ) -> None:
        self._workspace = Path(workspace).resolve()
        if not self._workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        self._working_directory = (
            Path(working_directory).resolve()
            if working_directory is not None
            else self._workspace
        )
        if not self._working_directory.is_relative_to(self._workspace):
            raise ValueError("working_directory must be inside the workspace")
        self._user_root = Path(user_root).resolve() if user_root is not None else None
        self._trust = trust
        self._approve_project = approve_project
        self._context_enabled = context_enabled
        self._explicit = tuple(explicit)
        self._builtins = tuple(builtins)

    @property
    def workspace(self) -> Path:
        return self._workspace

    def load(self) -> LoadedResources:
        resolved_by_kind: dict[ResourceKind, dict[str, Resource]] = {
            ResourceKind.SKILL: {},
            ResourceKind.PROMPT_TEMPLATE: {},
            ResourceKind.EXTENSION: {},
        }
        sources = (
            (ResourceScope.BUILTIN, self._builtins),
            (ResourceScope.USER, self._discover_scoped(self._user_root)),
            (
                ResourceScope.PROJECT,
                self._discover_scoped(self._workspace / ".omega")
                if self._project_is_trusted()
                else (),
            ),
            (ResourceScope.EXPLICIT, self._explicit),
        )
        for scope, scoped_resources in sources:
            for resource in scoped_resources:
                target = resolved_by_kind.get(resource.kind)
                if target is not None:
                    target[resource.name] = replace(resource, scope=scope)
        resolved = resolved_by_kind[ResourceKind.SKILL]
        prompt_templates = resolved_by_kind[ResourceKind.PROMPT_TEMPLATE]
        extensions = resolved_by_kind[ResourceKind.EXTENSION]
        context_files = self._discover_context_files() if self._context_enabled else ()
        system_replacement: Resource | None = None
        append_system: Resource | None = None
        settings: Resource | None = None
        skipped: tuple[str, ...] = ()
        if self._project_is_trusted():
            system_replacement = self._read_optional_project_instruction(
                "SYSTEM.md", ResourceKind.SYSTEM
            )
            append_system = self._read_optional_project_instruction(
                "APPEND_SYSTEM.md", ResourceKind.APPEND_SYSTEM
            )
            settings = self._read_optional_project_instruction(
                "settings.json", ResourceKind.SETTINGS
            )
            if settings is not None:
                try:
                    parsed_settings = json.loads(settings.content)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Project settings.json must contain valid JSON"
                    ) from error
                if not isinstance(parsed_settings, dict):
                    raise ValueError("Project settings.json must contain a JSON object")
        else:
            skipped = self._protected_project_paths()
        evidence = {
            f"skill:{name}": ResourceEvidence(
                resource.kind,
                resource.scope,
                resource.source,
                f"sha256:{sha256(resource.content.encode('utf-8')).hexdigest()}",
            )
            for name, resource in sorted(resolved.items())
        }
        for name, resource in sorted(resolved.items()):
            source_root = Path(resource.source).parent
            for relative, content in resource.related_resources.items():
                related = replace(
                    resource,
                    content=content,
                    source=str((source_root / relative).resolve()),
                    related_resources={},
                )
                evidence[f"skill:{name}:related:{relative}"] = self._evidence(related)
        for prefix, named_resources in (
            ("prompt", prompt_templates),
            ("extension", extensions),
        ):
            for name, resource in sorted(named_resources.items()):
                evidence[f"{prefix}:{name}"] = self._evidence(resource)
        for index, resource in enumerate(context_files):
            evidence[f"context:{index}:{resource.source}"] = self._evidence(resource)
        if system_replacement is not None:
            evidence["system:replacement"] = self._evidence(system_replacement)
        if append_system is not None:
            evidence["system:append"] = self._evidence(append_system)
        if settings is not None:
            evidence["settings:project"] = self._evidence(settings)
        return LoadedResources(
            resolved,
            evidence,
            context_files,
            system_replacement,
            append_system,
            prompt_templates,
            extensions,
            settings=settings,
            skipped_protected=skipped,
        )

    def _project_is_trusted(self) -> bool:
        return self._approve_project or (
            self._trust is not None and self._trust.is_trusted(self._workspace)
        )

    @staticmethod
    def _evidence(resource: Resource) -> ResourceEvidence:
        return ResourceEvidence(
            resource.kind,
            resource.scope,
            resource.source,
            f"sha256:{sha256(resource.content.encode('utf-8')).hexdigest()}",
        )

    def _discover_context_files(self) -> tuple[Resource, ...]:
        relative = self._working_directory.relative_to(self._workspace)
        directories = [self._workspace]
        current = self._workspace
        for part in relative.parts:
            current = current / part
            directories.append(current)
        contexts: list[Resource] = []
        for directory in directories:
            path = directory / "AGENTS.md"
            if path.is_file():
                contexts.append(
                    Resource(
                        ResourceKind.CONTEXT,
                        path.parent.relative_to(self._workspace).as_posix() or ".",
                        "Hierarchical project context",
                        self._read_scoped_text(path, self._workspace),
                        str(path.resolve()),
                        ResourceScope.PROJECT,
                    )
                )
        return tuple(contexts)

    def _read_optional_project_instruction(
        self, name: str, kind: ResourceKind
    ) -> Resource | None:
        path = self._workspace / ".omega" / name
        if not path.is_file():
            return None
        return Resource(
            kind,
            name,
            f"Trusted project {name}",
            self._read_scoped_text(path, self._workspace),
            str(path.resolve()),
            ResourceScope.PROJECT,
        )

    def _protected_project_paths(self) -> tuple[str, ...]:
        root = self._workspace / ".omega"
        candidates = [
            *(root / "skills").glob("*/SKILL.md"),
            *(root / "prompts").glob("*.md"),
            *(root / "extensions").glob("*.py"),
            root / "SYSTEM.md",
            root / "APPEND_SYSTEM.md",
            root / "settings.json",
        ]
        return tuple(
            str(path.resolve()) for path in sorted(candidates) if path.is_file()
        )

    @staticmethod
    def _discover_scoped(root: Path | None) -> tuple[Resource, ...]:
        if root is None:
            return ()
        skill_root = root / "skills"
        resources: list[Resource] = []
        if skill_root.is_dir():
            for path in sorted(skill_root.glob("*/SKILL.md")):
                resources.append(ResourceLoader._read_skill(path, root))
        for path in sorted((root / "prompts").glob("*.md")):
            resources.append(
                Resource.prompt_template(
                    path.stem,
                    ResourceLoader._read_scoped_text(path, root),
                    source=str(path.resolve()),
                )
            )
        for path in sorted((root / "extensions").glob("*.py")):
            resources.append(
                Resource.extension_source(
                    path.stem,
                    ResourceLoader._read_scoped_text(path, root),
                    source=str(path.resolve()),
                )
            )
        return tuple(resources)

    @staticmethod
    def _read_scoped_text(path: Path, scope_root: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(scope_root.resolve()):
            raise ValueError(f"local resource escapes its scope: {path}")
        return resolved.read_text(encoding="utf-8")

    @staticmethod
    def _read_skill(path: Path, scope_root: Path) -> Resource:
        text = ResourceLoader._read_scoped_text(path, scope_root)
        if not text.startswith("---\n"):
            raise ValueError(f"Skill {path} requires YAML-style front matter")
        try:
            header, content = text[4:].split("\n---\n", 1)
        except ValueError as error:
            raise ValueError(f"Skill {path} has unterminated front matter") from error
        metadata: dict[str, str] = {}
        for line in header.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                metadata[key.strip()] = value.strip()
        name = metadata.get("name", "")
        description = metadata.get("description", "")
        root = path.parent.resolve()
        related: dict[str, str] = {}
        for candidate in sorted(path.parent.rglob("*")):
            if not candidate.is_file() or candidate == path:
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Skill resource escapes its directory: {candidate}")
            raw = resolved.read_bytes()
            if b"\x00" in raw:
                raise ValueError(f"Skill related resource must be text: {candidate}")
            try:
                related[resolved.relative_to(root).as_posix()] = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"Skill related resource must be UTF-8 text: {candidate}"
                ) from error
        return Resource.skill(
            name,
            description,
            content.strip(),
            source=str(path.resolve()),
            related_resources=related,
        )
