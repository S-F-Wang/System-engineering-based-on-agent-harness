"""Bounded, explicitly supplied Extension registration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
import json
import re

from .tools import Tool


ExtensionConfigurator = Callable[["ExtensionAPI"], None]
EventSubscriber = Callable[[object], Awaitable[None]]
CommandHandler = Callable[[str], Awaitable[object]]
LifecycleHandler = Callable[[], Awaitable[None]]
_COMMAND_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_NAMESPACE = re.compile(
    r"[a-z][a-z0-9_-]{0,63}(?:\.[a-z][a-z0-9_-]{0,63})*\Z"
)


@dataclass(frozen=True, slots=True)
class SubscriberRegistration:
    extension_name: str
    callback: EventSubscriber


class HookPoint(str, Enum):
    BEFORE_RUN = "before_run"
    BEFORE_MODEL_REQUEST = "before_model_request"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_COMPACTION = "before_compaction"


@dataclass(frozen=True, slots=True)
class HookContext:
    point: HookPoint
    value: object
    snapshot: object | None = None


HookCallback = Callable[[HookContext], Awaitable[object | None]]


@dataclass(frozen=True, slots=True)
class HookRegistration:
    extension_name: str
    point: HookPoint
    callback: HookCallback


class HookExecutionError(RuntimeError):
    def __init__(self, extension_name: str, point: HookPoint, detail: str) -> None:
        self.extension_name = extension_name
        self.point = point
        super().__init__(
            f"Extension {extension_name!r} {point.value} Hook failed: {detail}"
        )


@dataclass(frozen=True, slots=True)
class CompactionHookRequest:
    trigger: object
    focus: str | None


@dataclass(frozen=True, slots=True)
class CommandRegistration:
    extension_name: str
    name: str
    handler: CommandHandler


@dataclass(frozen=True, slots=True)
class ReplacementRecord:
    kind: str
    name: str
    previous_owner: str
    replacement_owner: str


class CommandExecutionError(RuntimeError):
    def __init__(self, command: str, detail: str) -> None:
        self.command = command
        super().__init__(f"Command /{command} failed: {detail}")


@dataclass(frozen=True, slots=True)
class CustomEntry:
    namespace: str
    version: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not _NAMESPACE.fullmatch(self.namespace):
            raise ValueError("Custom Entry namespace must be a safe dotted name")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValueError("Custom Entry version must be a positive integer")
        try:
            canonical = json.loads(
                json.dumps(
                    dict(self.payload),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Custom Entry payload must be JSON-serializable") from error
        if not isinstance(canonical, dict):
            raise ValueError("Custom Entry payload must be a JSON object")
        object.__setattr__(self, "payload", MappingProxyType(canonical))


@dataclass(frozen=True, slots=True)
class RunAnnotation:
    run_id: str
    namespace: str
    version: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("Run Annotation requires a Run id")
        custom = CustomEntry(self.namespace, self.version, self.payload)
        object.__setattr__(self, "payload", custom.payload)


CustomEntrySink = Callable[[CustomEntry], None]


class _ExtensionState:
    def __init__(self) -> None:
        self.entries: list[CustomEntry] = []
        self.pending: list[CustomEntry] = []
        self.sink: CustomEntrySink | None = None
        self.annotations: list[RunAnnotation] = []

    def append(self, entry: CustomEntry) -> None:
        self.entries.append(entry)
        if self.sink is None:
            self.pending.append(entry)
        else:
            self.sink(entry)

    def activate(self, sink: CustomEntrySink) -> None:
        self.sink = sink
        pending = tuple(self.pending)
        self.pending.clear()
        for entry in pending:
            sink(entry)

    def deactivate(self) -> None:
        self.sink = None


class ExtensionInitializationError(RuntimeError):
    """Identify an Extension that could not be configured."""

    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        super().__init__(f"Extension {source!r} initialization failed: {detail}")


@dataclass(frozen=True, slots=True)
class Extension:
    name: str
    version: str
    configure: ExtensionConfigurator
    source: str | None = None
    startup: LifecycleHandler | None = None
    teardown: LifecycleHandler | None = None

    def __post_init__(self) -> None:
        if not _COMMAND_NAME.fullmatch(self.name):
            raise ValueError("Extension name must be a safe lowercase identifier")
        if not self.version.strip():
            raise ValueError("Extension version cannot be empty")
        if not callable(self.configure):
            raise TypeError("Extension configure must be callable")
        if self.startup is not None and not callable(self.startup):
            raise TypeError("Extension startup must be callable")
        if self.teardown is not None and not callable(self.teardown):
            raise TypeError("Extension teardown must be callable")

    @property
    def identity(self) -> str:
        return f"{self.name}@{self.version}"


class ExtensionAPI:
    """The small registration interface visible to one trusted Extension."""

    def __init__(
        self,
        extension_name: str,
        tools: dict[str, Tool],
        tool_owners: dict[str, str],
        subscribers: list[SubscriberRegistration],
        hooks: list[HookRegistration],
        commands: dict[str, CommandRegistration],
        state: _ExtensionState,
        replacements: list[ReplacementRecord],
    ) -> None:
        self._extension_name = extension_name
        self._tools = tools
        self._tool_owners = tool_owners
        self._subscribers = subscribers
        self._hooks = hooks
        self._commands = commands
        self._state = state
        self._replacements = replacements

    def register_tool(self, tool: Tool, *, replace: bool = False) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Extensions can register only Tool values")
        if tool.name in self._tools and not replace:
            raise ValueError(
                f"Tool {tool.name!r} is already registered; use replace=True"
            )
        if tool.name in self._tools:
            self._replacements.append(
                ReplacementRecord(
                    "tool",
                    tool.name,
                    self._tool_owners[tool.name],
                    self._extension_name,
                )
            )
        self._tools[tool.name] = tool
        self._tool_owners[tool.name] = self._extension_name

    def subscribe(self, subscriber: EventSubscriber) -> None:
        if not callable(subscriber):
            raise TypeError("Event subscriber must be callable")
        self._subscribers.append(
            SubscriberRegistration(self._extension_name, subscriber)
        )

    def register_hook(self, point: HookPoint, hook: HookCallback) -> None:
        if not isinstance(point, HookPoint):
            raise TypeError("Hook point must be a HookPoint")
        if not callable(hook):
            raise TypeError("Hook must be callable")
        self._hooks.append(HookRegistration(self._extension_name, point, hook))

    def register_command(
        self,
        name: str,
        handler: CommandHandler,
        *,
        replace: bool = False,
    ) -> None:
        if not _COMMAND_NAME.fullmatch(name):
            raise ValueError("command name must be a safe lowercase identifier")
        if not callable(handler):
            raise TypeError("command handler must be callable")
        if name in self._commands and not replace:
            raise ValueError(
                f"Command {name!r} is already registered; use replace=True"
            )
        if name in self._commands:
            self._replacements.append(
                ReplacementRecord(
                    "command",
                    name,
                    self._commands[name].extension_name,
                    self._extension_name,
                )
            )
        self._commands[name] = CommandRegistration(
            self._extension_name,
            name,
            handler,
        )

    def append_custom_entry(
        self,
        namespace: str,
        version: int,
        payload: Mapping[str, object],
    ) -> None:
        if namespace != self._extension_name and not namespace.startswith(
            f"{self._extension_name}."
        ):
            raise ValueError(
                "Custom Entry namespace must belong to the registering Extension"
            )
        self._state.append(CustomEntry(namespace, version, payload))

    def add_run_annotation(
        self,
        run_id: str,
        namespace: str,
        version: int,
        payload: Mapping[str, object],
    ) -> None:
        if namespace != self._extension_name and not namespace.startswith(
            f"{self._extension_name}."
        ):
            raise ValueError(
                "Run Annotation namespace must belong to the registering Extension"
            )
        self._state.annotations.append(
            RunAnnotation(run_id, namespace, version, payload)
        )


class ExtensionHost:
    """Resolve explicit Extensions into one immutable effective registration set."""

    def __init__(
        self,
        base_tools: Sequence[Tool],
        extensions: Sequence[Extension],
    ) -> None:
        tools = {tool.name: tool for tool in base_tools}
        tool_owners = {tool.name: "runtime" for tool in base_tools}
        subscribers: list[SubscriberRegistration] = []
        hooks: list[HookRegistration] = []
        commands: dict[str, CommandRegistration] = {}
        state = _ExtensionState()
        replacements: list[ReplacementRecord] = []
        identities: list[str] = []
        names: set[str] = set()
        for extension in extensions:
            if not isinstance(extension, Extension):
                raise TypeError("extensions must contain Extension values")
            if extension.name in names:
                raise ValueError(f"duplicate Extension name: {extension.name!r}")
            names.add(extension.name)
            source = extension.source or extension.identity
            try:
                extension.configure(
                    ExtensionAPI(
                        extension.name,
                        tools,
                        tool_owners,
                        subscribers,
                        hooks,
                        commands,
                        state,
                        replacements,
                    )
                )
            except Exception as error:
                raise ExtensionInitializationError(
                    source, f"{type(error).__name__}: {error}"
                ) from error
            identities.append(extension.identity)
        self._tools = MappingProxyType(dict(tools))
        self._extensions = tuple(extensions)
        self._identities = tuple(identities)
        self._subscribers = tuple(subscribers)
        self._hooks = tuple(hooks)
        self._commands = MappingProxyType(dict(commands))
        self._state = state
        self._started = False
        self._replacements = tuple(replacements)

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    @property
    def extensions(self) -> tuple[Extension, ...]:
        return self._extensions

    @property
    def identities(self) -> tuple[str, ...]:
        return self._identities

    @property
    def subscribers(self) -> tuple[SubscriberRegistration, ...]:
        return self._subscribers

    @property
    def hooks(self) -> tuple[HookRegistration, ...]:
        return self._hooks

    @property
    def commands(self) -> tuple[CommandRegistration, ...]:
        return tuple(self._commands.values())

    @property
    def replacements(self) -> tuple[ReplacementRecord, ...]:
        return self._replacements

    def command(self, name: str) -> CommandRegistration:
        try:
            return self._commands[name]
        except KeyError:
            raise KeyError(f"unknown chat command: /{name}") from None

    @property
    def custom_entries(self) -> tuple[CustomEntry, ...]:
        return tuple(self._state.entries)

    def activate_custom_entry_sink(self, sink: CustomEntrySink) -> None:
        self._state.activate(sink)

    def deactivate_custom_entry_sink(self) -> None:
        self._state.deactivate()

    def run_annotations(self, run_id: str | None = None) -> tuple[RunAnnotation, ...]:
        return tuple(
            annotation
            for annotation in self._state.annotations
            if run_id is None or annotation.run_id == run_id
        )

    async def startup(self) -> None:
        if self._started:
            return
        for extension in self._extensions:
            if extension.startup is None:
                continue
            source = extension.source or extension.identity
            try:
                await extension.startup()
            except Exception as error:
                raise ExtensionInitializationError(
                    source,
                    f"startup {type(error).__name__}",
                ) from error
        self._started = True

    async def teardown(self) -> tuple["LifecycleWarning", ...]:
        warnings: list[LifecycleWarning] = []
        for extension in reversed(self._extensions):
            if extension.teardown is None:
                continue
            try:
                await extension.teardown()
            except Exception as error:
                warnings.append(
                    LifecycleWarning(
                        extension.source or extension.identity,
                        "teardown",
                        type(error).__name__,
                    )
                )
        self._started = False
        return tuple(warnings)


@dataclass(frozen=True, slots=True)
class LifecycleWarning:
    source: str
    phase: str
    diagnostic: str


@dataclass(frozen=True, slots=True)
class ReloadResult:
    replaced_extensions: tuple[str, ...]
    warnings: tuple[LifecycleWarning, ...] = ()
    registration_replacements: tuple[ReplacementRecord, ...] = ()
