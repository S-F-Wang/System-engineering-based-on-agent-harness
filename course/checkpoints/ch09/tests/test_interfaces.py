from __future__ import annotations

import asyncio
from pathlib import Path

from agent_harness import (
    ModelEnd,
    ModelSpec,
    NonTerminalAdapter,
    Extension,
    ScriptedModelAdapter,
    StopReason,
    TextDelta,
    create_session,
)


def test_non_terminal_adapter_drives_snapshot_events_steering_and_follow_up(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        model = ScriptedModelAdapter(
            [
                [TextDelta("initial"), ModelEnd(StopReason.COMPLETE)],
                [TextDelta("steered"), ModelEnd(StopReason.COMPLETE)],
                [TextDelta("followed"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        ui = NonTerminalAdapter(
            create_session(
                model,
                ModelSpec("scripted/non-terminal"),
                workspace=workspace,
                no_save=True,
            )
        )

        ui.start("begin")
        snapshot = ui.snapshot
        ui.steer("adjust now")
        ui.follow_up("then continue")
        events = [event async for event in ui.events()]
        result = await ui.result()

        assert snapshot.model.model_id == "scripted/non-terminal"
        assert events[0].type == "agent_start"
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert result.final_text == "followed"
        assert len(model.received_requests) == 3
        request_text = repr(model.received_requests)
        assert "adjust now" in request_text
        assert "then continue" in request_text

    asyncio.run(scenario())


def test_non_terminal_adapter_executes_the_same_session_commands(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        async def inspect(arguments: str) -> object:
            return {"arguments": arguments}

        def configure(api) -> None:
            api.register_command("inspect", inspect)

        ui = NonTerminalAdapter(
            create_session(
                ScriptedModelAdapter(
                    [TextDelta("unused"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/commands"),
                workspace=workspace,
                no_save=True,
                extensions=(Extension("terminal", "1", configure),),
            )
        )

        assert await ui.command("inspect", "state") == {"arguments": "state"}

    asyncio.run(scenario())


def test_non_terminal_adapter_coordinates_cancellation(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        ui = NonTerminalAdapter(
            create_session(
                ScriptedModelAdapter(
                    [
                        [TextDelta("retry answer"), ModelEnd(StopReason.COMPLETE)],
                        [TextDelta("restored answer"), ModelEnd(StopReason.COMPLETE)],
                    ]
                ),
                ModelSpec("scripted/cancel"),
                workspace=workspace,
                no_save=True,
            )
        )

        ui.start("cancel this")
        ui.follow_up("restore this input")
        ui.cancel()
        result = await ui.result()

        assert result.status == "cancelled"
        assert result.stop_reason == "aborted"
        assert ui.restored_inputs[0].message.content[0].text == "restore this input"

        ui.start("retry")
        restored = await ui.result()

        assert restored.final_text == "restored answer"
        assert ui.restored_inputs == ()

    asyncio.run(scenario())
