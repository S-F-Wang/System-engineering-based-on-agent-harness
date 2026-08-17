from __future__ import annotations

import pytest

from agent_harness import AgentRuntime, JSONLSessionStore, ModelSpec, ScriptedModelAdapter
from agent_harness import create_agent_session


def test_persistence_inputs_fail_before_session_work_is_accepted(tmp_path) -> None:
    store = JSONLSessionStore(tmp_path)
    with pytest.raises(ValueError, match="safe local identifier"):
        store.create("../escape")
    with pytest.raises(ValueError, match="continuation requires"):
        create_agent_session(
            AgentRuntime(ScriptedModelAdapter([]), ModelSpec("scripted/contract")),
            session_id="missing-store",
        )
