from __future__ import annotations

import asyncio
import os

import pytest

from agent_harness import (
    AgentMessage,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
    Role,
    complete,
)


@pytest.mark.skipif(
    os.environ.get('AGENT_HARNESS_REAL_SMOKE') != '1',
    reason='set AGENT_HARNESS_REAL_SMOKE=1 to opt into the credential-gated smoke test',
)
def test_explicit_real_openai_compatible_endpoint():
    required = ('AGENT_HARNESS_BASE_URL', 'AGENT_HARNESS_API_KEY', 'AGENT_HARNESS_MODEL')
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail('missing explicit smoke configuration: ' + ', '.join(missing))
    adapter = OpenAICompatibleAdapter(
        OpenAICompatibleConfig(
            base_url=os.environ['AGENT_HARNESS_BASE_URL'],
            api_key=os.environ['AGENT_HARNESS_API_KEY'],
        )
    )
    result = asyncio.run(
        complete(
            adapter,
            [AgentMessage.text(Role.USER, 'Reply with the word ready.')],
            ModelSpec(os.environ['AGENT_HARNESS_MODEL'], max_output_tokens=32),
        )
    )
    assert result.message.content
