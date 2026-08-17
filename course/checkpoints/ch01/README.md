# Agent Harness — Chapter 1 Checkpoint

This immutable Checkpoint contains the typed provider-neutral model seam built by Chapter 1. It supports Python 3.11 or newer and one production transport: explicitly configured OpenAI-compatible streaming Chat Completions.

Ordinary tests use `ScriptedModelAdapter` or the local fake endpoint. The real-endpoint smoke test runs only when `AGENT_HARNESS_REAL_SMOKE=1` and all three explicit endpoint variables are supplied.
