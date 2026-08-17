# System Engineering Based on an Agent Harness

This repository is a nine-chapter, cumulative construction of a reusable Python agent harness. The Authoritative Course Tree is [`course/`](course/): each Chapter Notebook owns its source, exports an immutable Checkpoint Package, and runs the offline Checkpoint Gate before publication.

The nine cumulative Chapters establish the model, Runtime, Tool, Session-control, persistence, context-management, Extension, Coding Agent, public interface, and observability seams:

- [`course/notebooks/01_model_boundary.ipynb`](course/notebooks/01_model_boundary.ipynb) is the literate source.
- [`course/checkpoints/ch01/`](course/checkpoints/ch01/) is the generated, installable Python 3.11+ package.
- [`course/notebooks/02_async_runtime.ipynb`](course/notebooks/02_async_runtime.ipynb) adds streaming, retries, Events, and cancellation settlement.
- [`course/notebooks/03_structured_tools.ipynb`](course/notebooks/03_structured_tools.ipynb) adds explicit JSON-Schema Tools, replaceable execution, deterministic batches, bounded results, and the complete model–Tool round trip.
- [`course/notebooks/04_run_control.ipynb`](course/notebooks/04_run_control.ipynb) adds one-active-Run Sessions, Steering and Follow-up queues, coordinated cancellation settlement, real-process cancellation, and opt-in Run Guards.
- [`course/notebooks/05_durable_sessions.ipynb`](course/notebooks/05_durable_sessions.ipynb) adds tree-structured durable Sessions, cross-process writer leases, settled-boundary recovery, and nondestructive migration.
- [`course/notebooks/06_context_compaction.ipynb`](course/notebooks/06_context_compaction.ipynb) adds durable Compaction, adaptive thresholds, and one classified overflow-recovery attempt.
- [`course/notebooks/07_extensions.ipynb`](course/notebooks/07_extensions.ipynb) adds ordered Event barriers, fail-safe Hooks, explicitly supplied Extensions, lifecycle reload, Custom Entries, Run Annotations, and frozen Run Snapshots.
- [`course/notebooks/08_coding_agent.ipynb`](course/notebooks/08_coding_agent.ipynb) adds trusted local resources, deterministic Prompt Assembly, progressive Skill disclosure, and the explicit workspace-aware Coding Tool Preset.
- [`course/notebooks/09_interfaces_and_release.ipynb`](course/notebooks/09_interfaces_and_release.ipynb) freezes the Python factories, Omega CLI, local Run evidence, and terminal-neutral adapter contracts.
- [`course/checkpoints/ch09/`](course/checkpoints/ch09/) is the final generated cumulative Checkpoint Package.
- [`src/agent_harness/`](src/agent_harness/) is the production package generated only from Chapter 9; drift is checked against the final Checkpoint.
- [`course/tools/`](course/tools/) implements clean-kernel, deterministic export, drift, compilation, installation, import, and test gates.

Generated Checkpoint files are not edited by hand. The older notebooks under `notebooks/` remain historical evidence and are not production or export inputs.

## Offline verification

```bash
uv sync --python 3.11
uv run python -m pytest -q tests/test_checkpoint_export.py tests/test_release_gate.py
PIP_NO_INDEX=1 AGENT_HARNESS_REAL_SMOKE=0 \
  uv run python -m course.tools.release
uv run mypy src/agent_harness course/tools
```

The release command validates the one authoritative artifact chain, executes all
nine Chapter Notebooks in fresh credential-free kernels with external network
access blocked, verifies every cumulative Checkpoint without republishing it,
and regenerates the production package twice. The same command is the required
Python 3.11 matrix job on Linux, Windows, and macOS in
[`release-gate.yml`](.github/workflows/release-gate.yml). See the
[`version-one verification report`](docs/release/v1-verification.md) for exact
evidence, platform status, and known limitations.

The ordinary suite uses `ScriptedModelAdapter` and a loopback fake OpenAI-compatible Chat Completions endpoint. A real endpoint is supplementary and opt-in; see the generated Checkpoint README for its explicit credential gate.

The installed distribution exposes `create_session()`, `create_coding_session()`, and `omega`. The library requires explicit adapter/configuration values and never reads environment state. The CLI translates `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`, with `--base-url` and `--model` taking precedence and no API-key flag:

```bash
omega exec "inspect this workspace"
omega exec "machine task" --format json
omega chat
omega sessions list
omega runs list
```

Durable execution is local by default. Use `--no-save` to disable Session, Trace, Annotation, and Artifact persistence together. Version one deliberately excludes a full TUI, RPC/server mode, MCP, subagents, multimodal content, an optimizer, remote Extension installation, a model catalog, and a built-in sandbox.
