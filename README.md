# System Engineering Based on an Agent Harness

This repository is a nine-chapter, cumulative construction of a reusable Python agent harness. The Authoritative Course Tree is [`course/`](course/): each Chapter Notebook owns its source, exports an immutable Checkpoint Package, and runs the offline Checkpoint Gate before publication.

The first eight cumulative Chapters now establish the model, Runtime, Tool, Session-control, persistence, context-management, Extension, and Coding Agent seams:

- [`course/notebooks/01_model_boundary.ipynb`](course/notebooks/01_model_boundary.ipynb) is the literate source.
- [`course/checkpoints/ch01/`](course/checkpoints/ch01/) is the generated, installable Python 3.11+ package.
- [`course/notebooks/02_async_runtime.ipynb`](course/notebooks/02_async_runtime.ipynb) adds streaming, retries, Events, and cancellation settlement.
- [`course/notebooks/03_structured_tools.ipynb`](course/notebooks/03_structured_tools.ipynb) adds explicit JSON-Schema Tools, replaceable execution, deterministic batches, bounded results, and the complete model–Tool round trip.
- [`course/notebooks/04_run_control.ipynb`](course/notebooks/04_run_control.ipynb) adds one-active-Run Sessions, Steering and Follow-up queues, coordinated cancellation settlement, real-process cancellation, and opt-in Run Guards.
- [`course/notebooks/05_durable_sessions.ipynb`](course/notebooks/05_durable_sessions.ipynb) adds tree-structured durable Sessions, cross-process writer leases, settled-boundary recovery, and nondestructive migration.
- [`course/notebooks/06_context_compaction.ipynb`](course/notebooks/06_context_compaction.ipynb) adds durable Compaction, adaptive thresholds, and one classified overflow-recovery attempt.
- [`course/notebooks/07_extensions.ipynb`](course/notebooks/07_extensions.ipynb) adds ordered Event barriers, fail-safe Hooks, explicitly supplied Extensions, lifecycle reload, Custom Entries, Run Annotations, and frozen Run Snapshots.
- [`course/notebooks/08_coding_agent.ipynb`](course/notebooks/08_coding_agent.ipynb) adds trusted local resources, deterministic Prompt Assembly, progressive Skill disclosure, and the explicit workspace-aware Coding Tool Preset.
- [`course/checkpoints/ch08/`](course/checkpoints/ch08/) is the latest generated cumulative Checkpoint Package.
- [`course/tools/`](course/tools/) implements clean-kernel, deterministic export, drift, compilation, installation, import, and test gates.

Generated Checkpoint files are not edited by hand. The older notebooks under `notebooks/` remain historical evidence and are not production or export inputs.

## Offline verification

```bash
uv sync --python 3.11
uv run python -m pytest
PYTHONPATH=course/checkpoints/ch08/src \
  uv run python -m pytest -p no:cacheprovider course/checkpoints/ch08/tests
```

The ordinary suite uses `ScriptedModelAdapter` and a loopback fake OpenAI-compatible Chat Completions endpoint. A real endpoint is supplementary and opt-in; see the generated Checkpoint README for its explicit credential gate.
