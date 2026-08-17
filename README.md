# System Engineering Based on an Agent Harness

This repository is a nine-chapter, cumulative construction of a reusable Python agent harness. The Authoritative Course Tree is [`course/`](course/): each Chapter Notebook owns its source, exports an immutable Checkpoint Package, and runs the offline Checkpoint Gate before publication.

Chapter 1 establishes the provider-neutral model seam:

- [`course/notebooks/01_model_boundary.ipynb`](course/notebooks/01_model_boundary.ipynb) is the literate source.
- [`course/checkpoints/ch01/`](course/checkpoints/ch01/) is the generated, installable Python 3.11+ package.
- [`course/tools/`](course/tools/) implements clean-kernel, deterministic export, drift, compilation, installation, import, and test gates.

Generated Checkpoint files are not edited by hand. The older notebooks under `notebooks/` remain historical evidence and are not production or export inputs.

## Offline verification

```bash
uv sync --python 3.11
uv run python -m pytest
PYTHONPATH=course/checkpoints/ch01/src \
  uv run python -m pytest -p no:cacheprovider course/checkpoints/ch01/tests
```

The ordinary suite uses `ScriptedModelAdapter` and a loopback fake OpenAI-compatible Chat Completions endpoint. A real endpoint is supplementary and opt-in; see the generated Checkpoint README for its explicit credential gate.
