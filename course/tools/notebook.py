"""Structural and clean-kernel gates for Chapter Notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from nbclient import NotebookClient
import nbformat


REQUIRED_SECTIONS = (
    "Goal and Previous Limitation",
    "Conceptual Model",
    "Minimal Execution",
    "Staged Construction",
    "Observable Trace",
    "Failure Boundaries and Trade-offs",
    "Checkpoint Export and Verification",
    "Public API Summary",
)

_OFFLINE_SITECUSTOMIZE = '''\
"""Injected by the course Release Gate to keep notebook kernels offline."""

import ipaddress
import socket


class OfflineNetworkError(OSError):
    pass


def _is_local(host):
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if not isinstance(host, str):
        return False
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_original_getaddrinfo = socket.getaddrinfo
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


def _offline_getaddrinfo(host, *args, **kwargs):
    if not _is_local(host):
        raise OfflineNetworkError(f"offline Release Gate blocked host: {host}")
    return _original_getaddrinfo(host, *args, **kwargs)


def _offline_connect(instance, address):
    if isinstance(address, tuple) and not _is_local(address[0]):
        raise OfflineNetworkError(f"offline Release Gate blocked host: {address[0]}")
    return _original_connect(instance, address)


def _offline_connect_ex(instance, address):
    if isinstance(address, tuple) and not _is_local(address[0]):
        raise OfflineNetworkError(f"offline Release Gate blocked host: {address[0]}")
    return _original_connect_ex(instance, address)


socket.getaddrinfo = _offline_getaddrinfo
socket.socket.connect = _offline_connect
socket.socket.connect_ex = _offline_connect_ex
'''

_CREDENTIAL_MARKERS = ("API_KEY", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN")


@dataclass(frozen=True, slots=True)
class NotebookReport:
    sections: tuple[str, ...]
    export_cells: int


def validate_chapter_notebook(path: str | Path) -> NotebookReport:
    """Validate the stable Chapter Template without executing the notebook."""

    notebook_path = Path(path)
    notebook = nbformat.read(notebook_path, as_version=4)
    headings: list[str] = []
    export_cells = 0
    all_source: list[str] = []

    for cell in notebook.cells:
        source = str(cell.source)
        all_source.append(source)
        if cell.cell_type == "markdown":
            headings.extend(
                match.group(1).strip()
                for match in re.finditer(r"^##\s+(.+?)\s*$", source, re.MULTILINE)
            )
        marker = cell.metadata.get("agent_harness_export")
        if marker is not None:
            if cell.cell_type != "code":
                raise ValueError("Export Cells must be code cells")
            export_cells += 1

    positions: list[int] = []
    for required in REQUIRED_SECTIONS:
        try:
            positions.append(headings.index(required))
        except ValueError as error:
            raise ValueError(f"missing Chapter Template section: {required}") from error
    if positions != sorted(positions):
        raise ValueError("Chapter Template sections are out of narrative order")

    combined = "\n".join(all_source)
    has_todo = re.search(r"\bTODOs?\b", combined, re.IGNORECASE)
    has_exercise_section = any(
        re.search(r"\bexercises?\b", heading, re.IGNORECASE) for heading in headings
    )
    if has_todo or has_exercise_section:
        raise ValueError("Chapter Notebooks cannot contain TODOs or exercises")
    historical_inputs = ("notebooks/raw", "notebooks/lessons", "mini_harness.ipynb")
    if any(token in combined for token in historical_inputs):
        raise ValueError("Chapter Notebook refers to a historical course input")

    return NotebookReport(sections=REQUIRED_SECTIONS, export_cells=export_cells)


def execute_notebook(
    path: str | Path,
    *,
    cwd: str | Path,
    timeout: int = 300,
    offline: bool = True,
) -> Any:
    """Execute a Chapter Notebook in a fresh, credential-free kernel.

    Offline execution permits only loopback sockets so Jupyter kernel transport
    and deterministic local provider fakes continue to work.
    """

    notebook = nbformat.read(Path(path), as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
    )
    environment = dict(os.environ)
    for name in tuple(environment):
        if any(marker in name.upper() for marker in _CREDENTIAL_MARKERS):
            environment.pop(name)
    environment.pop("AGENT_HARNESS_REAL_SMOKE", None)
    if not offline:
        return client.execute(cwd=str(Path(cwd).resolve()), env=environment)

    with tempfile.TemporaryDirectory(prefix="notebook-offline-") as temporary:
        hook_root = Path(temporary)
        (hook_root / "sitecustomize.py").write_text(
            _OFFLINE_SITECUSTOMIZE,
            encoding="utf-8",
        )
        previous_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(hook_root), previous_pythonpath) if item
        )
        return client.execute(cwd=str(Path(cwd).resolve()), env=environment)
