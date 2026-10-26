"""Run directories of format 1, which the earlier model-calling passes wrote; the package now only reads them.

A format-1 run recorded the model's answers to the classify, premise and check passes, and replayed
them to rebuild its graph. The passes are gone: runs are now made by an agent's tool calls
(``session``, format 2). ``verify`` still runs a format-1 run's deterministic checks again, and reads
its manifest and files through this module. The layout of such a run::

    <run>/
      manifest.json   format "credencegraph-extract-run", version 1
      graph.json      the graph the run produced
      rubric.toml     byte copy of the rubric used
      prompts/        the template of each pass, and the rendered prompt of every call
      raw/            the record of every model call
      rejections.json the records the core refused, each with its error code
      report.json     the graph's coverage, and what each pass reported
      checks/c-NNN/   for each check: its inputs, the tool's output and a log
      source/         a copy of the source document, when it was stored

``RunError`` and run ids are shared with the tool runs.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT = "credencegraph-extract-run"
VERSION = 1

MANIFEST = "manifest.json"
GRAPH = "graph.json"
RUBRIC = "rubric.toml"
CHECKS = "checks"


class RunError(Exception):
    """A run cannot be made or replayed.

    Attributes:
        code: A stable error code.
    """

    def __init__(self, code: str, message: str) -> None:
        """Create the error.

        Args:
            code: A stable error code.
            message: What went wrong.
        """
        super().__init__(message)
        self.code = code


def new_run_id() -> str:
    """Return a fresh run id: the UTC time and a random suffix.

    Returns:
        The id, such as ``20261024T153000Z-1a2b3c4d``.
    """
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


RAW_PATH = re.compile(r"raw/[0-9]{4,}\.json")


def read_run_file(run_dir: Path, name: str) -> bytes:
    """Read a file of a run directory.

    Args:
        run_dir: The run directory.
        name: The file's path inside it.

    Returns:
        The file's bytes.

    Raises:
        RunError: If the file cannot be read.
    """
    try:
        return (run_dir / name).read_bytes()
    except OSError as error:
        raise RunError("incomplete-run", f"cannot read {name} of {str(run_dir)!r}: {error.strerror or error}") from None


def _field(value: Any, key: str, kind: type, where: str) -> Any:
    """Return ``value[key]`` if ``value`` is an object whose ``key`` is a ``kind``; else refuse the manifest."""
    if not isinstance(value, dict) or not isinstance(value.get(key), kind):
        raise RunError("invalid-manifest", f"{MANIFEST}: {where}{key} must be a {kind.__name__}")
    return value[key]


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """Read and check a format-1 run's manifest: its format, its version, and every field the run recorded.

    Args:
        run_dir: The run directory.

    Returns:
        The manifest.

    Raises:
        RunError: If there is no manifest, it is not JSON, not of this format and version, or a
            field the run recorded is missing or of the wrong type.
    """
    if not (run_dir / MANIFEST).is_file():
        raise RunError("not-a-run", f"{str(run_dir)!r} has no {MANIFEST}")
    try:
        manifest = json.loads(read_run_file(run_dir, MANIFEST))
    except ValueError as error:
        raise RunError("invalid-manifest", f"{MANIFEST} of {str(run_dir)!r} is not JSON: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise RunError("not-a-run", f"{MANIFEST} of {str(run_dir)!r} is not a {FORMAT} manifest")
    if manifest.get("version") != VERSION:
        raise RunError("unsupported-version", f"{MANIFEST} is version {manifest.get('version')!r}, not {VERSION}")
    _field(manifest, "run", str, "")
    _field(manifest, "graph_digest", str, "")
    if not all(isinstance(value, str) for value in _field(manifest, "versions", dict, "").values()):
        raise RunError("invalid-manifest", f"{MANIFEST}: every version must be a string")
    source = _field(manifest, "source", dict, "")
    for key in ("document", "file", "digest"):
        _field(source, key, str, "source.")
    _field(source, "stored", bool, "source.")
    if Path(source["file"]).name != source["file"] or source["file"] in {"", ".", ".."}:
        raise RunError("invalid-manifest", f"{MANIFEST}: source.file must be a plain file name")
    _field(_field(manifest, "rubric", dict, ""), "digest", str, "rubric.")
    for i, entry in enumerate(_field(manifest, "passes", list, "")):
        _field(entry, "name", str, f"passes[{i}].")
        for j, call in enumerate(_field(entry, "calls", list, f"passes[{i}].")):
            where = f"passes[{i}].calls[{j}]."
            _field(call, "raw_digest", str, where)
            if not RAW_PATH.fullmatch(_field(call, "raw", str, where)):
                raise RunError("invalid-manifest", f"{MANIFEST}: {where}raw must name a file raw/NNNN.json")
    return manifest
