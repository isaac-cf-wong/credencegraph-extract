"""Verifying runs: the deterministic checks run again, and rubric versions compared across runs.

``replay`` shows that a run's graph follows from its recorded calls; ``verify`` looks at the checks
one by one, from the run directory alone, without the source. Each ``recompute-numbers``,
``rederive-maths`` and ``rederive-maths-numeric`` check of the graph is run again from the inputs its
record holds, and any field of the record the re-run does not reproduce is reported: a result, a
detail, the digest of the check's files. The files in ``checks/c-NNN/`` are compared with the digest
the record holds, and every check record, citation checks included, with the ``inputs.json`` and
``output.json`` its check wrote; a check of a type the runner does not have is reported. Citation checks are judged by a model, so they are not run again: ``replay``
answers them from their recorded responses.

A rubric's ``version`` names its values, so two runs that record one rubric name and version with
different file digests used values that changed without a new version. ``rubric_reuse`` reports
each such name and version.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from credencegraph.core import CredenceGraphError, Graph, loads
from credencegraph.rubric import Rubric, RubricError, loads_rubric

from credencegraph_extract import session
from credencegraph_extract.checks import (
    CITATION,
    DETERMINISTIC,
    IMPLEMENTED,
    INPUTS,
    LOG,
    OUTPUT,
    directory_digest,
    plain,
    rerun,
)
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import CHECKS, GRAPH, MANIFEST, RUBRIC, RunError, read_manifest, read_run_file


@dataclass(frozen=True)
class Difference:
    """A field of a check record that differs from what verifying it gives.

    Attributes:
        check: The check node's id.
        field: The record's field, or ``files`` for the files of the check's directory.
        recorded: The recorded value; for ``files``, the record's log digest.
        found: The value the re-run gives; for ``files``, the digest of the files in the directory.
    """

    check: str
    field: str
    recorded: Any
    found: Any


@dataclass(frozen=True)
class Verification:
    """What verifying a run found.

    Attributes:
        rerun: The ids of the checks run again, in graph order.
        not_rerun: The ids of the checks judged by a model, which are not run again.
        differences: Every difference found, in graph order.
    """

    rerun: tuple[str, ...]
    not_rerun: tuple[str, ...]
    differences: tuple[Difference, ...]


def _files(run_dir: Path, check_id: str) -> dict[str, bytes] | None:
    """The files of a check's directory by name, or ``None`` when one cannot be read."""
    try:
        return {name: (run_dir / CHECKS / check_id / name).read_bytes() for name in (INPUTS, OUTPUT, LOG)}
    except OSError:
        return None


def _file_differences(check_id: str, recorded: Mapping[str, Any], files: Mapping[str, bytes]) -> list[Difference]:
    """The fields of a check record that differ from what its ``inputs.json`` and ``output.json`` say.

    A record whose fields were edited after the run disagrees with the files the check wrote, even when the
    files themselves still match the record's log digest.
    """
    try:
        written = {INPUTS: json.loads(files[INPUTS]), OUTPUT: json.loads(files[OUTPUT])}
    except ValueError:
        return [Difference(check_id, f"{INPUTS} and {OUTPUT}", "JSON", "not JSON")]
    fields = {INPUTS: ("type", "target", "artifact", "inputs"), OUTPUT: ("result", "detail")}
    return [
        Difference(check_id, f"{key} ({name})", recorded.get(key), written[name].get(key))
        for name, keys in fields.items()
        if isinstance(written[name], dict)
        for key in keys
        if plain(recorded.get(key)) != plain(written[name].get(key))
    ]


def run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read the manifest of a run of either format: a model-calling run (1) or an agent-tool run (2).

    An agent-tool run is first reconciled with its call log, which rewrites any derived file a crash left
    behind; it has a graph only while it is finished, so an unfinished one is refused.

    Args:
        run_dir: The run directory.

    Returns:
        The manifest.

    Raises:
        RunError: If the directory is not a run, or is an unfinished agent-tool run.
    """
    try:
        version = json.loads((run_dir / MANIFEST).read_bytes()).get("version")
    except (OSError, ValueError, AttributeError):
        version = None
    if version != session.VERSION:
        return read_manifest(run_dir)
    session.reconcile(run_dir)
    manifest = session.read_manifest(run_dir)
    if not manifest.get("finished"):
        raise RunError("unfinished-run", "the run is not finished: call finish_run before verifying it")
    return manifest


def verify_run(run_dir: Path) -> Verification:
    """Run a run's deterministic checks again and compare each with its record.

    Args:
        run_dir: The run directory.

    Returns:
        The checks run again, those not, and every difference found.

    Raises:
        RunError: If the run directory is malformed or incomplete, or its ``graph.json`` or
            ``rubric.toml`` differs from the one the manifest records.
    """
    manifest = run_manifest(run_dir)
    graph_bytes = read_run_file(run_dir, GRAPH)
    if bytes_digest(graph_bytes) != manifest["graph_digest"]:
        raise RunError("record-mismatch", f"{GRAPH} differs from the graph the run wrote")
    rubric_bytes = read_run_file(run_dir, RUBRIC)
    if bytes_digest(rubric_bytes) != manifest["rubric"]["digest"]:
        raise RunError("record-mismatch", f"{RUBRIC} differs from the rubric the run used")
    try:
        graph: Graph = loads(graph_bytes.decode("utf-8"))
        rubric: Rubric = loads_rubric(rubric_bytes.decode("utf-8"))
    except (UnicodeDecodeError, CredenceGraphError, RubricError) as error:
        raise RunError("invalid-record", f"cannot read the run's graph or rubric: {error}") from None
    rerun_ids: list[str] = []
    not_rerun: list[str] = []
    differences: list[Difference] = []
    for node in graph:
        recorded = node.attributes.get("check")
        if not isinstance(recorded, Mapping):
            continue
        kind = recorded.get("type")
        if kind == CITATION:
            not_rerun.append(node.id)
        elif kind not in DETERMINISTIC:
            differences.append(Difference(node.id, "type", kind, f"one of {list(IMPLEMENTED)}"))
        else:
            rerun_ids.append(node.id)
            found, _ = rerun(graph, rubric, node)
            differences.extend(
                Difference(node.id, key, recorded.get(key), value)
                for key, value in found.items()
                if plain(recorded.get(key)) != plain(value)
            )
        log = recorded.get("log")
        digest = log.get("digest") if isinstance(log, Mapping) else None
        files = _files(run_dir, node.id)
        on_disk = directory_digest(files) if files is not None else None
        if on_disk != digest:
            differences.append(Difference(node.id, "files", digest, on_disk))
        if files is not None:
            differences.extend(_file_differences(node.id, recorded, files))
    return Verification(tuple(rerun_ids), tuple(not_rerun), tuple(differences))


def rubric_reuse(manifests: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """Report each rubric name and version that runs record with more than one file digest.

    Args:
        manifests: Each run's name, as the report should give it, and its manifest.

    Returns:
        One warning per such name and version, naming each digest and the runs that record it.
    """
    seen: dict[tuple[str, str], dict[str, list[str]]] = {}
    for name, manifest in manifests:
        rubric = manifest["rubric"]
        key = (str(rubric.get("name")), str(rubric.get("version")))
        seen.setdefault(key, {}).setdefault(rubric["digest"], []).append(name)
    warnings = []
    for (rubric_name, version), digests in seen.items():
        if len(digests) > 1:
            listing = "; ".join(f"{digest} in {', '.join(runs)}" for digest, runs in digests.items())
            warnings.append(
                f"rubric {rubric_name!r} version {version!r} was used with {len(digests)} different files "
                f"({listing}): values changed without a new version"
            )
    return warnings
