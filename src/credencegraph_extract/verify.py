"""Verifying runs: the deterministic checks run again, and rubric versions compared across runs.

``replay`` shows that a run's graph follows from its recorded calls; ``verify`` looks at the checks
one by one, from the run directory alone, without the source. Each ``recompute-numbers``,
``rederive-maths`` and ``rederive-maths-numeric`` check of the graph is run again from the inputs its
record holds, and any field of the record the re-run does not reproduce is reported: a result, a
detail, the digest of the check's files. The files in ``checks/c-NNN/`` are compared with the digest
the record holds. Citation checks are judged by a model, so they are not run again: ``replay``
answers them from their recorded responses.

A rubric's ``version`` names its values, so two runs that record one rubric name and version with
different file digests used values that changed without a new version. ``rubric_reuse`` reports
each such name and version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from credencegraph.core import CredenceGraphError, Graph, loads
from credencegraph.rubric import Rubric, RubricError, loads_rubric

from credencegraph_extract.checks import DETERMINISTIC, INPUTS, LOG, OUTPUT, directory_digest, plain, rerun
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import CHECKS, GRAPH, RUBRIC, RunError, read_manifest, read_run_file


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


def _files_digest(run_dir: Path, check_id: str) -> str | None:
    """The digest of the files in a check's directory, or ``None`` when one cannot be read."""
    try:
        files = {name: (run_dir / CHECKS / check_id / name).read_bytes() for name in (INPUTS, OUTPUT, LOG)}
    except OSError:
        return None
    return directory_digest(files)


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
    manifest = read_manifest(run_dir)
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
        if recorded.get("type") not in DETERMINISTIC:
            not_rerun.append(node.id)
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
        on_disk = _files_digest(run_dir, node.id)
        if on_disk != digest:
            differences.append(Difference(node.id, "files", digest, on_disk))
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
