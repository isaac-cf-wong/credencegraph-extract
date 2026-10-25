"""Extraction runs and the run directory that records them.

A graph is never stored without the run that produced it. A run ingests a LaTeX document into the
core's chunk graph and then applies each pass of the pipeline. A pass asks the model through
``ask``; every call is written to the run directory as it happens, so the model's answers survive
whatever fails after them. The layout is::

    <run>/
      manifest.json   format "credencegraph-extract-run", version 1
      graph.json      the graph the run produced
      rubric.toml     byte copy of the rubric used
      prompts/        the template of each pass as shipped, and the rendered prompt of every call
      raw/            the record of every call: the provider's response verbatim, with its request metadata
      rejections.json the records the core refused, each with its error code
      report.json     the graph's coverage, and what each pass reports, such as hedge phrases the rubric does not map
      checks/c-NNN/   for each check the check pass ran: its inputs, the tool's output and a log
      source/         a copy of the source document, only with ``store_source``

Every node a run creates or changes carries ``attributes.extraction = {"run": <run id>, "pass":
<pass>}``; ingested nodes name the pass ``ingest``. Everything after the model calls is
deterministic, so ``replay`` rebuilds ``graph.json`` from ``raw/`` byte for byte.
"""

from __future__ import annotations

import json
import platform
import re
import secrets
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from credencegraph.core import Graph, dumps
from credencegraph.ingest.chunks import chunks_to_graph
from credencegraph.ingest.latex import LatexError, read_latex
from credencegraph.rubric import Rubric, RubricError, coverage, loads_rubric

from credencegraph_extract import provider
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.version import __version__

FORMAT = "credencegraph-extract-run"
VERSION = 1
INGEST = "ingest"

MANIFEST = "manifest.json"
GRAPH = "graph.json"
RUBRIC = "rubric.toml"
PROMPTS = "prompts"
RAW = "raw"
REJECTIONS = "rejections.json"
REPORT = "report.json"
SOURCE = "source"
CHECKS = "checks"

# The paths a pass may not write: those the run writes itself.
RESERVED = (MANIFEST, GRAPH, RUBRIC, PROMPTS, RAW, REJECTIONS, REPORT, SOURCE)

# Send a rendered prompt under an output schema and return the model's structured output with the
# request metadata of the call.
Send = Callable[[str, Mapping[str, Any]], tuple[Any, Mapping[str, Any]]]

# ``provider.call`` with the model and sampling parameters bound: prompt and schema to a record.
Call = Callable[[str, Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Outcome:
    """What a pass leads to.

    Attributes:
        graph: The graph after the pass.
        rejections: The records the pass did not apply, each a JSON object naming why.
        report: What the pass reports beside the graph, as JSON.
        files: Files to write into the run directory, by relative path; deterministic, like the graph.
    """

    graph: Graph
    rejections: Sequence[Mapping[str, Any]] = ()
    report: Mapping[str, Any] = field(default_factory=dict)
    files: Mapping[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class Ask:
    """How a pass reaches the model, and the versions the run records.

    Calling it sends one prompt and returns the model's structured output; ``answer`` returns the
    request metadata of the call beside it. A run sends each call to the provider and records it; a
    replay answers it from the record.

    Attributes:
        send: Sends one call and returns the output with the call's request metadata.
        versions: The versions the run records, by name; a replay reads them from the manifest.
    """

    send: Send
    versions: Mapping[str, str]

    def __call__(self, prompt: str, schema: Mapping[str, Any]) -> Any:
        """Send one prompt under an output schema.

        Args:
            prompt: The rendered prompt.
            schema: The output schema.

        Returns:
            The model's structured output.
        """
        return self.send(prompt, schema)[0]

    def answer(self, prompt: str, schema: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        """Send one prompt under an output schema, and say which model answered.

        Args:
            prompt: The rendered prompt.
            schema: The output schema.

        Returns:
            The model's structured output, and the request metadata of the call: provider, model,
            request id and the rest of what ``provider.call`` records.
        """
        return self.send(prompt, schema)


@dataclass(frozen=True)
class Built:
    """The result of ingesting a document and applying each pass.

    Attributes:
        graph: The graph.
        rejections: Every pass's rejections, in pass order, each with ``"pass"`` set to its pass's name.
        report: Each pass's report, by pass name.
        files: The files the passes write into the run directory, by path inside it.
    """

    graph: Graph
    rejections: list[dict[str, Any]]
    report: dict[str, Any]
    files: dict[str, bytes]


class Pass(Protocol):
    """A pass of the pipeline: it reads the graph, asks the model, and returns what it leads to.

    Attributes:
        name: The pass's name, recorded in the manifest and in ``attributes.extraction``.
        template: The prompt template as shipped, stored with the run.
    """

    name: str
    template: str

    def run(self, graph: Graph, rubric: Rubric, ask: Ask) -> Outcome:
        """Apply the pass.

        Args:
            graph: The graph so far.
            rubric: The run's rubric.
            ask: Sends one prompt to the model; the only way a pass may reach it.

        Returns:
            The graph after the pass, with what it rejected and reports.
        """
        ...


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


def graph_text(graph: Graph) -> str:
    """Return the text of ``graph.json``: the core's serialisation and a final newline.

    Args:
        graph: The graph.

    Returns:
        The text.
    """
    return dumps(graph) + "\n"


def stamp(graph: Graph, before: Graph, run_id: str, pass_name: str) -> Graph:
    """Stamp every node of ``graph`` that is new or changed since ``before`` with ``attributes.extraction``.

    Args:
        graph: The graph after a pass.
        before: The graph before it.
        run_id: The run's id.
        pass_name: The pass's name.

    Returns:
        The stamped graph.
    """
    stamped = Graph()
    mark = {"run": run_id, "pass": pass_name}
    for node in graph:
        changed = before.nodes.get(node.id) != node
        stamped.add_node(replace(node, attributes={**node.attributes, "extraction": mark}) if changed else node)
    for relation in graph.relations.values():
        stamped.add_relation(relation)
    return stamped


def build_graph(  # noqa: PLR0913, PLR0917 - the run's inputs, each needed by ingest or the passes
    source: Path,
    document: str,
    rubric: Rubric,
    run_id: str,
    passes: Sequence[Pass],
    asker: Callable[[Pass], Ask],
) -> Built:
    """Ingest a document and apply each pass, stamping what each one creates or changes.

    Args:
        source: The main ``.tex`` file.
        document: The document id of every anchor.
        rubric: The rubric; its ``initial`` table types the chunks.
        run_id: The run's id.
        passes: The pipeline.
        asker: Gives each pass its ``ask``.

    Returns:
        The graph, with what the passes rejected and report.

    Raises:
        RunError: If the document cannot be read.
    """
    try:
        chunks = read_latex(source)
    except (OSError, UnicodeDecodeError, LatexError) as error:
        raise RunError("invalid-source", f"cannot read {str(source)!r}: {error}") from None
    graph, _ = chunks_to_graph(chunks, document, rubric.initial)
    graph = stamp(graph, Graph(), run_id, INGEST)
    rejections: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    files: dict[str, bytes] = {}
    for item in passes:
        outcome = item.run(graph, rubric, asker(item))
        graph = stamp(outcome.graph, graph, run_id, item.name)
        rejections.extend({"pass": item.name, **rejection} for rejection in outcome.rejections)
        report[item.name] = dict(outcome.report)
        for name, data in outcome.files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] in RESERVED:
                raise RunError("invalid-file", f"pass {item.name!r} may not write {name!r} in the run directory")
            files[name] = data
    return Built(graph, rejections, report, files)


def _write_json(path: Path, value: Any) -> bytes:
    """Write ``value`` as indented JSON with a final newline, and return the bytes written."""
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


def versions() -> dict[str, str]:
    """Return the versions a run depends on: Python, the core, this package, the provider SDKs and sympy.

    Returns:
        The versions by name.
    """
    return {
        "python": platform.python_version(),
        "credencegraph": version("credencegraph"),
        "credencegraph-extract": __version__,
        **{sdk: version(sdk) for sdk in provider.PROVIDERS},
        "sympy": version("sympy"),
    }


def new_run_id() -> str:
    """Return a fresh run id: the UTC time and a random suffix.

    Returns:
        The id, such as ``20261024T153000Z-1a2b3c4d``.
    """
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def run_extraction(  # noqa: PLR0913 - each argument is a separate choice of the caller
    source: Path,
    run_dir: Path,
    rubric_path: Path,
    *,
    call: Call,
    passes: Sequence[Pass],
    document: str | None = None,
    store_source: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the pipeline on a document and write its run directory.

    Args:
        source: The main ``.tex`` file of the document.
        run_dir: The run directory to create; it must not exist or be empty.
        rubric_path: The rubric file.
        call: Sends one call to the provider and returns its record.
        passes: The pipeline, such as ``pipeline.PASSES``.
        document: The document id of every anchor; the source's file name by default.
        store_source: Copy the source document into the run directory.
        run_id: The run's id; a fresh one by default.

    Returns:
        The manifest.

    Raises:
        RunError: If the run directory is in use, or the source or the rubric cannot be read.
    """
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunError("run-exists", f"{str(run_dir)!r} exists and is not empty")
    if source.suffix.lower() != ".tex":
        raise RunError("invalid-source", f"{str(source)!r} is not a LaTeX file")
    run_id = run_id or new_run_id()
    created_at = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}"
    try:
        rubric_bytes = rubric_path.read_bytes()
        rubric = loads_rubric(rubric_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, RubricError) as error:
        raise RunError("invalid-rubric", f"cannot read {str(rubric_path)!r}: {error}") from None
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise RunError("invalid-source", f"cannot read {str(source)!r}: {error}") from None
    for name in (PROMPTS, RAW):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / RUBRIC).write_bytes(rubric_bytes)
    if store_source:
        (run_dir / SOURCE).mkdir()
        shutil.copyfile(source, run_dir / SOURCE / source.name)

    pass_entries: list[dict[str, Any]] = []
    count = 0
    recorded_versions = versions()

    def asker(item: Pass) -> Ask:
        template = item.template.encode("utf-8")
        (run_dir / PROMPTS / f"{item.name}.template.txt").write_bytes(template)
        entry: dict[str, Any] = {"name": item.name, "template_digest": bytes_digest(template), "calls": []}
        pass_entries.append(entry)

        def send(prompt: str, schema: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
            nonlocal count
            count += 1
            prompt_path = f"{PROMPTS}/{count:04d}.txt"
            raw_path = f"{RAW}/{count:04d}.json"
            prompt_bytes = prompt.encode("utf-8")
            (run_dir / prompt_path).write_bytes(prompt_bytes)
            record = {"pass": item.name, **call(prompt, schema)}
            raw_bytes = _write_json(run_dir / raw_path, record)
            request = record["request"]
            entry["calls"].append(
                {
                    "prompt": prompt_path,
                    "prompt_digest": bytes_digest(prompt_bytes),
                    "raw": raw_path,
                    "raw_digest": bytes_digest(raw_bytes),
                    **{key: request[key] for key in ("provider", "base_url", "model", "parameters", "sdk_version")},
                }
            )
            return provider.output(record), request

        return Ask(send, recorded_versions)

    built = build_graph(source, document or source.name, rubric, run_id, passes, asker)
    graph_bytes = graph_text(built.graph).encode("utf-8")
    (run_dir / GRAPH).write_bytes(graph_bytes)
    for name, data in built.files.items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_bytes(data)
    _write_json(run_dir / REJECTIONS, built.rejections)
    _write_json(run_dir / REPORT, {"coverage": coverage(built.graph, rubric), "passes": built.report})
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "run": run_id,
        "created_at": created_at,
        "source": {
            "document": document or source.name,
            "file": source.name,
            "digest": bytes_digest(source_bytes),
            "stored": store_source,
        },
        "versions": recorded_versions,
        "rubric": {**rubric.identity(), "digest": bytes_digest(rubric_bytes)},
        "passes": pass_entries,
        "graph_digest": bytes_digest(graph_bytes),
    }
    _write_json(run_dir / MANIFEST, manifest)
    return manifest


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
    """Read and check a run's manifest: its format, its version, and every field ``replay`` reads.

    Args:
        run_dir: The run directory.

    Returns:
        The manifest.

    Raises:
        RunError: If there is no manifest, it is not JSON, not of this format and version, or a
            field ``replay`` reads is missing or of the wrong type.
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


def _recorded_output(
    record: Any, prompt: str, schema: Mapping[str, Any], name: str, where: str
) -> tuple[Any, Mapping[str, Any]]:
    """Return the output and request metadata of a recorded call, after checking it answers ``prompt`` under ``schema``."""
    request = record.get("request") if isinstance(record, dict) else None
    if not isinstance(request, dict) or "prompt_digest" not in request or "output_schema" not in request:
        raise RunError("invalid-record", f"{where} has no request metadata")
    if bytes_digest(prompt.encode("utf-8")) != request["prompt_digest"] or dict(schema) != request["output_schema"]:
        raise RunError("unrecorded-call", f"pass {name!r} asks a call that differs from {where}")
    try:
        return provider.output(record), request
    except provider.ProviderError as error:
        raise RunError("invalid-record", f"{where}: {error}") from None


def replay(run_dir: Path, *, passes: Sequence[Pass], source: Path | None = None) -> bytes:
    """Rebuild a run's graph from its recorded calls, without calling the provider.

    The source is the stored copy if the run kept one, else ``source``; its digest must match the
    manifest's. Each pass's calls are answered from ``raw/`` in order, and each must ask the prompt
    and schema that was recorded.

    Args:
        run_dir: The run directory.
        passes: The pipeline; its pass names must be the manifest's, in order.
        source: The source document, for a run that did not store it.

    Returns:
        The rebuilt ``graph.json`` bytes.

    Raises:
        RunError: If the run directory is malformed or incomplete; if its rubric, source or records
            do not match the manifest; or if a pass asks a call that was not recorded.
    """
    manifest = read_manifest(run_dir)
    recorded = manifest["source"]
    if recorded["stored"]:
        source = run_dir / SOURCE / recorded["file"]
    elif source is None:
        raise RunError("missing-source", "the run did not store its source: pass the source document")
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise RunError("missing-source", f"cannot read {str(source)!r}: {error.strerror or error}") from None
    if bytes_digest(source_bytes) != recorded["digest"]:
        raise RunError("source-mismatch", f"{str(source)!r} is not the document the run read")
    rubric_bytes = read_run_file(run_dir, RUBRIC)
    if bytes_digest(rubric_bytes) != manifest["rubric"]["digest"]:
        raise RunError("record-mismatch", f"{RUBRIC} differs from the rubric the run used")
    try:
        rubric = loads_rubric(rubric_bytes.decode("utf-8"))
    except (UnicodeDecodeError, RubricError) as error:
        raise RunError("invalid-rubric", f"cannot read {RUBRIC}: {error}") from None
    names = [entry["name"] for entry in manifest["passes"]]
    if names != [item.name for item in passes]:
        raise RunError("unknown-pass", f"the run's passes are {names}, not {[item.name for item in passes]}")
    entries = {entry["name"]: entry for entry in manifest["passes"]}
    recorded_versions = manifest["versions"]

    pending: dict[str, Iterator[dict[str, Any]]] = {}

    def asker(item: Pass) -> Ask:
        calls = pending[item.name] = iter(entries[item.name]["calls"])

        def send(prompt: str, schema: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
            entry = next(calls, None)
            if entry is None:
                raise RunError("unrecorded-call", f"pass {item.name!r} asks more calls than were recorded")
            raw_bytes = read_run_file(run_dir, entry["raw"])
            if bytes_digest(raw_bytes) != entry["raw_digest"]:
                raise RunError("record-mismatch", f"{entry['raw']} differs from the record the run wrote")
            try:
                record = json.loads(raw_bytes)
            except ValueError as error:
                raise RunError("invalid-record", f"{entry['raw']} is not JSON: {error}") from None
            return _recorded_output(record, prompt, schema, item.name, entry["raw"])

        return Ask(send, recorded_versions)

    graph = build_graph(source, recorded["document"], rubric, manifest["run"], passes, asker).graph
    for name, calls in pending.items():
        if next(calls, None) is not None:
            raise RunError("unrecorded-call", f"pass {name!r} asks fewer calls than were recorded")
    return graph_text(graph).encode("utf-8")
