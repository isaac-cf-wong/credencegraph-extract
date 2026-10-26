"""Agent-tool runs: the run directory of format 2, its call log, its lock and its replay.

An agent in its own harness builds a graph through the package's tools. Each tool call that changes
the run is one **write**: it takes the run's lock, applies an operation to the current state, and
commits. The commit point is one line appended to ``calls.jsonl``, so the log is the record of truth
and every other file is derived from it:

1. the new state is written to ``state.json.tmp``;
2. the call's line is appended to ``calls.jsonl`` and flushed to disk;
3. ``state.json.tmp`` is renamed to ``state.json``, and ``manifest.json`` is rewritten the same way.

A process that dies before step 2 leaves no trace; one that dies after it leaves a ``state.json``
that names fewer calls than the log, which the next tool call finds and rebuilds by applying the
logged calls to the graph ``start_run`` ingested (``ingested.json``). Ids are allocated from the
state under the lock, so the calls of several agents, applied in log order, give the same ids again.

The layout::

    <run>/
      manifest.json   format "credencegraph-extract-run", version 2
      calls.jsonl     every write, in the order it took effect
      state.json      the state after the last call, and how many calls it reflects
      ingested.json   the graph start_run ingested, the start of every rebuild
      rubric.toml     byte copy of the rubric
      cited/          byte copies of the cited works, made by start_run
      source/         a copy of the document, only with store_source
      graph.json      the graph, only while the run is finished
      rejections.json every refused record, only while the run is finished
      report.json     the core's coverage report, only while the run is finished

``replay`` re-ingests the document and applies every logged call again, checking each call's result
digest; the graph it ends with must be byte-identical to the run's. The log records no local path:
a file is named by its name and digest.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PureWindowsPath
from typing import Any

from credencegraph.core import Graph, dumps
from credencegraph.core.serialization import graph_from_dict, graph_to_dict
from credencegraph.ingest.chunks import chunks_to_graph
from credencegraph.ingest.latex import LatexError, _Reader
from credencegraph.rubric import Rubric, RubricError, coverage, loads_rubric
from credencegraph.rubric.model import DOCUMENT
from credencegraph.rubric.validate import origin_of

from credencegraph_extract.assemble import edge_rules_not_requires, premises_rubric_problem
from credencegraph_extract.checks import IMPLEMENTED
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import CHECKS, RunError, new_run_id
from credencegraph_extract.version import __version__

FORMAT = "credencegraph-extract-run"
VERSION = 2

MANIFEST = "manifest.json"
CALLS = "calls.jsonl"
STATE = "state.json"
INGESTED = "ingested.json"
RUBRIC = "rubric.toml"
CITED = "cited"
SOURCE = "source"
LOCK = ".lock"
GRAPH = "graph.json"
REJECTIONS = "rejections.json"
REPORT = "report.json"
# The files a finished run has, derived from its state.
OUTPUTS = (GRAPH, REJECTIONS, REPORT)

START = "start_run"
FINISH = "finish_run"

# How long a tool call waits for the run's lock before it reports the run busy.
LOCK_TIMEOUT = 30.0


@dataclass(frozen=True)
class State:
    """What a run holds between tool calls.

    Attributes:
        graph: The graph so far.
        rejections: Every refused record or proposal, with the call that sent it.
        finished: Whether the last write was ``finish_run``; any other write makes it ``False``.
        files: The files of the checks run, by path in the run (``checks/c-NNN/<name>``), as text.
    """

    graph: Graph
    rejections: tuple[Mapping[str, Any], ...] = ()
    finished: bool = False
    files: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    """What an operation may read beside the state: the run and its fixed inputs.

    Attributes:
        run_dir: The run directory.
        run_id: The run's id.
        rubric: The run's rubric.
        call: The number of the call being applied, from 1 for ``start_run``.
        sources: Where a replay finds fetched sources the run did not store, as files named by their
            digest; ``None`` in a live run.
    """

    run_dir: Path
    run_id: str
    rubric: Rubric
    call: int
    sources: Path | None = None


# An operation applies one write's arguments to the state and returns the new state and the tool's
# result. It must be deterministic: replay applies it again and compares the result's digest.
Operation = Callable[[Context, State, Mapping[str, Any]], tuple[State, dict[str, Any]]]

OPERATIONS: dict[str, Operation] = {}


def operation(name: str) -> Callable[[Operation], Operation]:
    """Register a write tool's operation under its name.

    Args:
        name: The tool's name, as the call log records it.

    Returns:
        A decorator registering the operation.
    """

    def register(fn: Operation) -> Operation:
        OPERATIONS[name] = fn
        return fn

    return register


def stamp(graph: Graph, before: Graph, mark: Mapping[str, Any]) -> Graph:
    """Stamp every node of ``graph`` that is new or changed since ``before`` with ``attributes.extraction``.

    Args:
        graph: The graph after a change.
        before: The graph before it.
        mark: The stamp, such as ``{"run": <run id>, "tool": <tool>}``.

    Returns:
        The stamped graph.
    """
    stamped = Graph()
    for node in graph:
        changed = before.nodes.get(node.id) != node
        attributes = {**node.attributes, "extraction": dict(mark)}
        stamped.add_node(replace(node, attributes=attributes) if changed else node)
    for relation in graph.relations.values():
        stamped.add_relation(relation)
    return stamped


def versions() -> dict[str, str]:
    """Return the versions a run depends on: Python, the core, this package and sympy.

    Returns:
        The versions by name.
    """
    return {
        "python": platform.python_version(),
        "credencegraph": version("credencegraph"),
        "credencegraph-extract": __version__,
        "sympy": version("sympy"),
    }


def _json(value: Any) -> bytes:
    """Encode a value as the indented JSON a run writes."""
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _line(value: Any) -> bytes:
    """Encode a value as one line of ``calls.jsonl``."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def _durable_write(path: Path, data: bytes) -> None:
    """Write a file and flush it to disk."""
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _replace(path: Path, data: bytes) -> None:
    """Write a file by writing a temporary one beside it and renaming it into place."""
    temporary = path.with_name(path.name + ".tmp")
    _durable_write(temporary, data)
    temporary.replace(path)


@contextmanager
def lock(run_dir: Path, timeout: float = LOCK_TIMEOUT) -> Iterator[None]:
    """Hold the run's lock: one tool call at a time, across processes.

    The operating system releases the lock when its process exits, so a crashed agent leaves none
    behind. A call that cannot take the lock within ``timeout`` seconds reports the run busy.

    Args:
        run_dir: The run directory.
        timeout: How long to wait, in seconds.

    Yields:
        Nothing; the lock is held until the block ends.

    Raises:
        RunError: If the lock is not free within ``timeout``.
    """
    with (run_dir / LOCK).open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while not _try_lock(handle):
            if time.monotonic() >= deadline:
                raise RunError("run-busy", f"another tool call holds the lock of {run_dir.name!r}")
            time.sleep(0.05)
        try:
            yield
        finally:
            _unlock(handle)


if sys.platform == "win32":  # pragma: no cover - the lock of the platform the tests do not run on
    import msvcrt

    def _try_lock(handle: Any) -> bool:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(handle: Any) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: Any) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(handle: Any) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(run_dir: Path, name: str) -> bytes:
    """Read a file of a run directory, or refuse an incomplete run."""
    try:
        return (run_dir / name).read_bytes()
    except OSError as error:
        raise RunError("incomplete-run", f"cannot read {name} of the run: {error.strerror or error}") from None


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a format-2 run's manifest.

    Args:
        run_dir: The run directory.

    Returns:
        The manifest.

    Raises:
        RunError: If the directory is not a run of this format and version.
    """
    if not (run_dir / MANIFEST).is_file():
        raise RunError("not-a-run", f"{run_dir.name!r} has no {MANIFEST}")
    try:
        manifest = json.loads(_read(run_dir, MANIFEST))
    except ValueError as error:
        raise RunError("invalid-manifest", f"{MANIFEST} is not JSON: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise RunError("not-a-run", f"{MANIFEST} is not a {FORMAT} manifest")
    if manifest.get("version") != VERSION:
        raise RunError("unsupported-version", f"{MANIFEST} is version {manifest.get('version')!r}, not {VERSION}")
    return manifest


def read_calls(run_dir: Path) -> list[dict[str, Any]]:
    """Read the call log.

    Args:
        run_dir: The run directory.

    Returns:
        The calls, in the order they took effect.

    Raises:
        RunError: If a line is not a call object, or the calls are not numbered 1, 2, … from ``start_run``.
    """
    calls = []
    for number, line in enumerate(_read(run_dir, CALLS).splitlines(), start=1):
        try:
            call = json.loads(line)
        except ValueError as error:
            raise RunError("invalid-record", f"{CALLS} line {number} is not JSON: {error}") from None
        if not isinstance(call, dict) or call.get("seq") != number or not isinstance(call.get("tool"), str):
            raise RunError("invalid-record", f"{CALLS} line {number} is not call {number}")
        calls.append(call)
    if not calls or calls[0]["tool"] != START:
        raise RunError("invalid-record", f"{CALLS} does not begin with {START}")
    return calls


def _rubric(run_dir: Path, manifest: Mapping[str, Any]) -> Rubric:
    """The run's rubric, refused if its copy differs from the digest the manifest records."""
    data = _read(run_dir, RUBRIC)
    if bytes_digest(data) != manifest["rubric"]["digest"]:
        raise RunError("record-mismatch", f"{RUBRIC} differs from the rubric the run started with")
    try:
        return loads_rubric(data.decode("utf-8"))
    except (UnicodeDecodeError, RubricError) as error:
        raise RunError("invalid-rubric", f"cannot read {RUBRIC}: {error}") from None


def _graph(data: bytes, name: str) -> Graph:
    """Parse a graph stored in a run file."""
    try:
        return graph_from_dict(json.loads(data)["graph"])
    except (ValueError, KeyError, TypeError) as error:
        raise RunError("invalid-record", f"{name} holds no graph: {error}") from None


def _state_file(state: State, calls: int, log_digest: str) -> bytes:
    """The content of ``state.json``: the state and the calls it reflects."""
    return _json(
        {
            "calls": calls,
            "log_digest": log_digest,
            "finished": state.finished,
            "rejections": [dict(item) for item in state.rejections],
            "files": dict(sorted(state.files.items())),
            "graph": graph_to_dict(state.graph),
        }
    )


def _operations() -> dict[str, Operation]:
    """The registered operations, the tools' included.

    The tools register their operations when their module is imported; a process that only writes,
    replays or rebuilds through this module may not have imported it.
    """
    from credencegraph_extract import tools  # noqa: F401, PLC0415 - registers the operations; tools imports this module

    return OPERATIONS


def _apply(context: Context, state: State, call: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Apply one logged call's operation."""
    fn = _operations().get(call["tool"])
    if fn is None:
        raise RunError("unknown-tool", f"call {call['seq']} is of the unknown tool {call['tool']!r}")
    after, result = fn(context, state, call.get("args") or {})
    mark = {"run": context.run_id, "tool": call["tool"]}
    finished = after.finished if call["tool"] == FINISH else False
    return replace(after, graph=stamp(after.graph, state.graph, mark), finished=finished), result


def _rebuild(run_dir: Path, manifest: Mapping[str, Any], rubric: Rubric, calls: list[dict[str, Any]]) -> State:
    """The state the log leads to, from the ingested graph, each call checked against its log line.

    A log whose calls no longer give the results and graphs their lines record (edited by hand) is
    refused rather than adopted.
    """
    graph = _graph(_read(run_dir, INGESTED), INGESTED)
    state = State(graph)
    _check_digests(calls[0], state, _start_result(graph, rubric, manifest["run"]))
    for call in calls[1:]:
        context = Context(run_dir, manifest["run"], rubric, call["seq"])
        state, result = _apply(context, state, call)
        _check_digests(call, state, result)
    return state


def _load(run_dir: Path, manifest: Mapping[str, Any], calls: list[dict[str, Any]]) -> tuple[State, bool]:
    """The state after every logged call, and whether ``state.json`` held it as the log's last line records.

    ``state.json`` is used only when it was written for this exact log (its digest) and its graph,
    rejections and finished flag have the digests the last line records; otherwise the state is rebuilt
    from the log, each call checked against its line.
    """
    try:
        saved = json.loads(_read(run_dir, STATE))
    except (RunError, ValueError):
        saved = None
    log_digest = bytes_digest(_read(run_dir, CALLS))
    if isinstance(saved, dict) and saved.get("calls") == len(calls) and saved.get("log_digest") == log_digest:
        try:
            state = State(
                graph_from_dict(saved["graph"]),
                tuple(saved.get("rejections") or ()),
                bool(saved.get("finished")),
                dict(saved.get("files") or {}),
            )
        except (KeyError, TypeError, ValueError):
            state = None
        last = calls[-1]
        if state is not None and all(last.get(name) == value for name, value in _state_digests(state).items()):
            return state, True
    return _rebuild(run_dir, manifest, _rubric(run_dir, manifest), calls), False


def current_state(run_dir: Path) -> State:
    """Return the run's state, rebuilt from the log when ``state.json`` does not reflect all of it.

    The caller holds the lock.

    Args:
        run_dir: The run directory.

    Returns:
        The state after every logged call.

    Raises:
        RunError: If the run directory is malformed or incomplete.
    """
    manifest = read_manifest(run_dir)
    return _load(run_dir, manifest, read_calls(run_dir))[0]


def _manifest(manifest: Mapping[str, Any], calls: list[Mapping[str, Any]], state: State, log_digest: str) -> bytes:
    """The manifest after a commit: the fixed fields and what the log and state now say."""
    models = sorted({call["model"] for call in calls if isinstance(call.get("model"), str)})
    agents = sorted({call["agent"] for call in calls if isinstance(call.get("agent"), str)})
    graph_digest = bytes_digest(_graph_text(state.graph))
    return _json(
        {
            **manifest,
            "calls": len(calls),
            "log_digest": log_digest,
            "models": models,
            "agents": agents,
            "finished": state.finished,
            "graph_digest": graph_digest,
        }
    )


def _identity(model: Any, agent: Any, note: Any) -> dict[str, Any]:
    """Check and return who made a write: the model the harness reports, an agent name and a note."""
    if not isinstance(model, str) or not model.strip():
        raise RunError(
            "invalid-argument", "model must be the id of the model making the call, as the harness reports it"
        )
    for name, value in (("agent", agent), ("note", note)):
        if value is not None and not isinstance(value, str):
            raise RunError("invalid-argument", f"{name} must be a string")
    return {"model": model, "agent": agent, "note": note}


def write(  # noqa: PLR0913 - the call and who makes it
    run_dir: Path,
    tool: str,
    args: Mapping[str, Any],
    *,
    model: str,
    agent: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Make one write to a run: apply a tool's operation under the lock and commit it.

    Args:
        run_dir: The run directory.
        tool: The tool, a registered operation.
        args: Its arguments, as JSON values.
        model: The model making the call, as the harness reports it.
        agent: A name for the agent, for runs with several.
        note: Free text kept with the call, never used in inference.

    Returns:
        The tool's result.

    Raises:
        RunError: If the tool is unknown, the run is busy or malformed, or the operation refuses the
            call as a whole.
    """
    identity = _identity(model, agent, note)
    if tool not in _operations():
        raise RunError("unknown-tool", f"{tool!r} is not a tool that changes a run")
    with lock(run_dir):
        manifest = read_manifest(run_dir)
        rubric = _rubric(run_dir, manifest)
        calls = read_calls(run_dir)
        state = _reconcile(run_dir, manifest, rubric, calls)
        seq = len(calls) + 1
        # The call is applied to its arguments as the log will hold them, so a replay, which reads them
        # back from the log, gives the same result whatever order the agent wrote their keys in.
        call = {"seq": seq, "tool": tool, "args": json.loads(_line(dict(args))), **identity}
        after, result = _apply(Context(run_dir, manifest["run"], rubric, seq), state, call)
        line = _line({**call, "call_digest": _call_digest(call), **_digests(after, result), "at": _now()})
        log_digest = bytes_digest(_read(run_dir, CALLS) + line)
        temporary = run_dir / (STATE + ".tmp")
        _durable_write(temporary, _state_file(after, seq, log_digest))
        with (run_dir / CALLS).open("ab") as log:
            log.write(line)
            log.flush()
            os.fsync(log.fileno())
        temporary.replace(run_dir / STATE)
        _replace(run_dir / MANIFEST, _manifest(manifest, [*calls, json.loads(line)], after, log_digest))
        _check_files(run_dir, after)
        _outputs(run_dir, rubric, after)
    return result


def _graph_text(graph: Graph) -> bytes:
    """A graph as the core serialises it, with a final newline."""
    return (dumps(graph) + "\n").encode("utf-8")


def _digests(state: State, result: Mapping[str, Any]) -> dict[str, str]:
    """What a log line records of a call's effect: the digests of its result and of the graph it left.

    The result is digested with sorted keys, so its digest does not depend on how its own keys were
    built; the call itself is applied to its arguments as the log holds them (``write``).
    """
    canonical = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
    return {"result_digest": bytes_digest(canonical), **_state_digests(state)}


def _state_digests(state: State) -> dict[str, Any]:
    """What a log line records of the state a call left: its graph, its rejections, its check files and whether it is finished.

    A cached ``state.json`` is trusted only when these match the last line, so a derived file cannot
    stand in for what the log leads to.
    """
    rejections = json.dumps([dict(item) for item in state.rejections], ensure_ascii=False, sort_keys=True)
    files = json.dumps(dict(state.files), ensure_ascii=False, sort_keys=True)
    return {
        "graph_digest": bytes_digest(_graph_text(state.graph)),
        "rejections_digest": bytes_digest(rejections.encode("utf-8")),
        "files_digest": bytes_digest(files.encode("utf-8")),
        "finished": state.finished,
    }


# The fields of a log line that make up the call itself, which ``call_digest`` covers.
CALL_FIELDS = ("seq", "tool", "args", "model", "agent", "note")


def _call_digest(call: Mapping[str, Any]) -> str:
    """The digest of a call's own fields, so a line whose call was edited by hand is refused.

    It covers what no result digest does, such as the harness, the model and the note. A line edited
    together with this digest is a forgery of the run's files, which the run directory does not guard
    against: it has no key.
    """
    fields = {name: call.get(name) for name in CALL_FIELDS}
    return bytes_digest(json.dumps(fields, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8"))


def _check_digests(call: Mapping[str, Any], state: State, result: Mapping[str, Any]) -> None:
    """Refuse a replayed call whose own fields, result or state differ from what its log line recorded."""
    if call.get("call_digest") != _call_digest(call):
        raise RunError("record-mismatch", f"call {call['seq']} ({call['tool']}) was changed after it was logged")
    for name, digest in _digests(state, result).items():
        if call.get(name) != digest:
            what = name.removesuffix("_digest")
            msg = f"call {call['seq']} ({call['tool']}) gives another {what} than its log line records"
            raise RunError("record-mismatch", msg)


def _reconcile(run_dir: Path, manifest: Mapping[str, Any], rubric: Rubric, calls: list[dict[str, Any]]) -> State:
    """Bring every file derived from the log back in line with it, and return the state; the caller holds the lock.

    A process that dies after a call's log line is appended can leave ``state.json``, ``manifest.json``
    or the outputs behind the log: the state rebuilt, the manifest saying the run is unfinished while
    its last call finished it, or a finished run without its ``graph.json``. Each is rewritten from
    the state the log leads to.
    """
    state, trusted = _load(run_dir, manifest, calls)
    log_digest = bytes_digest(_read(run_dir, CALLS))
    if not trusted:
        _replace(run_dir / STATE, _state_file(state, len(calls), log_digest))
    if manifest.get("log_digest") != log_digest or manifest.get("finished") != state.finished:
        _replace(run_dir / MANIFEST, _manifest(manifest, calls, state, log_digest))
    _check_files(run_dir, state)
    _outputs(run_dir, rubric, state)
    return state


def reconcile(run_dir: Path) -> None:
    """Bring a run's state, manifest and outputs back in line with its call log, under the lock.

    Args:
        run_dir: The run directory.

    Raises:
        RunError: If the run directory is malformed or incomplete.
    """
    with lock(run_dir):
        manifest = read_manifest(run_dir)
        _reconcile(run_dir, manifest, _rubric(run_dir, manifest), read_calls(run_dir))


def _check_files(run_dir: Path, state: State) -> None:
    """Write the files of the checks run from the state, each that is missing or differs.

    Checks are only ever added, so the files of a check, once written, stay; a file edited by hand is
    written again from the state.
    """
    for path, text in state.files.items():
        target = run_dir / path
        if not path.startswith(f"{CHECKS}/") or ".." in Path(path).parts:
            raise RunError("invalid-record", f"the check file {path!r} is not under {CHECKS}/")
        data = text.encode("utf-8")
        if not target.is_file() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            _replace(target, data)


def _outputs(run_dir: Path, rubric: Rubric, state: State) -> None:
    """Write a finished run's outputs from its state, or remove them from a run a later write unfinished.

    They are derived from the state, never read back: ``graph.json`` (the graph as the core serialises
    it), ``rejections.json`` (every refused record, with the call that sent it) and ``report.json`` (the
    core's coverage report). An unfinished run has none, so a stale graph is never taken for its result.
    """
    if not state.finished:
        for name in OUTPUTS:
            (run_dir / name).unlink(missing_ok=True)
        return
    _replace(run_dir / GRAPH, _graph_text(state.graph))
    _replace(run_dir / REJECTIONS, _json([dict(item) for item in state.rejections]))
    _replace(run_dir / REPORT, _json({"coverage": coverage(state.graph, rubric)}))


def _now() -> str:
    """The time of a call, in UTC to the second."""
    return f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}"


def rubric_problems(rubric: Rubric) -> list[str]:
    """Return why a rubric cannot be used for an agent-tool run, or nothing if it can.

    Args:
        rubric: The rubric.

    Returns:
        Every problem: an edge rule that creates a relation other than ``requires``, and a
        ``[premises]`` table without the analyst ``assumption`` type it needs.
    """
    problems = []
    refused = edge_rules_not_requires(rubric)
    if refused:
        problems.append(
            f"the edge rules of {refused} create relations other than requires, which would let a document "
            "raise the credence of its own claims"
        )
    if rubric.premises is not None:
        problem = premises_rubric_problem(rubric)
        if problem:
            problems.append(f"{problem}, which its [premises] table needs")
    return problems


def _cited_copies(cited: Path) -> dict[str, bytes]:
    """Read the cited works to copy: the regular files directly in a directory, never a link."""
    if not cited.is_dir():
        raise RunError("invalid-cited", f"the cited works {cited.name!r} are not a directory")
    copies = {}
    for entry in sorted(cited.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            kind = "a link" if entry.is_symlink() else "not a regular file"
            raise RunError("invalid-cited", f"{entry.name!r} among the cited works is {kind}; copy only files")
        copies[entry.name] = entry.read_bytes()
    return copies


def _ingest(source: Path, data: bytes, document: str, rubric: Rubric) -> Graph:
    """Read a LaTeX document into the core's chunk graph, parsing ``data`` as its main file.

    The main file is parsed from the bytes already read and digested, never read again, so the graph,
    the recorded digest and any stored copy are of the same bytes. Files it inputs are read from the
    main file's directory as usual. The core's reader keeps each file's text once; it is given the main
    file's text through that cache, which is not public API (a public ``text`` argument is requested of
    the core).
    """
    try:
        reader = _Reader(source)
        reader.texts[reader.main] = data.decode("utf-8")
        reader.collect_definitions(reader.main)
        reader.install_definitions()
        reader.read_file(reader.main, body_only=True)
    except (OSError, UnicodeDecodeError, LatexError) as error:
        raise RunError("invalid-source", f"cannot read {source.name!r}: {error}") from None
    return chunks_to_graph(reader.chunks, document, rubric.initial)[0]


def _start_result(graph: Graph, rubric: Rubric, run_id: str) -> dict[str, Any]:
    """What ``start_run`` tells the agent: the chunks, the types, the hedges and the check types."""
    chunks = []
    for node in graph:
        if origin_of(node) != DOCUMENT:
            continue
        chunk = {"id": node.id, "unit": node.attributes.get("unit"), "type": node.kind, "text": node.statement}
        for key in ("label", "key"):
            if node.attributes.get(key):
                chunk[key] = node.attributes[key]
        chunks.append(chunk)
    types = []
    for name, spec in rubric.types.items():
        if DOCUMENT not in spec.origins:
            continue
        entry: dict[str, Any] = {"name": name, "description": spec.description, "assess": spec.assess}
        entry["form"] = {
            field.name: {"type": field.type, "required": field.required, "values": list(field.values)}
            for field in spec.form.values()
        }
        entry["rests_on"] = list(spec.rests_on.types) if spec.rests_on is not None else None
        types.append(entry)
    checks = [{"name": name, "targets": list(check.targets)} for name, check in rubric.checks.items()]
    return {
        "run": run_id,
        "chunks": chunks,
        "types": types,
        "hedges": list(rubric.hedges),
        "checks": [check for check in checks if check["name"] in IMPLEMENTED],
        "not_runnable": [check["name"] for check in checks if check["name"] not in IMPLEMENTED],
    }


@dataclass(frozen=True)
class _Start:
    """What ``start_run`` read, before anything is written."""

    rubric_bytes: bytes
    rubric: Rubric
    source_bytes: bytes
    cited: dict[str, bytes] = field(default_factory=dict)


def _read_start(source: Path, rubric_path: Path, cited: Path | None) -> _Start:
    """Read and check everything ``start_run`` needs, refusing before the run directory is touched."""
    if source.suffix.lower() != ".tex":
        raise RunError("invalid-source", f"{source.name!r} is not a LaTeX file")
    try:
        rubric_bytes = rubric_path.read_bytes()
        rubric = loads_rubric(rubric_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, RubricError) as error:
        raise RunError("invalid-rubric", f"cannot read {rubric_path.name!r}: {error}") from None
    problems = rubric_problems(rubric)
    if problems:
        raise RunError("invalid-rubric", f"rubric {rubric.name!r} cannot be used: " + "; ".join(problems))
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise RunError("invalid-source", f"cannot read {source.name!r}: {error.strerror or error}") from None
    return _Start(rubric_bytes, rubric, source_bytes, _cited_copies(cited) if cited is not None else {})


def _populate(  # noqa: PLR0913, PLR0917 - what start_run read and decided, written into the run
    run: Path,
    read: _Start,
    graph: Graph,
    result: Mapping[str, Any],
    args: Mapping[str, Any],
    identity: Mapping[str, Any],
    run_id: str,
    store_source: bool,
    source_name: str,
    document: str,
) -> None:
    """Write a new run's files.

    They are the rubric, the cited copies, the document if stored, the ingested graph, the first log
    line, the state and the manifest.
    """
    (run / RUBRIC).write_bytes(read.rubric_bytes)
    (run / CITED).mkdir()
    for name, data in read.cited.items():
        (run / CITED / name).write_bytes(data)
    if store_source:
        (run / SOURCE).mkdir()
        (run / SOURCE / source_name).write_bytes(read.source_bytes)
    state = State(graph)
    (run / INGESTED).write_bytes(_json({"graph": graph_to_dict(graph)}))
    call = {"seq": 1, "tool": START, "args": json.loads(_line(args)), **identity}
    line = _line({**call, "call_digest": _call_digest(call), **_digests(state, result), "at": _now()})
    _durable_write(run / CALLS, line)
    log_digest = bytes_digest(line)
    _replace(run / STATE, _state_file(state, 1, log_digest))
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "run": run_id,
        "created_at": _now(),
        "harness": args["harness"],
        "source": args["source"] | {"document": document},
        "rubric": {**read.rubric.identity(), "digest": args["rubric_digest"]},
        "cited": args["cited"],
        "versions": versions(),
    }
    _replace(run / MANIFEST, _manifest(manifest, [json.loads(line)], state, log_digest))


def start_run(  # noqa: PLR0913 - each argument is a separate choice of the caller
    source: Path,
    rubric: Path,
    run: Path,
    *,
    harness: str,
    model: str,
    agent: str | None = None,
    note: str | None = None,
    document: str | None = None,
    cited: Path | None = None,
    store_source: bool = False,
) -> dict[str, Any]:
    """Start a run: check the rubric, ingest the document, copy the cited works, and log the call.

    Everything is read and checked before the run directory is written, so a rubric the tools cannot
    use, an unreadable document or a cited directory holding a link leaves nothing behind.

    Args:
        source: The document's main ``.tex`` file.
        rubric: The rubric file.
        run: The run directory to create; it must not exist or be empty.
        harness: The agent harness, as it reports itself.
        model: The model making the call.
        agent: A name for the agent.
        note: Free text kept with the call.
        document: The document id of every anchor; the source's file name by default.
        cited: The directory of cited works, each a file named by its bibliography key; copied into
            the run with each file's digest, and the copy is the only one the citation tool reads.
        store_source: Copy the document into the run.

    Returns:
        The run's id, the chunks, the document types with their form fields and edge rules, the hedge
        phrases, the check types the tools run, and those the rubric declares but the tools do not.

    Raises:
        RunError: If the run directory is in use, the rubric cannot be used, the document cannot be
            read, or the cited works are not plain files.
    """
    identity = _identity(model, agent, note)
    if not isinstance(harness, str) or not harness.strip():
        raise RunError("invalid-argument", "harness must name the agent harness")
    if run.exists() and any(run.iterdir()):
        raise RunError("run-exists", f"{run.name!r} exists and is not empty")
    read = _read_start(source, rubric, cited)
    if document is not None and (
        not document.strip() or any(sep in document for sep in ("/", "\\")) or PureWindowsPath(document).drive
    ):
        raise RunError("invalid-argument", f"document {document!r} must be an id for the document, not a path")
    document = document or source.name
    run_id = new_run_id()
    graph = stamp(_ingest(source, read.source_bytes, document, read.rubric), Graph(), {"run": run_id, "tool": START})
    args = {
        "document": document,
        "source": {"file": source.name, "digest": bytes_digest(read.source_bytes), "stored": store_source},
        "rubric_digest": bytes_digest(read.rubric_bytes),
        "cited": {name: bytes_digest(data) for name, data in read.cited.items()},
        "harness": harness,
    }
    result = _start_result(graph, read.rubric, run_id)
    # Build the run in a temporary directory beside it and rename it into place, so a start that fails
    # part-way leaves nothing that a second start_run would refuse as a run in use.
    building = run.parent / f".{run.name}.{run_id}.starting"
    building.mkdir(parents=True)
    try:
        _populate(building, read, graph, result, args, identity, run_id, store_source, source.name, document)
        if run.exists():
            run.rmdir()
        building.rename(run)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return result


def get_state(run_dir: Path) -> dict[str, Any]:
    """Return what a run holds now, for an agent deciding its next call.

    Args:
        run_dir: The run directory.

    Returns:
        Every node's id, type and statement, with its assessment where it has one; the relations; the
        refused records so far; how many calls the log holds; and whether the run is finished.

    Raises:
        RunError: If the run directory is malformed or incomplete.
    """
    with lock(run_dir):
        manifest = read_manifest(run_dir)
        all_calls = read_calls(run_dir)
        state = _reconcile(run_dir, manifest, _rubric(run_dir, manifest), all_calls)
        calls = len(all_calls)
    nodes = []
    for node in state.graph:
        item: dict[str, Any] = {"id": node.id, "type": node.kind, "text": node.statement}
        if "assessment" in node.attributes:
            item["assessment"] = node.attributes["assessment"]
        if "check" in node.attributes:
            item["result"] = node.attributes["check"].get("result")
        nodes.append(item)
    relations = [
        {"type": relation.type, "from": relation.source, "to": relation.target}
        for relation in state.graph.relations.values()
    ]
    return {
        "nodes": nodes,
        "relations": relations,
        "rejections": [dict(item) for item in state.rejections],
        "calls": calls,
        "finished": state.finished,
    }


def replay(run_dir: Path, *, source: Path | None = None, sources: Path | None = None) -> bytes:
    """Re-ingest a run's document and apply every logged call again, without the network.

    Each call's result must have the digest the log recorded; the graph the last call leaves is
    returned, as ``state.json`` holds it, for the caller to compare with the run's.

    Args:
        run_dir: The run directory.
        source: The document, for a run that did not store it.
        sources: The fetched sources the run did not store, as files named by their digest.

    Returns:
        The rebuilt graph, as the core serialises it, with a final newline.

    Raises:
        RunError: If the run directory is malformed or incomplete, the document, rubric or a cited
            copy differs from what the run recorded, or a call's result differs from its record.
    """
    manifest = read_manifest(run_dir)
    calls = read_calls(run_dir)
    start = calls[0]["args"]
    recorded = start["source"]
    if recorded.get("stored"):
        source = run_dir / SOURCE / recorded["file"]
    elif source is None:
        raise RunError("missing-source", "the run did not store its document: pass the source document")
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise RunError("missing-source", f"cannot read {source.name!r}: {error.strerror or error}") from None
    if bytes_digest(source_bytes) != recorded["digest"]:
        raise RunError("source-mismatch", f"{source.name!r} is not the document the run read")
    rubric = _rubric(run_dir, manifest)
    for name, digest in start["cited"].items():
        if bytes_digest(_read(run_dir, f"{CITED}/{name}")) != digest:
            raise RunError("record-mismatch", f"the cited copy {name!r} differs from the one start_run made")
    graph = _ingest(source, source_bytes, start["document"], rubric)
    graph = stamp(graph, Graph(), {"run": manifest["run"], "tool": START})
    state = State(graph)
    _check_digests(calls[0], state, _start_result(graph, rubric, manifest["run"]))
    for call in calls[1:]:
        state, result = _apply(Context(run_dir, manifest["run"], rubric, call["seq"], sources), state, call)
        _check_digests(call, state, result)
    return _graph_text(state.graph)


def recorded_graph(run_dir: Path) -> bytes:
    """Return the run's graph after every logged call, as ``replay`` returns it.

    Args:
        run_dir: The run directory.

    Returns:
        The graph, as the core serialises it, with a final newline.
    """
    with lock(run_dir):
        state = current_state(run_dir)
    return _graph_text(state.graph)
