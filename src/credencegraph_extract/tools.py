"""The tools an agent calls to build a graph: classify, add premises, and finish a run.

Each tool is one write to a run directory (``session.write``): it takes the run's lock, applies its
operation to the current state, and is logged. A tool never writes graph JSON and never sets a credence:
records go through the assembler, so every base and strength is the rubric's. A refused record is a
result, not an error: it comes back with its code, the core's message and a ``fix`` saying what to
change, and is kept in the run's rejections; the agent may simply call again.

- ``classify`` types, splits and links chunks. A node already typed may be typed again, which first
  returns it to unassigned (``assemble``), but not once a check targets it: a failed check cannot be
  escaped by making its node a type no check applies to. A chunk already split cannot be split again.
- ``add_premises`` adds the implicit premises of conclusions, as the premise records do.
- ``run_checks`` checks and runs proposals of the maths check types (``checks.run_maths``), and adds a
  check node for each that runs, with its files in the run. A check is never withdrawn or run again on
  its target, nor run again on another target as another encoding of the same step.
- ``judge_citation`` reads a citation's source itself, finds the agent's passage in it, and records a
  check that passes or fails as the agent judged only when the source is tied to the reference
  (``citation``); any other source gives an ``error``, which moves no credence.
- ``finish_run`` derives every assessed node's verdict from the checks that observe it, runs the core's
  check at the ``assessed`` level with rubric credences, and marks the run finished; the run's outputs
  are written then. Any later write marks it unfinished again.

``start_run`` and ``get_state`` are in ``session``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from credencegraph.cli.common import UNKNOWN_NODE
from credencegraph.core import Graph
from credencegraph.core.relation import OBSERVES
from credencegraph.rubric import check_graph

from credencegraph_extract.assemble import (
    DUPLICATE_RECORD,
    INVALID_RECORD,
    assemble,
    assemble_premises,
    fixes,
    premises_rubric_problem,
)
from credencegraph_extract.checks import CITATION, add_checks, assess, citation_refusal, judged, run_maths
from credencegraph_extract.checks import fix as check_fix
from credencegraph_extract.citation import SourceError, read_cited, read_url, tie, verify_observation
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.fetch import fetch
from credencegraph_extract.run import RunError
from credencegraph_extract.session import (
    FINISH,
    SOURCES,
    Context,
    State,
    current_state,
    lock,
    operation,
    read_calls,
    versions,
    write,
)

CLASSIFY = "classify"
ADD_PREMISES = "add_premises"
RUN_CHECKS = "run_checks"
JUDGE_CITATION = "judge_citation"

# The rejection code of a record on a node a check targets.
CHECKED = "checked-node"


def _records(args: Mapping[str, Any], name: str = "records") -> list[Any]:
    """The list of records a tool call sends, or refuse the call."""
    records = args.get(name)
    if not isinstance(records, list):
        raise RunError("invalid-argument", f"{name} must be a list of records")
    return records


def _sent(records: Any) -> Any:
    """The records as a call logs them: a tuple as a list, anything else as given, for the operation to refuse.

    A string or a mapping is not turned into a list of its characters or keys.
    """
    return list(records) if isinstance(records, tuple) else records


def _versions(args: Mapping[str, Any]) -> dict[str, str]:
    """The versions of the code that ran a call, as the tool records them in its arguments."""
    found = args.get("versions")
    if not isinstance(found, Mapping) or not all(isinstance(value, str) for value in found.values()):
        raise RunError("invalid-argument", "versions must name the version of each piece of code that ran")
    return dict(found)


def _changed(before: Graph, after: Graph) -> list[str]:
    """The nodes a call created or changed, in graph order."""
    return [node.id for node in after if before.nodes.get(node.id) != node]


def _kept(context: Context, tool: str, rejections: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The refused records as the run keeps them: each with the call and tool that sent it."""
    return tuple({"call": context.call, "tool": tool, **rejection} for rejection in rejections)


def _fallback(rejection: Mapping[str, Any]) -> list[str]:
    """What to change in a refused record when no closer diagnosis applies: from its code, else its message."""
    node = rejection.get("node")
    if rejection.get("code") == DUPLICATE_RECORD:
        return [f"send one record per node in a call: an earlier record in this call already names {node!r}"]
    if rejection.get("code") == UNKNOWN_NODE:
        return ["name only nodes the run has: the chunk ids start_run listed, or ids earlier calls created"]
    return [f"change the record so that this no longer holds: {rejection.get('message')}"]


def _premise_fix(context: Context, rejection: Mapping[str, Any]) -> list[str]:
    """What to change in a refused premise record."""
    if rejection.get("code") == INVALID_RECORD:
        conclusions = list(context.rubric.premises.for_types) if context.rubric.premises is not None else []
        shape = (
            f'{{"conclusion": <id of a node of a type in {conclusions}>, '
            '"premises": [{"stated_at": <chunk id>} or {"statement": <the premise in words>}, ...]}'
        )
        return [f"{rejection.get('message')}; a premise record is {shape}"]
    return _fallback(rejection)


def _check_fix(rejection: Mapping[str, Any]) -> list[str]:
    """What to change in a refused check proposal: what the refusal knew, else what its code says, else the fallback."""
    found = rejection.get("fix") or check_fix(rejection["code"], rejection["record"])
    return [found] if found else _fallback(rejection)


def checked(graph: Graph) -> set[str]:
    """Return the nodes a check targets: the sources of ``observes`` relations.

    Args:
        graph: The graph.

    Returns:
        Their ids.
    """
    return {relation.source for relation in graph.relations.values() if relation.type == OBSERVES}


@operation(CLASSIFY)
def _classify(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Apply classify records; refuse those on a node a check targets."""
    targeted = checked(state.graph)
    kept: list[Any] = []
    refused: list[dict[str, Any]] = []
    for record in _records(args):
        node = record.get("node") if isinstance(record, Mapping) else None
        if isinstance(node, str) and node in targeted:
            message = f"a check targets {node!r}, so it can no longer be typed again or split"
            refused.append({"node": node, "code": CHECKED, "message": message, "record": record})
        else:
            kept.append(record)
    assembly = assemble(state.graph, context.rubric, kept)
    for rejection in refused:
        rejection["fix"] = [f"leave {rejection['node']!r} as it is; send records only for nodes no check targets"]
    for rejection in assembly.rejections:
        found = fixes(assembly.graph, context.rubric, rejection["record"])
        refused.append({**rejection, "fix": found or _fallback(rejection)})
    after = replace(state, graph=assembly.graph, rejections=(*state.rejections, *_kept(context, CLASSIFY, refused)))
    result = {
        "applied": _changed(state.graph, assembly.graph),
        "refused": refused,
        "unmapped_hedges": assembly.unmapped,
    }
    return after, result


@operation(ADD_PREMISES)
def _add_premises(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Apply premise records."""
    problem = premises_rubric_problem(context.rubric)
    if problem is not None:
        raise RunError("invalid-argument", f"no premises can be added: {problem}")
    assembly = assemble_premises(state.graph, context.rubric, _records(args))
    refused = [{**rejection, "fix": _premise_fix(context, rejection)} for rejection in assembly.rejections]
    after = replace(state, graph=assembly.graph, rejections=(*state.rejections, *_kept(context, ADD_PREMISES, refused)))
    result = {
        "applied": _changed(state.graph, assembly.graph),
        "refused": refused,
        "overflow": assembly.overflow,
    }
    return after, result


@operation(RUN_CHECKS)
def _run_checks(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Check and run maths proposals, adding a check node for each that runs."""
    graph, files, checks, rejections = run_maths(
        state.graph, context.rubric, _records(args, "proposals"), _versions(args)
    )
    refused = [{**rejection, "fix": _check_fix(rejection)} for rejection in rejections]
    texts = {path: data.decode("utf-8") for path, data in files.items()}
    after = replace(
        state,
        graph=graph,
        rejections=(*state.rejections, *_kept(context, RUN_CHECKS, refused)),
        files={**state.files, **texts},
    )
    return after, {"checks": checks, "refused": refused}


def _judge_arguments(args: Mapping[str, Any]) -> None:
    """Refuse a citation call whose arguments are not of their types."""
    for name in ("target", "reference", "source", "passage"):
        if not isinstance(args.get(name), str) or not args[name].strip():
            raise RunError("invalid-argument", f"{name} must be a non-empty string")
    if not isinstance(args.get("supports"), bool):
        raise RunError("invalid-argument", "supports must be true or false")
    observation = args.get("read")
    if observation is not None and not isinstance(observation, Mapping):
        raise RunError("invalid-argument", "read must be what the tool read")


def _source_bytes(context: Context, observation: Mapping[str, Any]) -> bytes:
    """The bytes a replayed citation call read, found by their digest: in the cited copy, the run's sources, or ``--sources``."""
    digest = str(observation.get("digest"))
    name = digest.removeprefix("sha256:")
    places = [context.run_dir / SOURCES / name]
    if observation.get("kind") == "cited":
        places.insert(0, context.run_dir / "cited" / str(observation.get("file")))
    if context.sources is not None:
        places.append(context.sources / name)
    for place in places:
        if place.is_file() and bytes_digest(data := place.read_bytes()) == digest:
            return data
    what = observation.get("file") if observation.get("kind") == "cited" else observation.get("url")
    msg = f"call {context.call} read {what!r} ({digest}), whose bytes the run did not store: pass them with --sources"
    raise RunError("missing-source", msg)


@operation(JUDGE_CITATION)
def _judge_citation(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Record a citation check from what the tool read: tied to the reference, or an error."""
    _judge_arguments(args)
    target, reference = args["target"], args["reference"]
    observation = args.get("read")
    refusal = citation_refusal(state.graph, context.rubric, target, reference)
    if refusal is None and not isinstance(observation, Mapping):
        raise RunError("invalid-argument", "the call records nothing read")
    if refusal is None and "error" in observation:
        refusal = {"node": target, "code": observation.get("code"), "message": observation["error"]}
        refusal["record"] = {"target": target, "type": CITATION, "reference": reference, "source": args["source"]}
        refusal["fix"] = (
            "give a public http or https URL, or the name of a file of the run's cited works"
            if observation.get("code") == "refused-url"
            else "give a source that can be read: a text, LaTeX, PDF or HTML file of the cited work"
        )
    if refusal is not None:
        refused = [{**refusal, "fix": _check_fix(refusal)}]
        after = replace(state, rejections=(*state.rejections, *_kept(context, JUDGE_CITATION, refused)))
        return after, {"check": None, "refused": refused}
    if context.replaying:
        differ = verify_observation(observation, _source_bytes(context, observation), args["passage"])
        if differ:
            msg = f"call {context.call} recorded a reading of its source that its bytes do not give: {differ}"
            raise RunError("record-mismatch", msg)
    untied = tie(state.graph.nodes[reference], context.cited, observation)
    result = judged(
        state.graph,
        target,
        reference,
        observation,
        passage=args["passage"],
        supports=args["supports"],
        untied=untied,
        model=context.model,
    )
    graph, files, [entry] = add_checks(state.graph, context.rubric, [result], _versions(args))
    texts = {path: data.decode("utf-8") for path, data in files.items()}
    after = replace(state, graph=graph, files={**state.files, **texts})
    return after, {"check": entry["id"], "result": result.result, "tied": untied is None, "detail": result.detail}


@operation(FINISH)
def _finish(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Derive the verdicts, run the core's check, and mark the run finished."""
    graph = assess(state.graph, context.rubric)
    findings = check_graph(graph, context.rubric, "assessed", rubric_credences=True)
    result = {
        "findings": [
            {"code": finding.diagnostic, "nodes": list(finding.nodes), "message": finding.message}
            for finding in findings
        ],
        "ok": not findings,
    }
    return replace(state, graph=graph, finished=True), result


def classify(
    run: Path, records: Sequence[Any], *, model: str, agent: str | None = None, note: str | None = None
) -> dict[str, Any]:
    """Type, split and link chunks of a run's document.

    Args:
        run: The run directory.
        records: Classify records, each ``{"node", "type", "form"?, "rests_on"?, "hedge"?}`` or
            ``{"node", "split", "children"}``.
        model: The model making the call, as the harness reports it.
        agent: A name for the agent, for runs with several.
        note: Free text kept with the call, never used in inference.

    Returns:
        ``applied``, the nodes created or changed; ``refused``, each refused record with its code,
        message and ``fix``; and ``unmapped_hedges``, hedge phrases the rubric does not list.

    Raises:
        RunError: If the call as a whole is refused (``records`` not a list, the run busy or malformed).
    """
    return write(run, CLASSIFY, {"records": _sent(records)}, model=model, agent=agent, note=note)


def add_premises(
    run: Path, records: Sequence[Any], *, model: str, agent: str | None = None, note: str | None = None
) -> dict[str, Any]:
    """Add the implicit premises of conclusions.

    Args:
        run: The run directory.
        records: Premise records, each ``{"conclusion", "premises": [{"stated_at"} | {"statement"}, ...]}``.
        model: The model making the call, as the harness reports it.
        agent: A name for the agent.
        note: Free text kept with the call.

    Returns:
        ``applied``, the nodes created or changed (analyst premises included); ``refused``, each
        refused record with its code and message; and ``overflow``, premises beyond the rubric's limit.

    Raises:
        RunError: If the rubric takes no premises, or the call as a whole is refused.
    """
    return write(run, ADD_PREMISES, {"records": _sent(records)}, model=model, agent=agent, note=note)


def run_checks(
    run: Path, proposals: Sequence[Any], *, model: str, agent: str | None = None, note: str | None = None
) -> dict[str, Any]:
    """Check and run proposals of the maths check types: ``recompute-numbers`` and ``rederive-maths``.

    A check, once run, is part of the run: it cannot be withdrawn or run again on the same target, nor
    run again on another target in another encoding of the same step. The versions of the code that
    runs the checks are recorded with the call.

    Args:
        run: The run directory.
        proposals: Check proposals, each ``{"target", "type": "recompute-numbers", "nodes", "expression",
            "expected"}`` or ``{"target", "type": "rederive-maths", "nodes", "lhs", "rhs", "symbols",
            "assumption"}``.
        model: The model making the call, as the harness reports it.
        agent: A name for the agent.
        note: Free text kept with the call.

    Returns:
        ``checks``, each check run with its id, type, target, result and detail (a ``rederive-maths``
        that errs is followed by its numeric fallback when the rubric declares one); and ``refused``,
        each refused proposal with its code, message and ``fix``.

    Raises:
        RunError: If the call as a whole is refused.
    """
    args = {"proposals": _sent(proposals), "versions": versions()}
    return write(run, RUN_CHECKS, args, model=model, agent=agent, note=note)


def _is_url(source: str) -> bool:
    """Whether a source is given as a URL, of any scheme, rather than as a path."""
    return "://" in source or source.split(":", 1)[0].lower() in {"http", "https", "file"}


def _observe(
    run: Path, reference: Any, source: Any, passage: Any, fetcher: Callable[[str], Any]
) -> dict[str, Any] | None:
    """Read a citation's source, as the call records it; ``None`` when the arguments name nothing to read."""
    if not all(isinstance(value, str) and value.strip() for value in (reference, source, passage)):
        return None
    with lock(run):
        node = current_state(run).graph.nodes.get(reference)
        stored = bool(read_calls(run)[0]["args"]["source"].get("stored"))
    if node is None or node.attributes.get("unit") != "reference":
        return None
    try:
        if _is_url(source):
            observation, data = read_url(node, source, passage, fetcher)
        else:
            observation, data = read_cited(run, source, passage)
    except SourceError as error:
        return {"error": str(error), "code": error.code}
    if stored and observation["kind"] == "url":
        path = run / SOURCES / observation["digest"].removeprefix("sha256:")
        path.parent.mkdir(exist_ok=True)
        if not path.is_file():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
    return observation


def judge_citation(  # noqa: PLR0913 - the check, what the agent judged, and who makes the call
    run: Path,
    target: str,
    reference: str,
    source: str,
    passage: str,
    *,
    supports: bool,
    model: str,
    agent: str | None = None,
    note: str | None = None,
    fetcher: Callable[[str], Any] = fetch,
) -> dict[str, Any]:
    """Record a citation check: read the source, find the passage, and decide whether the source is tied to the reference.

    The tool reads the source itself; it never takes the text from the agent. A local source is a file
    of the run's copy of the cited works; a URL is fetched under the fetch policy (``fetch``). What was
    read (the file or URLs, the digests of the bytes and of their text, where the passage is, and what
    ties the source to the reference) is logged with the call, not the bytes, which are kept in
    ``sources/`` only when the run stores its sources. The check passes or fails, as the agent judged,
    only when the passage is in the source and the source is tied to the reference
    (``citation.tie``); otherwise it is an ``error``, which moves no credence. A target whose citation
    check passed or failed takes no other; one whose checks erred may be checked again.

    Args:
        run: The run directory.
        target: The citing node.
        reference: The id of the bibliography entry (``r-N``), not its key.
        source: A file of the run's cited works, by name or path, or an ``http`` or ``https`` URL.
        passage: The passage of the source the target rests on, verbatim.
        supports: Whether the passage supports the target.
        model: The model making the call, as the harness reports it.
        agent: A name for the agent.
        note: Free text kept with the call.
        fetcher: Fetches a URL under the fetch policy; ``fetch.fetch`` by default.

    Returns:
        ``check``, the id of the check, ``result``, ``tied`` and ``detail``; or ``check`` ``None`` and
        ``refused``, with the refusal's code, message and ``fix``.

    Raises:
        RunError: If an argument is not of its type, or the call as a whole is refused.
    """
    args = {
        "target": target,
        "reference": reference,
        "source": source,
        "passage": passage,
        "supports": supports,
        "read": _observe(run, reference, source, passage, fetcher),
        "versions": versions(),
    }
    return write(run, JUDGE_CITATION, args, model=model, agent=agent, note=note)


def finish_run(run: Path, *, model: str, agent: str | None = None, note: str | None = None) -> dict[str, Any]:
    """Derive the verdicts, check the graph with the core, and write the run's outputs.

    It may be called again after later writes, which mark the run unfinished.

    Args:
        run: The run directory.
        model: The model making the call, as the harness reports it.
        agent: A name for the agent.
        note: Free text kept with the call.

    Returns:
        ``findings``, every violation the core's check reports at the ``assessed`` level with rubric
        credences, and ``ok``, whether there are none.
    """
    return write(run, FINISH, {}, model=model, agent=agent, note=note)
