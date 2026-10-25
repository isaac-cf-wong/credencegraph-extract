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
- ``finish_run`` derives every assessed node's verdict from the checks that observe it, runs the core's
  check at the ``assessed`` level with rubric credences, and marks the run finished; the run's outputs
  are written then. Any later write marks it unfinished again.

``start_run`` and ``get_state`` are in ``session``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from credencegraph.core import Graph
from credencegraph.core.relation import OBSERVES
from credencegraph.rubric import check_graph

from credencegraph_extract.assemble import assemble, assemble_premises, fixes, premises_rubric_problem
from credencegraph_extract.checks import assess
from credencegraph_extract.run import RunError
from credencegraph_extract.session import FINISH, Context, State, operation, write

CLASSIFY = "classify"
ADD_PREMISES = "add_premises"

# The rejection code of a record on a node a check targets.
CHECKED = "checked-node"


def _records(args: Mapping[str, Any]) -> list[Any]:
    """The list of records a tool call sends, or refuse the call."""
    records = args.get("records")
    if not isinstance(records, list):
        raise RunError("invalid-argument", "records must be a list of records")
    return records


def _changed(before: Graph, after: Graph) -> list[str]:
    """The nodes a call created or changed, in graph order."""
    return [node.id for node in after if before.nodes.get(node.id) != node]


def _kept(context: Context, tool: str, rejections: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The refused records as the run keeps them: each with the call and tool that sent it."""
    return tuple({"call": context.call, "tool": tool, **rejection} for rejection in rejections)


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
    for rejection in assembly.rejections:
        found = fixes(assembly.graph, context.rubric, rejection["record"])
        refused.append({**rejection, "fix": found} if found else rejection)
    for rejection in refused:
        if rejection["code"] == CHECKED:
            rejection["fix"] = [f"leave {rejection['node']!r} as it is; send records only for nodes no check targets"]
    after = State(assembly.graph, (*state.rejections, *_kept(context, CLASSIFY, refused)), state.finished)
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
    after = State(
        assembly.graph, (*state.rejections, *_kept(context, ADD_PREMISES, assembly.rejections)), state.finished
    )
    result = {
        "applied": _changed(state.graph, assembly.graph),
        "refused": assembly.rejections,
        "overflow": assembly.overflow,
    }
    return after, result


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
    return write(run, CLASSIFY, {"records": list(records)}, model=model, agent=agent, note=note)


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
    return write(run, ADD_PREMISES, {"records": list(records)}, model=model, agent=agent, note=note)


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
