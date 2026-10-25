"""Write tools for the tests of the run machinery: ``note`` sets ``attributes.note`` on a node, and
``observe`` adds a check on a target.

They stand in for the real tools so the call log, the lock, crash recovery and replay can be tested
on their own. ``register`` registers them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from credencegraph.core import Graph

from credencegraph_extract.session import Context, State, operation


def note(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Set a note on a node, or refuse a node the graph lacks."""
    node_id, text = args.get("node"), args.get("text")
    if node_id not in state.graph.nodes:
        refusal = {
            "call": context.call,
            "node": node_id,
            "code": "unknown-node",
            "message": f"{node_id!r} is not a node",
        }
        return replace(state, rejections=(*state.rejections, refusal)), {"applied": [], "refused": [refusal]}
    graph = Graph()
    for node in state.graph:
        graph.add_node(replace(node, attributes={**node.attributes, "note": text}) if node.id == node_id else node)
    for relation in state.graph.relations.values():
        graph.add_relation(relation)
    return replace(state, graph=graph), {"applied": [node_id], "refused": []}


def observe(context: Context, state: State, args: Mapping[str, Any]) -> tuple[State, dict[str, Any]]:
    """Add a check of the rubric's first check type on a target, as the check tools will: a test stand-in."""
    from credencegraph.core import Node
    from credencegraph.core.relation import OBSERVES, Relation

    target, result = args["target"], args.get("result", "failed")
    name, check = next(iter(context.rubric.checks.items()))
    check_id = f"c-{len([n for n in state.graph if n.id.startswith('c-')]) + 1:03d}"
    graph = Graph()
    for node in state.graph:
        graph.add_node(node)
    record = {"type": name, "target": target, "result": result}
    attributes = {"origin": "evidence", "check": record, "observed": result == "passed"}
    graph.add_node(Node(check_id, kind=name, statement=f"{name} on {target}: {result}", attributes=attributes))
    for relation in state.graph.relations.values():
        graph.add_relation(relation)
    relation_id = f"{target}-{OBSERVES}-{check_id}"
    graph.add_relation(
        Relation(relation_id, OBSERVES, target, check_id, sensitivity=check.sensitivity, false_pass=check.false_pass)
    )
    return replace(state, graph=graph), {"check": check_id}


def register() -> None:
    """Register the test tools with the run machinery."""
    operation("note")(note)
    operation("observe")(observe)
