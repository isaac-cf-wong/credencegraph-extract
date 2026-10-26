"""A write tool for the tests of the run machinery: it sets ``attributes.note`` on a node.

It stands in for the real tools so the call log, the lock, crash recovery and replay can be tested
on their own. ``register`` registers it.
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


def register() -> None:
    """Register the test tool with the run machinery."""
    operation("note")(note)
