"""The assembler: it applies classify records to a graph through the core's ``split`` and ``annotate``.

A classify record types one chunk of the document, or splits it into clauses and types each clause.
Its fields are the type, the form, the nodes it rests on, and a hedge phrase; it has no field for a
credence. Every ``base`` and ``strength`` the assembler writes is therefore the rubric's: ``annotate``
gives a typed node its type's base and a rests-on relation its edge rule's strength. The one value
the text sets is ``Node.stated``, read from the rubric's ``[hedges]`` table for the phrase the model
located; ``stated`` is used by diagnostics only, never by inference.

Records are applied in document order, each as a whole or not at all. A record the core refuses is
not applied and not dropped: it is returned as a rejection with the core's error code, and its node
stays as it was.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from credencegraph.cli.common import UNKNOWN_NODE, translate
from credencegraph.core import Graph, Node
from credencegraph.core.errors import ValidationError
from credencegraph.core.relation import REQUIRES
from credencegraph.rubric import Rubric, annotate_node, split_node
from credencegraph.rubric.faithfulness import unfaithful
from credencegraph.rubric.model import DOCUMENT, VERBATIM
from credencegraph.rubric.validate import node_text, origin_of

# Rejection codes of the assembler's own, beside the core's error codes.
INVALID_RECORD = "invalid-record"
DUPLICATE_RECORD = "duplicate-record"
NOT_REJECTED = "not-rejected"

# The fields of a record that types one node, and of a record that splits a chunk into typed clauses.
TYPED_FIELDS = frozenset({"type", "form", "rests_on", "hedge"})
SPLIT_FIELDS = frozenset({"node", "split", "children"})


class _RecordError(Exception):
    """A record is not applied, with the code and message of a rejection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Assembly:
    """The result of applying a set of records.

    Attributes:
        graph: The graph with every accepted record applied.
        rejections: The records not applied, in the order they were tried, each as ``{"node", "code",
            "message", "record"}``; ``node`` is ``None`` when the record names none.
        unmapped: The hedge phrases found in their node's text that the rubric's ``[hedges]`` does not
            list, each as ``{"node", "phrase"}``.
    """

    graph: Graph
    rejections: list[dict[str, Any]]
    unmapped: list[dict[str, str]]


def edge_rules_not_requires(rubric: Rubric) -> list[str]:
    """Return the types whose edge rule creates a relation other than ``requires``.

    Text may only add ``requires`` relations: with no other parents, a node's credence cannot exceed
    its type's base, whatever the text says. A ``supports`` edge rule would let a document raise the
    credence of its own claims, so a rubric with one cannot be used for extraction.

    Args:
        rubric: The rubric.

    Returns:
        The names of those types, in declaration order.
    """
    return [
        name for name, spec in rubric.types.items() if spec.rests_on is not None and spec.rests_on.relation != REQUIRES
    ]


def _check_typed(where: str, value: Any, fields: Collection[str]) -> None:
    """Refuse a typed record or child whose shape is not ``{"type", "form"?, "rests_on"?, "hedge"?}``."""
    if not isinstance(value, Mapping):
        raise _RecordError(INVALID_RECORD, f"{where} is not an object")
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise _RecordError(INVALID_RECORD, f"{where} has unknown fields {unknown}")
    if not isinstance(value.get("type"), str):
        raise _RecordError(INVALID_RECORD, f"{where} needs a type, as a string")
    if not isinstance(value.get("form", {}), Mapping):
        raise _RecordError(INVALID_RECORD, f"{where}: form must be an object")
    rests_on = value.get("rests_on", [])
    if not isinstance(rests_on, list) or not all(isinstance(item, str) for item in rests_on):
        raise _RecordError(INVALID_RECORD, f"{where}: rests_on must be a list of node ids")
    if value.get("hedge") is not None and not isinstance(value["hedge"], str):
        raise _RecordError(INVALID_RECORD, f"{where}: hedge must be a string or null")


def _targets(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Check a record's shape and return its typed parts: the record itself, or the children of a split."""
    if "split" not in record:
        _check_typed("the record", record, TYPED_FIELDS | {"node"})
        return [record]
    unknown = sorted(set(record) - SPLIT_FIELDS)
    if unknown:
        raise _RecordError(INVALID_RECORD, f"a split record has unknown fields {unknown}")
    clauses, children = record["split"], record.get("children")
    if not isinstance(clauses, list) or not clauses or not all(isinstance(clause, str) for clause in clauses):
        raise _RecordError(INVALID_RECORD, "split must be a non-empty list of clauses")
    if not isinstance(children, list) or len(children) != len(clauses):
        raise _RecordError(INVALID_RECORD, f"a split into {len(clauses)} clauses needs {len(clauses)} children")
    for i, child in enumerate(children):
        _check_typed(f"children[{i}]", child, TYPED_FIELDS)
    return children


def _replaced(graph: Graph, node: Node) -> Graph:
    """Copy a graph with one node replaced."""
    out = Graph()
    for item in graph:
        out.add_node(node if item.id == node.id else item)
    for relation in graph.relations.values():
        out.add_relation(relation)
    return out


def _apply(graph: Graph, rubric: Rubric, record: Mapping[str, Any]) -> tuple[Graph, list[dict[str, str]]]:
    """Apply one record, as a whole; return the graph and the hedge phrases the rubric does not map.

    Raises:
        _RecordError: If the record is malformed, names a node the graph lacks, or the core refuses an edit.
    """
    targets = _targets(record)
    node_id = record["node"]
    try:
        if "split" in record:
            graph, ids = split_node(graph, node_id, record["split"])
        else:
            ids = [node_id]
        unmapped = []
        for target_id, part in zip(ids, targets, strict=True):
            missing = [source for source in part.get("rests_on", []) if source not in graph]
            if missing:
                raise _RecordError(UNKNOWN_NODE, f"node {target_id!r} rests on nodes not in the graph: {missing}")
            graph = annotate_node(
                graph,
                rubric,
                target_id,
                node_type=part["type"],
                settings=part.get("form") or {},
                rests_on=part.get("rests_on", []),
            )
            phrase = part.get("hedge")
            if phrase:
                node = graph.nodes[target_id]
                problem = unfaithful(VERBATIM, phrase, node_text(graph, node))
                if problem:
                    msg = f"node {target_id!r}: hedge {problem}"
                    raise ValidationError(msg)
                stated = rubric.hedge(phrase)
                if stated is None:
                    unmapped.append({"node": target_id, "phrase": phrase})
                else:
                    graph = _replaced(graph, replace(node, stated=stated))
    except ValidationError as error:
        raise _RecordError(translate(error).code, str(error)) from None
    return graph, unmapped


def _in_document_order(graph: Graph, records: Iterable[Any]) -> list[Any]:
    """Sort records by the position of their node in the graph; records naming no known node go last."""
    order = {node_id: i for i, node_id in enumerate(graph.nodes)}
    end = len(order)

    def position(record: Any) -> int:
        node_id = record.get("node") if isinstance(record, Mapping) else None
        return order.get(node_id, end) if isinstance(node_id, str) else end

    return sorted(records, key=position)


def assemble(graph: Graph, rubric: Rubric, records: Sequence[Any], *, only: Collection[str] | None = None) -> Assembly:
    """Apply classify records in document order through the core's ``split`` and ``annotate``.

    Each record is applied as a whole or not at all. One is rejected, and its node left as it was,
    when it is malformed, names a node the graph lacks or one that is not a document chunk, repeats
    the node of an earlier record, falls outside ``only``, or the core refuses one of its edits: an
    unknown type, a form not found in the text, a clause that does not occur exactly once, a
    rests-on node of a type the edge rule does not allow. A hedge phrase that is not a substring of
    its node's text rejects the record too. One the text contains sets ``Node.stated`` to the
    rubric's credence for it, or, if ``[hedges]`` does not list it, is returned as unmapped.

    Args:
        graph: The graph, left unchanged.
        rubric: The rubric.
        records: The records, as the model returned them.
        only: The nodes records may name; any node by default.

    Returns:
        The graph, the rejections and the unmapped hedge phrases.
    """
    rejections: list[dict[str, Any]] = []
    unmapped: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in _in_document_order(graph, records):
        node_id = record.get("node") if isinstance(record, Mapping) else None
        try:
            if not isinstance(node_id, str):
                raise _RecordError(INVALID_RECORD, "the record names no node")
            if node_id not in graph:
                raise _RecordError(UNKNOWN_NODE, f"node {node_id!r} is not in the graph")
            if origin_of(graph.nodes[node_id]) != DOCUMENT:
                raise _RecordError(INVALID_RECORD, f"node {node_id!r} is not a chunk of the document")
            if node_id in seen:
                raise _RecordError(DUPLICATE_RECORD, f"node {node_id!r} has an earlier record")
            seen.add(node_id)
            if only is not None and node_id not in only:
                raise _RecordError(NOT_REJECTED, f"node {node_id!r} had no rejected record to repair")
            graph, phrases = _apply(graph, rubric, record)
        except _RecordError as refused:
            node = node_id if isinstance(node_id, str) else None
            rejections.append({"node": node, "code": refused.code, "message": str(refused), "record": record})
        else:
            unmapped.extend(phrases)
    return Assembly(graph, rejections, unmapped)
