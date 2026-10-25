"""The assembler: it applies classify and premise records to a graph through the core's edits.

A classify record types one chunk of the document, or splits it into clauses and types each clause.
Its fields are the type, the form, the nodes it rests on, and a hedge phrase; it has no field for a
credence. Every ``base`` and ``strength`` the assembler writes is therefore the rubric's: ``annotate``
gives a typed node its type's base and a rests-on relation its edge rule's strength. The one value
the text sets is ``Node.stated``, read from the rubric's ``[hedges]`` table for the phrase the model
located; ``stated`` is used by diagnostics only, never by inference.

A premise record names a conclusion and the premises it needs beside those it rests on. A premise
``stated_at`` a chunk becomes a ``requires`` from that chunk, through ``annotate``, with the edge
rule's strength. A ``statement`` becomes an analyst node of type ``assumption``, with no source and
the rubric's ``premises.base``, joined to the conclusion by a ``requires`` of ``premises.strength``.
Analyst nodes whose normalised statements are identical are one node with several ``requires``.
Each is named ``a-<n>``: ``n`` counts from 1, in the order the nodes are created (conclusions in
document order, then each record's premises in its order), skipping any id the graph already has.
The number is not zero-padded, since it is assigned before the total is known. A replay creates the
same nodes in the same order, so it gives them the same ids.

Records are applied in document order, each as a whole or not at all. A record the core refuses is
not applied and not dropped: it is returned as a rejection with the core's error code, and its node
stays as it was.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import count
from typing import Any

from credencegraph.cli.common import UNKNOWN_NODE, translate
from credencegraph.core import Graph, Node
from credencegraph.core.errors import ValidationError
from credencegraph.core.relation import REQUIRES, Relation
from credencegraph.rubric import FormField, NodeType, Rubric, annotate_node, normalise_text, split_node
from credencegraph.rubric.faithfulness import unfaithful
from credencegraph.rubric.model import ANALYST, DOCUMENT, ENUM, EQREF, NUMBER, QUANTITY, VERBATIM
from credencegraph.rubric.validate import node_text, origin_of

# Rejection codes of the assembler's own, beside the core's error codes.
INVALID_RECORD = "invalid-record"
DUPLICATE_RECORD = "duplicate-record"
NOT_REJECTED = "not-rejected"

# The fields of a record that types one node, and of a record that splits a chunk into typed clauses.
TYPED_FIELDS = frozenset({"type", "form", "rests_on", "hedge"})
SPLIT_FIELDS = frozenset({"node", "split", "children"})

# The fields of a premise record, and the two kinds of premise it lists.
PREMISE_FIELDS = frozenset({"conclusion", "premises"})
STATED_AT = "stated_at"
STATEMENT = "statement"

# The type of the analyst nodes the premise records create, and the prefix of their ids.
ASSUMPTION = "assumption"
ANALYST_PREFIX = "a"


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


@dataclass(frozen=True)
class PremiseAssembly:
    """The result of applying a set of premise records.

    Attributes:
        graph: The graph with every accepted record applied.
        rejections: The records not applied, as in ``Assembly``; ``node`` is the conclusion.
        overflow: The statements not added because their conclusion already had
            ``premises.max_per_conclusion`` analyst premises, each as ``{"conclusion", "statement"}``.
    """

    graph: Graph
    rejections: list[dict[str, Any]]
    overflow: list[dict[str, str]]


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


def _in_document_order(graph: Graph, records: Iterable[Any], key: str) -> list[Any]:
    """Sort records by the position of the node they name under ``key``; records naming no known node go last."""
    order = {node_id: i for i, node_id in enumerate(graph.nodes)}
    end = len(order)

    def position(record: Any) -> int:
        node_id = record.get(key) if isinstance(record, Mapping) else None
        return order.get(node_id, end) if isinstance(node_id, str) else end

    return sorted(records, key=position)


# Refuse a record's node before it is applied, or apply a record: the graph and what it reports.
Admit = Callable[[Node], None]
Apply = Callable[[Graph, Mapping[str, Any]], tuple[Graph, list[dict[str, str]]]]


def _apply_all(  # noqa: PLR0913 - the record kind is described by its key and its two steps
    graph: Graph,
    records: Sequence[Any],
    key: str,
    admit: Admit,
    apply: Apply,
    *,
    only: Collection[str] | None,
) -> tuple[Graph, list[dict[str, Any]], list[dict[str, str]]]:
    """Apply records in document order, each as a whole or not at all.

    Returns:
        The graph, the rejections, and what the applied records report.
    """
    rejections: list[dict[str, Any]] = []
    reported: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in _in_document_order(graph, records, key):
        node_id = record.get(key) if isinstance(record, Mapping) else None
        try:
            if not isinstance(node_id, str):
                raise _RecordError(INVALID_RECORD, "the record names no node")
            if node_id not in graph:
                raise _RecordError(UNKNOWN_NODE, f"node {node_id!r} is not in the graph")
            admit(graph.nodes[node_id])
            if node_id in seen:
                raise _RecordError(DUPLICATE_RECORD, f"node {node_id!r} has an earlier record")
            seen.add(node_id)
            if only is not None and node_id not in only:
                raise _RecordError(NOT_REJECTED, f"node {node_id!r} had no rejected record to repair")
            graph, items = apply(graph, record)
        except _RecordError as refused:
            node = node_id if isinstance(node_id, str) else None
            rejections.append({"node": node, "code": refused.code, "message": str(refused), "record": record})
        else:
            reported.extend(items)
    return graph, rejections, reported


def _admit_chunk(node: Node) -> None:
    """Refuse a classify record for a node that is not a chunk of the document."""
    if origin_of(node) != DOCUMENT:
        raise _RecordError(INVALID_RECORD, f"node {node.id!r} is not a chunk of the document")


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
    graph, rejections, unmapped = _apply_all(
        graph, records, "node", _admit_chunk, lambda graph, record: _apply(graph, rubric, record), only=only
    )
    return Assembly(graph, rejections, unmapped)


# The Python types a form value of each field type has, before its faithfulness can be checked.
_VALUE_TYPES: dict[str, tuple[type, ...]] = {
    VERBATIM: (str,),
    EQREF: (str,),
    NUMBER: (int, float),
    QUANTITY: (Mapping,),
}


def _rests_on_fixes(graph: Graph, spec: NodeType, label: str, rests_on: Sequence[Any]) -> list[str]:
    """What to change in the ``rests_on`` of a typed part."""
    if spec.rests_on is None:
        return [f"{label} rests on nothing: leave out rests_on"] if rests_on else []
    allowed = list(spec.rests_on.types)
    out = []
    for source in rests_on:
        if not isinstance(source, str) or source not in graph:
            out.append(f"{source!r} in rests_on is not a node id: use ids listed under Chunks")
        elif graph.nodes[source].kind not in allowed:
            out.append(
                f"{source!r} in rests_on is of type {graph.nodes[source].kind!r}, and {label} rests only on "
                f"{allowed}: type {source!r} as one of those in the same answer, or leave it out of rests_on"
            )
    if len(rests_on) < spec.rests_on.min:
        out.append(f"{label} must rest on at least {spec.rests_on.min} chunk(s) of the types {allowed}")
    return out


def _value_fix(field: FormField, label: str, value: Any, *, text: str, where: str) -> str | None:
    """What to change in one form value, or ``None`` if the core accepts it."""
    name = field.name
    if field.type == ENUM:
        return None if value in field.values else f"form field {name!r} of {label} must be one of {list(field.values)}"
    if field.type not in _VALUE_TYPES:
        return None
    if not isinstance(value, _VALUE_TYPES[field.type]) or isinstance(value, bool):
        return f"form field {name!r} of {label} must be a {field.type}"
    if field.type == QUANTITY and not (
        isinstance(value.get("value"), int | float)
        and not isinstance(value.get("value"), bool)
        and isinstance(value.get("unit"), str)
    ):
        return f'form field {name!r} of {label} must be an object with a number "value" and a string "unit"'
    problem = unfaithful(field.type, value, text)
    return f"form field {name!r}: {problem}; copy it exactly from {where}" if problem else None


def _form_fixes(spec: NodeType, label: str, form: Mapping[str, Any], text: str, where: str) -> list[str]:
    """What to change in the ``form`` of a typed part."""
    out = []
    extra = sorted(set(form) - set(spec.form))
    if extra:
        fields = list(spec.form) or "none"
        out.append(f"{label} has no form field {extra}: its form fields are {fields}; leave the others out")
    out.extend(
        f"{label} needs the form field {name!r}"
        for name, field in spec.form.items()
        if field.required and name not in form
    )
    for name, value in form.items():
        field = spec.form.get(name)
        found = _value_fix(field, label, value, text=text, where=where) if field is not None else None
        if found:
            out.append(found)
    return out


def _part_fixes(graph: Graph, rubric: Rubric, part: Any, *, text: str, where: str = "the chunk's text") -> list[str]:
    """What to change in one typed part of a classify record: the record itself, or one child of a split.

    ``text`` is what the part's values must be found in, named ``where`` in the instructions: the chunk's
    text for a record, the clause's for a split's child.
    """
    if not isinstance(part, Mapping):
        return ["send each typed record as a JSON object"]
    kind = part.get("type")
    spec = rubric.types.get(kind) if isinstance(kind, str) else None
    if spec is None or DOCUMENT not in spec.origins:
        types = [name for name, item in rubric.types.items() if DOCUMENT in item.origins]
        return [f"{kind!r} is not a type a chunk may have: use one of {types}"]
    label = f"type {kind!r}"
    out = []
    rests_on = part.get("rests_on", [])
    if not isinstance(rests_on, list) or not all(isinstance(item, str) for item in rests_on):
        out.append('rests_on must be a list of chunk ids, such as ["s-2"]')
    else:
        out.extend(_rests_on_fixes(graph, spec, label, rests_on))
    form = part.get("form", {})
    if not isinstance(form, Mapping):
        fields = list(spec.form) or "none"
        out.append(f"form must be an object of field names and values; the form fields of {label} are {fields}")
    else:
        out.extend(_form_fixes(spec, label, form, text, where))
    hedge = part.get("hedge")
    if isinstance(hedge, str) and hedge and hedge not in text:
        out.append(f"hedge {hedge!r} is not in {where}: copy the phrase exactly, or leave hedge out")
    return out


def _node_fix(graph: Graph, node_id: Any) -> str | None:
    """What to change in the node a classify record names, or ``None`` if it is a chunk of the document."""
    if isinstance(node_id, str) and node_id in graph:
        if origin_of(graph.nodes[node_id]) == DOCUMENT:
            return None
        return f"{node_id!r} is not a chunk of the document: send records only for the ids listed under Chunks"
    base = node_id[:-1] if isinstance(node_id, str) and node_id[-1:].isalpha() else None
    if base is not None and base in graph:
        return (
            f"{node_id!r} is not a chunk id: a clause id such as {node_id!r} exists only once a split of "
            f'{base!r} is applied. To type the clauses of {base!r}, send one record with "node": {base!r}, '
            f'"split" and "children"; to type {base!r} whole, send "node": {base!r}'
        )
    return f"{node_id!r} is not a chunk id: use one of the ids listed under Chunks"


def fixes(graph: Graph, rubric: Rubric, record: Any) -> list[str]:
    """Say what to change in a classify record so the assembler can apply it, as instructions to the model.

    The record is read against the graph it was applied to and the rubric, independently of the error the
    core gave: a node id that is not a chunk (and, for an id such as ``s-2a``, how a chunk is split), a type
    a chunk may not have, ``rests_on`` on a type that rests on nothing or naming nodes of other types, form
    fields of another type, a required one missing, a value or hedge not found in the text, and a clause
    that does not occur exactly once.

    Args:
        graph: The graph the record was applied to.
        rubric: The rubric.
        record: The record, as the model returned it.

    Returns:
        One instruction per problem found, in the record's order; empty when none is found.
    """
    if not isinstance(record, Mapping):
        return ["send each record as a JSON object"]
    node_fix = _node_fix(graph, record.get("node"))
    if node_fix is not None:
        return [node_fix]
    node_id = record["node"]
    text = node_text(graph, graph.nodes[node_id])
    if "split" not in record:
        return _part_fixes(graph, rubric, record, text=text)
    clauses, children = record.get("split"), record.get("children")
    if not isinstance(clauses, list) or not isinstance(children, list) or len(clauses) != len(children):
        return ['a split record needs "split", a list of clauses, and "children", one typed record per clause']
    out = []
    for i, (clause, child) in enumerate(zip(clauses, children, strict=True)):
        found = text.count(clause) if isinstance(clause, str) and clause else 0
        if found != 1:
            out.append(
                f"clause {i + 1} occurs {found} times in the text of {node_id!r}: copy a clause that occurs once"
            )
        clause_text = clause if isinstance(clause, str) else ""
        found = _part_fixes(graph, rubric, child, text=clause_text, where="the clause's text")
        out.extend(f"children[{i}]: {fix}" for fix in found)
    return out


def premises_rubric_problem(rubric: Rubric) -> str | None:
    """Say why a rubric cannot take premise records, or return ``None`` if it can.

    Args:
        rubric: The rubric.

    Returns:
        The problem: the rubric has no ``[premises]`` table, or does not declare the ``assumption``
        type for analyst nodes.
    """
    if rubric.premises is None:
        return f"rubric {rubric.name!r} has no [premises] table"
    spec = rubric.types.get(ASSUMPTION)
    if spec is None or ANALYST not in spec.origins:
        return f"rubric {rubric.name!r} does not declare the type {ASSUMPTION!r} with origin {ANALYST!r}"
    return None


def _premise_parts(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Check a premise record's shape; return its linked chunks and its normalised statements, each once.

    Raises:
        _RecordError: If the record has a field other than ``conclusion`` and ``premises``, or a premise
            is not an object with exactly one of ``stated_at`` and ``statement``, as a non-empty string.
    """
    unknown = sorted(set(record) - PREMISE_FIELDS)
    if unknown:
        raise _RecordError(INVALID_RECORD, f"a premise record has unknown fields {unknown}")
    premises = record.get("premises")
    if not isinstance(premises, list):
        raise _RecordError(INVALID_RECORD, "premises must be a list")
    links: list[str] = []
    statements: list[str] = []
    for i, premise in enumerate(premises):
        if not isinstance(premise, Mapping) or len(premise) != 1 or not set(premise) <= {STATED_AT, STATEMENT}:
            msg = f"premises[{i}] must be an object with exactly one of {STATED_AT} and {STATEMENT}"
            raise _RecordError(INVALID_RECORD, msg)
        ((name, value),) = premise.items()
        if not isinstance(value, str) or not normalise_text(value):
            raise _RecordError(INVALID_RECORD, f"premises[{i}].{name} must be a non-empty string")
        if name == STATED_AT:
            links.append(value)
        else:
            statements.append(normalise_text(value))
    return links, list(dict.fromkeys(statements))


def _apply_premises(graph: Graph, rubric: Rubric, record: Mapping[str, Any]) -> tuple[Graph, list[dict[str, str]]]:
    """Apply one premise record, as a whole; return the graph and the statements over the cap.

    Raises:
        _RecordError: If the record is malformed, links a node the graph lacks, or the core refuses an edit.
    """
    premises = rubric.premises
    assert premises is not None  # noqa: S101 - assemble_premises refuses a rubric without [premises]
    conclusion = record["conclusion"]
    links, statements = _premise_parts(record)
    missing = [node_id for node_id in links if node_id not in graph]
    if missing:
        raise _RecordError(UNKNOWN_NODE, f"conclusion {conclusion!r} is stated at nodes not in the graph: {missing}")
    held = sum(
        relation.type == REQUIRES
        and relation.target == conclusion
        and origin_of(graph.nodes[relation.source]) == ANALYST
        for relation in graph.relations.values()
    )
    room = max(premises.max_per_conclusion - held, 0)
    analysts = {node.statement: node.id for node in graph if origin_of(node) == ANALYST}
    fresh = (node_id for n in count(1) if (node_id := f"{ANALYST_PREFIX}-{n}") not in graph)
    try:
        if links:
            graph = annotate_node(graph, rubric, conclusion, rests_on=links)
        out = Graph()
        for node in graph:
            out.add_node(node)
        for relation in graph.relations.values():
            out.add_relation(relation)
        for statement in statements[:room]:
            node_id = analysts.get(statement)
            if node_id is None:
                node_id = next(fresh)
                out.add_node(
                    Node(
                        node_id,
                        kind=ASSUMPTION,
                        statement=statement,
                        base=premises.base,
                        attributes={"origin": ANALYST},
                    )
                )
            relation_id = f"{node_id}-{REQUIRES}-{conclusion}"
            out.add_relation(Relation(relation_id, REQUIRES, node_id, conclusion, strength=premises.strength))
    except ValidationError as error:
        raise _RecordError(translate(error).code, str(error)) from None
    return out, [{"conclusion": conclusion, "statement": statement} for statement in statements[room:]]


def assemble_premises(
    graph: Graph, rubric: Rubric, records: Sequence[Any], *, only: Collection[str] | None = None
) -> PremiseAssembly:
    """Apply premise records in document order of their conclusions.

    Each record is applied as a whole or not at all. One is rejected, and its conclusion left as it
    was, when it is malformed, names a node the graph lacks or one that is not a document node of a
    type in ``premises.for``, repeats the conclusion of an earlier record, falls outside ``only``, or
    the core refuses a ``stated_at`` link: a node of a type the edge rule does not allow, a relation
    that already exists or would close a cycle. A conclusion takes at most
    ``premises.max_per_conclusion`` analyst premises; the statements beyond are returned as overflow,
    not added. A statement repeated within one record is one premise.

    Args:
        graph: The graph, left unchanged.
        rubric: The rubric; it must pass ``premises_rubric_problem``.
        records: The records, as the model returned them.
        only: The conclusions records may name; any by default.

    Returns:
        The graph, the rejections and the overflow.

    Raises:
        ValueError: If the rubric cannot take premise records.
    """
    problem = premises_rubric_problem(rubric)
    if problem is not None:
        raise ValueError(problem)
    for_types = rubric.premises.for_types  # type: ignore[union-attr]

    def admit(node: Node) -> None:
        if origin_of(node) != DOCUMENT or node.kind not in for_types:
            msg = f"node {node.id!r} is a {node.kind!r}, not a conclusion of a type in premises.for {list(for_types)}"
            raise _RecordError(INVALID_RECORD, msg)

    graph, rejections, overflow = _apply_all(
        graph, records, "conclusion", admit, lambda graph, record: _apply_premises(graph, rubric, record), only=only
    )
    return PremiseAssembly(graph, rejections, overflow)
