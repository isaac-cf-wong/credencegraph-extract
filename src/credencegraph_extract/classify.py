"""The classify pass: the model types each chunk, splits mixed ones, links them and locates hedges.

The pass sends the chunks of the document and the rubric's types in one prompt, under an output
schema built from the rubric, and the model answers with one classify record per chunk::

    {"node": "s-3", "type": "claim", "form": {...}, "rests_on": ["s-2"], "hedge": "suggests"}

or, for a chunk that makes several statements::

    {"node": "s-2", "split": ["clause one", "clause two"], "children": [{"type": ...}, {"type": ...}]}

A record has no credence field. The assembler applies the records; what the core refuses goes back
to the model once, with the error each record got, and what still fails stays rejected. Both answers
are recorded like any other call.
"""

from __future__ import annotations

import json
from string import Template
from typing import TYPE_CHECKING, Any

from credencegraph.core import Graph
from credencegraph.rubric import FormField, Rubric
from credencegraph.rubric.model import BOOL, DOCUMENT, ENUM, EQREF, NUMBER, QUANTITY, SCOPE, VERBATIM
from credencegraph.rubric.validate import origin_of

from credencegraph_extract.assemble import assemble, edge_rules_not_requires
from credencegraph_extract.run import Outcome, RunError

if TYPE_CHECKING:
    from credencegraph_extract.run import Ask

NAME = "classify"
INITIAL = "initial"
REPAIR = "repair"

TEMPLATE = """\
You are classifying the chunks of a document against a rubric. Each chunk is a verbatim sentence,
equation or other unit of the document, with an id. Return one record for every chunk you can type.

A record types one chunk: "type" is one of the types below, "form" fills in that type's fields,
"rests_on" lists the ids of the chunks the statement depends on, and "hedge" is the hedging phrase
the chunk uses, such as "suggests" or "may", if it uses one.
A chunk that makes several statements of different types is split instead: "split" lists clauses
copied exactly from its text, each occurring exactly once in it, and "children" gives one typed
record (without "node") per clause, in the same order. Its children are then named by the chunk's
id followed by a, b, c, ..., and other records may rest on them.

Rules:
- Copy every text value exactly from the chunk: a verbatim field and a hedge must be substrings of
  the chunk's text, a number or quantity must appear in it as a numeral, and an equation reference
  must be referenced in it.
- Only list in "rests_on" chunks of the types the record's type may rest on; leave it out for a type
  that rests on nothing.
- Do not give probabilities, confidence levels or any other judgement of whether a statement is
  true. The rubric sets those.

Types:
$types

Hedge phrases the rubric knows (report any other you find; it is listed for review, not dropped):
$hedges

Chunks, in document order, as id (current type): text
$chunks

Records from your previous answer that were not applied, with the error each one got. If this list
is not empty, return corrected records for these chunks only, and nothing else:
$rejected
"""


def _field_text(field: FormField) -> str:
    """Describe a form field for the prompt."""
    values = f" one of {list(field.values)}" if field.values else ""
    return f"{field.name} ({field.type}{values}{', required' if field.required else ''})"


def _types_text(rubric: Rubric) -> str:
    """Describe each type a chunk of the document may take."""
    lines = []
    for name, spec in rubric.types.items():
        if DOCUMENT not in spec.origins:
            continue
        lines.append(f"- {name}: {spec.description}")
        if spec.form:
            lines.append(f"  form: {'; '.join(_field_text(field) for field in spec.form.values())}")
        if spec.rests_on is not None:
            lines.append(f"  rests on at least {spec.rests_on.min} of: {', '.join(spec.rests_on.types)}")
    return "\n".join(lines)


def render(graph: Graph, rubric: Rubric, rejected: list[dict[str, Any]]) -> str:
    """Render the prompt of the pass.

    Args:
        graph: The graph.
        rubric: The rubric.
        rejected: The rejections to repair; empty for the first call.

    Returns:
        The prompt.
    """
    chunks = [f"{node.id} ({node.kind}): {node.statement or ''}" for node in graph if origin_of(node) == DOCUMENT]
    repair = [{key: rejection[key] for key in ("record", "code", "message")} for rejection in rejected]
    return Template(TEMPLATE).substitute(
        types=_types_text(rubric),
        hedges=", ".join(json.dumps(phrase, ensure_ascii=False) for phrase in rubric.hedges) or "none",
        chunks="\n".join(chunks),
        rejected=json.dumps(repair, indent=2, ensure_ascii=False),
    )


_STRING = {"type": "string"}


def _field_schema(field: FormField, rubric: Rubric) -> dict[str, Any]:
    """Return the JSON Schema of one form field's value."""
    if field.type in {VERBATIM, EQREF}:
        return dict(_STRING)
    if field.type == NUMBER:
        return {"type": "number"}
    if field.type == QUANTITY:
        return {
            "type": "object",
            "properties": {"value": {"type": "number"}, "unit": dict(_STRING)},
            "required": ["value", "unit"],
            "additionalProperties": False,
        }
    if field.type == ENUM:
        return {"type": "string", "enum": list(field.values)}
    if field.type == BOOL:
        return {"type": "boolean"}
    if field.type == SCOPE:
        bound = {"anyOf": [{"type": "number"}, {"type": "null"}]}
        return {
            "type": "object",
            "properties": {name: {"type": "array", "items": bound} for name in rubric.parameters},
            "additionalProperties": False,
        }
    msg = f"form field {field.name!r} has the unknown type {field.type!r}"
    raise RunError("invalid-rubric", msg)


def output_schema(rubric: Rubric) -> dict[str, Any]:
    """Return the output schema of the pass, built from the rubric.

    The form allows every field some type declares, each with the schema of its field type (or any
    of them, where types declare one name differently); the core checks that a field belongs to the
    record's type. No record has a field for a credence.

    Args:
        rubric: The rubric.

    Returns:
        The JSON Schema.
    """
    types = [name for name, spec in rubric.types.items() if DOCUMENT in spec.origins]
    fields: dict[str, list[dict[str, Any]]] = {}
    for name in types:
        for field in rubric.types[name].form.values():
            schema = _field_schema(field, rubric)
            if schema not in fields.setdefault(field.name, []):
                fields[field.name].append(schema)
    form = {
        "type": "object",
        "properties": {
            name: options[0] if len(options) == 1 else {"anyOf": options} for name, options in fields.items()
        },
        "additionalProperties": False,
    }
    typed = {
        "type": {"type": "string", "enum": types},
        "form": form,
        "rests_on": {"type": "array", "items": dict(_STRING)},
        "hedge": dict(_STRING),
    }
    single = {
        "type": "object",
        "properties": {"node": dict(_STRING), **typed},
        "required": ["node", "type"],
        "additionalProperties": False,
    }
    child = {"type": "object", "properties": typed, "required": ["type"], "additionalProperties": False}
    split = {
        "type": "object",
        "properties": {
            "node": dict(_STRING),
            "split": {"type": "array", "items": dict(_STRING)},
            "children": {"type": "array", "items": child},
        },
        "required": ["node", "split", "children"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"records": {"type": "array", "items": {"anyOf": [single, split]}}},
        "required": ["records"],
        "additionalProperties": False,
    }


def _records(output: Any) -> list[Any]:
    """Return the records of the model's answer.

    Raises:
        RunError: If the answer is not an object with a list of records.
    """
    if not isinstance(output, dict) or not isinstance(output.get("records"), list):
        raise RunError("invalid-record", "the classify answer is not an object with a list of records")
    return output["records"]


class ClassifyPass:
    """Type, split and link the chunks of the document, and set ``stated`` from hedges.

    Attributes:
        name: ``classify``.
        template: The prompt template, rendered once per call.
    """

    name = NAME
    template = TEMPLATE

    def run(self, graph: Graph, rubric: Rubric, ask: Ask) -> Outcome:
        """Apply the pass: one call, and one repair call if the core refused any record.

        Args:
            graph: The graph so far.
            rubric: The run's rubric.
            ask: Sends a prompt to the model.

        Returns:
            The graph; the rejections of both rounds, each marked with its round; and the report,
            whose ``hedge-unmapped`` lists the hedge phrases the rubric does not map.

        Raises:
            RunError: Before any call, if an edge rule of the rubric creates a relation other than
                ``requires``.
        """
        refused = edge_rules_not_requires(rubric)
        if refused:
            msg = (
                f"rubric {rubric.name!r} cannot be used for extraction: the edge rules of {refused} create "
                "relations other than requires, which would let a document raise the credence of its own claims"
            )
            raise RunError("invalid-rubric", msg)
        schema = output_schema(rubric)
        first = assemble(graph, rubric, _records(ask(render(graph, rubric, []), schema)))
        rejections = [{"round": INITIAL, **rejection} for rejection in first.rejections]
        unmapped = list(first.unmapped)
        result = first
        if first.rejections:
            answer = ask(render(first.graph, rubric, first.rejections), schema)
            only = {rejection["node"] for rejection in first.rejections if rejection["node"] is not None}
            result = assemble(first.graph, rubric, _records(answer), only=only)
            rejections.extend({"round": REPAIR, **rejection} for rejection in result.rejections)
            unmapped.extend(result.unmapped)
        return Outcome(result.graph, rejections, {"hedge-unmapped": unmapped})
