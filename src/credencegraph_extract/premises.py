"""The premise pass: the model names what else must be true for each conclusion to follow.

The pass runs after classification, in a fresh prompt. For every node whose type is in the rubric's
``premises.for`` it sends the conclusion's text and the texts of the chunks it already rests on, and
once, the id and text of every chunk of the document. It sends nothing else of the classification:
no types, forms, hedges or reasoning, so its mistakes do not repeat the classify pass's. The model
answers with one premise record per conclusion::

    {"conclusion": "s-3", "premises": [{"stated_at": "s-1"}, {"statement": "The noise is stationary."}]}

A premise ``stated_at`` a chunk is one the document states; a ``statement`` is one it leaves unsaid,
which the assembler adds as an analyst node. A record has no credence field. What the core refuses
goes back to the model once, as in the classify pass.
"""

from __future__ import annotations

import json
from string import Template
from typing import TYPE_CHECKING, Any

from credencegraph.core import Graph
from credencegraph.core.relation import REQUIRES
from credencegraph.rubric import Rubric
from credencegraph.rubric.model import COMPOUND, DOCUMENT
from credencegraph.rubric.validate import node_text, origin_of

from credencegraph_extract.assemble import STATED_AT, STATEMENT, assemble_premises, premises_rubric_problem
from credencegraph_extract.classify import INITIAL, REPAIR
from credencegraph_extract.run import Outcome, RunError

if TYPE_CHECKING:
    from credencegraph_extract.run import Ask

NAME = "premises"

TEMPLATE = """\
You are reading the conclusions of a document. For each conclusion below you are given its text and
the texts of the premises it is currently known to rest on. Ask: what else must be true for this
conclusion to follow from these premises?

Return one record per conclusion, with "conclusion" set to its id and "premises" listing each further
premise it needs:
- {"stated_at": "<id>"} when the document itself states the premise: give the id of that chunk from
  the list of chunks below. Do not repeat a premise the conclusion is already known to rest on.
- {"statement": "<text>"} when the document does not state it: write the premise as one short,
  self-contained sentence. Use the same sentence, word for word, for the same premise of several
  conclusions.
Give the premises that matter most first. A conclusion that needs nothing further gets an empty list.

Do not give probabilities, confidence levels or any other judgement of whether a premise is true.

Conclusions, each as id: text, then the premises it rests on:
$conclusions

Chunks of the document, in document order, as id: text
$chunks

Records from your previous answer that were not applied, with the error each one got. If this list
is not empty, return corrected records for these conclusions only, and nothing else:
$rejected
"""


def conclusions(graph: Graph, rubric: Rubric) -> list[str]:
    """Return the ids of the conclusions the pass asks about: document nodes of a type in ``premises.for``.

    Args:
        graph: The graph.
        rubric: The rubric.

    Returns:
        The ids, in document order; empty if the rubric has no ``[premises]`` table.
    """
    if rubric.premises is None:
        return []
    return [node.id for node in graph if origin_of(node) == DOCUMENT and node.kind in rubric.premises.for_types]


def render(graph: Graph, rubric: Rubric, rejected: list[dict[str, Any]]) -> str:
    """Render the prompt of the pass.

    Args:
        graph: The graph.
        rubric: The rubric.
        rejected: The rejections to repair; empty for the first call.

    Returns:
        The prompt.
    """
    blocks = []
    for node_id in conclusions(graph, rubric):
        lines = [f"{node_id}: {node_text(graph, graph.nodes[node_id])}"]
        parents = [
            relation.source
            for relation in graph.relations.values()
            if relation.type == REQUIRES and relation.target == node_id
        ]
        lines.extend(f"  rests on {parent}: {graph.nodes[parent].statement or ''}" for parent in parents)
        lines.extend([] if parents else ["  rests on nothing yet"])
        blocks.append("\n".join(lines))
    chunks = [
        f"{node.id}: {node.statement}"
        for node in graph
        if origin_of(node) == DOCUMENT and node.kind != COMPOUND and node.statement
    ]
    repair = [{key: rejection[key] for key in ("record", "code", "message")} for rejection in rejected]
    return Template(TEMPLATE).substitute(
        conclusions="\n".join(blocks),
        chunks="\n".join(chunks),
        rejected=json.dumps(repair, indent=2, ensure_ascii=False),
    )


_STRING = {"type": "string"}


def output_schema() -> dict[str, Any]:
    """Return the output schema of the pass. No record has a field for a credence.

    Returns:
        The JSON Schema.
    """
    premise = {
        "anyOf": [
            {
                "type": "object",
                "properties": {name: dict(_STRING)},
                "required": [name],
                "additionalProperties": False,
            }
            for name in (STATED_AT, STATEMENT)
        ]
    }
    record = {
        "type": "object",
        "properties": {"conclusion": dict(_STRING), "premises": {"type": "array", "items": premise}},
        "required": ["conclusion", "premises"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"records": {"type": "array", "items": record}},
        "required": ["records"],
        "additionalProperties": False,
    }


def _records(output: Any) -> list[Any]:
    """Return the records of the model's answer.

    Raises:
        RunError: If the answer is not an object with a list of records.
    """
    if not isinstance(output, dict) or not isinstance(output.get("records"), list):
        raise RunError("invalid-record", "the premises answer is not an object with a list of records")
    return output["records"]


class PremisePass:
    """Link the premises a conclusion's text states, and add those it leaves unsaid as analyst nodes.

    Attributes:
        name: ``premises``.
        template: The prompt template, rendered once per call.
    """

    name = NAME
    template = TEMPLATE

    def run(self, graph: Graph, rubric: Rubric, ask: Ask) -> Outcome:
        """Apply the pass: one call, and one repair call if the core refused any record.

        A rubric without a ``[premises]`` table, or a graph with no conclusion of a type it names,
        needs no call: the graph is returned unchanged.

        Args:
            graph: The graph after classification.
            rubric: The run's rubric.
            ask: Sends a prompt to the model.

        Returns:
            The graph; the rejections of both rounds, each marked with its round; and the report, whose
            ``overflow`` lists the statements not added because their conclusion had reached
            ``premises.max_per_conclusion``.

        Raises:
            RunError: Before any call, if the rubric has ``[premises]`` but does not declare the
                ``assumption`` type for analyst nodes.
        """
        if not conclusions(graph, rubric):
            return Outcome(graph)
        problem = premises_rubric_problem(rubric)
        if problem is not None:
            raise RunError("invalid-rubric", f"{problem}, so the premise pass cannot add implicit premises")
        schema = output_schema()
        first = assemble_premises(graph, rubric, _records(ask(render(graph, rubric, []), schema)))
        rejections = [{"round": INITIAL, **rejection} for rejection in first.rejections]
        overflow = list(first.overflow)
        result = first
        if first.rejections:
            answer = ask(render(first.graph, rubric, first.rejections), schema)
            only = {rejection["node"] for rejection in first.rejections if rejection["node"] is not None}
            result = assemble_premises(first.graph, rubric, _records(answer), only=only)
            rejections.extend({"round": REPAIR, **rejection} for rejection in result.rejections)
            overflow.extend(result.overflow)
        return Outcome(result.graph, rejections, {"overflow": overflow})
