"""A pass for the tests: it asks the model for a note on each sentence and stores it on the node.

It stands in for the real passes so the run machinery can be tested end to end on recorded calls.
``fake_client`` answers ``provider.call`` offline with the response bodies the fixture run recorded.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from credencegraph.core import Graph
from credencegraph.rubric import Rubric

from credencegraph_extract.run import Ask, Outcome

MODEL = "claude-fixture-20261024"
SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}},
    "required": ["note"],
    "additionalProperties": False,
}


class NotesPass:
    """Ask for a note on every sentence node; set it as ``attributes.note``."""

    name = "notes"
    template = "Write one short note on this sentence of a paper.\n\nSentence: {text}\n"

    def run(self, graph: Graph, rubric: Rubric, ask: Ask) -> Outcome:
        """Apply the pass.

        Args:
            graph: The graph so far.
            rubric: The run's rubric.
            ask: Sends a prompt to the model.

        Returns:
            The graph with notes.
        """
        out = Graph()
        for node in graph:
            if node.attributes.get("unit") != "sentence":
                out.add_node(node)
                continue
            note = ask(self.template.format(text=node.statement), SCHEMA)["note"]
            out.add_node(replace(node, attributes={**node.attributes, "note": note}))
        for relation in graph.relations.values():
            out.add_relation(relation)
        return Outcome(out)


def message(index: int, text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    """Return an Anthropic Messages response body whose text is ``text``."""
    return {
        "id": f"msg_fixture{index:04d}",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 40, "output_tokens": 12},
    }


def fake_client(bodies: list[dict[str, Any]]) -> Any:
    """Return a stand-in for ``anthropic.Anthropic`` that answers each call with the next body.

    Its ``sent`` list keeps the keyword arguments of every request.
    """
    sent: list[dict[str, Any]] = []
    queue = iter(bodies)

    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        body = next(queue)
        return SimpleNamespace(request_id=f"req_{body['id']}", json=lambda: json.loads(json.dumps(body)))

    return SimpleNamespace(messages=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)), sent=sent)


NOTES = [
    "States the sample size used for the noise floor.",
    "Cites prior work for the vanishing bias.",
    "Draws the conclusion from the bias result.",
]
BODIES = [message(i, json.dumps({"note": note})) for i, note in enumerate(NOTES, start=1)]
