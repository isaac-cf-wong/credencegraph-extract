"""A pass for the tests: it asks the model for a note on each sentence and stores it on the node.

It stands in for the real passes so the run machinery can be tested end to end on recorded calls.
``fake_client`` answers ``provider.call`` offline with the response bodies the fixture run recorded;
``fake_openai_client`` does the same for the OpenAI provider, with Chat Completions bodies.
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
# The endpoint the fake clients name: the Anthropic SDK's default.
BASE_URL = "https://api.anthropic.com"
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


def fake_client(bodies: list[dict[str, Any]], *, request_ids: bool = True) -> Any:
    """Return a stand-in for ``anthropic.Anthropic`` that answers each call with the next body.

    Its ``sent`` list keeps the keyword arguments of every request. Without ``request_ids`` it sends no
    request-id header, as some compatible servers do.
    """
    sent: list[dict[str, Any]] = []
    queue = iter(bodies)

    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        body = next(queue)
        request_id = f"req_{body['id']}" if request_ids else None
        return SimpleNamespace(request_id=request_id, json=lambda: json.loads(json.dumps(body)))

    return SimpleNamespace(
        messages=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)),
        sent=sent,
        base_url=BASE_URL,
        api_key="test-key",
        auth_token=None,
    )


NOTES = [
    "States the sample size used for the noise floor.",
    "Cites prior work for the vanishing bias.",
    "Draws the conclusion from the bias result.",
]
BODIES = [message(i, json.dumps({"note": note})) for i, note in enumerate(NOTES, start=1)]


# The endpoint the fake OpenAI client names: the OpenAI SDK's default.
OPENAI_BASE_URL = "https://api.openai.com/v1/"


def completion(index: int, text: str | None, finish_reason: str = "stop", refusal: str | None = None) -> dict[str, Any]:
    """Return an OpenAI Chat Completions response body whose message text is ``text``."""
    return {
        "id": f"chatcmpl-fixture{index:04d}",
        "object": "chat.completion",
        "created": 1761400000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": refusal},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


def fake_openai_client(bodies: list[dict[str, Any]], *, request_ids: bool = True) -> Any:
    """Return a stand-in for ``openai.OpenAI`` that answers each call with the next body.

    Its ``sent`` list keeps the keyword arguments of every request. Without ``request_ids`` it sends no
    request-id header, as some compatible servers do.
    """
    sent: list[dict[str, Any]] = []
    queue = iter(bodies)

    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        body = next(queue)
        request_id = f"req_{body['id']}" if request_ids else None
        return SimpleNamespace(request_id=request_id, text=json.dumps(body))

    completions = SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), sent=sent, base_url=OPENAI_BASE_URL)
