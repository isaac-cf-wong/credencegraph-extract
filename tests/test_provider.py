"""The provider module: a rendered prompt and an output schema in, the raw response and request metadata out."""

from __future__ import annotations

import json

import anthropic
import pytest
from notes_pass import MODEL, SCHEMA, fake_client, message

from credencegraph_extract import provider
from credencegraph_extract.digest import bytes_digest


def test_call_returns_the_response_verbatim_with_request_metadata() -> None:
    body = message(1, '{"note": "a note"}')
    client = fake_client([body])
    record = provider.call("Prompt text", SCHEMA, model=MODEL, max_tokens=512, client=client)
    assert record["response"] == body
    assert record["request"] == {
        "provider": "anthropic",
        "sdk_version": anthropic.__version__,
        "model": MODEL,
        "parameters": {"max_tokens": 512},
        "output_schema": SCHEMA,
        "prompt_digest": bytes_digest(b"Prompt text"),
        "request_id": "req_msg_fixture0001",
    }
    assert client.sent == [
        {
            "model": MODEL,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Prompt text"}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        }
    ]


def test_output_parses_the_text_blocks() -> None:
    body = message(1, "")
    body["content"] = [{"type": "text", "text": '{"note": '}, {"type": "text", "text": '"joined"}'}]
    assert provider.output({"response": body}) == {"note": "joined"}


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_output_refuses_a_response_that_did_not_end_normally(stop_reason: str) -> None:
    with pytest.raises(provider.ProviderError, match=stop_reason):
        provider.output({"response": message(1, '{"note": "x"}', stop_reason)})


def test_output_refuses_text_that_is_not_json() -> None:
    with pytest.raises(provider.ProviderError, match="not JSON"):
        provider.output({"response": message(1, "no json here")})


def test_record_survives_a_json_round_trip() -> None:
    record = provider.call("p", SCHEMA, model=MODEL, max_tokens=8, client=fake_client([message(1, "{}")]))
    assert json.loads(json.dumps(record)) == record
