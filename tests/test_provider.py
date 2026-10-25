"""The provider module: a rendered prompt and an output schema in, the raw response and request metadata out, for either API."""

from __future__ import annotations

import json

import anthropic
import openai
import pytest
from notes_pass import (
    BASE_URL,
    MODEL,
    OPENAI_BASE_URL,
    SCHEMA,
    completion,
    fake_client,
    fake_openai_client,
    message,
)

from credencegraph_extract import provider
from credencegraph_extract.digest import bytes_digest


def anthropic_record(body: dict) -> dict:
    return {"request": {"provider": "anthropic"}, "response": body}


def openai_record(body: dict) -> dict:
    return {"request": {"provider": "openai"}, "response": body}


def test_call_returns_the_response_verbatim_with_request_metadata() -> None:
    body = message(1, '{"note": "a note"}')
    client = fake_client([body])
    record = provider.call("Prompt text", SCHEMA, model=MODEL, max_tokens=512, client=client)
    assert record["response"] == body
    assert record["request"] == {
        "provider": "anthropic",
        "sdk_version": anthropic.__version__,
        "base_url": BASE_URL,
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
    assert provider.output(anthropic_record(body)) == {"note": "joined"}


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_output_refuses_a_response_that_did_not_end_normally(stop_reason: str) -> None:
    with pytest.raises(provider.ProviderError, match=stop_reason):
        provider.output(anthropic_record(message(1, '{"note": "x"}', stop_reason)))


def test_output_refuses_text_that_is_not_json() -> None:
    with pytest.raises(provider.ProviderError, match="not JSON"):
        provider.output(anthropic_record(message(1, "no json here")))


def test_record_survives_a_json_round_trip() -> None:
    record = provider.call("p", SCHEMA, model=MODEL, max_tokens=8, client=fake_client([message(1, "{}")]))
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"response": []},
        anthropic_record({"stop_reason": "end_turn", "content": "text"}),
        anthropic_record({"stop_reason": "end_turn", "content": [{"type": "text"}]}),
        anthropic_record({"stop_reason": "end_turn", "content": ["text"]}),
        openai_record({"choices": []}),
        openai_record({"choices": ["choice"]}),
        openai_record({"choices": [{"finish_reason": "stop"}]}),
        openai_record({"choices": [{"finish_reason": "stop", "message": {"content": None}}]}),
    ],
    ids=[
        "no-response",
        "response-not-object",
        "content-not-list",
        "text-block-without-text",
        "block-not-object",
        "no-choice",
        "choice-not-object",
        "no-message",
        "message-without-text",
    ],
)
def test_output_refuses_a_malformed_response(record: dict) -> None:
    with pytest.raises(provider.ProviderError):
        provider.output(record)


def test_an_openai_call_returns_the_response_verbatim_with_request_metadata() -> None:
    body = completion(1, '{"note": "a note"}')
    client = fake_openai_client([body])
    record = provider.call("Prompt text", SCHEMA, model=MODEL, max_tokens=512, provider="openai", client=client)
    assert record["response"] == body
    assert record["request"] == {
        "provider": "openai",
        "sdk_version": openai.__version__,
        "base_url": OPENAI_BASE_URL,
        "model": MODEL,
        "parameters": {"max_tokens": 512},
        "output_schema": SCHEMA,
        "prompt_digest": bytes_digest(b"Prompt text"),
        "request_id": "req_chatcmpl-fixture0001",
    }
    assert client.sent == [
        {
            "model": MODEL,
            "max_completion_tokens": 512,
            "messages": [{"role": "user", "content": "Prompt text"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": SCHEMA}},
        }
    ]
    assert provider.output(record) == {"note": "a note"}


@pytest.mark.parametrize(("name", "sdk"), [("anthropic", anthropic), ("openai", openai)])
def test_a_default_client_is_built_for_the_endpoint(name: str, sdk: object, monkeypatch: pytest.MonkeyPatch) -> None:
    made = []
    body = completion(1, "{}") if name == "openai" else message(1, "{}")
    client = fake_openai_client([body]) if name == "openai" else fake_client([body])

    def build(base_url: str | None) -> object:
        made.append(base_url)
        return client

    monkeypatch.setattr(sdk, "OpenAI" if name == "openai" else "Anthropic", build)
    record = provider.call("p", SCHEMA, model=MODEL, max_tokens=8, provider=name, base_url="http://localhost:8000/v1")
    assert made == ["http://localhost:8000/v1"]
    assert record["request"]["provider"] == name


def test_an_unknown_provider_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown provider 'other'"):
        provider.call("p", SCHEMA, model=MODEL, max_tokens=8, provider="other")


def test_a_record_is_read_as_the_provider_it_names() -> None:
    # The same text read as the other provider's body is malformed.
    with pytest.raises(provider.ProviderError, match="no choice"):
        provider.output(openai_record(message(1, "{}")))
    with pytest.raises(provider.ProviderError, match="stopped with None"):
        provider.output(anthropic_record(completion(1, "{}")))
    with pytest.raises(provider.ProviderError, match="unknown provider 'other'"):
        provider.output({"request": {"provider": "other"}, "response": message(1, "{}")})
    with pytest.raises(provider.ProviderError, match="unknown provider None"):
        provider.output({"response": message(1, "{}")})


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_openai_output_refuses_a_response_that_did_not_end_normally(finish_reason: str) -> None:
    with pytest.raises(provider.ProviderError, match=finish_reason):
        provider.output(openai_record(completion(1, '{"note": "x"}', finish_reason)))


def test_openai_output_refuses_a_refusal() -> None:
    with pytest.raises(provider.ProviderError, match="is a refusal: I cannot help"):
        provider.output(openai_record(completion(1, None, refusal="I cannot help")))


def test_openai_output_refuses_text_that_is_not_json() -> None:
    with pytest.raises(provider.ProviderError, match="not JSON"):
        provider.output(openai_record(completion(1, "no json here")))
