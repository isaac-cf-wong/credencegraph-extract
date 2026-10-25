"""The one module that calls a model provider: an Anthropic-compatible or an OpenAI-compatible endpoint.

``call`` sends a rendered prompt under a JSON output schema and returns a *record*: the raw response
body, verbatim, with the metadata of the request that produced it. ``output`` reads the structured
output back out of a record, as the provider the record names would. A replay reads recorded
records with the same ``output``, so a recorded response is interpreted exactly as it was when it
arrived.

Two wire formats are spoken, each through its provider's own SDK:

- ``anthropic``: the Messages API, with the schema as ``output_config``.
- ``openai``: the Chat Completions API, with the schema as a ``json_schema`` response format. It is
  the format most OpenAI-compatible servers implement.

Either may be pointed at another server that speaks it with ``base_url``; otherwise the SDK's own
environment variables choose the endpoint and the credentials (``ANTHROPIC_API_KEY`` and
``ANTHROPIC_BASE_URL``, or ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``). Every record names the
endpoint the call went to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import anthropic
import openai

from credencegraph_extract.digest import bytes_digest

ANTHROPIC = "anthropic"
OPENAI = "openai"
# The providers, each named as its SDK's distribution.
PROVIDERS = (ANTHROPIC, OPENAI)

# The name an OpenAI response format gives its schema.
SCHEMA_NAME = "output"

# The body fields each call sets itself, which request options may not override, with the other spelling
# of a field the call sets: OpenAI's deprecated max_tokens would sit beside max_completion_tokens in the
# body, and a server that reads it would apply a limit the record does not show.
_SET_BY_CALL = {
    ANTHROPIC: frozenset({"model", "max_tokens", "messages", "output_config"}),
    OPENAI: frozenset({"model", "max_completion_tokens", "max_tokens", "messages", "response_format"}),
}

_NO_CREDENTIALS = "no API key: set {variable}; a server that checks none still needs some value there"


class ProviderError(Exception):
    """A response carries no structured output: it is malformed, the model refused or ran out of tokens, or it is not JSON."""


def check_options(provider: str, options: Mapping[str, Any]) -> None:
    """Refuse an unknown provider, or request options that set a field a call to it sets itself.

    Args:
        provider: The provider.
        options: The request options.

    Raises:
        ValueError: If ``provider`` is not one of ``PROVIDERS``, or ``options`` sets such a field.
    """
    if provider not in PROVIDERS:
        msg = f"unknown provider {provider!r}: not one of {list(PROVIDERS)}"
        raise ValueError(msg)
    taken = sorted(set(options) & _SET_BY_CALL[provider])
    if taken:
        msg = f"request options may not set {taken}: the call sets them"
        raise ValueError(msg)


def call(  # noqa: PLR0913 - each argument is a separate choice of the caller
    prompt: str,
    schema: Mapping[str, Any],
    *,
    model: str,
    max_tokens: int,
    provider: str = ANTHROPIC,
    base_url: str | None = None,
    options: Mapping[str, Any] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Send one prompt and return the record of the call.

    Args:
        prompt: The rendered prompt, sent as the single user message.
        schema: The JSON Schema the response must follow.
        model: The exact model id, dated where the provider dates its ids.
        max_tokens: The output token limit; sent to an OpenAI-compatible endpoint as
            ``max_completion_tokens``.
        provider: ``anthropic`` or ``openai``: the wire format, and the SDK that speaks it.
        base_url: The endpoint of a server that speaks it; the SDK's default by default.
        options: Further fields of the request body, for what a server accepts beyond the API, such as
            ``{"thinking_budget": 4000}``; recorded with the sampling parameters. They may not set a
            field the call sets itself.
        client: The client to send with; one of the provider's SDK, built from ``base_url`` and the
            environment's credentials, by default.

    Returns:
        ``{"request": ..., "response": ...}``: the request metadata (provider, SDK version, endpoint,
        model, sampling parameters, output schema, prompt digest, request id) and the response body
        as the provider sent it. The request id is the one the provider's request-id header gives;
        from a server that sends none, it is the ``id`` the response body gives itself.

    Raises:
        ValueError: If ``provider`` is not one of ``PROVIDERS``, or ``options`` sets a field the call sets.
        ProviderError: If no client is given and the environment has no credentials for the provider.
    """
    options = dict(options or {})
    check_options(provider, options)
    extra = {"extra_body": options} if options else {}
    if provider == ANTHROPIC:
        if client is None:
            client = anthropic.Anthropic(base_url=base_url)
            # The SDK builds a client without credentials and fails only when it sends, with a TypeError.
            if client.api_key is None and client.auth_token is None:
                raise ProviderError(_NO_CREDENTIALS.format(variable="ANTHROPIC_API_KEY"))
        sdk_version = anthropic.__version__
        raw = client.messages.with_raw_response.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": dict(schema)}},
            **extra,
        )
        response = raw.json()
    else:
        if client is None:
            try:
                client = openai.OpenAI(base_url=base_url)
            except openai.OpenAIError:
                raise ProviderError(_NO_CREDENTIALS.format(variable="OPENAI_API_KEY")) from None
        sdk_version = openai.__version__
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "schema": dict(schema)}},
            **extra,
        )
        response = json.loads(raw.text)
    request = {
        "provider": provider,
        "sdk_version": sdk_version,
        "base_url": str(client.base_url),
        "model": model,
        "parameters": {"max_tokens": max_tokens, **({"options": options} if options else {})},
        "output_schema": dict(schema),
        "prompt_digest": bytes_digest(prompt.encode("utf-8")),
        # A server that sends no request-id header still names its response; that name stands in.
        "request_id": raw.request_id or (response.get("id") if isinstance(response, dict) else None),
    }
    return {"request": request, "response": response}


def _json(text: str, response: Mapping[str, Any]) -> Any:
    """Parse the text of a response as JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"response {response.get('id')!r} is not JSON: {error}"
        raise ProviderError(msg) from None


def _anthropic_output(response: Mapping[str, Any]) -> Any:
    """The structured output of an Anthropic Messages response."""
    stop = response.get("stop_reason")
    if stop != "end_turn":
        msg = f"response {response.get('id')!r} stopped with {stop!r}, not 'end_turn'"
        raise ProviderError(msg)
    blocks = response.get("content")
    if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
        msg = f"response {response.get('id')!r} has no list of content blocks"
        raise ProviderError(msg)
    texts = [block.get("text") for block in blocks if block.get("type") == "text"]
    if not all(isinstance(text, str) for text in texts):
        msg = f"response {response.get('id')!r} has a text block without text"
        raise ProviderError(msg)
    return _json("".join(texts), response)


def _openai_output(response: Mapping[str, Any]) -> Any:
    """The structured output of an OpenAI Chat Completions response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        msg = f"response {response.get('id')!r} has no choice"
        raise ProviderError(msg)
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        msg = f"response {response.get('id')!r} has no message"
        raise ProviderError(msg)
    if message.get("refusal"):
        msg = f"response {response.get('id')!r} is a refusal: {message['refusal']}"
        raise ProviderError(msg)
    stop = choice.get("finish_reason")
    if stop != "stop":
        msg = f"response {response.get('id')!r} stopped with {stop!r}, not 'stop'"
        raise ProviderError(msg)
    content = message.get("content")
    if not isinstance(content, str):
        msg = f"response {response.get('id')!r} has a message without text"
        raise ProviderError(msg)
    return _json(content, response)


def output(record: Mapping[str, Any]) -> Any:
    """Return the structured output of a recorded call, read as its provider's wire format.

    Args:
        record: A record returned by ``call``.

    Returns:
        The JSON value the model returned.

    Raises:
        ProviderError: If the record names an unknown provider, or the response is malformed, was
            refused, did not end normally, or its text is not JSON.
    """
    response = record.get("response")
    if not isinstance(response, dict):
        msg = "the record has no response object"
        raise ProviderError(msg)
    request = record.get("request")
    provider = request.get("provider") if isinstance(request, dict) else None
    if provider == ANTHROPIC:
        return _anthropic_output(response)
    if provider == OPENAI:
        return _openai_output(response)
    msg = f"the record names an unknown provider {provider!r}"
    raise ProviderError(msg)
