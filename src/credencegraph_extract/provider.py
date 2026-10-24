"""The one module that calls a model provider: Anthropic.

``call`` sends a rendered prompt under a JSON output schema and returns a *record*: the raw response
body, verbatim, with the metadata of the request that produced it. ``output`` reads the structured
output back out of a record. A replay reads recorded records with the same ``output``, so a recorded
response is interpreted exactly as it was when it arrived.

No provider interface is defined: a second provider is a second module with the same two functions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import anthropic

from credencegraph_extract.digest import bytes_digest

PROVIDER = "anthropic"


class ProviderError(Exception):
    """A response carries no structured output: it is malformed, the model refused or ran out of tokens, or it is not JSON."""


def call(
    prompt: str,
    schema: Mapping[str, Any],
    *,
    model: str,
    max_tokens: int,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    """Send one prompt and return the record of the call.

    Args:
        prompt: The rendered prompt, sent as the single user message.
        schema: The JSON Schema the response must follow.
        model: The exact dated model id.
        max_tokens: The output token limit.
        client: The client to send with; one reading the environment's credentials by default.

    Returns:
        ``{"request": ..., "response": ...}``: the request metadata (provider, SDK version, model,
        sampling parameters, output schema, prompt digest, request id) and the response body as the
        provider sent it.
    """
    client = client or anthropic.Anthropic()
    raw = client.messages.with_raw_response.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": dict(schema)}},
    )
    request = {
        "provider": PROVIDER,
        "sdk_version": anthropic.__version__,
        "model": model,
        "parameters": {"max_tokens": max_tokens},
        "output_schema": dict(schema),
        "prompt_digest": bytes_digest(prompt.encode("utf-8")),
        "request_id": raw.request_id,
    }
    return {"request": request, "response": raw.json()}


def output(record: Mapping[str, Any]) -> Any:
    """Return the structured output of a recorded call.

    Args:
        record: A record returned by ``call``.

    Returns:
        The JSON value the model returned.

    Raises:
        ProviderError: If the response is malformed, did not end normally, or its text is not JSON.
    """
    response = record.get("response")
    if not isinstance(response, dict):
        msg = "the record has no response object"
        raise ProviderError(msg)
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
    text = "".join(texts)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"response {response.get('id')!r} is not JSON: {error}"
        raise ProviderError(msg) from None
