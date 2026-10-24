"""Digests of recorded bytes, in the ``sha256:<hex>`` form the core uses for text."""

from __future__ import annotations

import hashlib

from credencegraph.rubric.text import DIGEST_PREFIX


def bytes_digest(data: bytes) -> str:
    """Return ``sha256:`` and the hex SHA-256 of ``data``, unnormalised.

    Args:
        data: The bytes.

    Returns:
        The digest.
    """
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()
