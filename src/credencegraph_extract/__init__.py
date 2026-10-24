"""credencegraph-extract: an automated analyst for credencegraph, with every model call recorded and replayable."""

from __future__ import annotations

# PEP 810: defer importlib.metadata (loaded by version.py) on 3.15+.
# Ignored on 3.13/3.14 where the lazy import machinery does not exist.
__lazy_modules__ = ["credencegraph_extract.version"]

from credencegraph_extract.version import __version__

__all__ = ["__version__"]
