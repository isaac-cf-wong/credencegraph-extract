"""Run the ``credencegraph-extract`` command as ``python -m credencegraph_extract``."""

from __future__ import annotations

if __name__ == "__main__":
    from credencegraph_extract.cli import app

    app()
