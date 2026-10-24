"""The ``credencegraph-extract`` command.

``run`` extracts a graph from a LaTeX document into a new run directory. ``replay`` rebuilds a run's
graph from its recorded calls, without the network, and reports whether it is byte-identical to the
recorded ``graph.json``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from credencegraph_extract import provider
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import GRAPH, RunError, read_manifest, replay, run_extraction

DEFAULT_MAX_TOKENS = 16000


def _no_model(prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a model call when no ``--model`` was given."""
    raise RunError("missing-model", "a pass needs the model: pass --model with the exact dated model id")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credencegraph-extract", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Extract a graph from a LaTeX document into a new run directory.")
    run.add_argument("source", type=Path, help="The main .tex file of the document.")
    run.add_argument("run_dir", type=Path, help="The run directory to create; it must not exist or be empty.")
    run.add_argument("--rubric", type=Path, required=True, help="The TOML rubric.")
    run.add_argument("--document", help="The document id of every anchor; the source's file name by default.")
    run.add_argument("--model", help="The exact dated model id the passes call.")
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="The output token limit per call.")
    run.add_argument(
        "--store-source",
        action="store_true",
        help="Copy the source document into the run directory (it is not copied by default; its digest always is).",
    )

    rerun = commands.add_parser("replay", help="Rebuild a run's graph from raw/ and compare it with graph.json.")
    rerun.add_argument("run_dir", type=Path, help="The run directory.")
    rerun.add_argument("--source", type=Path, help="The source document, for a run that did not store it.")
    rerun.add_argument("--output", type=Path, help="Also write the rebuilt graph to this file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: The arguments; ``sys.argv[1:]`` by default.

    Returns:
        The exit status: 0 on success, 1 if a replayed graph differs, 2 on an error.
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            call = partial(provider.call, model=args.model, max_tokens=args.max_tokens) if args.model else _no_model
            manifest = run_extraction(
                args.source,
                args.run_dir,
                args.rubric,
                call=call,
                document=args.document,
                store_source=args.store_source,
            )
            print(f"run {manifest['run']} written to {args.run_dir}: graph {manifest['graph_digest']}")
            return 0
        rebuilt = replay(args.run_dir, source=args.source)
    except RunError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_bytes(rebuilt)
    recorded = read_manifest(args.run_dir)["graph_digest"]
    if rebuilt == (args.run_dir / GRAPH).read_bytes() and bytes_digest(rebuilt) == recorded:
        print(f"replayed {args.run_dir}: {GRAPH} is byte-identical ({recorded})")
        return 0
    print(f"replayed {args.run_dir}: {GRAPH} differs: recorded {recorded}, rebuilt {bytes_digest(rebuilt)}")
    return 1


def app() -> None:
    """Entry point of the ``credencegraph-extract`` script."""
    sys.exit(main())
