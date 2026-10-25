"""The ``credencegraph-extract`` command.

``run`` extracts a graph from a LaTeX document into a new run directory. ``replay`` rebuilds a run's
graph from its recorded calls, without the network, and reports whether it is byte-identical to the
recorded ``graph.json``. A run's citation checks read the cited works from ``--cited``; a replay
needs the same directory, and re-runs every check that a run allowed, uncalibrated ones included. The exit status is 0 on success, 1 if a replayed graph differs from the
recorded one, and 2 on any error, reported as one line on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import anthropic
import openai

from credencegraph_extract import cited, provider
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.pipeline import passes
from credencegraph_extract.run import GRAPH, RunError, read_manifest, read_run_file, replay, run_extraction

DEFAULT_MAX_TOKENS = 16000


def _no_model(prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a model call when no ``--model`` was given."""
    raise RunError("missing-model", "a pass needs the model: pass --model with the exact dated model id")


def _options(text: str) -> dict[str, Any]:
    """Read ``--request-options``: a JSON object."""
    try:
        value = json.loads(text)
    except ValueError as error:
        msg = f"not JSON: {error}"
        raise argparse.ArgumentTypeError(msg) from None
    if not isinstance(value, dict):
        msg = "not a JSON object"
        raise argparse.ArgumentTypeError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credencegraph-extract", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Extract a graph from a LaTeX document into a new run directory.")
    run.add_argument("source", type=Path, help="The main .tex file of the document.")
    run.add_argument("run_dir", type=Path, help="The run directory to create; it must not exist or be empty.")
    run.add_argument("--rubric", type=Path, required=True, help="The TOML rubric.")
    run.add_argument("--document", help="The document id of every anchor; the source's file name by default.")
    run.add_argument("--model", help="The exact model id the passes call, dated where the provider dates its ids.")
    run.add_argument(
        "--provider",
        choices=provider.PROVIDERS,
        default=provider.ANTHROPIC,
        help="The API the endpoint speaks: anthropic (Messages) or openai (Chat Completions); anthropic by default.",
    )
    run.add_argument(
        "--base-url",
        help="The endpoint of a server that speaks the provider's API; the provider SDK's default by default.",
    )
    run.add_argument(
        "--request-options",
        type=_options,
        help="Further fields of every request body, as a JSON object, such as '{\"thinking_budget\": 4000}'.",
    )
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="The output token limit per call.")
    run.add_argument(
        "--store-source",
        action="store_true",
        help="Copy the source document into the run directory (it is not copied by default; its digest always is).",
    )
    run.add_argument("--cited", type=Path, help="The directory of cited works, each named by its bibliography key.")
    run.add_argument(
        "--fetch-arxiv",
        action="store_true",
        help="Download a cited work missing from --cited from arXiv, into --cited, when its entry names one.",
    )
    run.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Run a model-judged check with a model its check type was not fitted with; the report lists each.",
    )

    rerun = commands.add_parser("replay", help="Rebuild a run's graph from raw/ and compare it with graph.json.")
    rerun.add_argument("run_dir", type=Path, help="The run directory.")
    rerun.add_argument("--source", type=Path, help="The source document, for a run that did not store it.")
    rerun.add_argument("--output", type=Path, help="Also write the rebuilt graph to this file.")
    rerun.add_argument("--cited", type=Path, help="The directory of cited works the run read.")
    return parser


def _replay(args: argparse.Namespace) -> int:
    """Replay a run and report whether its graph is byte-identical; 0 if it is, 1 if it differs."""
    resolve = cited.directory(args.cited) if args.cited else cited.no_sources
    rebuilt = replay(args.run_dir, passes=passes(cited=resolve, allow_uncalibrated=True), source=args.source)
    recorded = read_manifest(args.run_dir)["graph_digest"]
    if args.output:
        try:
            args.output.write_bytes(rebuilt)
        except OSError as error:
            raise RunError("io-error", f"cannot write {str(args.output)!r}: {error.strerror or error}") from None
    if rebuilt == read_run_file(args.run_dir, GRAPH) and bytes_digest(rebuilt) == recorded:
        print(f"replayed {args.run_dir}: {GRAPH} is byte-identical ({recorded})")
        return 0
    print(f"replayed {args.run_dir}: {GRAPH} differs: recorded {recorded}, rebuilt {bytes_digest(rebuilt)}")
    return 1


def _run(args: argparse.Namespace) -> int:
    """Run the pipeline into a new run directory."""
    call = (
        partial(
            provider.call,
            model=args.model,
            max_tokens=args.max_tokens,
            provider=args.provider,
            base_url=args.base_url,
            options=args.request_options,
        )
        if args.model
        else _no_model
    )
    try:
        provider.check_options(args.provider, args.request_options or {})
    except ValueError as error:
        raise RunError("invalid-argument", str(error)) from None
    if args.fetch_arxiv and not args.cited:
        raise RunError("invalid-argument", "--fetch-arxiv needs --cited, the directory to download into")
    resolve = cited.directory(args.cited, fetch_arxiv=args.fetch_arxiv) if args.cited else cited.no_sources
    manifest = run_extraction(
        args.source,
        args.run_dir,
        args.rubric,
        call=call,
        passes=passes(cited=resolve, allow_uncalibrated=args.allow_uncalibrated),
        document=args.document,
        store_source=args.store_source,
    )
    print(f"run {manifest['run']} written to {args.run_dir}: graph {manifest['graph_digest']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Every failure is reported as one ``error [<code>]: <message>`` line on stderr with status 2,
    never as a traceback: a run directory that is malformed or incomplete, a file that cannot be
    read or written, a source or record that does not match the manifest, and a provider error.

    Args:
        argv: The arguments; ``sys.argv[1:]`` by default.

    Returns:
        The exit status: 0 on success (for ``replay``, the rebuilt graph is byte-identical), 1 if
        ``replay`` rebuilt a graph that differs from the recorded one, 2 on any error.
    """
    args = _parser().parse_args(argv)
    try:
        return _run(args) if args.command == "run" else _replay(args)
    except RunError as error:
        code, message = error.code, str(error)
    except provider.ProviderError as error:
        code, message = "provider-error", str(error)
    except (anthropic.APIError, openai.APIError) as error:
        code, message = "provider-error", f"{type(error).__name__}: {error}"
    print(f"error [{code}]: {message}", file=sys.stderr)
    return 2


def app() -> None:
    """Entry point of the ``credencegraph-extract`` script."""
    sys.exit(main())
