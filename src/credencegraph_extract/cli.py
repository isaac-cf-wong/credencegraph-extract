"""The ``credencegraph-extract`` command.

``run`` extracts a graph from a LaTeX document into a new run directory. ``replay`` rebuilds a run's
graph from its recorded calls, without the network, and reports whether it is byte-identical to the
recorded ``graph.json``. A run's citation checks read the cited works from ``--cited``; a replay
needs the same directory, and re-runs every check that a run allowed, uncalibrated ones included.
``verify`` runs each run's deterministic checks again from their recorded inputs, reports every
check whose record the re-run does not reproduce, and warns when the runs record one rubric name and
version with different file digests. The exit status is 0 on success, 1 if a replayed graph or a
verified check differs from the recorded one, and 2 on any error, reported as one line on stderr.
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

from credencegraph_extract import cited, provider, session
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.pipeline import passes
from credencegraph_extract.run import GRAPH, MANIFEST, RunError, read_manifest, read_run_file, replay, run_extraction
from credencegraph_extract.verify import plain, rubric_reuse, run_manifest, verify_run

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
    rerun.add_argument(
        "--sources",
        type=Path,
        help="For an agent-tool run: the fetched sources it did not store, as files named by their digest.",
    )

    check = commands.add_parser(
        "verify", help="Run each run's deterministic checks again and compare them with their records."
    )
    check.add_argument("run_dirs", type=Path, nargs="+", metavar="run_dir", help="The run directories.")
    return parser


def _format_version(run_dir: Path) -> Any:
    """The format version a run directory's manifest names, or ``None`` when it cannot be read."""
    try:
        return json.loads((run_dir / MANIFEST).read_bytes()).get("version")
    except (OSError, ValueError, AttributeError):
        return None


def _replay_tools(args: argparse.Namespace) -> int:
    """Replay an agent-tool run's call log; 0 if the graph it leads to is byte-identical, 1 if it differs."""
    rebuilt = session.replay(args.run_dir, source=args.source, sources=args.sources)
    recorded = session.recorded_graph(args.run_dir)
    if args.output:
        try:
            args.output.write_bytes(rebuilt)
        except OSError as error:
            raise RunError("io-error", f"cannot write {str(args.output)!r}: {error.strerror or error}") from None
    calls = len(session.read_calls(args.run_dir))
    if rebuilt == recorded:
        print(f"replayed {args.run_dir}: the graph after {calls} calls is byte-identical ({bytes_digest(rebuilt)})")
        return 0
    print(
        f"replayed {args.run_dir}: the graph after {calls} calls differs: "
        f"recorded {bytes_digest(recorded)}, rebuilt {bytes_digest(rebuilt)}"
    )
    return 1


def _replay(args: argparse.Namespace) -> int:
    """Replay a run and report whether its graph is byte-identical; 0 if it is, 1 if it differs."""
    if _format_version(args.run_dir) == session.VERSION:
        return _replay_tools(args)
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


def _verify(args: argparse.Namespace) -> int:
    """Verify each run, and warn about rubric versions reused; 0 if every check agrees, 1 if one differs."""
    manifests = [(str(run_dir), run_manifest(run_dir)) for run_dir in args.run_dirs]
    status = 0
    for run_dir in args.run_dirs:
        found = verify_run(run_dir)
        for difference in found.differences:
            print(
                f"{run_dir}: {difference.check}: {difference.field} differs: "
                f"recorded {json.dumps(plain(difference.recorded), ensure_ascii=False)}, "
                f"verified {json.dumps(plain(difference.found), ensure_ascii=False)}"
            )
        differing = len({difference.check for difference in found.differences})
        status = max(status, 1 if differing else 0)
        print(
            f"verified {run_dir}: {len(found.rerun)} checks run again, {differing} differ; "
            f"{len(found.not_rerun)} model-judged checks not run again"
        )
    for warning in rubric_reuse(manifests):
        print(f"warning: {warning}")
    return status


def _run(args: argparse.Namespace) -> int:
    """Run the pipeline into a new run directory."""
    call = partial(provider.call, model=args.model, max_tokens=args.max_tokens) if args.model else _no_model
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
        The exit status: 0 on success (for ``replay``, the rebuilt graph is byte-identical; for
        ``verify``, every check agrees with its record), 1 if ``replay`` rebuilt a graph that differs
        from the recorded one or ``verify`` found a check that differs, 2 on any error.
    """
    args = _parser().parse_args(argv)
    try:
        command = {"run": _run, "replay": _replay, "verify": _verify}[args.command]
        return command(args)
    except RunError as error:
        code, message = error.code, str(error)
    except provider.ProviderError as error:
        code, message = "provider-error", str(error)
    except anthropic.APIError as error:
        code, message = "provider-error", f"{type(error).__name__}: {error}"
    print(f"error [{code}]: {message}", file=sys.stderr)
    return 2


def app() -> None:
    """Entry point of the ``credencegraph-extract`` script."""
    sys.exit(main())
