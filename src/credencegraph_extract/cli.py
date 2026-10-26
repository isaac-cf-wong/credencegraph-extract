"""The ``credencegraph-extract`` command.

``mcp`` serves the agent tools over MCP on stdio, for an agent harness to start (``mcp_server``).
``tool`` makes one of those calls from the command line, for a harness without MCP, and prints its JSON
result; ``unlock`` reports whether a tool call holds a run's lock, and which process. ``replay``
applies a run's logged calls again to its document, without the network, and reports whether the graph
is byte-identical to the run's. ``verify`` runs each run's deterministic checks again from their
recorded inputs, reports every check whose record the re-run does not reproduce, and warns when the runs
record one rubric name and version with different file digests; it reads runs of either format. The
exit status is 0 on success, 1 if a replayed graph or a verified check differs from the recorded one,
and 2 on any error, reported as one line on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from credencegraph_extract import session
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import MANIFEST, RunError
from credencegraph_extract.verify import plain, rubric_reuse, run_manifest, verify_run

# The agent tools, as ``tool`` takes them; the same as the MCP server's.
TOOL_NAMES = ("start_run", "get_state", "classify", "add_premises", "run_checks", "judge_citation", "finish_run")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credencegraph-extract", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    rerun = commands.add_parser(
        "replay", help="Apply a run's logged calls again to its document and compare the graph with the run's."
    )
    rerun.add_argument("run_dir", type=Path, help="The run directory.")
    rerun.add_argument("--source", type=Path, help="The source document, for a run that did not store it.")
    rerun.add_argument("--output", type=Path, help="Also write the rebuilt graph to this file.")
    rerun.add_argument(
        "--sources",
        type=Path,
        help="The fetched sources the run did not store, as files named by the hex of their digest.",
    )

    check = commands.add_parser(
        "verify", help="Run each run's deterministic checks again and compare them with their records."
    )
    check.add_argument("run_dirs", type=Path, nargs="+", metavar="run_dir", help="The run directories.")

    commands.add_parser("mcp", help="Serve the agent tools over MCP on stdio, for an agent harness to start.")

    tool = commands.add_parser(
        "tool",
        help="Make one agent-tool call on a run, as the MCP server would, and print its JSON result.",
        description=(
            "Make one agent-tool call on a run, for a harness without MCP. The result is printed as JSON on "
            "stdout; a call the run refuses is one 'error [<code>]: <message>' line on stderr, with status 2."
        ),
    )
    tool.add_argument("name", metavar="name", help=f"The tool: one of {', '.join(TOOL_NAMES)}.")
    tool.add_argument("--run", required=True, help="The run directory (for start_run, the one to create).")
    tool.add_argument(
        "--args",
        default="{}",
        help="The tool's other arguments as one JSON object, or '-' to read it from stdin.",
    )

    unlock = commands.add_parser(
        "unlock",
        help="Report whether a tool call holds a run's lock, and which process; a free lock's leftover record is removed.",
    )
    unlock.add_argument("run_dir", type=Path, help="The run directory.")
    return parser


def _format_version(run_dir: Path) -> Any:
    """The format version a run directory's manifest names, or ``None`` when it cannot be read."""
    try:
        return json.loads((run_dir / MANIFEST).read_bytes()).get("version")
    except (OSError, ValueError, AttributeError):
        return None


def _replay(args: argparse.Namespace) -> int:
    """Replay a run's call log; 0 if the graph it leads to is byte-identical, 1 if it differs."""
    if _format_version(args.run_dir) != session.VERSION:
        raise RunError(
            "unsupported-version",
            f"{str(args.run_dir)!r} is not an agent-tool run; a run of the earlier model-calling passes "
            "is no longer replayed, but verify still runs its checks again",
        )
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


def _tool(args: argparse.Namespace) -> int:
    """Make one agent-tool call through the same binding as the MCP server, and print its result."""
    from credencegraph_extract.mcp_server import TOOLS, refuse_invented  # noqa: PLC0415 - only this command needs the SDK

    refuse_invented(args.name, {})
    text = sys.stdin.read() if args.args == "-" else args.args
    try:
        arguments = json.loads(text)
    except ValueError as error:
        raise RunError("invalid-argument", f"--args is not JSON: {error}") from None
    if not isinstance(arguments, dict):
        raise RunError("invalid-argument", "--args must be a JSON object of the tool's arguments")
    if "run" in arguments:
        raise RunError("invalid-argument", "give the run directory with --run, not in --args")
    refuse_invented(args.name, arguments)
    result = TOOLS[args.name](run=args.run, **arguments)
    if isinstance(result, dict) and set(result) == {"error"}:
        raise RunError(result["error"]["code"], result["error"]["message"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _unlock(args: argparse.Namespace) -> int:
    """Report who holds a run's lock."""
    holder = session.lock_holder(args.run_dir)
    if holder is None:
        print(f"{args.run_dir}: no tool call holds the lock")
        return 0
    who = (
        f"process {holder['pid']}, since {holder.get('since')}" if "pid" in holder else "a process that wrote no record"
    )
    print(
        f"{args.run_dir}: the lock is held by {who}; the operating system releases it when that process "
        "exits, so stop it if it is hung"
    )
    return 1


def _mcp(args: argparse.Namespace) -> int:
    """Serve the agent tools over MCP until the harness closes stdin."""
    from credencegraph_extract.mcp_server import serve  # noqa: PLC0415 - only this command needs the MCP SDK

    serve()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Every failure is reported as one ``error [<code>]: <message>`` line on stderr with status 2,
    never as a traceback: a run directory that is malformed or incomplete, a file that cannot be
    read or written, a source or record that does not match the manifest, and a call the run refuses.

    Args:
        argv: The arguments; ``sys.argv[1:]`` by default.

    Returns:
        The exit status: 0 on success (for ``replay``, the rebuilt graph is byte-identical; for
        ``verify``, every check agrees with its record), 1 if ``replay`` rebuilt a graph that differs
        from the recorded one or ``verify`` found a check that differs, 2 on any error.
    """
    args = _parser().parse_args(argv)
    try:
        commands = {"replay": _replay, "verify": _verify, "mcp": _mcp, "tool": _tool, "unlock": _unlock}
        command = commands[args.command]
        return command(args)
    except RunError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        return 2


def app() -> None:
    """Entry point of the ``credencegraph-extract`` script."""
    sys.exit(main())
