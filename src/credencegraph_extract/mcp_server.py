"""The MCP server: the agent tools over stdio, for an agent harness to call.

``credencegraph-extract mcp`` serves one MCP tool per tool of ``session`` and ``tools``: ``start_run``,
``get_state``, ``classify``, ``add_premises``, ``run_checks``, ``judge_citation`` and ``finish_run``.
The server holds no state: every tool takes the run directory and reads and writes it, so a server
that is restarted, or several serving one run, lose nothing.

Each tool's description carries the rules its records must follow, so a model that reads only the
tool list can use them. A mistake of the model's is never a protocol error: an argument of the wrong
type, a run that does not exist, a call the run refuses, each comes back as a JSON result
``{"error": {"code", "message"}}``, and a refused record comes back in the tool's ``refused`` list
with what to change, as the tools return it. Arguments are therefore accepted as any JSON value, every
one optional to the protocol, and checked by the tools rather than by the protocol's schema, which
would answer a mistake with a validation error instead; the descriptions say which are required.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from credencegraph_extract import session, tools
from credencegraph_extract.run import RunError
from credencegraph_extract.version import __version__

NAME = "credencegraph-extract"

INSTRUCTIONS = """\
Build the credence graph of a LaTeX paper, one tool call at a time; every call is recorded in the run \
directory, which holds the whole state.

1. start_run: ingest the paper. It lists the chunks with their ids, the document types (each with its \
form fields and what it rests on), the hedge phrases and the check types.
2. classify every chunk; split a chunk that mixes kinds of statement into clauses and type those.
3. add_premises: the premises each conclusion rests on, stated or left unsaid.
4. run_checks for arithmetic and maths steps, and judge_citation for citations, a few at a time; read \
each result before the next.
5. finish_run, and fix what it reports.

Write node ids exactly as listed. Never give a probability: credences come from the rubric. Never \
judge whether a statement is true outside a check. A refused record comes back with a fix: apply it and \
call again. A citation counts only when its source is tied to the reference: the file of the run's \
cited/ copy that the entry's key names (never edit it), the entry's arXiv paper, or where the entry's \
DOI resolves; any other source gives a check that moves no credence.
"""

Run = Annotated[Any, Field(description="The run directory, as a path.")]
Model = Annotated[Any, Field(description="The id of the model making this call, as the harness reports it.")]
Agent = Annotated[Any, Field(description="A name for the agent making the call, when several share the run.")]
Note = Annotated[Any, Field(description="Free text kept with the call (reasoning, if any); never used in inference.")]

server = MCPServer(NAME, instructions=INSTRUCTIONS, version=__version__)


def _error(code: str, message: str) -> dict[str, Any]:
    """A call the run refused, as the JSON result the model reads."""
    return {"error": {"code": code, "message": message}}


def _path(name: str, value: Any) -> Path:
    """A path argument, or refuse the call."""
    if not isinstance(value, str) or not value.strip():
        raise RunError("invalid-argument", f"{name} must be a path")
    return Path(value)


def _run(value: Any) -> Path:
    """The run directory of a call on an existing run, or refuse the call."""
    path = _path("run", value)
    if not path.is_dir():
        raise RunError("missing-run", f"{path.name!r} is not a run directory: start_run makes one")
    return path


def _answer(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Make a tool call; a call the run refuses comes back as a JSON error, never as a protocol error."""
    try:
        return call()
    except RunError as error:
        return _error(error.code, str(error))
    except OSError as error:
        name = Path(error.filename).name if error.filename else "a file"
        return _error("invalid-argument", f"cannot open {name!r}: {error.strerror or error}")


def _optional(name: str, value: Any) -> Path | None:
    """An optional path argument."""
    return None if value is None else _path(name, value)


@server.tool(
    description=(
        "Start a run: check the rubric, ingest the LaTeX paper, copy the cited works into the run, and "
        "return the chunks (id, unit, type, text, label or bibliography key), the document types with "
        "their form fields and what each rests on, the hedge phrases, and the check types with their "
        "targets. `run` must not exist or be empty. `cited` is a directory of the cited works, each a "
        "file named by its bibliography key (.txt, .tex or .pdf); citation checks read only the run's "
        "copy of it. `harness` names the agent harness."
    )
)
def start_run(  # noqa: PLR0913, PLR0917 - the tool's arguments, each a separate choice
    source: Annotated[Any, Field(description="The paper's main .tex file.")] = None,
    rubric: Annotated[Any, Field(description="The rubric file (TOML).")] = None,
    run: Run = None,
    harness: Annotated[Any, Field(description="The agent harness, as it reports itself.")] = None,
    model: Model = None,
    agent: Agent = None,
    note: Note = None,
    document: Annotated[Any, Field(description="An id for the document; the source's file name by default.")] = None,
    cited: Annotated[Any, Field(description="The directory of cited works, named by bibliography key.")] = None,
    store_source: Annotated[Any, Field(description="Keep the paper and fetched sources in the run.")] = False,
) -> dict[str, Any]:
    """Start a run."""

    def call() -> dict[str, Any]:
        if not isinstance(store_source, bool):
            raise RunError("invalid-argument", "store_source must be true or false")
        return session.start_run(
            _path("source", source),
            _path("rubric", rubric),
            _path("run", run),  # start_run makes the run
            harness=harness,
            model=model,
            agent=agent,
            note=note,
            document=document,
            cited=_optional("cited", cited),
            store_source=store_source,
        )

    return _answer(call)


@server.tool(
    description=(
        "Read the run as it stands: every node's id, type and text, with its verdict or check result "
        "where it has one; the relations; every refused record so far, with the call that sent it; "
        "how many calls the run has; and whether it is finished."
    )
)
def get_state(run: Run = None) -> dict[str, Any]:
    """Read the run."""
    return _answer(lambda: session.get_state(_run(run)))


@server.tool(
    description=(
        'Type, split and link chunks. Each record is {"node": <chunk id>, "type": <document type>, '
        '"form": {<that type\'s fields>}, "rests_on": [<ids>], "hedge": <phrase>}, with `form` '
        "only for a type that has fields (every required one given) and `rests_on` only for a type "
        "that rests on something, naming nodes of the types it may rest on; or a split "
        '{"node": <chunk id>, "split": true, "children": [<records without node>]} for a chunk '
        "that mixes kinds of statement. `node` is always a chunk id exactly as start_run listed it, "
        "never a clause id or an id with its type. A value of a form field must be written in the "
        "chunk's text. A node may be typed again until a check targets it; a split chunk is not split "
        "again (type its clauses). Refused records come back with a `fix`: apply it and call again."
    )
)
def classify(
    run: Run = None,
    records: Annotated[Any, Field(description="A list of classify records.")] = None,
    model: Model = None,
    agent: Agent = None,
    note: Note = None,
) -> dict[str, Any]:
    """Type, split and link chunks."""
    return _answer(lambda: tools.classify(_run(run), records, model=model, agent=agent, note=note))


@server.tool(
    description=(
        'Add the premises a conclusion rests on. Each record is {"conclusion": <id of a node of a '
        'type the rubric\'s premises apply to>, "premises": [{"stated_at": <chunk id>} or '
        '{"statement": <a premise the text leaves unsaid, in words>}]}. A stated premise links the '
        "chunk that states it; an unsaid one becomes an analyst assumption. Refused records come back "
        "with a `fix`."
    )
)
def add_premises(
    run: Run = None,
    records: Annotated[Any, Field(description="A list of premise records.")] = None,
    model: Model = None,
    agent: Agent = None,
    note: Note = None,
) -> dict[str, Any]:
    """Add premises."""
    return _answer(lambda: tools.add_premises(_run(run), records, model=model, agent=agent, note=note))


@server.tool(
    description=(
        "Run arithmetic and maths checks on nodes of a type the rubric checks. A proposal is "
        '{"target": <id>, "type": "recompute-numbers", "nodes": [<chunk ids>], "expression": '
        '<arithmetic>, "expected": <the reported value as written>} or {"target": <id>, "type": '
        '"rederive-maths", "nodes": [<equation chunk ids>], "lhs": <expression>, "rhs": '
        '<expression>, "symbols": [{"name", "assumptions": [...], "domain": [low, high]}], '
        '"assumption": <in words>}. Every number in a recompute-numbers expression, the expected '
        "value included, must be written in the cited chunks: no constant they do not write (write "
        "a - a * p, not a * (1 - p)). Checks are final: a check is never withdrawn or run again on its "
        "target, nor run on another target as another encoding of the same step, and a failed check "
        "stays. Citations go to judge_citation. Refused proposals come back with a `fix`."
    )
)
def run_checks(
    run: Run = None,
    proposals: Annotated[Any, Field(description="A list of check proposals.")] = None,
    model: Model = None,
    agent: Agent = None,
    note: Note = None,
) -> dict[str, Any]:
    """Run maths checks."""
    return _answer(lambda: tools.run_checks(_run(run), proposals, model=model, agent=agent, note=note))


@server.tool(
    description=(
        "Record a citation check: whether a passage of the cited work supports the citing node. "
        "`reference` is the bibliography entry's node id (r-N), not its key. `source` is the file of "
        "the run's cited/ copy (its name) or an http(s) URL; the tool reads it itself, so never paste "
        "its text. `passage` is quoted verbatim from the source. The check passes or fails as "
        "`supports` says only when the passage is in the source and the source is tied to the "
        "reference: the cited/ file the entry's key names, unchanged since start_run; the arXiv paper "
        "the entry names (give its arxiv.org URL); or the page the entry's DOI resolves to (give the "
        "doi.org URL). Any other source gives an error check, which moves no credence. A target whose "
        "citation check passed or failed takes no other; after one that errs, another may be made."
    )
)
def judge_citation(  # noqa: PLR0913, PLR0917 - the tool's arguments, each a separate choice
    run: Run = None,
    target: Annotated[Any, Field(description="The citing node's id.")] = None,
    reference: Annotated[Any, Field(description="The bibliography entry's node id, r-N.")] = None,
    source: Annotated[Any, Field(description="A file of the run's cited/ copy, or an http(s) URL.")] = None,
    passage: Annotated[Any, Field(description="The passage of the source, verbatim.")] = None,
    supports: Annotated[Any, Field(description="Whether the passage supports the citing node.")] = None,
    model: Model = None,
    agent: Agent = None,
    note: Note = None,
) -> dict[str, Any]:
    """Record a citation check."""
    return _answer(
        lambda: tools.judge_citation(
            _run(run),
            target,
            reference,
            source,
            passage,
            supports=supports,
            model=model,
            agent=agent,
            note=note,
        )
    )


@server.tool(
    description=(
        "Finish the run: derive every verdict from the checks, run the core's check of the graph, and "
        "write graph.json, rejections.json and report.json. Returns the core's findings; fix them and "
        "finish again. Any later call marks the run unfinished until the next finish_run."
    )
)
def finish_run(run: Run = None, model: Model = None, agent: Agent = None, note: Note = None) -> dict[str, Any]:
    """Finish the run."""
    return _answer(lambda: tools.finish_run(_run(run), model=model, agent=agent, note=note))


def serve() -> None:
    """Serve the tools over stdio until the harness closes the connection."""
    server.run("stdio")
