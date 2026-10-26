---
name: credencegraph-extract
description:
    Build the credencegraph graph of a LaTeX paper with the
    credencegraph-extract tools, which type its chunks, add the premises of its
    conclusions, check its arithmetic, maths and citations, and finish the run.
    Use when asked to extract, assess or check the claims of a paper into a
    credence graph, or when the credencegraph-extract MCP tools are available.
version: 0.1.0
license: BSD-3-Clause
---

# credencegraph-extract

You are the analyst. The tools turn your records into a graph; you never write
graph JSON and never give a probability. Every call that changes the run is
recorded in the run directory, refused records included, so a reader can see
what you did, rebuild the graph and check it. A read (`get_state`) and a call
refused as a whole, such as one with an argument of the wrong type, change
nothing and are not recorded.

## Setup

The tools are an MCP server started by your harness: the command
`credencegraph-extract`, the argument `mcp`, over stdio. Install the package
(`pip install credencegraph-extract`, or
`uv tool install credencegraph-extract`) and register the server once:

| Harness     | How                                                                 |
| ----------- | ------------------------------------------------------------------- |
| Claude Code | `claude mcp add credencegraph-extract -- credencegraph-extract mcp` |
| Codex       | `codex mcp add credencegraph-extract -- credencegraph-extract mcp`  |
| Pi          | `pi mcp add credencegraph-extract -- credencegraph-extract mcp`     |
| OpenCode    | the `opencode.json` entry below                                     |
| Cursor      | the `.cursor/mcp.json` entry below                                  |

OpenCode, in the project's `opencode.json`:

```json
{
    "$schema": "https://opencode.ai/config.json",
    "mcp": {
        "credencegraph-extract": {
            "type": "local",
            "command": ["credencegraph-extract", "mcp"],
            "enabled": true
        }
    }
}
```

Cursor, in the project's `.cursor/mcp.json` (or `~/.cursor/mcp.json`):

```json
{
    "mcpServers": {
        "credencegraph-extract": {
            "command": "credencegraph-extract",
            "args": ["mcp"]
        }
    }
}
```

In any other harness, make the same calls from a shell, one per command; the
result is JSON on stdout, and a refused call is one `error [<code>]: <message>`
line on stderr with status 2:

```shell
credencegraph-extract tool start_run --run runs/paper \
  --args '{"source": "paper.tex", "rubric": "rubric.toml", "harness": "<harness>", "model": "<model id>", "cited": "cited"}'
credencegraph-extract tool classify --run runs/paper --args - < records.json
```

## The workflow

1. **`start_run`** with the paper's main `.tex` file, the rubric, a new run
   directory, the harness's name and your model's id, and the directory of the
   cited works if you have them (files named by bibliography key: `<key>.txt`,
   `.tex` or `.pdf`). Read what it returns: the chunks with their ids, the
   document types with their form fields and what each rests on, the hedge
   phrases, and the check types with the types they target.
2. **`classify`** every chunk. Split a chunk that makes several kinds of
   statement into its clauses and type each clause (the shape is below). Send
   records in batches; the result lists the nodes `applied`, the records
   `refused` and the `unmapped_hedges`, hedge phrases the rubric gives no
   credence for. Send corrected records for what was refused.
3. **`add_premises`** for each conclusion of a type the rubric's premises apply
   to: the chunks that state what it rests on, and the premises the text leaves
   unsaid. A chunk a `rests_on` of yours already names is linked; do not send it
   again as `stated_at`. Read `overflow`: the unsaid premises not added because
   the conclusion already has as many as the rubric allows. Keep the ones that
   matter most rather than sending more.
4. **`run_checks`** for arithmetic (`recompute-numbers`) and maths steps
   (`rederive-maths`), and **`judge_citation`** for citations. A few at a time:
   read each result before the next.
5. **`finish_run`**, and fix what it reports. `get_state` shows the run as it
   stands at any point.

Pass `model` (your model's id, as your harness reports it) on every call that
changes the run, and `agent` when several agents share one run. Put any
reasoning you want kept in `note`; it is recorded and never used in inference.

## Rules

- **Ids exactly as `start_run` listed them**: `s-3`, never `s-3 (result)`. A
  record's `node` is a chunk id, or the id of a clause a split of yours has
  created (`s-2a` after you split `s-2`); a clause id before its split names
  nothing. A bibliography entry is named by its node id (`r-1`), not its key.
- **A split copies its clauses verbatim** from the chunk, each occurring once in
  it, with one child record per clause; the clauses become `s-2a`, `s-2b`, ...
  in order, and a child may rest on an earlier one:

    ```json
    {
        "node": "s-2",
        "split": [
            "The bias vanishes for long segments",
            "the estimator is therefore consistent"
        ],
        "children": [
            {
                "type": "result",
                "form": { "quantity": "The bias", "trend": "vanishes" }
            },
            {
                "type": "claim",
                "form": {
                    "shape": "universal",
                    "statement": "the estimator is therefore consistent"
                },
                "rests_on": ["s-2a"]
            }
        ]
    }
    ```

- **A record has only what its type allows**: `form` with that type's fields and
  its required ones; `rests_on` only on a type that rests on something, and only
  naming nodes of the types it may rest on. A value of a form field is copied
  from the chunk's text.
- **A hedge is a phrase of the chunk's own text**, such as "suggests".
- **Numbers come from the text**: every numeral of a `recompute-numbers`
  expression, the expected value included, is written in the chunks it cites. No
  constant the text does not write: `a - a * p`, not `a * (1 - p)`.
- **A maths step** is two sides, `lhs` and `rhs`, over declared symbols with
  their assumptions and a sampling domain; it passes when sympy shows them equal
  and fails when it shows them different. When sympy cannot decide, the check is
  an `error`, and if the rubric has `rederive-maths-numeric` the tool also
  compares the sides at sampled points of the domain.
- **Checks are final.** A maths or arithmetic check is never withdrawn or run
  again on its target, and one step is one check however it is written; a check
  that passed or failed stays. Pick the check that tests the claim, not the one
  likely to pass. (A citation check that erred may be tried again; see below.)
- **A typed node may be typed again until a check targets it**; after that it
  stays.
- **A refused record comes back with a `fix`.** Apply it and call again; do not
  resend the same record.

## Citations

`judge_citation` takes the citing node, the entry's node id, a source, the
passage you quote verbatim from it, and whether the passage supports the node.
The tool reads the source itself:

- give the name of the entry's file in the run's `cited/` copy, the entry's
  arXiv URL, or the entry's DOI URL (`https://doi.org/<doi>`): these are **tied
  to the reference**, and only these make the check count;
- any other URL is read and recorded, but its check is an `error` that moves no
  credence. Use web search to _find_ the work, then give its tied source.

A citation check that passed or failed stays; after one that erred, you may try
again.

## Never

- give a probability, a credence or a verdict: credences come from the rubric,
  verdicts from the checks;
- judge whether a statement is true outside a check;
- paste a source's text into `judge_citation`: give its file name or URL;
- edit, add or remove anything in the run directory, least of all
  `<run>/cited/`: a changed cited copy is no longer tied to its reference, and
  every file is derived from the call log;
- start a second run in the same directory.

## Finishing

`finish_run` derives every verdict from the checks and runs credencegraph's own
check of the graph. Fix what it reports (an untyped chunk, a missing form field)
and finish again; any later call marks the run unfinished until you do. The run
is done when `finish_run` returns `"ok": true`. Then
`credencegraph-extract verify <run>` runs the maths checks again, and
`credencegraph-extract replay <run> --source paper.tex` rebuilds the graph from
the log.
