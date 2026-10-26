# credencegraph-extract

[![Python CI](https://github.com/isaac-cf-wong/credencegraph-extract/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-cf-wong/credencegraph-extract/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isaac-cf-wong/credencegraph-extract/main.svg)](https://results.pre-commit.ci/latest/github/isaac-cf-wong/credencegraph-extract/main)
[![Documentation Status](https://github.com/isaac-cf-wong/credencegraph-extract/actions/workflows/documentation.yml/badge.svg)](https://isaac-cf-wong.github.io/credencegraph-extract/)
[![codecov](https://codecov.io/gh/isaac-cf-wong/credencegraph-extract/graph/badge.svg)](https://codecov.io/gh/isaac-cf-wong/credencegraph-extract)
[![PyPI Version](https://img.shields.io/pypi/v/credencegraph-extract)](https://pypi.org/project/credencegraph-extract/)
[![Python Versions](https://img.shields.io/pypi/pyversions/credencegraph-extract)](https://pypi.org/project/credencegraph-extract/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

**The tools an agent uses to build the
[credencegraph](https://github.com/isaac-cf-wong/credencegraph) graph of a
paper, with every call recorded, replayable and checkable.**

credencegraph turns a document into a graph of verbatim chunks, and an analyst
types, splits, links and assesses them. This package lets that analyst be an
agent in its own harness, such as Claude Code, Codex, OpenCode, Cursor or Pi,
with its own model, web search and file tools. The agent calls the package's
tools; it never writes graph JSON and never sets a credence. Everything that
turns its records into a graph is deterministic code built on credencegraph's
public API, and every call is logged so the graph can be rebuilt and checked.

> **Status: experimental.** An agent ingests a document, classifies its chunks,
> adds the implicit premises of its conclusions, runs checks on the nodes the
> rubric marks for assessment, and finishes the run.

## Installation

```shell
pip install credencegraph-extract
```

It depends on `credencegraph[ingest]` (one 0.x minor series), `mcp` (the server)
and `sympy` (the maths checks), and needs Python 3.13 or later. It calls no
model: the agent's harness does.

## Connecting an agent

`credencegraph-extract mcp` serves the tools over MCP on stdio. Register it in
the harness as a stdio server with the command `credencegraph-extract` and the
argument `mcp`. The server's instructions give the workflow, and each tool's
description carries the rules its records must follow:

1. `start_run` ingests the paper and lists its chunks, the document types (each
   with its form fields and what it rests on), the hedge phrases and the check
   types.
2. `classify` types every chunk, splitting one that mixes kinds of statement.
3. `add_premises` adds the premises each conclusion rests on.
4. `run_checks` and `judge_citation` check arithmetic, maths steps and
   citations, a few at a time.
5. `finish_run` derives the verdicts, checks the graph and writes it.

`get_state` reads the run as it stands. The server keeps no state of its own:
every tool takes the run directory. A mistake of the model's, such as an
argument of the wrong type, an argument the tool does not take, an unknown tool
or a run that does not exist, comes back as a JSON result
`{"error": {"code", "message"}}`, never as a protocol error, and a refused
record comes back with a `fix` saying what to change, for the agent to call
again.

A harness without MCP makes the same calls from the command line:

```shell
credencegraph-extract tool classify --run runs/paper \
  --args '{"records": [{"node": "s-1", "type": "method"}], "model": "<model id>"}'
```

prints the tool's JSON result; a call the run refuses is one
`error [<code>]: <message>` line on stderr, with status 2 (`--args -` reads the
arguments from stdin).

Every call that changes the run takes `model`, the model making it as the
harness reports it, and optionally `agent`, a name for one of several agents
sharing the run, and `note`, free text kept with the call and never used in
inference.

## Runs

A graph is never stored without the run that produced it. A run is a directory:

```text
runs/paper/
  manifest.json   format "credencegraph-extract-run", version 2
  calls.jsonl     every call that changed the run, in the order it took effect
  state.json      the state after the last call
  ingested.json   the graph start_run ingested, the start of every rebuild
  rubric.toml     byte copy of the rubric
  cited/          byte copies of the cited works, made by start_run
  checks/c-NNN/   each check's inputs, output and log
  source/         a copy of the document, only with store_source
  sources/        the bytes of each fetched source, only with store_source
  graph.json      the graph, only while the run is finished
  rejections.json every refused record, only while the run is finished
  report.json     coverage, untied citation checks, and checks made with a
                  model their type was not fitted with, only while finished
```

**The log is the record of truth.** Each call takes the run's lock, so several
agents can share one run, and is committed by one line appended to
`calls.jsonl`: the call, the model and agent that made it, and the digests of
its result and of the graph it left. A call interrupted after its line was
written is completed from the log by the next call, and every other file is
derived from the log. No local path is recorded: files are named by name and
digest. Every node a call creates or changes carries
`attributes.extraction = {"run": "<run id>", "tool": "<tool>"}`.

The source document is **not** copied by default, since it is usually someone
else's copyrighted work; its digest always is. `store_source` copies it, and the
sources fetched for citation checks, for private use. Both the digest and the
copy cover the main `.tex` file only: the files of a document split with
`\input` are not yet recorded.

`credencegraph-extract unlock runs/paper` reports whether a tool call holds the
run's lock, and which process. The lock is the operating system's, released when
its process exits, so a crashed agent never leaves one behind; a hung one holds
it until it is stopped.

## Classify records

`classify` takes one **classify record** per chunk:

```json
{
    "node": "s-3",
    "type": "claim",
    "form": {
        "shape": "existential",
        "statement": "the method may extend to coloured noise"
    },
    "rests_on": ["s-2a", "e-1"],
    "hedge": "suggests"
}
```

or, for a chunk that makes several statements, a split into verbatim clauses
with one typed child each:

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

A record has no field for a credence, and the assembler rejects a record that
carries a field the format does not have.

The **assembler** applies the records in document order through credencegraph's
`split` and `annotate`, each record whole or not at all. Every `base` and
`strength` it writes is therefore the rubric's: a typed node gets its type's
base, and a `rests_on` relation gets the edge rule's strength. A record the core
refuses (an unknown type, a form value not found in the text, a clause that does
not occur exactly once, a `rests_on` node of a type the edge rule does not
allow) is not applied: it comes back in `refused` with the core's error code and
a `fix`, read from the record and the rubric, such as "type 'derivation' rests
on nothing: leave out rests_on" or how to split a chunk when a clause id such as
`s-2a` was written as a chunk. Its node stays as it was, and the refusal is kept
with the call that sent it. A typed node may be classified again, which first
returns it to unassigned, but not once a check targets it, and a split chunk is
not split again.

**Hedges.** `hedge` is a phrase of the node's own text, such as "suggests"; the
record is rejected if the text does not contain it. The rubric's `[hedges]`
table (rubric format 2) maps it to `Node.stated`, compared after text
normalisation and case-folding. The agent locates the phrase and never gives the
number. A phrase the table does not list sets nothing and is returned under
`unmapped_hedges`, which is how the table grows. `stated` is read only by the
overclaim and underclaim diagnostics, never by inference.

**Only `requires` from the text.** Text may add `requires` relations (through
the edge rule) and `part-of` relations (through `split`), nothing else: with
only `requires` parents, a node's credence cannot exceed its type's base,
whatever the text says. A rubric whose edge rules create `supports` relations
would let a document raise the credence of its own claims, so `start_run`
refuses it before anything is written.

## Premise records

`add_premises` takes, for nodes whose type is in the rubric's `premises.for`,
one **premise record** per conclusion: what else must be true for this
conclusion to follow from what it rests on?

```json
{
    "conclusion": "s-3",
    "premises": [
        { "stated_at": "s-1" },
        { "statement": "The noise is stationary over the observation." }
    ]
}
```

- `stated_at` names a chunk that states the premise, linked or not by the
  classify records. It becomes a `requires` relation from that chunk, with the
  edge rule's strength.
- `statement` is a premise the document leaves unsaid. It becomes an **analyst
  node**: origin `analyst`, type `assumption` (which the rubric must declare for
  origin `analyst`), no source, base `premises.base`, joined to the conclusion
  by a `requires` of `premises.strength`.

A conclusion takes at most `premises.max_per_conclusion` analyst premises; the
statements beyond are not added and are returned as `overflow`. Analyst nodes
whose statements are identical after text normalisation are one node with a
`requires` into each of its conclusions, so a shared assumption is a common
parent rather than independent copies. Records are applied whole or not at all,
and refused ones come back with a `fix`. A rubric without a `[premises]` table
takes no premise records.

Analyst nodes are named `a-1`, `a-2`, …, numbered in the order they are created
(conclusions in document order, then each record's premises in the order given),
skipping any id the graph already has. An analyst node has no source and no
inferential parent, so credencegraph's `unanchored` diagnostic reports it: these
are the assumptions a reviewer should look at. As a `requires` premise it can
only lower the credence of its conclusion.

## Checks

The nodes of every type with `assess = true` can be checked. The agent
**proposes** checks; it never says whether a maths check passes. Four check
types are implemented, each run only if the rubric's `[checks]` table declares
it and lists the node's type in its `targets`:

- `recompute-numbers`: an arithmetic expression over numerals of the chunks the
  proposal cites, and the reported result. Every numeral, the result included,
  must be one those chunks' text contains, compared exactly as the decimal it
  writes (never as a float), so the agent cannot supply a number the document
  does not. The expression is parsed with `ast` against a whitelist (numerals,
  `+ - * / **`, signs and parentheses; never `eval`) and evaluated exactly with
  `fractions`. It passes when it falls within the rounding the reported
  numeral's own digits imply: `350` is `350 ± 0.5`.
- `rederive-maths`: a step between equation chunks, encoded as two expressions
  `lhs` and `rhs` over declared symbols with sympy assumptions. The expressions
  are built from the same kind of whitelist, with a fixed set of functions. The
  check passes when sympy simplifies `lhs - rhs` to 0, fails when sympy shows it
  is never 0, and is an `error` otherwise. The encoding is recorded in the check
  record's `inputs` for a human to audit.
- `rederive-maths-numeric`: a separate check type, with its own priors, run only
  when `rederive-maths` errs. It compares the two sides at 32 points drawn in
  each symbol's domain to a relative tolerance of `1e-12`. The points come from
  a generator seeded by the encoding, and the seed and the points are recorded
  with the check, so the same points can be drawn again.
- `citation-support`, through `judge_citation` (below).

`run_checks` runs the first three. A proposal that cannot be run, a second of
its type on one node, or the same step as a check on another target (the same
exact value, expected value and rounding of an arithmetic check, or the same two
sides of a maths step once expanded, however they are written) is refused with a
`fix`: checks that would pass or fail together are one observation, not two.
Checks are append-only: a check is never withdrawn, nor run again on its target,
so a failed check stays observed. Every check finishes in bounded time: an
exponent above 100 or a maths step of degree above 1000 is refused, and a step
too large to simplify symbolically is decided by the numeric fallback.

Each check that runs becomes an **evidence node** `c-NNN` (numbered in the order
run) of its check type, with a statement from a fixed template
(`recompute-numbers on s-3: passed`), and the check record in
`attributes.check`: `type`, `target`, `artifact`, `inputs`, `result`, `detail`,
`tool` (this package, Python and the libraries used), `model` (the model the
call reported, for a citation check, or `null`) and `log` (`checks/c-NNN/` and
its digest). It is joined to its target by `target --observes--> c-NNN` with the
check type's `sensitivity` and `false_pass`. `attributes.observed` is `true` for
a pass and `false` for a fail; an errored check carries none, so it is recorded
but changes no credence. `finish_run` derives each target's
`attributes.assessment` from its checks: `fails`, `holds`, `undetermined` (only
errors), or `not_assessed` with the reason `no v1 check applies` or
`not selected`. The finished graph passes
`credencegraph check GRAPH --rubric RUBRIC --level assessed --rubric-credences`.

**Citation checks.** `judge_citation` takes the citing node, the bibliography
entry's id, a source, the passage the agent found and whether it supports the
node. The tool reads the source itself, never text the agent pastes: a file of
the run's copy of the cited works (named by bibliography key: `<key>.txt`,
`.tex` or `.pdf`), or an `http` or `https` URL, fetched from public addresses
only. It connects to the address it checked, on every redirect hop, so a URL
cannot reach the machine's own network. The check passes or fails, as the agent
judged, only when the passage is in the source and the source is **tied to the
reference**: the cited copy the entry's key names, unchanged since `start_run`;
the arXiv paper the entry names, fetched by the tool at the entry's version or
the current one; or the page the entry's DOI resolves to. Any other source, such
as a page the agent wrote, gives an `error`, which moves no credence, and
`report.json` lists it. A citation check that passed or failed stays; one that
erred may be followed by another. A check type that lists `models` was fitted
with those models; a check made with another is run and listed in `report.json`
under `uncalibrated`.

## Replay

```shell
credencegraph-extract replay runs/paper --source paper.tex
```

re-ingests the document and applies every logged call again, checking each
call's result and graph against the digests its line recorded, and reports
whether the graph is byte-identical to the run's. What `judge_citation` read is
logged with the call (the URLs, the digests of the bytes and of their text, and
where the passage is), not the bytes, so rebuilding a run reads nothing again;
`replay` reads every source again from its bytes and refuses a reading they do
not give. A fetched source the run did not store is passed with `--sources DIR`,
as files named by the hex of their digest. `--source` is not needed when the run
stored its document. Replay needs no network.

## Verify

```shell
credencegraph-extract verify runs/paper runs/other-paper
```

runs every deterministic check of each run again — `recompute-numbers`,
`rederive-maths` and `rederive-maths-numeric` — from the inputs its record in
`graph.json` holds, with the code installed now, and prints each field of a
check record the re-run does not reproduce: the result, the detail, the inputs
(a numeric check draws its points again from its seed), and the digest of the
check's files. It also compares the files in `checks/c-NNN/` with that digest,
and every check record, citation checks included, with the `inputs.json` and
`output.json` its check wrote, so a record edited after the run is reported even
where its files were left alone. A check of a type the runner does not have is
reported too. Recorded inputs are checked again as the run checked the proposal,
so an operand the document's text does not contain makes the re-run an `error`.
Citation checks rest on an agent's judgement and are not run again. `verify`
needs neither the source nor the network, and refuses a run whose `graph.json`
or `rubric.toml` differs from the manifest's digest. It also reads the runs of
format 1 that earlier versions of this package made by calling a model; those
can no longer be replayed.

A rubric's `version` names its values. When the runs given record one rubric
name and version with different file digests, `verify` prints a `warning:` line
naming each digest and its runs: the values changed without a new version.

`replay` and `verify` exit 0 on success; `replay` exits 1 when the rebuilt graph
differs from the recorded one, and `verify` when a check differs from its
record. Every error exits 2 with one `error [<code>]: <message>` line on stderr:
a run directory that is malformed or incomplete, a file that cannot be read or
written, or a source or record that does not match the run.

## Development

```shell
uv sync --group dev
uv run prek install       # run the pre-commit hooks on every commit
uv run pytest
uv run prek run --all-files
```

The tests are offline: no network and no model. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).
