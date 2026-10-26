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

**An automated analyst for
[credencegraph](https://github.com/isaac-cf-wong/credencegraph), with every
model call recorded and replayable.**

credencegraph turns a document into a graph of verbatim chunks, and an analyst
types, splits, links and assesses them. This package is the analyst when that
analyst is a language model. The model never writes graph JSON and never sets a
credence: it returns records, and everything that turns those records into a
graph is deterministic code built on credencegraph's public API.

> **Status: experimental.** A run ingests the document, classifies its chunks,
> adds the implicit premises of its conclusions, and runs checks on the nodes
> the rubric marks for assessment.

## Installation

```shell
pip install credencegraph-extract
```

It depends on `credencegraph[ingest]` (one 0.x minor series), `anthropic` and
`sympy` (the maths checks), and needs Python 3.13 or later.

## Runs

LLM output cannot be regenerated, so a graph is never stored without the run
that produced it:

```shell
credencegraph-extract run paper.tex runs/paper --rubric rubric.toml --model <exact dated model id>
```

writes a **run directory**:

```text
runs/paper/
  manifest.json   format "credencegraph-extract-run", version 1
  graph.json      the graph the run produced
  rubric.toml     byte copy of the rubric used
  prompts/        each pass's prompt template as shipped, and the rendered prompt of every call
  raw/            every provider response verbatim, one file per call, with its request metadata
  rejections.json the records the core refused, each with its error code
  report.json     the graph's coverage, and what each pass reports
  checks/c-NNN/   each check's inputs, the tool's output and a log
  source/         a copy of the source document, only with --store-source
```

`manifest.json` records the run id and time; the source document's id, its
SHA-256 digest and whether a copy is stored; the versions of Python,
`credencegraph`, this package, the provider SDK and `sympy`; the rubric's name,
version and digest; for each pass, the digest of its template and, for every
call, the provider, the exact model id, the sampling parameters, the SDK version
and the digests of the rendered prompt and the raw record; and the digest of
`graph.json`.

Every node a run creates or changes carries
`attributes.extraction = {"run": "<run id>", "pass": "<pass>"}`, so a graph that
leaves its run directory still names its run. Ingested nodes name the pass
`ingest`.

The source document is **not** copied by default, since it is usually someone
else's copyrighted work; its digest always is. `--store-source` copies it, for
private use. Both the digest and the copy cover the main `.tex` file only: the
files of a document split with `\input` are not yet recorded.

## Classify pass

The classify pass sends the document's chunks and the rubric's types to the
model in one prompt and gets back one **classify record** per chunk:

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

A record has no field for a credence: the output schema, built from the rubric,
has none, and the assembler rejects a record that carries a field the format
does not have. The schema has one branch per type, so a server that enforces it
keeps each record to what its type allows: a `form` with only that type's
fields, its required ones required, and `rests_on` only on a type with an edge
rule. A run replays only with the schema it was recorded with.

The **assembler** applies the records in document order through credencegraph's
`split` and `annotate`, each record whole or not at all. Every `base` and
`strength` it writes is therefore the rubric's: a typed node gets its type's
base, and a `rests_on` relation gets the edge rule's strength. A record the core
refuses (an unknown type, a form value not found in the text, a clause that does
not occur exactly once, a `rests_on` node of a type the edge rule does not
allow) is not applied and not dropped: it goes to `rejections.json` with the
core's error code, and its node stays as it was. The refused records and their
errors go back to the model once, each with a `fix`: what to change, read from
the record and the rubric, such as "type 'derivation' rests on nothing: leave
out rests_on" or how to split a chunk when the model wrote a clause id such as
`s-2a` as a chunk. The prompt also states each type's form fields and what it
rests on ("rests on nothing" where it has no edge rule), the JSON shape of a
record, and that `node` is always a listed chunk id, so a model that is never
shown the output schema still has the rules. That repair answer is recorded in
`raw/` like any other call, and what it does not fix stays rejected. Each entry
of `rejections.json` names its round, `initial` or `repair`, so a node is still
rejected exactly when it has a `repair` entry; it shows as untyped in the
report's coverage.

**Hedges.** `hedge` is a phrase of the node's own text, such as "suggests"; the
record is rejected if the text does not contain it. The rubric's `[hedges]`
table (rubric format 2) maps it to `Node.stated`, compared after text
normalisation and case-folding. The model locates the phrase and never gives the
number. A phrase the table does not list sets nothing and is listed in
`report.json` under `passes.classify.hedge-unmapped`, which is how the table
grows. `stated` is read only by the overclaim and underclaim diagnostics, never
by inference.

**Only `requires` from the text.** Text may add `requires` relations (through
the edge rule) and `part-of` relations (through `split`), nothing else: with
only `requires` parents, a node's credence cannot exceed its type's base,
whatever the text says. A rubric whose edge rules create `supports` relations
would let a document raise the credence of its own claims, so a run refuses it
before any model call. An assembled graph can be checked independently with

```shell
credencegraph check runs/paper/graph.json --rubric rubric.toml --level typed --rubric-credences
```

## Premise pass

After classification, the premise pass asks, for every node whose type is in the
rubric's `premises.for`: what else must be true for this conclusion to follow
from what it rests on? It runs in a fresh prompt that holds each conclusion's
text, the texts of the chunks it already rests on, and the id and text of every
chunk of the document. It is given nothing else of the classification (no types,
forms, hedges or reasoning), so its mistakes are not the classify pass's. It
answers with one **premise record** per conclusion:

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
  classify pass. It becomes a `requires` relation from that chunk, with the edge
  rule's strength.
- `statement` is a premise the document leaves unsaid. It becomes an **analyst
  node**: origin `analyst`, type `assumption` (which the rubric must declare for
  origin `analyst`), no source, base `premises.base`, joined to the conclusion
  by a `requires` of `premises.strength`.

A conclusion takes at most `premises.max_per_conclusion` analyst premises; the
statements beyond are not added and are listed in `report.json` under
`passes.premises.overflow`. Analyst nodes whose statements are identical after
text normalisation are one node with a `requires` into each of its conclusions,
so a shared assumption is a common parent rather than independent copies.
Records are applied whole or not at all, refused ones go to `rejections.json`,
and the model gets one repair round, as in the classify pass. A rubric without a
`[premises]` table runs no premise call.

**Analyst node ids.** Analyst nodes are named `a-1`, `a-2`, …: the number counts
from 1 in the order the nodes are created (conclusions in document order, then
each record's premises in the order given; the repair round continues the
count), skipping any id the graph already has. It is not zero-padded, since it
is assigned before the total is known. `replay` applies the same records in the
same order, so it gives the same ids.

An analyst node has no source and no inferential parent, so credencegraph's
`unanchored` diagnostic reports it: these are the assumptions a reviewer should
look at. As a `requires` premise it can only lower the credence of its
conclusion.

## Check pass

The last pass checks the nodes of every type with `assess = true`. It sends
those a rubric check type may run on, with the document's chunks, and the model
**proposes** checks; it never says whether one passes. Four check types are
implemented, each run only if the rubric's `[checks]` table declares it and
lists the node's type in its `targets`:

- `recompute-numbers`: an arithmetic expression over numerals of the chunks the
  proposal cites, and the reported result. Every numeral, the result included,
  must be one those chunks' text contains, compared exactly as the decimal it
  writes (never as a float), so the model cannot supply a number the document
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
- `citation-support`: the cited work's text is read from `--cited DIR`, named by
  its bibliography key (`<key>.txt`, `.tex` or `.pdf`), and with
  `--fetch-arxiv`, downloaded from arXiv into that directory when the entry
  names an identifier. The model copies the passage the judgement rests on and
  says whether it supports the citing node. A passage not found in the work's
  normalised text is an `error`, not a fail. A check type that lists `models`
  was fitted with those models: the pass refuses to run it with another unless
  `--allow-uncalibrated` is given, and then `report.json` lists each such check
  under `passes.checks.uncalibrated`.

A proposal that cannot be run, a second of its type on one node, or the same
step as another proposal on another target (the same expected value and exact
value of the expression, or the same two sides of a maths step however they are
written) is rejected with a code in `rejections.json`: checks that would pass or
fail together are one observation, not two. The refused proposals go back to the
model once, each with its error and a `fix` (for an unfaithful numeral: cite
every chunk whose numeral is used, and write the expression with the text's own
numerals only, as `a - a * p` rather than `a * (1 - p)`; for a citation naming a
bibliography key, the id of its entry); the corrections are checked the same
way, and each rejection names its round, `initial` or `repair`. Each check that
runs becomes an **evidence node** `c-NNN` (in document order of its target, then
run order) of its check type, with a statement from a fixed template
(`recompute-numbers on s-3: passed`), and the check record in
`attributes.check`: `type`, `target`, `artifact`, `inputs`, `result`, `detail`,
`tool` (this package, Python and the libraries used), `model` (provider, model
id and request id, or `null`) and `log` (`checks/c-NNN/` and its digest). It is
joined to its target by `target --observes--> c-NNN` with the check type's
`sensitivity` and `false_pass`. `attributes.observed` is `true` for a pass and
`false` for a fail; an errored check carries none, so it is recorded but changes
no credence. The target's `attributes.assessment` is derived from its checks:
`fails`, `holds`, `undetermined` (only errors), or `not_assessed` with the
reason `no v1 check applies` or `not selected`. The result passes
`credencegraph check GRAPH --rubric RUBRIC --level assessed --rubric-credences`.

## Replay

The model calls are recorded, not reproduced. Everything after them is
deterministic, so

```shell
credencegraph-extract replay runs/paper --source paper.tex
```

re-runs ingestion and every pass, answering each model call from `raw/` instead
of the provider, and checks that the rebuilt `graph.json` is byte-identical to
the recorded one. It needs no network and no API key. `--source` is not needed
when the run stored its source; a run with citation checks needs the same
`--cited` directory, and a replay re-runs the checks a run allowed, uncalibrated
ones included. Replay refuses a source, rubric or record whose digest differs
from the manifest's, and a pass that asks a call that was not recorded.

## Agent-tool runs

The package is becoming a set of tools an agent calls from its own harness,
instead of a program that calls a model. A run made that way is a directory of
format 2 (`credencegraph_extract.session`): `start_run` checks the rubric,
ingests the document and copies the cited works into the run, each with its
digest; every later tool call that changes the run takes the run's lock, so
several agents can share one run, and is committed by one line appended to
`calls.jsonl`. The log is the record of truth: a call interrupted after its line
was written is completed from the log by the next call, and

```shell
credencegraph-extract replay runs/paper --source paper.tex
```

re-ingests the document and applies every logged call again, checking each
call's result and graph against the digests its line recorded. No local path is
recorded: files are named by name and digest.

The graph tools (`credencegraph_extract.tools`) are `classify`, `add_premises`
and `finish_run`. Records go through the same assembler as the passes, so every
base and strength is the rubric's, and a refused record comes back with a `fix`
for the agent to act on, as often as it needs. A typed node may be classified
again, which first returns it to unassigned, but not once a check targets it,
and a split chunk is not split again. `finish_run` derives every verdict from
the checks, runs the core's check at the `assessed` level with rubric credences,
and writes `graph.json`, `rejections.json` and `report.json`; any later write
removes them until the next `finish_run`. `verify` takes a finished run of
either format.

`run_checks` runs proposals of the maths check types (`recompute-numbers`,
`rederive-maths` and its numeric fallback), checked as the check pass checks
them, and adds a check node and its files under `checks/` for each. Checks are
append-only: a check is never withdrawn, nor run again on its target, nor run on
another target as another encoding of the same step, so a failed check stays
observed. The versions of the code that ran a call's checks are logged with it.

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
Citation checks are judged by a model and are not run again; `replay` answers
them from their recorded responses. `verify` needs neither the source nor the
network, and refuses a run whose `graph.json` or `rubric.toml` differs from the
manifest's digest.

A rubric's `version` names its values. When the runs given record one rubric
name and version with different file digests, `verify` prints a `warning:` line
naming each digest and its runs: the values changed without a new version.

All three commands exit 0 on success; `replay` exits 1 when the rebuilt graph
differs from the recorded one, and `verify` when a check differs from its
record. Every error exits 2 with one `error [<code>]: <message>` line on stderr:
a run directory that is malformed or incomplete (an unreadable or invalid
`manifest.json`, a missing `graph.json`, `rubric.toml` or `raw/` record), a file
that cannot be read or written, a source or record that does not match the
manifest, and a provider error.

## Development

```shell
uv sync --group dev
uv run prek install       # run the pre-commit hooks on every commit
uv run pytest
uv run prek run --all-files
```

The tests run on recorded responses only: no network and no live model calls.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).
