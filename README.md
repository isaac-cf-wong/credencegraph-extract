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

> **Status: experimental.** A run ingests the document and classifies its
> chunks. The passes that add implicit premises and run checks are not built
> yet.

## Installation

```shell
pip install credencegraph-extract
```

It depends on `credencegraph[ingest]` (one 0.x minor series) and `anthropic`,
and needs Python 3.13 or later.

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
  source/         a copy of the source document, only with --store-source
```

`manifest.json` records the run id and time; the source document's id, its
SHA-256 digest and whether a copy is stored; the versions of Python,
`credencegraph`, this package and the provider SDK; the rubric's name, version
and digest; for each pass, the digest of its template and, for every call, the
provider, the exact model id, the sampling parameters, the SDK version and the
digests of the rendered prompt and the raw record; and the digest of
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
does not have.

The **assembler** applies the records in document order through credencegraph's
`split` and `annotate`, each record whole or not at all. Every `base` and
`strength` it writes is therefore the rubric's: a typed node gets its type's
base, and a `rests_on` relation gets the edge rule's strength. A record the core
refuses (an unknown type, a form value not found in the text, a clause that does
not occur exactly once, a `rests_on` node of a type the edge rule does not
allow) is not applied and not dropped: it goes to `rejections.json` with the
core's error code, and its node stays as it was. The refused records and their
errors go back to the model once; that repair answer is recorded in `raw/` like
any other call, and what it does not fix stays rejected. Each entry of
`rejections.json` names its round, `initial` or `repair`, so a node is still
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

## Replay

The model calls are recorded, not reproduced. Everything after them is
deterministic, so

```shell
credencegraph-extract replay runs/paper --source paper.tex
```

re-runs ingestion and every pass, answering each model call from `raw/` instead
of the provider, and checks that the rebuilt `graph.json` is byte-identical to
the recorded one. It needs no network and no API key. `--source` is not needed
when the run stored its source. Replay refuses a source, rubric or record whose
digest differs from the manifest's, and a pass that asks a call that was not
recorded.

Both commands exit 0 on success, and `replay` exits 1 when the rebuilt graph
differs from the recorded one. Every error exits 2 with one
`error [<code>]: <message>` line on stderr: a run directory that is malformed or
incomplete (an unreadable or invalid `manifest.json`, a missing `graph.json`,
`rubric.toml` or `raw/` record), a file that cannot be read or written, a source
or record that does not match the manifest, and a provider error.

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
