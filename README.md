# credencegraph-extract

**An automated analyst for [credencegraph](https://github.com/isaac-cf-wong/credencegraph), with every
model call recorded and replayable.**

credencegraph turns a document into a graph of verbatim chunks, and an analyst types, splits, links
and assesses them. This package is the analyst when that analyst is a language model. The model
never writes graph JSON and never sets a credence: it returns records, and everything that turns
those records into a graph is deterministic code built on credencegraph's public API.

> **Status: experimental.** The package is at its scaffold stage: it provides the provider call,
> the run directory and replay. The passes that classify chunks, add implicit premises and run
> checks are not built yet, so a run today ingests the document and records it.

## Installation

```shell
pip install credencegraph-extract
```

It depends on `credencegraph[ingest]` (one 0.x minor series) and `anthropic`, and needs Python 3.13
or later.

## Runs

LLM output cannot be regenerated, so a graph is never stored without the run that produced it:

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
  source/         a copy of the source document, only with --store-source
```

`manifest.json` records the run id and time; the source document's id, its SHA-256 digest and
whether a copy is stored; the versions of Python, `credencegraph`, this package and the provider
SDK; the rubric's name, version and digest; for each pass, the digest of its template and, for
every call, the provider, the exact model id, the sampling parameters, the SDK version and the
digests of the rendered prompt and the raw record; and the digest of `graph.json`.

Every node a run creates or changes carries `attributes.extraction = {"run": "<run id>", "pass":
"<pass>"}`, so a graph that leaves its run directory still names its run. Ingested nodes name the
pass `ingest`.

The source document is **not** copied by default, since it is usually someone else's copyrighted
work; its digest always is. `--store-source` copies it, for private use.
Both the digest and the copy cover the main `.tex` file only: the files of a document split with
`\input` are not yet recorded.

## Replay

The model calls are recorded, not reproduced. Everything after them is deterministic, so

```shell
credencegraph-extract replay runs/paper --source paper.tex
```

re-runs ingestion and every pass, answering each model call from `raw/` instead of the provider,
and checks that the rebuilt `graph.json` is byte-identical to the recorded one. It needs no network
and no API key. `--source` is not needed when the run stored its source. Replay refuses a source,
rubric or record whose digest differs from the manifest's, and a pass that asks a call that was not
recorded.

Both commands exit 0 on success, and `replay` exits 1 when the rebuilt graph differs from the
recorded one. Every error exits 2 with one `error [<code>]: <message>` line on stderr: a run
directory that is malformed or incomplete (an unreadable or invalid `manifest.json`, a missing
`graph.json`, `rubric.toml` or `raw/` record), a file that cannot be read or written, a source or
record that does not match the manifest, and a provider error.

## Development

```shell
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

The tests run on recorded responses only: no network and no live model calls.

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).
