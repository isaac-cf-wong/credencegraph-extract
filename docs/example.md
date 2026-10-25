---
title: End-to-end example
description:
    A live run on an invented paper, from ingestion through the classify,
    premise and check passes to conditioning on the checks' observations.
---

`examples/end-to-end/` holds an invented paper, its rubric, the one work it
cites, and the run directory of a live extraction made with a local model. The
run is committed whole, so everything below can be repeated offline from it.

The paper, the study it describes, its numbers and its reference are invented.
One number is wrong on purpose: removing 5% of 1036800 readings leaves 984960,
and the paper says 984460.

## The run

The run used `Ornith-1.5-35B-A3B-MLX-4bit`, a small mixture-of-experts model
served locally through an OpenAI-compatible API, with its reasoning capped at
16384 tokens:

```shell
cd examples/end-to-end
credencegraph-extract run paper.tex run --rubric rubric.toml \
    --provider openai --base-url <server>/v1 \
    --model Ornith-1.5-35B-A3B-MLX-4bit \
    --request-options '{"thinking_budget": 16384}' --max-tokens 32768 \
    --cited cited --allow-uncalibrated
```

`--allow-uncalibrated` is needed because the rubric's `citation-support` check
lists only an invented model id: its values are illustrative and were fitted
with no model. The run made four calls, about 14 minutes in all.

The endpoint each call went to is recorded with the call. The committed run
replaces the address of the machine it ran on with
`http://local-openai-compatible-server.invalid/v1/`, in `raw/` and in
`manifest.json`, with the records' digests recomputed; replay does not read the
endpoint, and `graph.json` does not contain it.

## What each pass did

**Classify** typed all twelve chunks in one call, with no record refused. It
split `s-7` into the measured drift rate (`s-7a`, a result) and the drift over
30 days (`s-7b`, a derivation), and made the conclusion `s-9` rest on `s-7a`,
`s-7b` and Lee's finding `s-8`, with the hedge "suggest".

**Premises** added one implicit premise of the conclusion, the analyst node
`a-1`: "The array needs recalibration once its drift reaches 0.6 K."

**Checks.** The plan proposed nine checks. One, the recomputation of `s-4`, was
admitted; eight were refused: numerals taken from a chunk the proposal did not
cite, a step of mathematics citing sentences rather than equations, a function
`T(...)` the expression language does not have, and a citation naming the
bibliography key `lee2019` instead of the node `r-1`. They went back to the
model once, each with its error and what to change. Five corrections were
admitted; three were refused again and are in `rejections.json` with round
`repair`. Six checks ran:

| Check                     | Target | Inputs                                                    | Result              |
| ------------------------- | ------ | --------------------------------------------------------- | ------------------- |
| `c-001` recompute-numbers | `s-2`  | `1440 * 30` = 43200                                       | passed              |
| `c-002` recompute-numbers | `s-3`  | `24 * 43200` = 1036800                                    | passed              |
| `c-003` recompute-numbers | `s-4`  | `1036800 - 1036800 * 0.05` = 984460                       | **failed** (984960) |
| `c-004` rederive-maths    | `e-1`  | `(T_0 + alpha*(t + tau)) - (T_0 + alpha*t)` = `alpha*tau` | passed              |
| `c-005` rederive-maths    | `s-6`  | the same encoding                                         | passed              |
| `c-006` rederive-maths    | `e-2`  | the same encoding                                         | passed              |

Two things the example shows a reader should weigh. The three maths checks are
one encoding, proposed for three targets: one piece of evidence observed on
three nodes. And no check reached `s-9` or its supports: `s-7a`, `s-7b` and
`s-8` are `not_assessed` ("not selected"), so the conclusion's credence is not
moved by any observation.

## Checking the run offline

```text
$ credencegraph-extract replay run --source paper.tex --cited cited
replayed run: graph.json is byte-identical (sha256:6a32faa9d4bb36b3d3af3036f9db21f698c6538b0d52eac97a156a900af8a7d4)
$ credencegraph-extract verify run
verified run: 6 checks run again, 0 differ; 0 model-judged checks not run again
$ credencegraph check run/graph.json --rubric rubric.toml --level assessed --rubric-credences
run/graph.json: ok at level assessed with rubric credences against rubric end-to-end-example 1
```

## Conditioning on the checks

```text
$ credencegraph query run/graph.json marginal s-4 --draws 0
P(s-4=true) = 0.8
$ credencegraph query run/graph.json marginal s-4 --observed --draws 0
P(s-4=true | c-001=true, c-002=true, c-003=false, c-004=true, c-005=true, c-006=true) = 0.181818
```

The failed check takes the wrong number from its type's base 0.8 to 0.18: with
the rubric's mean sensitivity 0.95 and false-pass rate 0.1, 0.8 × 0.05 / (0.8 ×
0.05 + 0.2 × 0.9) = 0.04 / 0.22. The conclusion `s-9` stays at 0.183565 with or
without `--observed`, since no check bears on it.
