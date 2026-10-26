---
title: End-to-end example
description:
    A run an agent made through a harness on the tools, on an invented paper,
    from start_run to finish_run, checked offline and conditioned on its checks.
---

`examples/end-to-end/` holds an invented paper, its rubric, the one work it
cites, and the run directory an agent made of it through the tools. The run is
committed whole, so everything below can be repeated offline from it.

The paper, the study it describes, its numbers and its reference are invented.
One number is wrong on purpose: removing 5% of 1036800 readings leaves 984960,
and the paper says 984460.

## The run

The agent was Claude Code (`claude-opus-5-5`), in a project holding the
example's paper, rubric and cited work, with the server registered and the skill
installed there, given this task:

```text
Use the credencegraph-extract skill to build the credence graph of paper.tex with rubric.toml.
Make the run in the directory `run`, passing the paths relative as written here: source `paper.tex`,
rubric `rubric.toml`, run `run`, cited `cited`, harness `claude-code`.
Classify every chunk, add the premises of the conclusions, propose every check the rubric allows
for the nodes it targets (arithmetic, maths steps and citations), and finish the run until
finish_run reports ok. Do not edit anything inside run/. Report the run directory, the checks
and their results, and what finish_run returned.
```

It made ten calls, each recorded in `run/calls.jsonl` with its arguments and the
digest of its result. The paths are the relative ones it was given, so the run
names no directory of the machine it ran on.

The rubric's `citation-support` check lists only an invented model id: its
values are illustrative and were fitted with no model, so `report.json` lists
the judge as uncalibrated.

## What each call did

**`start_run`** listed twelve chunks, the rubric's types and hedges, and four
check types, and copied the cited work into `run/cited/`.

**`classify`** typed every chunk in one call, with no record refused: the counts
`s-2`, `s-3` and `s-4` and the drift `s-7` as results, the equation `e-2` as a
derivation, Lee's finding `s-8` as a literature claim, and the conclusion `s-9`
as a claim with the hedge "suggest", resting on `s-7` and `s-8`.

**`add_premises`** linked the derivation `e-2` as a stated premise of `s-9` and
added two the text leaves unsaid, the analyst nodes `a-1` ("A drift of 0.6
kelvin over a month exceeds the accuracy the array is required to keep.") and
`a-2` ("The drift continues linearly beyond the 30 days observed, …").

**`run_checks`** and **`judge_citation`** ran six checks. Two proposals for
`s-4` were refused first and are kept in `rejections.json`: `5%` is not an
expression, and `5 / 100` uses a 100 the text does not write. The third,
`1036800 - 1036800 * 0.05`, was admitted.

| Check                     | Target | Inputs                                                    | Result              |
| ------------------------- | ------ | --------------------------------------------------------- | ------------------- |
| `c-001` recompute-numbers | `s-2`  | `1440 * 30` = 43200                                       | passed              |
| `c-002` recompute-numbers | `s-3`  | `43200 * 24` = 1036800                                    | passed              |
| `c-003` recompute-numbers | `s-7`  | `0.02 * 30` = 0.6                                         | passed              |
| `c-004` recompute-numbers | `s-4`  | `1036800 - 1036800 * 0.05` = 984460                       | **failed** (984960) |
| `c-005` rederive-maths    | `e-2`  | `(T_0 + alpha*(t + tau)) - (T_0 + alpha*t)` = `alpha*tau` | passed              |
| `c-006` citation-support  | `s-8`  | `lee2019.txt`, the passage below, supports                | passed              |

The citation check quotes "Over a month, drifts between 0.3 K and 1 K were
typical of uncalibrated thermistors". It counts because its source is the
entry's file among the cited works, tied to `r-1`; the tool found the passage in
it and anchored the check to it.

**`finish_run`** returned `{"ok": true, "findings": []}` on its first call.

## Checking the run offline

```text
$ cd examples/end-to-end
$ credencegraph-extract replay run --source paper.tex
replayed run: the graph after 10 calls is byte-identical (sha256:dd3160b6b6e727b61a67b53ba1cf442aea8c612b9fb7c56530fb6f22360786ba)
$ credencegraph-extract verify run
verified run: 5 checks run again, 0 differ; 1 model-judged checks not run again
$ credencegraph check run/graph.json --rubric rubric.toml --level assessed --rubric-credences
run/graph.json: ok at level assessed with rubric credences against rubric end-to-end-example 1
```

`verify` does not run the citation check again: its verdict was the model's.

## Conditioning on the checks

```text
$ credencegraph query run/graph.json marginal s-4 --draws 0
P(s-4=true) = 0.8
$ credencegraph query run/graph.json marginal s-4 --observed --draws 0
P(s-4=true | c-001=true, c-002=true, c-003=true, c-004=false, c-005=true, c-006=true) = 0.181818
```

The failed check takes the wrong number from its type's base 0.8 to 0.18: with
the rubric's mean sensitivity 0.95 and false-pass rate 0.1, 0.8 × 0.05 / (0.8 ×
0.05 + 0.2 × 0.9) = 0.04 / 0.22.

`diagnose --observed` takes the same observations as evidence, and names the
failed check's sensitivity as the crux of `s-4`, ahead of its base:

```text
$ credencegraph diagnose run/graph.json --target s-4 --observed
...
crux:sensitivity:s-4-observes-c-004: crux 0.141 for the sensitivity of observes relation 's-4-observes-c-004': P('s-4' | c-001=true, c-002=true, c-003=true, c-004=false, c-005=true, c-006=true) moves 2.98 per unit of it and its standard deviation is 0.0476
crux:base:s-4: crux 0.112 for the base of 's-4': P('s-4' | c-001=true, c-002=true, c-003=true, c-004=false, c-005=true, c-006=true) moves 0.93 per unit of it and its standard deviation is 0.121
...
```

It also reports what a reader of the conclusion should weigh: the two premises
the analyst added have no source (`unanchored:a-1`, `unanchored:a-2`), and `s-9`
is overclaimed. Its hedge "suggest" states it at 0.6, while its premises give
0.166 given the observations. No check reaches `s-9` itself: its verdict is
`not_assessed` ("no v1 check applies").
