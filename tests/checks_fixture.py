"""The recorded answers of a run with the check pass on the invented fixture paper, ``fixtures/checks/paper.tex``.

The classify answer types every sentence and equation; references take their type from the
rubric's ``[initial]`` table. The rubric has no ``[premises]``, so the premise pass makes no call.

The check plan proposes, out of document order:

- ``recompute-numbers`` on ``s-2``: ``400 * 250`` is 100000, not the reported 90000, so it fails;
- ``recompute-numbers`` on ``s-3``: ``400 - 400 * 0.125`` is 350, so it passes;
- ``rederive-maths`` from ``e-1`` to ``e-2``: ``c / N * N`` equals ``c``, which sympy proves;
- ``rederive-maths`` on ``e-3``: ``atan(x) + atan(1/x) = pi/2`` for positive ``x``, which sympy cannot
  decide, so it errs and the numeric check passes in its place;
- ``citation-support`` on ``s-7``: the cited work is in ``fixtures/checks/cited``, and the judgement
  quotes a passage of it that supports the sentence;
- ``citation-support`` on ``s-8``: the cited work is missing, so the check errs without a judgement;
- two proposals that are rejected: one whose numeral 300 no cited chunk contains, and one on the
  method ``s-1``, which is not assessed.

``e-1`` gets no check and ``s-9``, a claim, is of a type no implemented check runs on.
"""

from __future__ import annotations

import json
from typing import Any

from conftest import FIXTURES
from notes_pass import message

PAPER = FIXTURES / "checks" / "paper.tex"
RUBRIC = FIXTURES / "checks" / "rubric.toml"
CITED = FIXTURES / "checks" / "cited"
RUN = FIXTURES / "checks" / "run"
RUN_ID = "20261025T120000Z-checks"

CLASSIFY: list[dict[str, Any]] = [
    {"node": "s-1", "type": "method"},
    {"node": "s-2", "type": "result", "form": {"quantity": "segments"}},
    {"node": "s-3", "type": "result", "form": {"quantity": "realisations"}},
    {"node": "s-4", "type": "context"},
    {"node": "e-1", "type": "derivation"},
    {"node": "s-5", "type": "context"},
    {"node": "e-2", "type": "derivation"},
    {"node": "s-6", "type": "context"},
    {"node": "e-3", "type": "derivation"},
    {"node": "s-7", "type": "literature_claim"},
    {"node": "s-8", "type": "literature_claim"},
    {"node": "s-9", "type": "claim", "rests_on": ["s-3", "e-2"]},
]

PRODUCT = {
    "target": "e-2",
    "type": "rederive-maths",
    "nodes": ["e-1", "e-2"],
    "lhs": "c / N * N",
    "rhs": "c",
    "symbols": [
        {"name": "c", "assumptions": ["real"], "domain": [-10, 10]},
        {"name": "N", "assumptions": ["positive"], "domain": [1, 100]},
    ],
    "assumption": "N is the number of segments, so it is positive.",
}

PHASE = {
    "target": "e-3",
    "type": "rederive-maths",
    "nodes": ["e-3"],
    "lhs": "atan(x) + atan(1/x)",
    "rhs": "pi/2",
    "symbols": [{"name": "x", "assumptions": ["positive"], "domain": [0.01, 100]}],
    "assumption": "x is positive.",
}

PLAN: list[dict[str, Any]] = [
    {"target": "s-7", "type": "citation-support", "reference": "r-1"},
    {
        "target": "s-3",
        "type": "recompute-numbers",
        "nodes": ["s-3"],
        "expression": "400 - 400 * 0.125",
        "expected": "350",
    },
    {
        "target": "s-2",
        "type": "recompute-numbers",
        "nodes": ["s-1", "s-2"],
        "expression": "400 * 250",
        "expected": "90000",
    },
    PRODUCT,
    PHASE,
    {"target": "s-8", "type": "citation-support", "reference": "r-2"},
    {"target": "s-2", "type": "recompute-numbers", "nodes": ["s-2"], "expression": "300 * 300", "expected": "90000"},
    {"target": "s-1", "type": "recompute-numbers", "nodes": ["s-1"], "expression": "400", "expected": "400"},
]

PASSAGE = "In comparable settings the bias stays below 1% for all segment lengths we tried"
JUDGEMENT = {"passage": PASSAGE, "supports": True}

BODIES = [
    message(1, json.dumps({"records": CLASSIFY})),
    message(2, json.dumps({"checks": PLAN})),
    message(3, json.dumps(JUDGEMENT)),
]

# What each check finds, in id order: its type, target and result.
RESULTS = [
    ("c-001", "recompute-numbers", "s-2", "failed"),
    ("c-002", "recompute-numbers", "s-3", "passed"),
    ("c-003", "rederive-maths", "e-2", "passed"),
    ("c-004", "rederive-maths", "e-3", "error"),
    ("c-005", "rederive-maths-numeric", "e-3", "passed"),
    ("c-006", "citation-support", "s-7", "passed"),
    ("c-007", "citation-support", "s-8", "error"),
]

VERDICTS = {
    "s-2": {"verdict": "fails"},
    "s-3": {"verdict": "holds"},
    "e-1": {"verdict": "not_assessed", "reason": "not selected"},
    "e-2": {"verdict": "holds"},
    "e-3": {"verdict": "holds"},
    "s-7": {"verdict": "holds"},
    "s-8": {"verdict": "undetermined"},
    "s-9": {"verdict": "not_assessed", "reason": "no v1 check applies"},
}
