"""The classify and premise records of a run on the invented fixture paper, ``fixtures/classify/paper.tex``.

``FIRST`` types every chunk, splits ``s-2`` into a result and a claim, and locates two hedges:
``suggests``, which the fixture rubric maps, and ``We expect``, which it does not. Two of its records
are refused: ``s-4``'s form quotes a statement the text does not contain, and ``s-5`` rests on
``s-4``, which is still unassigned when ``s-5`` is applied. ``REPAIR`` corrects both. The records are
listed out of document order, which the assembler restores.

The premise records concern the three claims, ``s-2b``, ``s-3`` and ``s-5``. ``PREMISES_FIRST`` links
``s-2b`` to ``s-1``, a chunk the classify records did not link, and gives ``s-2b`` three unstated
premises, one more than the fixture rubric's ``max_per_conclusion`` of 2, so the third is reported
as overflow. ``s-3`` needs the stationarity premise of ``s-2b`` again, with different whitespace,
so both share one analyst node. ``s-5``'s record is refused, as it links ``s-4``, which ``s-5``
already rests on; ``PREMISES_REPAIR`` drops that link. ``RUN`` is a run of these records made by the
earlier model-calling passes (format 1).
"""

from __future__ import annotations

from typing import Any

from conftest import FIXTURES

PAPER = FIXTURES / "classify" / "paper.tex"
RUBRIC = FIXTURES / "classify" / "rubric.toml"
RUN = FIXTURES / "classify" / "run"
RUN_ID = "20261025T120000Z-classify"

S5 = {
    "node": "s-5",
    "type": "claim",
    "form": {"shape": "bound", "statement": "the estimator to stay unbiased"},
    "rests_on": ["s-4"],
    "hedge": "We expect",
}

FIRST: list[dict[str, Any]] = [
    S5,
    {"node": "s-1", "type": "method", "form": {"kind": "simulation"}},
    {
        "node": "s-2",
        "split": ["The bias vanishes for long segments", "the estimator is therefore consistent"],
        "children": [
            {"type": "result", "form": {"quantity": "The bias", "trend": "vanishes"}},
            {
                "type": "claim",
                "form": {"shape": "universal", "statement": "the estimator is therefore consistent"},
                "rests_on": ["s-2a"],
            },
        ],
    },
    {
        "node": "s-3",
        "type": "claim",
        "form": {"shape": "existential", "statement": "the method may extend to coloured noise"},
        "rests_on": ["s-2a", "e-1"],
        "hedge": "suggests",
    },
    {"node": "s-4", "type": "literature_claim", "form": {"statement": "a bias below one percent"}},
    {"node": "e-1", "type": "derivation"},
]

REPAIR: list[dict[str, Any]] = [
    {"node": "s-4", "type": "literature_claim", "form": {"statement": "a bias below 1\\% in comparable settings"}},
    S5,
]

STATIONARY = "The noise is stationary over the observation."

PREMISES_FIRST: list[dict[str, Any]] = [
    {
        "conclusion": "s-5",
        "premises": [{"statement": "The settings of Smith match ours."}, {"stated_at": "s-4"}],
    },
    {
        "conclusion": "s-2b",
        "premises": [
            {"stated_at": "s-1"},
            {"statement": "The segments are long compared with the noise correlation time."},
            {"statement": STATIONARY},
            {"statement": "The segments are independent."},
        ],
    },
    {"conclusion": "s-3", "premises": [{"statement": "The noise  is stationary\nover the observation. "}]},
]

PREMISES_REPAIR: list[dict[str, Any]] = [
    {"conclusion": "s-5", "premises": [{"statement": "The settings of Smith match ours."}]},
]
