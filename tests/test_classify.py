"""Classifying the invented fixture paper through the tools: the typed graph, hedges, and records left refused."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from classify_fixture import FIRST, PAPER, REPAIR, RUBRIC
from credencegraph.core import loads
from credencegraph.rubric import check_graph, load_rubric

from credencegraph_extract import session
from credencegraph_extract.tools import classify, finish_run

MODEL = "test-model-1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def started(run_dir: Path) -> Path:
    session.start_run(PAPER, RUBRIC, run_dir, harness="test-harness", model=MODEL)
    return run_dir


def test_the_fixture_paper_passes_the_typed_check_with_rubric_credences(tmp_path: Path) -> None:
    run_dir = started(tmp_path / "run")
    classify(run_dir, FIRST, model=MODEL)
    classify(run_dir, REPAIR, model=MODEL)
    finish_run(run_dir, model=MODEL)
    graph = loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    assert check_graph(graph, load_rubric(RUBRIC), "typed", rubric_credences=True) == []
    result = subprocess.run(  # noqa: S603 - the core's console script, fixed arguments
        [
            str(Path(sys.executable).parent / "credencegraph"),
            "check",
            str(run_dir / "graph.json"),
            "--rubric",
            str(RUBRIC),
            "--level",
            "typed",
            "--rubric-credences",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok at level typed with rubric credences" in result.stdout


def test_hedges_the_rubric_maps_set_stated_and_the_others_are_returned(tmp_path: Path) -> None:
    run_dir = started(tmp_path / "run")
    unmapped = classify(run_dir, FIRST, model=MODEL)["unmapped_hedges"]
    unmapped += classify(run_dir, REPAIR, model=MODEL)["unmapped_hedges"]
    assert unmapped == [{"node": "s-5", "phrase": "We expect"}]
    graph = json.loads(session.recorded_graph(run_dir))
    stated = {node["id"]: node.get("stated") for node in graph["nodes"] if node.get("stated") is not None}
    assert stated == {"s-3": {"alpha": 6.0, "beta": 4.0}}


def test_records_left_uncorrected_stay_refused_and_their_chunks_untyped(tmp_path: Path) -> None:
    run_dir = started(tmp_path / "run")
    classify(run_dir, FIRST, model=MODEL)
    unrepaired = {**REPAIR[0], "form": {"statement": "a bias of one percent"}}
    classify(run_dir, [unrepaired, REPAIR[1]], model=MODEL)
    finish_run(run_dir, model=MODEL)
    rejections = read_json(run_dir / "rejections.json")
    assert [(item["call"], item["node"], item["code"]) for item in rejections] == [
        (2, "s-4", "invalid-argument"),
        (2, "s-5", "invalid-argument"),
        (3, "s-4", "invalid-argument"),
        (3, "s-5", "invalid-argument"),
    ]
    assert all(item["fix"] for item in rejections)
    kinds = {node["id"]: node["kind"] for node in read_json(run_dir / "graph.json")["nodes"]}
    assert kinds["s-4"] == kinds["s-5"] == "unassigned"
    assert read_json(run_dir / "report.json")["coverage"]["untyped"] == ["s-4", "s-5"]
