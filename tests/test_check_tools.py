"""The check tools on an agent-tool run: run_checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from checks_fixture import CITED, CLASSIFY, PAPER, PHASE, PLAN, PRODUCT, RESULTS, RUBRIC, VERDICTS

from credencegraph_extract import session
from credencegraph_extract.checks import directory_digest
from credencegraph_extract.run import RunError
from credencegraph_extract.session import get_state, recorded_graph, replay, start_run
from credencegraph_extract.tools import classify, finish_run, run_checks
from credencegraph_extract.verify import verify_run

MODEL = "test-model-1"
# The maths proposals of the recorded check run: s-3 passes, s-2 fails, e-2 passes, e-3 errs and its
# numeric fallback passes.
MATHS = [PLAN[1], PLAN[2], PRODUCT, PHASE]
RECOMPUTE = PLAN[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run of the check fixture's paper, classified as the recorded run was."""
    target = tmp_path / "run"
    start_run(PAPER, RUBRIC, target, harness="test-harness", model=MODEL, cited=CITED)
    assert classify(target, CLASSIFY, model=MODEL)["refused"] == []
    return target


def test_the_maths_checks_give_what_the_check_pass_gave(run_dir: Path) -> None:
    result = run_checks(run_dir, MATHS, model=MODEL)
    assert result["refused"] == []
    found = [(item["type"], item["target"], item["result"]) for item in result["checks"]]
    recorded = [(check_type, target, outcome) for _, check_type, target, outcome in RESULTS[:5]]
    # The pass numbers checks in document order of their targets; the tool, in the order proposed.
    assert sorted(found) == sorted(recorded)
    assert [item["id"] for item in result["checks"]] == ["c-001", "c-002", "c-003", "c-004", "c-005"]
    assert all(item["detail"] for item in result["checks"])


def test_each_check_is_an_observed_evidence_node_with_its_files(run_dir: Path) -> None:
    run_checks(run_dir, MATHS, model=MODEL)
    graph = json.loads(recorded_graph(run_dir))
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {(edge["source"], edge["target"]) for edge in graph["relations"] if edge["type"] == "observes"}
    assert edges == {("s-3", "c-001"), ("s-2", "c-002"), ("e-2", "c-003"), ("e-3", "c-004"), ("e-3", "c-005")}
    assert [nodes[f"c-00{n}"]["attributes"].get("observed") for n in range(1, 6)] == [True, False, True, None, True]
    for check_id in ("c-001", "c-004"):
        record = nodes[check_id]["attributes"]["check"]
        files = {path.name: path.read_bytes() for path in (run_dir / "checks" / check_id).iterdir()}
        assert set(files) == {"inputs.json", "output.json", "log.txt"}
        assert record["log"] == {"path": f"checks/{check_id}/", "digest": directory_digest(files)}
        assert record["model"] is None
        assert record["tool"]["version"] == session.versions()["credencegraph-extract"]


def test_the_versions_that_ran_the_checks_are_logged_with_the_call(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    assert log(run_dir)[-1]["args"]["versions"] == session.versions()


def test_a_check_is_not_run_twice_on_its_target_in_a_later_call(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    again = run_checks(run_dir, [{**RECOMPUTE, "expression": "400 - 0.125 * 400"}], model=MODEL)
    assert again["checks"] == []
    [refused] = again["refused"]
    assert (refused["node"], refused["code"]) == ("s-3", "duplicate-record")
    assert refused["fix"] == ["propose at most one check of each type on a target"]


def test_one_step_is_not_checked_again_on_another_target_in_a_later_call(run_dir: Path) -> None:
    run_checks(run_dir, [PRODUCT], model=MODEL)
    again = run_checks(run_dir, [{**PRODUCT, "target": "e-1", "lhs": "c", "rhs": "N * c / N"}], model=MODEL)
    assert again["checks"] == []
    [refused] = again["refused"]
    assert (refused["node"], refused["code"]) == ("e-1", "duplicate-record")
    assert refused["fix"] == ["propose this check once, on the step's own target; 'e-2' already has it"]


def test_a_failed_check_stays_observed(run_dir: Path) -> None:
    failed = run_checks(run_dir, [PLAN[2]], model=MODEL)["checks"]
    assert [item["result"] for item in failed] == ["failed"]
    retry = run_checks(run_dir, [{**PLAN[2], "expression": "250 * 400"}], model=MODEL)
    assert [item["code"] for item in retry["refused"]] == ["duplicate-record"]
    finish_run(run_dir, model=MODEL)
    verdicts = {
        node["id"]: node["attributes"]["assessment"]
        for node in read_json(run_dir / "graph.json")["nodes"]
        if "assessment" in node["attributes"]
    }
    assert verdicts["s-2"] == VERDICTS["s-2"]


def test_a_citation_proposal_is_sent_to_the_citation_tool(run_dir: Path) -> None:
    result = run_checks(run_dir, [PLAN[0]], model=MODEL)
    [refused] = result["refused"]
    assert (refused["node"], refused["code"]) == ("s-7", "invalid-record")
    assert refused["fix"] == ["send a citation-support check to the citation tool, with the source and the passage"]


def test_every_refused_proposal_says_what_to_change(run_dir: Path) -> None:
    proposals = [
        {"target": "s-9", "type": "recompute-numbers"},
        {**RECOMPUTE, "expected": "349"},
        {**RECOMPUTE, "nodes": ["x-1"]},
        "not a proposal",
        {**RECOMPUTE, "type": "rerun-code"},
    ]
    result = run_checks(run_dir, proposals, model=MODEL)
    assert [item["code"] for item in result["refused"]] == [
        "not-a-target",
        "unfaithful-numeral",
        "unknown-node",
        "invalid-record",
        "invalid-record",
    ]
    assert all(
        item["fix"] and all(isinstance(line, str) and line for line in item["fix"]) for item in result["refused"]
    )
    assert [item["call"] for item in get_state(run_dir)["rejections"]] == [3] * 5


def test_the_node_a_check_targets_can_no_longer_be_typed_again(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    [refused] = classify(run_dir, [{"node": "s-3", "type": "context"}], model=MODEL)["refused"]
    assert refused["code"] == "checked-node"


def test_proposals_must_be_a_list(run_dir: Path) -> None:
    with pytest.raises(RunError, match="proposals must be a list") as raised:
        run_checks(run_dir, "check s-3", model=MODEL)  # type: ignore[arg-type]
    assert raised.value.code == "invalid-argument"


def test_a_finished_run_with_checks_verifies_and_replays(run_dir: Path, no_network: None) -> None:
    run_checks(run_dir, MATHS, model=MODEL)
    finish_run(run_dir, model=MODEL)
    found = verify_run(run_dir)
    assert found.rerun == ("c-001", "c-002", "c-003", "c-004", "c-005")
    assert found.differences == ()
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir) == (run_dir / "graph.json").read_bytes()


def test_check_files_lost_or_edited_are_written_again_from_the_log(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    expected = (run_dir / "checks" / "c-001" / "output.json").read_bytes()
    (run_dir / "checks" / "c-001" / "output.json").write_text("{}", encoding="utf-8")
    (run_dir / "checks" / "c-001" / "log.txt").unlink()
    (run_dir / "state.json").unlink()
    session.reconcile(run_dir)
    assert (run_dir / "checks" / "c-001" / "output.json").read_bytes() == expected
    assert (run_dir / "checks" / "c-001" / "log.txt").is_file()


def test_a_state_file_with_other_check_files_is_not_trusted(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    saved = read_json(run_dir / "state.json")
    path = "checks/c-001/output.json"
    original = saved["files"][path]
    saved["files"][path] = original.replace("passed", "failed")
    (run_dir / "state.json").write_text(json.dumps(saved), encoding="utf-8")
    session.reconcile(run_dir)
    assert read_json(run_dir / "state.json")["files"][path] == original
    assert (run_dir / path).read_text(encoding="utf-8") == original
