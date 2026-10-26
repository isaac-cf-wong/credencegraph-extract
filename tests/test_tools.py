"""The graph tools: classify, add_premises and finish_run, on an agent-tool run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import checks_fixture
import pytest
import session_ops
from classify_fixture import FIRST, PAPER, PREMISES_FIRST, PREMISES_REPAIR, REPAIR, RUBRIC, RUN

from credencegraph_extract import session
from credencegraph_extract.run import RunError
from credencegraph_extract.session import get_state, recorded_graph, replay, start_run, write
from credencegraph_extract.tools import add_premises, classify, finish_run
from credencegraph_extract.verify import verify_run

session_ops.register()

MODEL = "test-model-1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def calls(run_dir: Path) -> int:
    return len((run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines())


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    target = tmp_path / "run"
    start_run(PAPER, RUBRIC, target, harness="test-harness", model=MODEL)
    return target


@pytest.fixture
def checks_run(tmp_path: Path) -> Path:
    target = tmp_path / "checks-run"
    start_run(checks_fixture.PAPER, checks_fixture.RUBRIC, target, harness="h", model=MODEL, cited=checks_fixture.CITED)
    return target


def content(graph: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """A graph without its extraction stamps: what the records made, not which run made it."""
    nodes = []
    for node in graph["nodes"]:
        attributes = {key: value for key, value in node["attributes"].items() if key != "extraction"}
        nodes.append({**node, "attributes": attributes})
    return nodes, sorted(graph["relations"], key=lambda relation: relation["id"])


def test_the_tools_build_the_graph_the_passes_built_from_the_same_records(run_dir: Path, no_network: None) -> None:
    first = classify(run_dir, FIRST, model=MODEL)
    assert [(item["node"], item["code"]) for item in first["refused"]] == [
        ("s-4", "invalid-argument"),
        ("s-5", "invalid-argument"),
    ]
    assert all(item["fix"] for item in first["refused"])
    assert classify(run_dir, REPAIR, model=MODEL)["refused"] == []
    premises = add_premises(run_dir, PREMISES_FIRST, model=MODEL)
    assert [item["node"] for item in premises["refused"]] == ["s-5"]
    assert add_premises(run_dir, PREMISES_REPAIR, model=MODEL)["refused"] == []
    assert content(json.loads(recorded_graph(run_dir))) == content(read_json(RUN / "graph.json"))


def test_a_typed_node_typed_again_takes_the_new_types_base_form_and_links(run_dir: Path) -> None:
    classify(run_dir, FIRST, model=MODEL)
    classify(run_dir, REPAIR, model=MODEL)
    before = {node["id"]: node for node in json.loads(recorded_graph(run_dir))["nodes"]}
    assert (before["s-5"]["kind"], before["s-3"]["kind"]) == ("claim", "claim")
    result = classify(run_dir, [{"node": "s-3", "type": "result", "form": {"quantity": "the method"}}], model=MODEL)
    assert result["refused"] == []
    graph = json.loads(recorded_graph(run_dir))
    node = next(item for item in graph["nodes"] if item["id"] == "s-3")
    assert node["kind"] == "result"
    assert node["base"] == {"alpha": 8.0, "beta": 2.0}
    assert node["attributes"]["form"] == {"quantity": "the method"}
    assert not [relation for relation in graph["relations"] if relation["target"] == "s-3"]


def test_a_failed_retyping_leaves_the_node_as_it_was(run_dir: Path) -> None:
    classify(run_dir, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}], model=MODEL)
    before = next(node for node in json.loads(recorded_graph(run_dir))["nodes"] if node["id"] == "s-1")
    result = classify(run_dir, [{"node": "s-1", "type": "result", "form": {"quantity": "absent"}}], model=MODEL)
    assert [item["code"] for item in result["refused"]] == ["invalid-argument"]
    after = next(node for node in json.loads(recorded_graph(run_dir))["nodes"] if node["id"] == "s-1")
    assert after == before


def test_a_split_chunk_cannot_be_split_again(run_dir: Path) -> None:
    split = FIRST[2]
    classify(run_dir, [split], model=MODEL)
    result = classify(run_dir, [split], model=MODEL)
    [refused] = result["refused"]
    assert refused["code"] == "already-split"
    assert refused["fix"] == ["'s-2' is already split: send records for its clauses s-2a, s-2b, ... instead"]
    assert [node["id"] for node in get_state(run_dir)["nodes"] if node["id"].startswith("s-2")] == [
        "s-2",
        "s-2a",
        "s-2b",
    ]


def test_a_node_a_check_targets_cannot_be_typed_again(checks_run: Path) -> None:
    classify(checks_run, [{"node": "s-2", "type": "result", "form": {"quantity": "segments"}}], model=MODEL)
    write(checks_run, "observe", {"target": "s-2", "result": "failed"}, model=MODEL)
    result = classify(checks_run, [{"node": "s-2", "type": "context"}], model=MODEL)
    [refused] = result["refused"]
    assert (refused["node"], refused["code"]) == ("s-2", "checked-node")
    assert refused["fix"] == ["leave 's-2' as it is; send records only for nodes no check targets"]
    assert next(node for node in get_state(checks_run)["nodes"] if node["id"] == "s-2")["type"] == "result"
    assert [item["code"] for item in get_state(checks_run)["rejections"]] == ["checked-node"]


def test_finish_writes_the_outputs_and_a_later_write_removes_them(run_dir: Path) -> None:
    classify(run_dir, FIRST, model=MODEL)
    classify(run_dir, REPAIR, model=MODEL)
    result = finish_run(run_dir, model=MODEL)
    assert set(result) == {"findings", "ok"}
    assert result["ok"] == (result["findings"] == [])
    assert {name for name in ("graph.json", "rejections.json", "report.json") if (run_dir / name).exists()} == {
        "graph.json",
        "rejections.json",
        "report.json",
    }
    assert (run_dir / "graph.json").read_bytes() == recorded_graph(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["finished"] is True
    assert manifest["graph_digest"].startswith("sha256:")
    assert [item["code"] for item in read_json(run_dir / "rejections.json")] == ["invalid-argument"] * 2
    classify(run_dir, [{"node": "r-1", "type": "reference"}], model=MODEL)
    assert not (run_dir / "graph.json").exists()
    assert read_json(run_dir / "manifest.json")["finished"] is False
    with pytest.raises(RunError) as raised:
        verify_run(run_dir)
    assert raised.value.code == "unfinished-run"
    finish_run(run_dir, model=MODEL)
    assert (run_dir / "graph.json").exists()


def test_finish_derives_each_verdict_from_the_checks_that_observe_it(checks_run: Path) -> None:
    classify(
        checks_run,
        [
            {"node": "s-2", "type": "result", "form": {"quantity": "segments"}},
            {"node": "s-3", "type": "result", "form": {"quantity": "realisations"}},
        ],
        model=MODEL,
    )
    write(checks_run, "observe", {"target": "s-2", "result": "failed"}, model=MODEL)
    finish_run(checks_run, model=MODEL)
    verdicts = {
        node["id"]: node["attributes"]["assessment"]
        for node in read_json(checks_run / "graph.json")["nodes"]
        if "assessment" in node["attributes"]
    }
    assert verdicts["s-2"] == {"verdict": "fails"}
    assert verdicts["s-3"] == {"verdict": "not_assessed", "reason": "not selected"}


def test_finish_reports_what_the_core_check_finds(run_dir: Path) -> None:
    result = finish_run(run_dir, model=MODEL)
    assert result["ok"] is False
    assert {"untyped"} <= {finding["code"] for finding in result["findings"]}
    assert all(finding["nodes"] and finding["message"] for finding in result["findings"])


def test_a_finished_tool_run_verifies_and_replays(checks_run: Path, no_network: None) -> None:
    classify(checks_run, [{"node": "s-2", "type": "result", "form": {"quantity": "segments"}}], model=MODEL)
    finish_run(checks_run, model=MODEL)
    found = verify_run(checks_run)
    assert (found.rerun, found.differences) == ((), ())
    assert replay(checks_run, source=checks_fixture.PAPER) == recorded_graph(checks_run)


def test_premises_need_a_rubric_that_takes_them(checks_run: Path) -> None:
    before = calls(checks_run)
    with pytest.raises(RunError, match="no premises can be added") as raised:
        add_premises(checks_run, [{"conclusion": "s-9", "premises": [{"statement": "x"}]}], model=MODEL)
    assert raised.value.code == "invalid-argument"
    assert calls(checks_run) == before


def test_records_must_be_a_list(run_dir: Path) -> None:
    with pytest.raises(RunError, match="records must be a list") as raised:
        write(run_dir, "classify", {"records": {"node": "s-1"}}, model=MODEL)
    assert raised.value.code == "invalid-argument"
    assert calls(run_dir) == 1


def test_every_call_is_logged_with_its_model_and_agent(run_dir: Path) -> None:
    classify(run_dir, FIRST[:2], model="m-a", agent="typist", note="first pass")
    finish_run(run_dir, model="m-b")
    lines = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(line["tool"], line["model"], line["agent"]) for line in lines] == [
        ("start_run", MODEL, None),
        ("classify", "m-a", "typist"),
        ("finish_run", "m-b", None),
    ]
    assert read_json(run_dir / "manifest.json")["models"] == ["m-a", "m-b", MODEL]


def test_a_fresh_process_replays_a_tool_run_from_the_command_line(run_dir: Path) -> None:
    classify(run_dir, FIRST, model=MODEL)
    finish_run(run_dir, model=MODEL)
    done = subprocess.run(  # noqa: S603 - this interpreter, the package's own command
        [sys.executable, "-m", "credencegraph_extract", "replay", str(run_dir), "--source", str(PAPER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "byte-identical" in done.stdout


def test_a_crash_before_the_outputs_are_written_is_repaired_on_the_next_read(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classify(run_dir, FIRST, model=MODEL)
    outputs = session._outputs

    def crash_once_finished(run: Path, rubric: Any, state: Any) -> None:
        if state.finished:  # the outputs of the committed finish, not the reconcile before it
            raise OSError("simulated crash")
        outputs(run, rubric, state)

    monkeypatch.setattr(session, "_outputs", crash_once_finished)
    with pytest.raises(OSError, match="simulated crash"):
        finish_run(run_dir, model=MODEL)
    monkeypatch.undo()
    assert read_json(run_dir / "manifest.json")["finished"] is True
    assert not (run_dir / "graph.json").exists()
    verify_run(run_dir)
    assert (run_dir / "graph.json").read_bytes() == recorded_graph(run_dir)


def test_a_crash_before_stale_outputs_are_removed_is_repaired_on_the_next_read(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finish_run(run_dir, model=MODEL)
    assert (run_dir / "graph.json").exists()
    outputs = session._outputs

    def crash_once_unfinished(run: Path, rubric: Any, state: Any) -> None:
        if not state.finished:  # the cleanup after the committed write, not the reconcile before it
            raise OSError("simulated crash")
        outputs(run, rubric, state)

    monkeypatch.setattr(session, "_outputs", crash_once_unfinished)
    with pytest.raises(OSError, match="simulated crash"):
        classify(run_dir, FIRST[:1], model=MODEL)
    monkeypatch.undo()
    assert (run_dir / "graph.json").exists()
    assert get_state(run_dir)["finished"] is False
    assert not (run_dir / "graph.json").exists()


def test_a_crash_before_the_manifest_is_updated_is_repaired(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = session._replace

    def crash_on_manifest(path: Path, data: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("simulated crash")
        original(path, data)

    monkeypatch.setattr(session, "_replace", crash_on_manifest)
    with pytest.raises(OSError, match="simulated crash"):
        finish_run(run_dir, model=MODEL)
    monkeypatch.undo()
    assert read_json(run_dir / "manifest.json")["finished"] is False
    assert get_state(run_dir)["finished"] is True
    assert read_json(run_dir / "manifest.json")["finished"] is True
    assert (run_dir / "graph.json").exists()


def test_every_refused_record_says_what_to_change(run_dir: Path) -> None:
    duplicate = [FIRST[1], FIRST[1]]
    [refused] = classify(run_dir, duplicate, model=MODEL)["refused"]
    assert refused["code"] == "duplicate-record"
    assert refused["fix"] == ["send one record per node in a call: an earlier record in this call already names 's-1'"]
    classify(run_dir, FIRST, model=MODEL)
    classify(run_dir, REPAIR, model=MODEL)
    premises = add_premises(run_dir, PREMISES_FIRST, model=MODEL)["refused"]
    assert premises
    assert all(item["fix"] for item in premises)
    malformed = add_premises(run_dir, [{"conclusion": "s-5", "premises": "one"}], model=MODEL)["refused"]
    assert malformed[0]["fix"][0].endswith(
        "; a premise record is {\"conclusion\": <id of a node of a type in ['claim']>, "
        '"premises": [{"stated_at": <chunk id>} or {"statement": <the premise in words>}, ...]}'
    )
    rejections = read_json(run_dir / "state.json")["rejections"]
    assert all(item.get("fix") for item in rejections)


def test_a_call_replays_whatever_order_the_agent_wrote_its_keys_in(run_dir: Path, no_network: None) -> None:
    # The form's keys are not in sorted order and both its values are wrong: the refusal's fixes, and
    # which field the core names first, follow the order of the keys.
    record = {"type": "claim", "node": "s-1", "form": {"statement": "not present", "shape": "invalid"}}
    assert classify(run_dir, [record], model=MODEL)["refused"]
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir)


def test_reconciling_refuses_a_log_edited_by_hand(run_dir: Path) -> None:
    classify(run_dir, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}], model=MODEL)
    manifest_before = (run_dir / "manifest.json").read_bytes()
    log = run_dir / "calls.jsonl"
    log.write_text(log.read_text(encoding="utf-8").replace('"type": "method"', '"type": "context"'), encoding="utf-8")
    with pytest.raises(RunError, match=r"call 2 \(classify\) was changed after it was logged") as raised:
        session.reconcile(run_dir)
    assert raised.value.code == "record-mismatch"
    assert (run_dir / "manifest.json").read_bytes() == manifest_before


@pytest.mark.parametrize("damage", ["missing", "not-json"])
def test_a_missing_or_unreadable_state_file_is_rebuilt(run_dir: Path, damage: str) -> None:
    classify(run_dir, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}], model=MODEL)
    expected = recorded_graph(run_dir)
    if damage == "missing":
        (run_dir / "state.json").unlink()
    else:
        (run_dir / "state.json").write_text("{", encoding="utf-8")
    assert get_state(run_dir)["calls"] == 2
    assert read_json(run_dir / "state.json")["calls"] == 2
    assert recorded_graph(run_dir) == expected
    classify(run_dir, [{"node": "s-3", "type": "claim"}], model=MODEL)


def test_reconciling_refuses_an_edited_start_run_line(run_dir: Path) -> None:
    classify(run_dir, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}], model=MODEL)
    manifest_before = (run_dir / "manifest.json").read_bytes()
    log = run_dir / "calls.jsonl"
    log.write_text(log.read_text(encoding="utf-8").replace('"test-harness"', '"edited"', 1), encoding="utf-8")
    with pytest.raises(RunError, match=r"call 1 \(start_run\) was changed after it was logged"):
        session.reconcile(run_dir)
    assert (run_dir / "manifest.json").read_bytes() == manifest_before


def test_reconciling_refuses_a_forged_start_run_digest(run_dir: Path) -> None:
    classify(run_dir, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}], model=MODEL)
    log = run_dir / "calls.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["result_digest"] = "sha256:" + "ff" * 32
    log.write_text("\n".join([json.dumps(first, sort_keys=True), *lines[1:]]) + "\n", encoding="utf-8")
    (run_dir / "state.json").unlink()
    with pytest.raises(RunError, match=r"call 1 \(start_run\) gives another result"):
        session.reconcile(run_dir)


def test_a_state_file_whose_content_was_changed_is_rebuilt(run_dir: Path) -> None:
    classify(run_dir, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}], model=MODEL)
    expected = recorded_graph(run_dir)
    path = run_dir / "state.json"
    state = read_json(path)
    state["graph"]["nodes"][0]["statement"] = "tampered"
    path.write_text(json.dumps(state), encoding="utf-8")
    assert all(node["text"] != "tampered" for node in get_state(run_dir)["nodes"])
    assert "tampered" not in path.read_text(encoding="utf-8")
    assert recorded_graph(run_dir) == expected
