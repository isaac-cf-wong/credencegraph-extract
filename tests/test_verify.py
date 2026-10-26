"""``verify`` runs a run's deterministic checks again and warns about rubric versions reused with other values."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from checks_fixture import RUN
from conftest import FIXTURES
from credencegraph.core import loads
from credencegraph.rubric import load_rubric

from credencegraph_extract import evaluate
from credencegraph_extract.checks import rerun
from credencegraph_extract.cli import main
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import RunError
from credencegraph_extract.verify import rubric_reuse, verify_run


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A copy of the recorded run with checks, safe to modify."""
    target = tmp_path / "run"
    shutil.copytree(RUN, target)
    return target


def edit_check(run_dir: Path, check_id: str, change: Callable[[dict[str, Any]], None]) -> None:
    """Change one check record of a run's graph, and record the new graph's digest in the manifest."""
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    (node,) = [node for node in graph["nodes"] if node["id"] == check_id]
    change(node["attributes"]["check"])
    data = (json.dumps(graph, indent=2) + "\n").encode("utf-8")
    (run_dir / "graph.json").write_bytes(data)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["graph_digest"] = bytes_digest(data)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def fields(found: Any, check_id: str) -> list[str]:
    return [difference.field for difference in found.differences if difference.check == check_id]


def test_every_check_of_the_recorded_run_agrees(run_dir: Path, no_network: None) -> None:
    found = verify_run(run_dir)
    assert found.rerun == ("c-001", "c-002", "c-003", "c-004", "c-005")
    assert found.not_rerun == ("c-006", "c-007")
    assert found.differences == ()


def test_a_rerun_reproduces_each_record_and_its_files(run_dir: Path) -> None:
    graph = loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    rubric = load_rubric(run_dir / "rubric.toml")
    for check_id in ("c-001", "c-002", "c-003", "c-004", "c-005"):
        record, files = rerun(graph, rubric, graph.nodes[check_id])
        assert record == json.loads(json.dumps(dict(graph.nodes[check_id].attributes["check"]), default=dict))
        assert files == {name: (run_dir / "checks" / check_id / name).read_bytes() for name in files}


def test_a_citation_check_is_not_rerun(run_dir: Path) -> None:
    graph = loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not a check of a type"):
        rerun(graph, load_rubric(run_dir / "rubric.toml"), graph.nodes["c-006"])


def test_a_recorded_result_the_rerun_does_not_give_is_reported(run_dir: Path) -> None:
    def flip(record: dict[str, Any]) -> None:
        record["result"] = "failed"

    edit_check(run_dir, "c-002", flip)
    found = verify_run(run_dir)
    assert fields(found, "c-002") == ["result", "result (output.json)"]
    assert [(d.recorded, d.found) for d in found.differences] == [("failed", "passed"), ("failed", "passed")]


def test_a_changed_input_changes_the_result_detail_and_log(run_dir: Path) -> None:
    # 400 - 400 * 0.125 is 350, as reported; 400 * 0.125 is 50, so the check now fails.
    def change(record: dict[str, Any]) -> None:
        record["inputs"]["expression"] = "400 * 0.125"

    edit_check(run_dir, "c-002", change)
    found = verify_run(run_dir)
    assert fields(found, "c-002") == ["result", "detail", "log", "inputs (inputs.json)"]
    assert found.differences[0].found == "failed"


def test_an_unfaithful_recorded_numeral_cannot_be_rerun(run_dir: Path) -> None:
    def change(record: dict[str, Any]) -> None:
        record["inputs"]["expression"] = "400 - 50"

    edit_check(run_dir, "c-002", change)
    differences = {d.field: d for d in verify_run(run_dir).differences if d.check == "c-002"}
    assert differences["result"].found == "error"
    assert differences["detail"].found.startswith("the recorded inputs cannot be run: no numeral of ['s-3'] equals 50")


def test_a_record_naming_a_missing_node_cannot_be_rerun(run_dir: Path) -> None:
    def change(record: dict[str, Any]) -> None:
        record["artifact"]["nodes"] = ["s-99"]

    edit_check(run_dir, "c-001", change)
    differences = {d.field: d for d in verify_run(run_dir).differences if d.check == "c-001"}
    assert differences["result"].found == "error"
    assert "nodes not in the graph: ['s-99']" in differences["detail"].found


def test_a_symbolic_check_is_decided_again(run_dir: Path) -> None:
    def change(record: dict[str, Any]) -> None:
        record["inputs"]["rhs"] = "c + 1"

    edit_check(run_dir, "c-003", change)
    found = verify_run(run_dir)
    assert fields(found, "c-003")[:2] == ["result", "detail"]
    assert found.differences[0].found == "failed"


def test_numeric_points_drawn_differently_are_reported(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_of = evaluate.seed_of
    monkeypatch.setattr(evaluate, "seed_of", lambda encoding: seed_of(encoding) + 1)
    found = verify_run(run_dir)
    assert fields(found, "c-005") == ["inputs", "log"]
    assert {d.check for d in found.differences} == {"c-005"}


def test_check_files_that_differ_from_the_record_are_reported(run_dir: Path) -> None:
    (run_dir / "checks" / "c-001" / "log.txt").write_text("edited\n", encoding="utf-8")
    shutil.rmtree(run_dir / "checks" / "c-006")
    found = verify_run(run_dir)
    assert [(d.check, d.field) for d in found.differences] == [("c-001", "files"), ("c-006", "files")]
    assert found.differences[1].found is None


def test_a_check_retyped_as_a_citation_is_not_skipped(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Rewriting the type would otherwise move a deterministic check out of the re-run, and its files still
    # match the record's log digest.
    def retype(record: dict[str, Any]) -> None:
        record["type"] = "citation-support"
        record["result"] = "failed"

    edit_check(run_dir, "c-002", retype)
    found = verify_run(run_dir)
    assert "c-002" in found.not_rerun
    assert fields(found, "c-002") == ["type (inputs.json)", "result (output.json)"]
    assert main(["verify", str(run_dir)]) == 1


def test_a_check_of_a_type_the_runner_does_not_have_is_reported(run_dir: Path) -> None:
    def retype(record: dict[str, Any]) -> None:
        record["type"] = "rerun-code"

    edit_check(run_dir, "c-002", retype)
    found = verify_run(run_dir)
    assert "c-002" not in found.rerun + found.not_rerun
    assert fields(found, "c-002") == ["type", "type (inputs.json)"]
    assert found.differences[0].found.startswith("one of ['recompute-numbers'")


def test_an_edited_citation_record_is_reported(run_dir: Path) -> None:
    def edit(record: dict[str, Any]) -> None:
        record["artifact"]["quote"] = "The bias is always zero."

    edit_check(run_dir, "c-006", edit)
    assert fields(verify_run(run_dir), "c-006") == ["artifact (inputs.json)"]


def test_check_files_that_are_not_json_are_reported(run_dir: Path) -> None:
    (run_dir / "checks" / "c-001" / "output.json").write_text("{", encoding="utf-8")
    assert fields(verify_run(run_dir), "c-001") == ["files", "inputs.json and output.json"]


def test_a_graph_that_differs_from_the_manifest_is_refused(run_dir: Path) -> None:
    (run_dir / "graph.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RunError, match=r"graph.json differs") as raised:
        verify_run(run_dir)
    assert raised.value.code == "record-mismatch"


def test_a_rubric_that_differs_from_the_manifest_is_refused(run_dir: Path) -> None:
    with (run_dir / "rubric.toml").open("a", encoding="utf-8") as rubric:
        rubric.write("\n")
    with pytest.raises(RunError, match=r"rubric.toml differs") as raised:
        verify_run(run_dir)
    assert raised.value.code == "record-mismatch"


def manifest(name: str, version: str, digest: str) -> dict[str, Any]:
    return {"rubric": {"name": name, "version": version, "digest": digest}}


def test_a_rubric_version_used_with_two_files_is_reported() -> None:
    warnings = rubric_reuse(
        [
            ("a", manifest("m", "1", "sha256:1")),
            ("b", manifest("m", "1", "sha256:2")),
            ("c", manifest("m", "1", "sha256:1")),
        ]
    )
    assert warnings == [
        (
            "rubric 'm' version '1' was used with 2 different files (sha256:1 in a, c; sha256:2 in b): "
            "values changed without a new version"
        )
    ]


def test_one_file_per_rubric_version_is_not_reported() -> None:
    assert (
        rubric_reuse(
            [
                ("a", manifest("m", "1", "sha256:1")),
                ("b", manifest("m", "2", "sha256:2")),
                ("c", manifest("n", "1", "sha256:3")),
                ("d", manifest("m", "1", "sha256:1")),
            ]
        )
        == []
    )


def test_the_command_verifies_runs_and_warns_about_reuse(
    run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    other = tmp_path / "other"
    shutil.copytree(FIXTURES / "classify" / "run", other)
    path = other / "manifest.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    recorded["rubric"].update(name="checks-test", version="1")
    path.write_text(json.dumps(recorded), encoding="utf-8")
    assert main(["verify", str(run_dir), str(other)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"verified {run_dir}: 5 checks run again, 0 differ; 2 model-judged checks not run again"
    assert out[1] == f"verified {other}: 0 checks run again, 0 differ; 0 model-judged checks not run again"
    assert out[2].startswith("warning: rubric 'checks-test' version '1' was used with 2 different files")


def test_the_command_exits_1_and_names_each_difference(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def flip(record: dict[str, Any]) -> None:
        record["result"] = "failed"

    edit_check(run_dir, "c-002", flip)
    assert main(["verify", str(run_dir)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f'{run_dir}: c-002: result differs: recorded "failed", verified "passed"',
        f'{run_dir}: c-002: result (output.json) differs: recorded "failed", verified "passed"',
        f"verified {run_dir}: 5 checks run again, 1 differ; 2 model-judged checks not run again",
    ]


def test_the_command_reports_an_error_with_exit_2(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (run_dir / "graph.json").write_text("{}", encoding="utf-8")
    assert main(["verify", str(run_dir)]) == 2
    assert capsys.readouterr().err.startswith("error [record-mismatch]: graph.json differs")


def edit_manifest(run_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda m: m.update(format="something-else"), "not-a-run"),
        (lambda m: m.update(version=9), "unsupported-version"),
        (lambda m: m.pop("run"), "invalid-manifest"),
        (lambda m: m.update(graph_digest=7), "invalid-manifest"),
        (lambda m: m["versions"].update(python=3), "invalid-manifest"),
        (lambda m: m["source"].update(file="../paper.tex"), "invalid-manifest"),
        (lambda m: m["source"].update(stored="no"), "invalid-manifest"),
        (lambda m: m["rubric"].pop("digest"), "invalid-manifest"),
        (lambda m: m["passes"][0].pop("name"), "invalid-manifest"),
        (lambda m: m["passes"][0]["calls"][0].update(raw="elsewhere.json"), "invalid-manifest"),
    ],
    ids=[
        "another format",
        "another version",
        "no run id",
        "graph digest not a string",
        "a version not a string",
        "source file a path",
        "stored not a boolean",
        "no rubric digest",
        "a pass without a name",
        "a call record outside raw",
    ],
)
def test_a_malformed_manifest_of_an_earlier_run_is_refused(
    run_dir: Path, change: Callable[[dict[str, Any]], None], code: str
) -> None:
    edit_manifest(run_dir, change)
    with pytest.raises(RunError) as raised:
        verify_run(run_dir)
    assert raised.value.code == code


@pytest.mark.parametrize(("content", "code"), [(None, "not-a-run"), ("{", "invalid-manifest")])
def test_a_missing_or_unreadable_manifest_is_refused(run_dir: Path, content: str | None, code: str) -> None:
    path = run_dir / "manifest.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(RunError) as raised:
        verify_run(run_dir)
    assert raised.value.code == code
