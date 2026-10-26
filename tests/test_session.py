"""Agent-tool runs: start_run, the call log, the lock, crash recovery, several processes, and replay."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import session_ops
from checks_fixture import CITED, PAPER, RUBRIC
from conftest import FIXTURES

from credencegraph_extract import session
from credencegraph_extract.cli import main
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import RunError
from credencegraph_extract.session import get_state, lock, recorded_graph, replay, start_run, write

session_ops.register()

MODEL = "test-model-1"


def start(run_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return start_run(PAPER, RUBRIC, run_dir, harness="test-harness", model=MODEL, cited=CITED, **kwargs)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    target = tmp_path / "run"
    start(target)
    return target


def test_start_run_writes_the_layout_and_logs_the_call(run_dir: Path) -> None:
    names = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file())
    assert names == [
        "calls.jsonl",
        "cited/smith2020.txt",
        "ingested.json",
        "manifest.json",
        "rubric.toml",
        "state.json",
    ]
    [call] = log(run_dir)
    assert (call["seq"], call["tool"], call["model"], call["agent"]) == (1, "start_run", MODEL, None)
    assert call["args"]["cited"] == {"smith2020.txt": bytes_digest((CITED / "smith2020.txt").read_bytes())}
    manifest = read_json(run_dir / "manifest.json")
    assert (manifest["format"], manifest["version"], manifest["calls"], manifest["finished"]) == (
        "credencegraph-extract-run",
        2,
        1,
        False,
    )
    assert manifest["cited"] == call["args"]["cited"]
    assert manifest["models"] == [MODEL]
    assert manifest["harness"] == "test-harness"
    assert manifest["rubric"]["digest"] == bytes_digest(RUBRIC.read_bytes())


def test_no_local_path_is_recorded(run_dir: Path) -> None:
    for name in ("calls.jsonl", "manifest.json", "state.json", "ingested.json"):
        text = (run_dir / name).read_text(encoding="utf-8")
        assert str(FIXTURES) not in text
        assert str(run_dir.parent) not in text


def test_start_run_tells_the_agent_the_chunks_types_and_checks(tmp_path: Path) -> None:
    result = start(tmp_path / "run")
    ids = [chunk["id"] for chunk in result["chunks"]]
    assert ids[:3] == ["s-1", "s-2", "s-3"]
    assert {"id": "r-1", "unit": "reference", "type": "reference", "key": "smith2020"}.items() <= next(
        chunk for chunk in result["chunks"] if chunk["id"] == "r-1"
    ).items()
    claim = next(item for item in result["types"] if item["name"] == "claim")
    assert claim["rests_on"] == ["result", "derivation", "literature_claim"]
    result_type = next(item for item in result["types"] if item["name"] == "result")
    assert result_type["form"] == {"quantity": {"type": "verbatim", "required": True, "values": []}}
    assert {check["name"] for check in result["checks"]} == {
        "recompute-numbers",
        "rederive-maths",
        "rederive-maths-numeric",
        "citation-support",
    }
    assert result["not_runnable"] == []


def rubric_with(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "rubric.toml"
    text = RUBRIC.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('relation = "requires"', 'relation = "supports"', "create relations other than requires"),
        (
            "[checks.recompute-numbers]",
            '[premises]\nbase = "beta:2,2"\nstrength = "beta:8,2"\nfor = ["claim"]\nmax_per_conclusion = 2\n\n[checks.recompute-numbers]',
            "does not declare the type 'assumption'",
        ),
    ],
    ids=["supports-edge-rule", "premises-without-assumption"],
)
def test_a_rubric_the_tools_cannot_use_is_refused_before_anything_is_written(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    rubric = rubric_with(tmp_path, old, new)
    with pytest.raises(RunError, match=message) as raised:
        start_run(PAPER, rubric, tmp_path / "run", harness="h", model=MODEL)
    assert raised.value.code == "invalid-rubric"
    assert not (tmp_path / "run").exists()


def test_a_check_type_the_tools_do_not_run_is_reported(tmp_path: Path) -> None:
    rubric = rubric_with(
        tmp_path,
        "[checks.citation-support]",
        '[checks.rerun-code]\ndescription = "x"\ntargets = ["result"]\nsensitivity = "beta:9,1"\n'
        'false_pass = "beta:1,9"\nmodels = []\n\n[checks.citation-support]',
    )
    result = start_run(PAPER, rubric, tmp_path / "run", harness="h", model=MODEL)
    assert result["not_runnable"] == ["rerun-code"]


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_a_link_among_the_cited_works_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    cited = tmp_path / "cited"
    shutil.copytree(CITED, cited)
    (cited / "jones2021.txt").symlink_to(cited / "smith2020.txt")
    with pytest.raises(RunError, match=r"'jones2021\.txt' among the cited works is a link") as raised:
        start_run(PAPER, RUBRIC, tmp_path / "run", harness="h", model=MODEL, cited=cited)
    assert raised.value.code == "invalid-cited"
    assert not (tmp_path / "run").exists()


def test_a_directory_among_the_cited_works_is_refused(tmp_path: Path) -> None:
    cited = tmp_path / "cited"
    shutil.copytree(CITED, cited)
    (cited / "more").mkdir()
    with pytest.raises(RunError, match="'more' among the cited works is not a regular file"):
        start_run(PAPER, RUBRIC, tmp_path / "run", harness="h", model=MODEL, cited=cited)


def test_the_cited_copy_does_not_follow_later_changes_to_the_original(tmp_path: Path) -> None:
    cited = tmp_path / "cited"
    shutil.copytree(CITED, cited)
    start_run(PAPER, RUBRIC, tmp_path / "run", harness="h", model=MODEL, cited=cited)
    (cited / "smith2020.txt").write_text("A forged passage.\n", encoding="utf-8")
    assert (tmp_path / "run" / "cited" / "smith2020.txt").read_bytes() == (CITED / "smith2020.txt").read_bytes()


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"model": ""}, "invalid-argument"),
        ({"harness": " "}, "invalid-argument"),
        ({"agent": 3}, "invalid-argument"),
    ],
)
def test_start_run_needs_who_is_calling(tmp_path: Path, kwargs: dict[str, Any], code: str) -> None:
    arguments = {"harness": "h", "model": MODEL, **kwargs}
    with pytest.raises(RunError) as raised:
        start_run(PAPER, RUBRIC, tmp_path / "run", **arguments)
    assert raised.value.code == code
    assert not (tmp_path / "run").exists()


def test_start_run_refuses_a_run_directory_in_use(run_dir: Path) -> None:
    with pytest.raises(RunError) as raised:
        start(run_dir)
    assert raised.value.code == "run-exists"


def test_start_run_refuses_a_source_that_is_not_latex(tmp_path: Path) -> None:
    with pytest.raises(RunError) as raised:
        start_run(RUBRIC, RUBRIC, tmp_path / "run", harness="h", model=MODEL)
    assert raised.value.code == "invalid-source"


def test_a_write_is_logged_stamped_and_reflected_in_the_manifest(run_dir: Path) -> None:
    result = write(run_dir, "note", {"node": "s-2", "text": "first"}, model="m-2", agent="sub-1", note="why")
    assert result == {"applied": ["s-2"], "refused": []}
    call = log(run_dir)[1]
    assert (call["seq"], call["tool"], call["args"], call["model"], call["agent"], call["note"]) == (
        2,
        "note",
        {"node": "s-2", "text": "first"},
        "m-2",
        "sub-1",
        "why",
    )
    graph = read_json(run_dir / "state.json")["graph"]
    node = next(item for item in graph["nodes"] if item["id"] == "s-2")
    assert node["attributes"]["note"] == "first"
    assert node["attributes"]["extraction"]["tool"] == "note"
    manifest = read_json(run_dir / "manifest.json")
    assert (manifest["calls"], manifest["models"], manifest["agents"]) == (2, ["m-2", MODEL], ["sub-1"])
    assert manifest["log_digest"] == bytes_digest((run_dir / "calls.jsonl").read_bytes())


def test_a_refusal_is_a_result_and_is_kept(run_dir: Path) -> None:
    result = write(run_dir, "note", {"node": "x-9", "text": "?"}, model=MODEL)
    assert [item["code"] for item in result["refused"]] == ["unknown-node"]
    assert [item["node"] for item in get_state(run_dir)["rejections"]] == ["x-9"]


def test_a_write_needs_a_model_and_a_known_tool(run_dir: Path) -> None:
    with pytest.raises(RunError) as raised:
        write(run_dir, "note", {"node": "s-2"}, model="")
    assert raised.value.code == "invalid-argument"
    with pytest.raises(RunError) as raised:
        write(run_dir, "rewrite-graph", {}, model=MODEL)
    assert raised.value.code == "unknown-tool"
    assert len(log(run_dir)) == 1


def test_get_state_reports_the_nodes_rejections_and_calls(run_dir: Path) -> None:
    write(run_dir, "note", {"node": "s-2", "text": "first"}, model=MODEL)
    state = get_state(run_dir)
    assert (state["calls"], state["finished"]) == (2, False)
    assert {"id": "s-2", "type": "unassigned", "text": "In total we analyse 90000 segments."} in state["nodes"]


def test_replay_rebuilds_the_graph_byte_identically(run_dir: Path, no_network: None) -> None:
    for i, node in enumerate(["s-1", "s-3", "x-9", "s-1"]):
        write(run_dir, "note", {"node": node, "text": f"note {i}"}, model=MODEL)
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir)


def test_replay_reads_a_stored_document(tmp_path: Path) -> None:
    start(tmp_path / "run", store_source=True)
    write(tmp_path / "run", "note", {"node": "s-1", "text": "n"}, model=MODEL)
    assert replay(tmp_path / "run") == recorded_graph(tmp_path / "run")


def test_replay_needs_the_document_the_run_read(run_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(RunError) as raised:
        replay(run_dir)
    assert raised.value.code == "missing-source"
    other = tmp_path / "other.tex"
    other.write_text(PAPER.read_text(encoding="utf-8") + "%\n", encoding="utf-8")
    with pytest.raises(RunError) as raised:
        replay(run_dir, source=other)
    assert raised.value.code == "source-mismatch"


def test_replay_refuses_an_edited_cited_copy(run_dir: Path) -> None:
    (run_dir / "cited" / "smith2020.txt").write_text("A forged passage.\n", encoding="utf-8")
    with pytest.raises(RunError, match=r"the cited copy 'smith2020\.txt' differs") as raised:
        replay(run_dir, source=PAPER)
    assert raised.value.code == "record-mismatch"


def test_replay_refuses_a_call_whose_record_was_edited(run_dir: Path) -> None:
    write(run_dir, "note", {"node": "s-1", "text": "kept"}, model=MODEL)
    path = run_dir / "calls.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"kept"', '"edited"'), encoding="utf-8")
    with pytest.raises(RunError, match=r"call 2 \(note\) was changed after it was logged") as raised:
        replay(run_dir, source=PAPER)
    assert raised.value.code == "record-mismatch"


def test_a_crash_after_the_log_line_is_recovered_by_rebuilding(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(run_dir, "note", {"node": "s-1", "text": "before"}, model=MODEL)
    original = Path.replace

    def crash(self: Path, target: Any) -> Path:
        if self.name == "state.json.tmp":
            raise OSError("simulated crash")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        write(run_dir, "note", {"node": "s-3", "text": "logged"}, model=MODEL)
    monkeypatch.undo()
    assert len(log(run_dir)) == 3
    assert read_json(run_dir / "state.json")["calls"] == 2
    notes = {node["id"]: node.get("note") for node in get_state(run_dir)["nodes"]}
    assert notes == dict.fromkeys(notes) | {}  # get_state reports no notes; check the graph instead
    graph = json.loads(recorded_graph(run_dir))
    assert {node["id"]: node["attributes"].get("note") for node in graph["nodes"]}["s-3"] == "logged"
    write(run_dir, "note", {"node": "s-4", "text": "after"}, model=MODEL)
    assert read_json(run_dir / "state.json")["calls"] == 4
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir)


def test_a_crash_before_the_log_line_leaves_no_trace(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = (run_dir / "calls.jsonl").read_bytes()

    def crash(*args: Any) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "fsync", crash)
    with pytest.raises(OSError, match="simulated crash"):
        write(run_dir, "note", {"node": "s-3", "text": "lost"}, model=MODEL)
    monkeypatch.undo()
    assert (run_dir / "calls.jsonl").read_bytes() == before
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir)


def test_a_held_lock_makes_another_call_wait_and_then_report_the_run_busy(run_dir: Path) -> None:
    with lock(run_dir), pytest.raises(RunError) as raised, lock(run_dir, timeout=0.2):
        pass
    assert raised.value.code == "run-busy"


WRITER = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import session_ops
from credencegraph_extract.session import write
session_ops.register()
run, name = Path(sys.argv[2]), sys.argv[3]
for i in range(15):
    write(run, "note", {"node": "s-%d" % (1 + i % 9), "text": "%s %d" % (name, i)}, model="m", agent=name)
"""


def test_several_processes_write_one_run_in_turn(run_dir: Path, no_network: None) -> None:
    tests = str(Path(__file__).parent)
    writers = [
        subprocess.Popen([sys.executable, "-c", WRITER, tests, str(run_dir), name])  # noqa: S603 - this interpreter
        for name in ("a", "b", "c")
    ]
    assert [writer.wait(timeout=120) for writer in writers] == [0, 0, 0]
    calls = log(run_dir)
    assert [call["seq"] for call in calls] == list(range(1, 47))
    assert sorted(call["agent"] for call in calls[1:]) == sorted(["a"] * 15 + ["b"] * 15 + ["c"] * 15)
    assert read_json(run_dir / "state.json")["calls"] == 46
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir)


def test_a_run_of_another_format_is_refused(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "checks" / "run", tmp_path / "old")
    with pytest.raises(RunError) as raised:
        session.read_manifest(tmp_path / "old")
    assert raised.value.code == "unsupported-version"


def test_the_command_replays_an_agent_tool_run(
    run_dir: Path, capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    write(run_dir, "note", {"node": "s-1", "text": "n"}, model=MODEL)
    assert main(["replay", str(run_dir), "--source", str(PAPER)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"replayed {run_dir}: the graph after 2 calls is byte-identical (sha256:")


def test_a_forged_state_file_is_not_taken_for_the_run(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(run_dir, "note", {"node": "s-1", "text": "n"}, model=MODEL)
    path = run_dir / "state.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"note": "n"', '"note": "forged"'), encoding="utf-8")
    assert '"forged"' not in recorded_graph(run_dir).decode("utf-8")
    assert main(["replay", str(run_dir), "--source", str(PAPER)]) == 0
    get_state(run_dir)
    assert '"forged"' not in path.read_text(encoding="utf-8")


def test_the_command_reports_an_edited_log_in_one_line(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(run_dir, "note", {"node": "s-1", "text": "kept"}, model=MODEL)
    path = run_dir / "calls.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"kept"', '"edited"'), encoding="utf-8")
    assert main(["replay", str(run_dir), "--source", str(PAPER)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error [record-mismatch]: call 2 (note) was changed after it was logged")
    assert err.count("\n") == 1


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        (lambda run: (run / "manifest.json").unlink(), "not-a-run"),
        (lambda run: (run / "manifest.json").write_text("{", encoding="utf-8"), "invalid-manifest"),
        (lambda run: edit(run / "manifest.json", '"credencegraph-extract-run"', '"another"'), "not-a-run"),
        (lambda run: (run / "calls.jsonl").unlink(), "incomplete-run"),
        (lambda run: (run / "calls.jsonl").write_text("{\n", encoding="utf-8"), "invalid-record"),
        (lambda run: edit(run / "calls.jsonl", '"seq": 1', '"seq": 7'), "invalid-record"),
        (lambda run: edit(run / "calls.jsonl", '"tool": "start_run"', '"tool": "note"'), "invalid-record"),
        (lambda run: (run / "rubric.toml").write_text("# edited\n", encoding="utf-8"), "record-mismatch"),
        (lambda run: (run / "ingested.json").write_text("{}", encoding="utf-8"), "invalid-record"),
        (lambda run: edit(run / "calls.jsonl", '"tool": "note"', '"tool": "gone"'), "unknown-tool"),
    ],
    ids=[
        "no-manifest",
        "manifest-not-json",
        "manifest-other-format",
        "no-log",
        "log-not-json",
        "log-misnumbered",
        "log-not-from-start",
        "rubric-edited",
        "ingested-without-graph",
        "unknown-tool-in-log",
    ],
)
def test_a_damaged_run_is_refused_with_its_code(run_dir: Path, damage: Any, code: str) -> None:
    write(run_dir, "note", {"node": "s-1", "text": "n"}, model=MODEL)
    (run_dir / "state.json").unlink()  # force a rebuild, which reads every file
    damage(run_dir)
    with pytest.raises(RunError) as raised:
        write(run_dir, "note", {"node": "s-2", "text": "m"}, model=MODEL)
    assert raised.value.code == code


@pytest.mark.parametrize(
    "document", ["/home/someone/paper.tex", "papers/paper.tex", "C:\\papers\\paper.tex", "C:paper.tex", "d:x", " "]
)
def test_a_document_id_that_is_a_path_is_refused(tmp_path: Path, document: str) -> None:
    with pytest.raises(RunError, match="must be an id for the document, not a path") as raised:
        start(tmp_path / "run", document=document)
    assert raised.value.code == "invalid-argument"
    assert not (tmp_path / "run").exists()


def test_the_graph_is_ingested_from_the_bytes_whose_digest_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file is changed while start_run ingests it and restored before it returns: the graph must come
    # from the bytes start_run read and digested, not from the file as it was during the parse.
    paper = tmp_path / "paper.tex"
    shutil.copyfile(PAPER, paper)
    original = paper.read_text(encoding="utf-8")
    ingest = session._ingest

    def change_during_ingest(source: Path, data: bytes, document: str, rubric: Any) -> Any:
        source.write_text(original.replace("90000 segments", "80000 segments"), encoding="utf-8")
        try:
            return ingest(source, data, document, rubric)
        finally:
            source.write_text(original, encoding="utf-8")

    monkeypatch.setattr(session, "_ingest", change_during_ingest)
    start_run(paper, RUBRIC, tmp_path / "run", harness="h", model=MODEL, store_source=True)
    monkeypatch.undo()
    ingested = (tmp_path / "run" / "ingested.json").read_text(encoding="utf-8")
    assert "90000 segments" in ingested
    assert "80000 segments" not in ingested
    assert replay(tmp_path / "run") == recorded_graph(tmp_path / "run")


def test_a_stored_document_is_the_bytes_whose_digest_is_recorded(tmp_path: Path) -> None:
    start(tmp_path / "run", store_source=True)
    stored = (tmp_path / "run" / "source" / "paper.tex").read_bytes()
    assert bytes_digest(stored) == read_json(tmp_path / "run" / "manifest.json")["source"]["digest"]


def test_a_write_after_finishing_marks_the_run_unfinished(run_dir: Path) -> None:
    write(run_dir, "finish_run", {}, model=MODEL)
    assert get_state(run_dir)["finished"] is True
    write(run_dir, "note", {"node": "s-1", "text": "after finishing"}, model=MODEL)
    assert get_state(run_dir)["finished"] is False
    assert read_json(run_dir / "manifest.json")["finished"] is False
    assert read_json(run_dir / "state.json")["finished"] is False


def test_a_forged_finished_flag_is_not_taken_for_the_run(run_dir: Path) -> None:
    for name in ("state.json", "manifest.json"):
        edit(run_dir / name, '"finished": false', '"finished": true')
    assert get_state(run_dir)["finished"] is False
    assert read_json(run_dir / "manifest.json")["finished"] is False


def test_a_start_that_fails_part_way_leaves_nothing_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = session._durable_write

    def crash_on_log(path: Path, data: bytes) -> None:
        if path.name == "calls.jsonl":
            raise OSError("simulated crash")
        durable(path, data)

    monkeypatch.setattr(session, "_durable_write", crash_on_log)
    with pytest.raises(OSError, match="simulated crash"):
        start(tmp_path / "run")
    monkeypatch.undo()
    assert sorted(path.name for path in tmp_path.iterdir()) == []
    start(tmp_path / "run")
    assert (tmp_path / "run" / "calls.jsonl").is_file()


def test_start_run_may_use_an_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    start(tmp_path / "run")
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_the_lock_file_is_never_empty(run_dir: Path) -> None:
    with lock(run_dir):
        pass
    assert (run_dir / ".lock").stat().st_size >= 1


def test_a_document_id_with_a_colon_that_is_not_a_drive_is_kept(tmp_path: Path) -> None:
    start(tmp_path / "run", document="arXiv:2101.01234")
    assert read_json(tmp_path / "run" / "manifest.json")["source"]["document"] == "arXiv:2101.01234"


def test_a_failure_while_moving_the_run_into_place_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Path, target: Any) -> Path:
        raise OSError("simulated failure")

    monkeypatch.setattr(Path, "rename", refuse)
    with pytest.raises(OSError, match="simulated failure"):
        start(tmp_path / "run")
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_cited_works_that_are_not_a_directory_are_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError) as raised:
        start_run(PAPER, RUBRIC, tmp_path / "run", harness="h", model=MODEL, cited=CITED / "smith2020.txt")
    assert raised.value.code == "invalid-cited"
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("paper.tex", b"\xff\xfe not utf-8", "invalid-source"),
        ("rubric.toml", b"this is = = not toml", "invalid-rubric"),
    ],
)
def test_a_document_or_rubric_that_cannot_be_read_is_refused(
    tmp_path: Path, name: str, content: bytes, code: str
) -> None:
    (tmp_path / name).write_bytes(content)
    source = tmp_path / "paper.tex" if name == "paper.tex" else PAPER
    rubric = tmp_path / "rubric.toml" if name == "rubric.toml" else RUBRIC
    with pytest.raises(RunError) as raised:
        start_run(source, rubric, tmp_path / "run", harness="h", model=MODEL)
    assert raised.value.code == code
    assert not (tmp_path / "run").exists()


def test_the_command_refuses_to_replay_a_run_of_the_earlier_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(FIXTURES / "checks" / "run", tmp_path / "old")
    assert main(["replay", str(tmp_path / "old")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error [unsupported-version]: ")
    assert "verify still runs its checks again" in err


def test_the_command_writes_the_replayed_graph(
    run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "rebuilt.json"
    assert main(["replay", str(run_dir), "--source", str(PAPER), "--output", str(output)]) == 0
    assert output.read_bytes() == recorded_graph(run_dir)
    assert main(["replay", str(run_dir), "--source", str(PAPER), "--output", str(tmp_path / "no" / "dir.json")]) == 2
    assert "error [io-error]" in capsys.readouterr().err
