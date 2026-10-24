"""``replay`` rebuilds ``graph.json`` byte-identically from ``raw/``, with no network."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FIXTURES
from notes_pass import NotesPass

from credencegraph_extract import cli
from credencegraph_extract.run import RunError, replay

SOURCE = FIXTURES / "paper.tex"


def sha256(data: bytes) -> str:
    """The hex SHA-256 of ``data``, computed here independently of the package's digest helper."""
    return hashlib.sha256(data).hexdigest()


def test_no_network_fixture_refuses_connections(no_network: None) -> None:
    with pytest.raises(AssertionError, match="network access"):
        socket.create_connection(("api.anthropic.com", 443))


def test_replay_is_byte_identical_offline(recorded_run: Path, no_network: None) -> None:
    rebuilt = replay(recorded_run, source=SOURCE, passes=[NotesPass()])
    recorded = (recorded_run / "graph.json").read_bytes()
    assert rebuilt == recorded
    assert sha256(rebuilt) == sha256(recorded)
    manifest = json.loads((recorded_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["graph_digest"] == f"sha256:{sha256(rebuilt)}"


def test_replay_uses_the_recorded_answers(recorded_run: Path, no_network: None) -> None:
    graph = json.loads(replay(recorded_run, source=SOURCE, passes=[NotesPass()]))
    notes = [node["attributes"]["note"] for node in graph["nodes"] if "note" in node["attributes"]]
    assert notes == [
        "States the sample size used for the noise floor.",
        "Cites prior work for the vanishing bias.",
        "Draws the conclusion from the bias result.",
    ]


def test_replay_reads_a_stored_source(recorded_run: Path, no_network: None) -> None:
    manifest_path = recorded_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["stored"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (recorded_run / "source").mkdir()
    shutil.copyfile(SOURCE, recorded_run / "source" / "paper.tex")
    assert replay(recorded_run, passes=[NotesPass()]) == (recorded_run / "graph.json").read_bytes()


def test_replay_needs_the_source_when_not_stored(recorded_run: Path) -> None:
    with pytest.raises(RunError) as caught:
        replay(recorded_run, passes=[NotesPass()])
    assert caught.value.code == "missing-source"


def test_replay_refuses_another_document(recorded_run: Path, tmp_path: Path) -> None:
    other = tmp_path / "paper.tex"
    other.write_text(SOURCE.read_text(encoding="utf-8").replace("250", "251"), encoding="utf-8")
    with pytest.raises(RunError) as caught:
        replay(recorded_run, source=other, passes=[NotesPass()])
    assert caught.value.code == "source-mismatch"


def test_replay_refuses_an_edited_record(recorded_run: Path) -> None:
    raw = recorded_run / "raw" / "0002.json"
    raw.write_text(raw.read_text(encoding="utf-8").replace("vanishing bias", "vanishing noise"), encoding="utf-8")
    with pytest.raises(RunError) as caught:
        replay(recorded_run, source=SOURCE, passes=[NotesPass()])
    assert caught.value.code == "record-mismatch"


def test_replay_refuses_an_edited_rubric(recorded_run: Path) -> None:
    rubric = recorded_run / "rubric.toml"
    rubric.write_text(rubric.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RunError) as caught:
        replay(recorded_run, source=SOURCE, passes=[NotesPass()])
    assert caught.value.code == "record-mismatch"


def test_replay_refuses_a_changed_prompt(recorded_run: Path) -> None:
    changed = NotesPass()
    changed.template = "Annotate this sentence.\n\n{text}\n"
    with pytest.raises(RunError) as caught:
        replay(recorded_run, source=SOURCE, passes=[changed])
    assert caught.value.code == "unrecorded-call"


def test_replay_refuses_a_different_pipeline(recorded_run: Path) -> None:
    with pytest.raises(RunError) as caught:
        replay(recorded_run, source=SOURCE, passes=[])
    assert caught.value.code == "unknown-pass"


def test_replay_refuses_unused_records(recorded_run: Path) -> None:
    manifest_path = recorded_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calls = manifest["passes"][0]["calls"]
    calls.append(calls[0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunError, match="fewer calls") as caught:
        replay(recorded_run, source=SOURCE, passes=[NotesPass()])
    assert caught.value.code == "unrecorded-call"


def test_replay_refuses_missing_records(recorded_run: Path) -> None:
    manifest_path = recorded_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["passes"][0]["calls"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunError, match="more calls") as caught:
        replay(recorded_run, source=SOURCE, passes=[NotesPass()])
    assert caught.value.code == "unrecorded-call"


def test_replay_refuses_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    with pytest.raises(RunError) as caught:
        replay(tmp_path)
    assert caught.value.code == "not-a-run"


def test_replay_refuses_another_manifest_version(recorded_run: Path) -> None:
    manifest_path = recorded_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunError) as caught:
        replay(recorded_run, source=SOURCE, passes=[NotesPass()])
    assert caught.value.code == "unsupported-version"


def test_cli_replay_reports_byte_identity(
    recorded_run: Path, no_network: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "credencegraph_extract.cli.replay",
        lambda run_dir, source: replay(run_dir, source=source, passes=[NotesPass()]),
    )
    output = recorded_run.parent / "rebuilt.json"
    status = cli.main(["replay", str(recorded_run), "--source", str(SOURCE), "--output", str(output)])
    assert status == 0
    assert "byte-identical" in capsys.readouterr().out
    assert output.read_bytes() == (recorded_run / "graph.json").read_bytes()


def test_cli_replay_reports_a_difference(
    recorded_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "credencegraph_extract.cli.replay",
        lambda run_dir, source: replay(run_dir, source=source, passes=[NotesPass()]),
    )
    graph = recorded_run / "graph.json"
    graph.write_bytes(graph.read_bytes().replace(b"vanishing bias", b"vanishing noise"))
    assert cli.main(["replay", str(recorded_run), "--source", str(SOURCE)]) == 1
    assert "differs" in capsys.readouterr().out


def test_cli_replay_reports_an_error(recorded_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["replay", str(recorded_run)]) == 2
    assert "missing-source" in capsys.readouterr().err


@pytest.fixture
def notes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the command replay with the test pass, as the recorded run was made."""
    monkeypatch.setattr(
        "credencegraph_extract.cli.replay",
        lambda run_dir, source: replay(run_dir, source=source, passes=[NotesPass()]),
    )


def replay_status(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run ``replay`` through the command; return its status and its stderr."""
    status = cli.main(["replay", str(run_dir), "--source", str(SOURCE)])
    return status, capsys.readouterr().err


def edit_manifest(run_dir: Path, edit: object) -> None:
    """Apply ``edit`` to the parsed manifest and write it back."""
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)  # type: ignore[operator]
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_cli_replay_of_a_malformed_manifest_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    (recorded_run / "manifest.json").write_text("{", encoding="utf-8")
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith("error [invalid-manifest]: ")
    assert err.count("\n") == 1


def test_cli_replay_of_a_missing_record_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    (recorded_run / "raw" / "0002.json").unlink()
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith("error [incomplete-run]: cannot read raw/0002.json")
    assert err.count("\n") == 1


@pytest.mark.parametrize("name", ["graph.json", "rubric.toml"])
def test_cli_replay_of_a_missing_file_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    (recorded_run / name).unlink()
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith(f"error [incomplete-run]: cannot read {name}")


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.pop("source"),
        lambda m: m["source"].update(stored="yes"),
        lambda m: m["source"].update(file="../paper.tex"),
        lambda m: m.update(passes={}),
        lambda m: m["passes"][0]["calls"][0].pop("raw_digest"),
        lambda m: m["passes"][0]["calls"][0].update(raw="../manifest.json"),
        lambda m: m["rubric"].update(digest=None),
    ],
    ids=[
        "no-source",
        "stored-not-bool",
        "file-escapes",
        "passes-not-list",
        "no-raw-digest",
        "raw-escapes",
        "no-rubric-digest",
    ],
)
def test_cli_replay_of_an_incomplete_manifest_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str], edit: object
) -> None:
    edit_manifest(recorded_run, edit)
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith("error [invalid-manifest]: ")


def test_cli_replay_of_a_record_that_is_not_json_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = recorded_run / "raw" / "0001.json"
    raw.write_text("not json", encoding="utf-8")
    edit_manifest(recorded_run, lambda m: m["passes"][0]["calls"][0].update(raw_digest=f"sha256:{sha256(b'not json')}"))
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith("error [invalid-record]: raw/0001.json is not JSON")


@pytest.mark.parametrize(
    "record",
    [[], {"request": {}}, {"request": None}],
    ids=["not-an-object", "no-metadata", "metadata-not-an-object"],
)
def test_cli_replay_of_a_record_without_metadata_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str], record: object
) -> None:
    data = json.dumps(record).encode("utf-8")
    (recorded_run / "raw" / "0001.json").write_bytes(data)
    edit_manifest(recorded_run, lambda m: m["passes"][0]["calls"][0].update(raw_digest=f"sha256:{sha256(data)}"))
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith("error [invalid-record]: ")


def test_cli_replay_of_a_malformed_response_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    path = recorded_run / "raw" / "0001.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["response"]["content"] = "not blocks"
    data = json.dumps(record).encode("utf-8")
    path.write_bytes(data)
    edit_manifest(recorded_run, lambda m: m["passes"][0]["calls"][0].update(raw_digest=f"sha256:{sha256(data)}"))
    status, err = replay_status(recorded_run, capsys)
    assert status == 2
    assert err.startswith("error [invalid-record]: raw/0001.json: ")


def test_cli_replay_of_an_unreadable_source_is_an_error(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    status = cli.main(["replay", str(recorded_run), "--source", str(recorded_run)])
    assert status == 2
    assert capsys.readouterr().err.startswith("error [missing-source]: ")


def test_cli_replay_cannot_write_its_output(
    recorded_run: Path, notes_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    output = recorded_run / "no-such-dir" / "graph.json"
    status = cli.main(["replay", str(recorded_run), "--source", str(SOURCE), "--output", str(output)])
    assert status == 2
    assert capsys.readouterr().err.startswith("error [io-error]: ")


def test_console_script_reports_a_malformed_manifest_without_a_traceback(recorded_run: Path) -> None:
    (recorded_run / "manifest.json").write_text("{", encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - fixed interpreter and arguments
        [sys.executable, "-c", "from credencegraph_extract.cli import app; app()", "replay", str(recorded_run)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert result.stderr.startswith("error [invalid-manifest]: ")
