"""Runs: the run directory, its manifest, ``attributes.extraction`` stamping and ``--store-source``."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest
from conftest import FIXTURES
from notes_pass import BODIES, MODEL, NotesPass, fake_client

from credencegraph_extract import cli, provider
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.run import FORMAT, VERSION, RunError, replay, run_extraction

SOURCE = FIXTURES / "paper.tex"
RUBRIC = FIXTURES / "rubric.toml"
RUN_ID = "20261024T120000Z-fixture0"


def make_run(run_dir: Path, **kwargs: object) -> dict:
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=fake_client(BODIES))
    return run_extraction(SOURCE, run_dir, RUBRIC, call=call, passes=[NotesPass()], run_id=RUN_ID, **kwargs)


def test_run_writes_the_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    files = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())
    assert files == [
        "graph.json",
        "manifest.json",
        "prompts/0001.txt",
        "prompts/0002.txt",
        "prompts/0003.txt",
        "prompts/notes.template.txt",
        "raw/0001.json",
        "raw/0002.json",
        "raw/0003.json",
        "rubric.toml",
    ]
    assert (run_dir / "rubric.toml").read_bytes() == RUBRIC.read_bytes()
    assert (run_dir / "prompts" / "notes.template.txt").read_text(encoding="utf-8") == NotesPass.template


def test_run_reproduces_the_recorded_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    for name in ("graph.json", "raw/0001.json", "raw/0002.json", "raw/0003.json", "prompts/0002.txt"):
        assert (run_dir / name).read_bytes() == (FIXTURES / "run" / name).read_bytes(), name


def test_manifest_records_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = make_run(run_dir)
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["format"] == FORMAT == "credencegraph-extract-run"
    assert manifest["version"] == VERSION == 1
    assert manifest["run"] == RUN_ID
    assert manifest["source"] == {
        "document": "paper.tex",
        "file": "paper.tex",
        "digest": bytes_digest(SOURCE.read_bytes()),
        "stored": False,
    }
    assert set(manifest["versions"]) == {"python", "credencegraph", "credencegraph-extract", "anthropic"}
    assert manifest["rubric"] == {"name": "test-rubric", "version": "1", "digest": bytes_digest(RUBRIC.read_bytes())}
    assert manifest["graph_digest"] == bytes_digest((run_dir / "graph.json").read_bytes())
    [entry] = manifest["passes"]
    assert entry["name"] == "notes"
    assert entry["template_digest"] == bytes_digest(NotesPass.template.encode("utf-8"))
    assert len(entry["calls"]) == 3
    for call in entry["calls"]:
        assert call["prompt_digest"] == bytes_digest((run_dir / call["prompt"]).read_bytes())
        assert call["raw_digest"] == bytes_digest((run_dir / call["raw"]).read_bytes())
        assert call["provider"] == "anthropic"
        assert call["model"] == MODEL
        assert call["parameters"] == {"max_tokens": 1024}


def test_every_node_names_its_run_and_pass(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    stamps = {node["id"]: node["attributes"]["extraction"] for node in graph["nodes"]}
    assert stamps == {
        "s-1": {"run": RUN_ID, "pass": "notes"},
        "s-2": {"run": RUN_ID, "pass": "notes"},
        "e-1": {"run": RUN_ID, "pass": "ingest"},
        "s-3": {"run": RUN_ID, "pass": "notes"},
        "r-1": {"run": RUN_ID, "pass": "ingest"},
    }
    assert [relation["type"] for relation in graph["relations"]] == ["cites"]


def test_source_is_not_stored_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    assert not (run_dir / "source").exists()


def test_store_source_copies_the_document(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = make_run(run_dir, store_source=True)
    assert manifest["source"]["stored"] is True
    assert (run_dir / "source" / "paper.tex").read_bytes() == SOURCE.read_bytes()
    assert replay(run_dir, passes=[NotesPass()]) == (run_dir / "graph.json").read_bytes()


def test_document_id_is_recorded(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = make_run(run_dir, document="arXiv:0000.00000")
    assert manifest["source"]["document"] == "arXiv:0000.00000"
    graph = json.loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    assert {node["sources"][0]["document"] for node in graph["nodes"]} == {"arXiv:0000.00000"}
    assert replay(run_dir, source=SOURCE, passes=[NotesPass()]) == (run_dir / "graph.json").read_bytes()


def test_run_refuses_a_used_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("earlier work", encoding="utf-8")
    with pytest.raises(RunError) as caught:
        make_run(run_dir)
    assert caught.value.code == "run-exists"
    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "earlier work"


def test_run_refuses_a_source_that_is_not_latex(tmp_path: Path) -> None:
    other = tmp_path / "paper.pdf"
    other.write_bytes(b"%PDF-1.7")
    with pytest.raises(RunError) as caught:
        run_extraction(other, tmp_path / "run", RUBRIC, call=provider.call, passes=[])
    assert caught.value.code == "invalid-source"


def test_records_are_kept_when_a_response_is_unusable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bodies = [{**BODIES[0], "stop_reason": "refusal"}]
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=fake_client(bodies))
    with pytest.raises(provider.ProviderError):
        run_extraction(SOURCE, run_dir, RUBRIC, call=call, passes=[NotesPass()])
    assert json.loads((run_dir / "raw" / "0001.json").read_text(encoding="utf-8"))["response"] == bodies[0]


def test_cli_run_and_replay_without_passes(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    assert cli.main(["run", str(SOURCE), str(run_dir), "--rubric", str(RUBRIC), "--store-source"]) == 0
    assert (run_dir / "source" / "paper.tex").exists()
    assert cli.main(["replay", str(run_dir)]) == 0
    assert "byte-identical" in capsys.readouterr().out


def test_cli_run_needs_a_model_when_a_pass_calls_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("credencegraph_extract.cli.run_extraction", partial(run_extraction, passes=[NotesPass()]))
    assert cli.main(["run", str(SOURCE), str(tmp_path / "run"), "--rubric", str(RUBRIC)]) == 2
    assert "missing-model" in capsys.readouterr().err


def test_run_reports_a_missing_input_file(tmp_path: Path) -> None:
    main = tmp_path / "main.tex"
    main.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{missing}\n\\end{document}\n", encoding="utf-8"
    )
    with pytest.raises(RunError) as caught:
        run_extraction(main, tmp_path / "run", RUBRIC, call=provider.call, passes=[])
    assert caught.value.code == "invalid-source"


def test_run_refuses_an_unreadable_rubric(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.toml"
    rubric.write_text("[rubric]\nformat = 99\n", encoding="utf-8")
    with pytest.raises(RunError) as caught:
        run_extraction(SOURCE, tmp_path / "run", rubric, call=provider.call, passes=[])
    assert caught.value.code == "invalid-rubric"
    assert not (tmp_path / "run").exists()
