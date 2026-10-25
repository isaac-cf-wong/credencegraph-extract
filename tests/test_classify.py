"""The classify pass on the invented fixture paper: records, the repair round, rejections, hedges and replay."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from classify_fixture import BODIES, FIRST, PAPER, REPAIR, RUBRIC, RUN, RUN_ID
from credencegraph.core import loads
from credencegraph.rubric import check_graph, load_rubric
from notes_pass import MODEL, fake_client, message

from credencegraph_extract import cli, provider
from credencegraph_extract.classify import ClassifyPass, output_schema, render
from credencegraph_extract.pipeline import PASSES
from credencegraph_extract.run import RunError, replay, run_extraction


def make_run(run_dir: Path, bodies: list[dict[str, Any]] = BODIES, rubric: Path = RUBRIC, passes: Any = PASSES) -> Any:
    client = fake_client(bodies)
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=client)
    run_extraction(PAPER, run_dir, rubric, call=call, passes=passes, run_id=RUN_ID)
    return client


# The tests that change the classify answers run the classify pass alone, so the premise pass's
# recorded answers need not match the graph they lead to.
CLASSIFY_ONLY = (ClassifyPass(),)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_pipeline_runs_the_classify_pass_first() -> None:
    assert [item.name for item in PASSES] == ["classify", "premises", "checks"]


def test_the_fixture_paper_passes_the_typed_check_with_rubric_credences(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
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


def test_the_run_reproduces_the_recorded_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    for name in (
        "graph.json",
        "rejections.json",
        "report.json",
        "raw/0001.json",
        "raw/0002.json",
        "raw/0003.json",
        "raw/0004.json",
        "prompts/0001.txt",
        "prompts/0002.txt",
        "prompts/0003.txt",
        "prompts/0004.txt",
        "prompts/classify.template.txt",
        "prompts/premises.template.txt",
        "prompts/checks.template.txt",
    ):
        assert (run_dir / name).read_bytes() == (RUN / name).read_bytes(), name


def test_rejections_of_the_first_round_are_kept_and_repaired(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    rejections = read_json(run_dir / "rejections.json")
    assert [(item["pass"], item["round"], item["node"], item["code"]) for item in rejections][:2] == [
        ("classify", "initial", "s-4", "invalid-argument"),
        ("classify", "initial", "s-5", "invalid-argument"),
    ]
    graph = read_json(run_dir / "graph.json")
    kinds = {node["id"]: node["kind"] for node in graph["nodes"] if node["attributes"]["origin"] == "document"}
    assert kinds == {
        "s-1": "method",
        "s-2": "compound",
        "s-2a": "result",
        "s-2b": "claim",
        "e-1": "derivation",
        "s-3": "claim",
        "s-4": "literature_claim",
        "s-5": "claim",
        "r-1": "reference",
    }
    stamps = {node["id"]: node["attributes"]["extraction"]["pass"] for node in graph["nodes"]}
    assert stamps["s-4"] == stamps["s-2a"] == "classify"
    assert stamps["r-1"] == "ingest"


def test_the_repair_call_is_recorded_like_any_other(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    client = make_run(run_dir)
    assert len(client.sent) == 4
    manifest = read_json(run_dir / "manifest.json")
    entry = manifest["passes"][0]
    assert entry["name"] == "classify"
    assert [call["raw"] for call in entry["calls"]] == ["raw/0001.json", "raw/0002.json"]
    assert read_json(run_dir / "raw" / "0002.json")["response"] == BODIES[1]
    repair_prompt = (run_dir / "prompts" / "0002.txt").read_text(encoding="utf-8")
    assert "'a bias below one percent' is not a substring of the text" in repair_prompt
    assert "s-2a (result): The bias vanishes for long segments" in repair_prompt


def test_no_repair_call_when_nothing_is_rejected(tmp_path: Path) -> None:
    records = [record for record in FIRST if record["node"] not in {"s-4", "s-5"}] + REPAIR
    client = make_run(tmp_path / "run", [message(1, json.dumps({"records": records}))], passes=CLASSIFY_ONLY)
    assert len(client.sent) == 1
    assert read_json(tmp_path / "run" / "rejections.json") == []


def test_what_the_repair_does_not_fix_stays_rejected(tmp_path: Path) -> None:
    unrepaired = {**REPAIR[0], "form": {"statement": "a bias of one percent"}}
    stray = {"node": "s-1", "type": "method", "form": {"kind": "procedure"}}
    bodies = [BODIES[0], message(2, json.dumps({"records": [unrepaired, REPAIR[1], stray]}))]
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies, passes=CLASSIFY_ONLY)
    rejections = read_json(run_dir / "rejections.json")
    assert [(item["round"], item["node"], item["code"]) for item in rejections] == [
        ("initial", "s-4", "invalid-argument"),
        ("initial", "s-5", "invalid-argument"),
        ("repair", "s-1", "not-rejected"),
        ("repair", "s-4", "invalid-argument"),
        ("repair", "s-5", "invalid-argument"),
    ]
    graph = read_json(run_dir / "graph.json")
    kinds = {node["id"]: node["kind"] for node in graph["nodes"]}
    assert kinds["s-4"] == kinds["s-5"] == "unassigned"
    assert kinds["s-1"] == "method"
    assert read_json(run_dir / "report.json")["coverage"]["untyped"] == ["s-4", "s-5"]


def test_the_report_lists_unmapped_hedges(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    report = read_json(run_dir / "report.json")
    assert report["passes"]["classify"] == {"hedge-unmapped": [{"node": "s-5", "phrase": "We expect"}]}
    graph = read_json(run_dir / "graph.json")
    stated = {node["id"]: node.get("stated") for node in graph["nodes"] if node.get("stated") is not None}
    assert stated == {"s-3": {"alpha": 6.0, "beta": 4.0}}


def test_a_rubric_with_a_supports_edge_rule_is_refused_before_any_call(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(
        RUBRIC.read_text(encoding="utf-8").replace('relation = "requires"', 'relation = "supports"'), encoding="utf-8"
    )
    client = fake_client([])
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=client)
    with pytest.raises(RunError) as caught:
        run_extraction(PAPER, tmp_path / "run", rubric, call=call, passes=PASSES)
    assert caught.value.code == "invalid-rubric"
    assert "['claim']" in str(caught.value)
    assert client.sent == []
    assert list((tmp_path / "run" / "raw").iterdir()) == []


def test_the_command_reports_a_supports_rubric_as_one_error_line(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(
        RUBRIC.read_text(encoding="utf-8").replace('relation = "requires"', 'relation = "supports"'), encoding="utf-8"
    )
    status = cli.main(["run", str(PAPER), str(tmp_path / "run"), "--rubric", str(rubric), "--model", MODEL])
    assert status == 2
    err = capsys.readouterr().err
    assert err.startswith("error [invalid-rubric]: ")
    assert err.count("\n") == 1


def test_an_answer_without_records_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RunError) as caught:
        make_run(tmp_path / "run", [message(1, json.dumps({"nodes": []}))])
    assert caught.value.code == "invalid-record"


def branches(schema: dict) -> dict[str, dict]:
    """The single-record branches of the schema, by the type each fixes."""
    *singles, _ = schema["properties"]["records"]["items"]["anyOf"]
    return {branch["properties"]["type"]["enum"][0]: branch for branch in singles}


def test_the_schema_has_no_credence_field() -> None:
    schema = output_schema(load_rubric(RUBRIC))
    text = json.dumps(schema)
    for word in ("base", "strength", "stated", "credence", "probability"):
        assert f'"{word}"' not in text
    split = schema["properties"]["records"]["items"]["anyOf"][-1]
    assert set(split["properties"]) == {"node", "split", "children"}
    assert list(branches(schema)) == ["reference", "method", "result", "derivation", "literature_claim", "claim"]
    for branch in branches(schema).values():
        assert set(branch["properties"]) <= {"node", "type", "form", "rests_on", "hedge"}
        assert branch["additionalProperties"] is False
        assert branch["properties"]["type"]["enum"] == [branch["properties"]["type"]["enum"][0]]


def test_only_a_type_with_an_edge_rule_may_rest_on_chunks() -> None:
    by_type = branches(output_schema(load_rubric(RUBRIC)))
    assert [name for name, branch in by_type.items() if "rests_on" in branch["properties"]] == ["claim"]


def test_each_type_has_only_its_own_form_fields_and_requires_its_required_ones() -> None:
    by_type = branches(output_schema(load_rubric(RUBRIC)))
    assert "form" not in by_type["reference"]["properties"]
    method = by_type["method"]["properties"]["form"]
    assert method["properties"] == {"kind": {"type": "string", "enum": ["simulation", "dataset", "procedure"]}}
    assert method["required"] == ["kind"]
    derivation = by_type["derivation"]["properties"]["form"]
    assert derivation["properties"] == {"equation": {"type": "string"}}
    assert "required" not in derivation
    assert by_type["derivation"]["required"] == ["node", "type"]
    result = by_type["result"]["properties"]["form"]
    assert set(result["properties"]) == {"quantity", "trend"}
    assert result["required"] == ["quantity"]
    assert result["additionalProperties"] is False
    assert by_type["result"]["required"] == ["node", "type", "form"]


def test_split_children_use_the_branches_without_node() -> None:
    schema = output_schema(load_rubric(RUBRIC))
    children = schema["properties"]["records"]["items"]["anyOf"][-1]["properties"]["children"]["items"]["anyOf"]
    for child, single in zip(children, branches(schema).values(), strict=True):
        assert child["properties"] == {k: v for k, v in single["properties"].items() if k != "node"}
        assert child["required"] == [k for k in single["required"] if k != "node"]


def test_the_repair_call_sends_each_refused_record_with_what_to_change(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir)
    prompt = (run_dir / "prompts" / "0002.txt").read_text(encoding="utf-8")
    sent = json.loads(prompt[prompt.index("and\nnothing else:\n") + len("and\nnothing else:\n") :])
    by_node = {item["record"]["node"]: item for item in sent}
    assert set(by_node) == {"s-4", "s-5"}
    assert by_node["s-4"]["fix"] == [
        (
            "form field 'statement': 'a bias below one percent' is not a substring of the text; "
            "copy it exactly from the chunk's text"
        )
    ]
    assert by_node["s-5"]["fix"] == [
        (
            "'s-4' in rests_on is of type 'unassigned', and type 'claim' rests only on "
            "['result', 'method', 'derivation', 'literature_claim']: type 's-4' as one of those in the same answer, "
            "or leave it out of rests_on"
        )
    ]
    first = (run_dir / "prompts" / "0001.txt").read_text(encoding="utf-8")
    assert first.endswith("and\nnothing else:\n[]\n")


def test_the_prompt_lists_types_hedges_and_chunks() -> None:
    from credencegraph.ingest.chunks import chunks_to_graph
    from credencegraph.ingest.latex import read_latex

    rubric = load_rubric(RUBRIC)
    graph = chunks_to_graph(read_latex(PAPER), "paper.tex", rubric.initial)[0]
    prompt = render(graph, rubric, [])
    assert "- claim: A conclusion the document asserts." in prompt
    assert "rests on at least 1 of: result, method, derivation, literature_claim" in prompt
    assert (
        "- derivation: A step of mathematics the document carries out.\n  form: equation (eqref)\n  rests on nothing"
        in prompt
    )
    assert "- reference: A bibliography entry.\n  form: no fields\n  rests on nothing" in prompt
    assert '"node" is always one of the chunk ids listed below' in prompt
    assert '"suggests", "may"' in prompt
    assert "s-4 (unassigned): Smith reports a bias below 1\\% in comparable settings~\\cite{smith2020}." in prompt
    assert prompt.endswith("nothing else:\n[]\n")
    assert (RUN / "prompts" / "classify.template.txt").read_text(encoding="utf-8") == ClassifyPass.template


def test_replay_of_the_recorded_run_is_byte_identical(tmp_path: Path, no_network: None) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    assert replay(run_dir, passes=PASSES, source=PAPER) == (run_dir / "graph.json").read_bytes()


def test_the_command_replays_the_recorded_run(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    assert cli.main(["replay", str(run_dir), "--source", str(PAPER)]) == 0
    assert "byte-identical" in capsys.readouterr().out


def test_replay_refuses_a_changed_repair_record(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    raw = run_dir / "raw" / "0002.json"
    raw.write_text(raw.read_text(encoding="utf-8").replace("comparable settings", "other settings"), encoding="utf-8")
    with pytest.raises(RunError) as caught:
        replay(run_dir, passes=PASSES, source=PAPER)
    assert caught.value.code == "record-mismatch"
