"""The check pass on the invented fixture paper: proposals, the four runners, check nodes, verdicts and replay."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from checks_fixture import (
    BODIES,
    CITED,
    CLASSIFY,
    JUDGEMENT,
    PAPER,
    PHASE,
    PLAN,
    PRODUCT,
    REPAIR,
    RESULTS,
    RUBRIC,
    RUN,
    RUN_ID,
    VERDICTS,
)
from credencegraph.core import Graph, Node, loads
from credencegraph.core.relation import OBSERVES
from credencegraph.inference import conditional, marginal
from credencegraph.rubric import Rubric, check_graph, load_rubric
from credencegraph.semantics.compiler import compile_graph
from notes_pass import MODEL, fake_client, message

from credencegraph_extract import checks, cited, evaluate, provider
from credencegraph_extract.checks import CheckPass, directory_digest, fix, output_schema, render
from credencegraph_extract.cli import main
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.pipeline import passes
from credencegraph_extract.run import RunError, replay, run_extraction


def make_run(  # noqa: PLR0913 - each argument varies one input of the run
    run_dir: Path,
    bodies: list[dict[str, Any]] = BODIES,
    rubric: Path = RUBRIC,
    *,
    cited_dir: Path | None = CITED,
    allow_uncalibrated: bool = False,
    paper: Path = PAPER,
) -> Any:
    client = fake_client(bodies)
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=client)
    resolve = cited.directory(cited_dir) if cited_dir else cited.no_sources
    pipeline = passes(cited=resolve, allow_uncalibrated=allow_uncalibrated)
    run_extraction(paper, run_dir, rubric, call=call, passes=pipeline, run_id=RUN_ID)
    return client


def bodies_with(
    plan: list[dict[str, Any]], *judgements: dict[str, Any], repair: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """The fixture's classify answer, then this plan, the answer to its repair call if one is made, and these judgements."""
    plans = [plan] if repair is None else [plan, repair]
    answers = [*(json.dumps({"checks": item}) for item in plans), *(json.dumps(judgement) for judgement in judgements)]
    return [BODIES[0], *(message(i, text) for i, text in enumerate(answers, start=2))]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_graph(run_dir: Path) -> Graph:
    return loads((run_dir / "graph.json").read_text(encoding="utf-8"))


def checks_of(graph: Graph) -> dict[str, Any]:
    return {node.id: node for node in graph if node.attributes.get("origin") == "evidence"}


def records(run_dir: Path) -> dict[str, Any]:
    """The check record of each check node, as plain JSON."""
    graph = read_json(run_dir / "graph.json")
    return {node["id"]: node["attributes"]["check"] for node in graph["nodes"] if "check" in node["attributes"]}


def checks_report(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "report.json")["passes"]["checks"]


def check_rejections(run_dir: Path) -> list[dict[str, Any]]:
    # "pass" is the field naming the pipeline stage, not a credential.
    return [item for item in read_json(run_dir / "rejections.json") if item["pass"] == "checks"]  # noqa: S105


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric(RUBRIC)


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh run of the pipeline on the fixture answers."""
    run_dir = tmp_path_factory.mktemp("made") / "run"
    make_run(run_dir)
    return run_dir


def edited_rubric(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "rubric.toml"
    text = RUBRIC.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


# The acceptance check: the assessed graph passes the core's strictest check.


def test_the_graph_passes_the_assessed_check_with_rubric_credences(made: Path, rubric: Rubric) -> None:
    assert check_graph(read_graph(made), rubric, "assessed", rubric_credences=True) == []
    result = subprocess.run(  # noqa: S603 - the core's console script, fixed arguments
        [
            str(Path(sys.executable).parent / "credencegraph"),
            "check",
            str(made / "graph.json"),
            "--rubric",
            str(RUBRIC),
            "--level",
            "assessed",
            "--rubric-credences",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok at level assessed with rubric credences" in result.stdout


def test_each_check_is_numbered_in_document_order_of_its_target(made: Path) -> None:
    graph = read_graph(made)
    found = [
        (node.id, node.kind, node.attributes["check"]["target"], node.attributes["check"]["result"])
        for node in checks_of(graph).values()
    ]
    assert found == RESULTS
    assert [
        (item["id"], item["type"], item["target"], item["result"]) for item in checks_report(made)["checks"]
    ] == RESULTS


def test_verdicts_are_derived_from_the_checks(made: Path) -> None:
    graph = read_graph(made)
    assessed = {node.id: dict(node.attributes["assessment"]) for node in graph if "assessment" in node.attributes}
    assert assessed == VERDICTS


def test_observed_is_set_for_a_pass_or_fail_and_absent_for_an_error(made: Path) -> None:
    for node in checks_of(read_graph(made)).values():
        result = node.attributes["check"]["result"]
        if result == "error":
            assert "observed" not in node.attributes, node.id
        else:
            assert node.attributes["observed"] is (result == "passed"), node.id


def test_each_check_is_an_evidence_node_joined_by_the_rubric_check_edge(made: Path, rubric: Rubric) -> None:
    graph = read_graph(made)
    for node in checks_of(graph).values():
        target = node.attributes["check"]["target"]
        assert node.base is None
        assert node.statement == f"{node.kind} on {target}: {node.attributes['check']['result']}"
        [edge] = [relation for relation in graph.relations.values() if relation.target == node.id]
        assert (edge.type, edge.source) == (OBSERVES, target)
        check = rubric.checks[node.kind]
        assert (edge.sensitivity, edge.false_pass) == (check.sensitivity, check.false_pass)
        assert node.attributes["extraction"] == {"run": RUN_ID, "pass": "checks"}


def observations(graph: Graph) -> dict[str, bool]:
    return {node.id: bool(node.attributes["observed"]) for node in graph if "observed" in node.attributes}


def test_an_errored_check_is_not_observed_and_changes_no_credence(made: Path) -> None:
    graph = read_graph(made)
    observed = observations(graph)
    assert set(observed) == {"c-001", "c-002", "c-003", "c-005", "c-006"}
    network = compile_graph(graph)
    assert conditional(network, "s-8", observed, draws=0).point == pytest.approx(
        marginal(network, "s-8", draws=0).point
    )


def test_a_failed_check_lowers_and_a_passed_one_raises_its_target(made: Path) -> None:
    graph = read_graph(made)
    network = compile_graph(graph)
    observed = observations(graph)
    for node_id, direction in (("s-2", -1), ("s-3", 1), ("e-3", 1), ("s-7", 1)):
        before = marginal(network, node_id, draws=0).point
        after = conditional(network, node_id, observed, draws=0).point
        assert (after - before) * direction > 0, node_id


def test_each_check_records_its_tool_model_and_log(made: Path) -> None:
    manifest = read_json(made / "manifest.json")
    versions = manifest["versions"]
    for node in checks_of(read_graph(made)).values():
        record = records(made)[node.id]
        tool = record["tool"]
        assert tool["name"] == "credencegraph-extract"
        assert tool["version"] == versions["credencegraph-extract"]
        assert tool["python"] == versions["python"]
        assert ("sympy" in tool) is node.kind.startswith("rederive-maths")
        if node.id == "c-006":
            assert record["model"] == {
                "provider": "anthropic",
                "model": MODEL,
                "request_id": "req_msg_fixture0004",
            }
        else:
            assert record["model"] is None
        log = record["log"]
        assert log["path"] == f"checks/{node.id}/"
        files = {path.name: path.read_bytes() for path in (made / log["path"]).iterdir()}
        assert set(files) == {"inputs.json", "output.json", "log.txt"}
        assert log["digest"] == directory_digest(files)
        assert json.loads(files["inputs.json"])["inputs"] == record["inputs"]
        assert json.loads(files["output.json"])["result"] == record["result"]


def test_recompute_records_the_expression_and_the_reported_value(made: Path) -> None:
    record = records(made)["c-002"]
    assert record["artifact"] == {"kind": "numbers", "nodes": ["s-3"]}
    assert record["inputs"] == {"expression": "400 - 400 * 0.125", "expected": "350"}
    assert record["detail"] == "400 - 400 * 0.125 = 350 (exact, Fraction); within 350 ± 0.5"
    assert records(made)["c-001"]["detail"].endswith("outside 90000 ± 0.5")


def test_the_symbolic_encoding_is_recorded_for_audit(made: Path) -> None:
    record = records(made)["c-003"]
    assert record["inputs"] == {key: PRODUCT[key] for key in ("lhs", "rhs", "symbols", "assumption")}
    assert record["artifact"] == {
        "kind": "equation",
        "nodes": ["e-1", "e-2"],
        "labels": ["eq:bias", "eq:product"],
    }


def test_the_numeric_check_records_its_seed_and_points_so_they_can_be_drawn_again(made: Path) -> None:
    inputs = records(made)["c-005"]["inputs"]
    encoding = {key: PHASE[key] for key in ("lhs", "rhs", "symbols", "assumption")}
    assert inputs["seed"] == evaluate.seed_of(encoding)
    assert len(inputs["points"]) == 32
    assert inputs["points"] == evaluate.sample(encoding["symbols"], inputs["seed"])
    assert inputs["rtol"] == "1E-12"


def test_the_numeric_check_runs_only_when_the_symbolic_one_errs(made: Path) -> None:
    kinds = {(node.attributes["check"]["target"], node.kind) for node in checks_of(read_graph(made)).values()}
    assert ("e-2", "rederive-maths-numeric") not in kinds
    assert ("e-3", "rederive-maths-numeric") in kinds


def test_no_numeric_check_without_its_check_type(tmp_path: Path) -> None:
    rubric = edited_rubric(tmp_path, "[checks.rederive-maths-numeric]", "[checks.unused-numeric]")
    run_dir = tmp_path / "run"
    make_run(run_dir, rubric=rubric)
    kinds = [item["type"] for item in checks_report(run_dir)["checks"]]
    assert "rederive-maths-numeric" not in kinds
    assert read_graph(run_dir).nodes["e-3"].attributes["assessment"] == {"verdict": "undetermined"}


def test_a_citation_check_anchors_the_passage_it_rests_on(made: Path) -> None:
    node = checks_of(read_graph(made))["c-006"]
    [anchor] = node.sources
    text = cited.directory(CITED)(read_graph(made).nodes["r-1"]).text
    start, end = (int(part) for part in anchor.locator.removeprefix("chars=").split("-"))  # type: ignore[union-attr]
    assert text[start:end] == anchor.quote == JUDGEMENT["passage"]
    assert node.attributes["check"]["artifact"]["quote"] == anchor.quote


def test_a_citation_check_without_the_cited_work_errs_without_a_judgement(made: Path) -> None:
    node = checks_of(read_graph(made))["c-007"]
    assert node.attributes["check"]["result"] == "error"
    assert "jones2021" in node.attributes["check"]["detail"]
    assert node.sources == ()
    # The plan, its repair, and one judgement: none for the check without its cited work.
    assert len(read_json(made / "manifest.json")["passes"][2]["calls"]) == 3


def test_a_passage_not_in_the_cited_work_is_an_error_not_a_fail(tmp_path: Path) -> None:
    plan = [{"target": "s-7", "type": "citation-support", "reference": "r-1"}]
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with(plan, {"passage": "The bias is always below 1%.", "supports": False}))
    [node] = checks_of(read_graph(run_dir)).values()
    assert node.attributes["check"]["result"] == "error"
    assert "observed" not in node.attributes
    assert read_graph(run_dir).nodes["s-7"].attributes["assessment"] == {"verdict": "undetermined"}


def test_a_located_passage_that_does_not_support_is_a_fail(tmp_path: Path) -> None:
    plan = [{"target": "s-7", "type": "citation-support", "reference": "r-1"}]
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with(plan, {**JUDGEMENT, "supports": False}))
    [node] = checks_of(read_graph(run_dir)).values()
    assert node.attributes["observed"] is False
    assert read_graph(run_dir).nodes["s-7"].attributes["assessment"] == {"verdict": "fails"}


def test_an_uncalibrated_model_is_refused_before_the_judgement(tmp_path: Path) -> None:
    rubric = edited_rubric(tmp_path, 'models = ["claude-fixture-20261024"]', 'models = ["another-model-20260101"]')
    with pytest.raises(RunError) as raised:
        make_run(tmp_path / "run", rubric=rubric)
    assert raised.value.code == "uncalibrated-model"
    # Classify, the plan and its repair: the refusal comes before the judgement call.
    assert len(list((tmp_path / "run" / "raw").iterdir())) == 3


def test_allow_uncalibrated_runs_the_check_and_reports_it(tmp_path: Path) -> None:
    rubric = edited_rubric(tmp_path, 'models = ["claude-fixture-20261024"]', 'models = ["another-model-20260101"]')
    run_dir = tmp_path / "run"
    make_run(run_dir, rubric=rubric, allow_uncalibrated=True)
    model = {"provider": "anthropic", "model": MODEL, "request_id": "req_msg_fixture0004"}
    assert checks_report(run_dir)["uncalibrated"] == [
        {"check": "c-006", "type": "citation-support", "model": model},
        {"check": "c-007", "type": "citation-support", "model": None},
    ]


def test_a_check_type_without_models_needs_no_calibration_binding(made: Path) -> None:
    assert checks_report(made)["uncalibrated"] == []


# Proposals the runner refuses.


def test_rejected_proposals_are_kept_with_their_codes(made: Path) -> None:
    assert [(item["round"], item["node"], item["code"]) for item in check_rejections(made)] == [
        ("initial", "s-2", "unfaithful-numeral"),
        ("initial", "s-1", "not-a-target"),
        ("repair", "s-2", "duplicate-record"),
        ("repair", "e-1", "not-rejected"),
    ]
    assert check_rejections(made)[0]["record"] == PLAN[6]
    assert [item["record"] for item in check_rejections(made)[2:]] == REPAIR


def test_refused_proposals_go_back_once_with_what_to_change(made: Path) -> None:
    prompt = (made / "prompts" / "0003.txt").read_text(encoding="utf-8")
    sent = json.loads(prompt[prompt.index("and nothing\nelse:\n") + len("and nothing\nelse:\n") :])
    assert [(item["code"], item["record"]) for item in sent] == [
        ("unfaithful-numeral", PLAN[6]),
        ("not-a-target", PLAN[7]),
    ]
    assert "a constant no cited chunk writes, such as the 1 of 1 - p or a 100, is not allowed" in sent[0]["fix"]
    assert "0.05" not in sent[0]["fix"]
    assert sent[1]["fix"] == "propose a check only on a target listed under Targets, of a check type listed for it"
    first = (made / "prompts" / "0002.txt").read_text(encoding="utf-8")
    assert first.endswith("and nothing\nelse:\n[]\n")


def test_a_corrected_proposal_is_run(tmp_path: Path) -> None:
    refused = {**RECOMPUTE, "expected": "350", "expression": "400 * (1 - 0.125)"}
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([refused], repair=[{**RECOMPUTE, "expected": "350"}]))
    assert [(item["round"], item["code"]) for item in check_rejections(run_dir)] == [("initial", "unfaithful-numeral")]
    assert [node.attributes["check"]["result"] for node in checks_of(read_graph(run_dir)).values()] == ["passed"]


def test_a_target_written_with_its_label_is_told_the_id_and_its_correction_runs(tmp_path: Path) -> None:
    # A small model copied the target as the prompt lists it, "s-3 (result)", and its corrected id was then
    # refused as a target with nothing refused.
    labelled = {**RECOMPUTE, "target": "s-3 (result)", "expected": "350"}
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([labelled], repair=[{**RECOMPUTE, "expected": "350"}]))
    assert [(item["round"], item["code"]) for item in check_rejections(run_dir)] == [("initial", "not-a-target")]
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["s-3"]
    prompt = (run_dir / "prompts" / "0003.txt").read_text(encoding="utf-8")
    assert "'s-3 (result)' is not a target id: write the id alone, 's-3', without the type in parentheses" in prompt


def test_chunk_ids_written_with_their_labels_are_told_the_ids() -> None:
    record = {**RECOMPUTE, "nodes": ["s-3 (result)", "s-2"]}
    assert fix("unknown-node", record) == (
        "['s-3 (result)'] are not chunk ids: write each id alone, as ['s-3'], without the type in parentheses"
    )
    assert fix("unknown-node", RECOMPUTE) == "use only chunk ids listed under Chunks"


def test_a_refused_unknown_target_does_not_open_the_repair_to_other_targets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([{**RECOMPUTE, "target": "ghost-99"}], repair=[{**RECOMPUTE, "expected": "350"}]))
    assert [(item["round"], item["node"], item["code"]) for item in check_rejections(run_dir)] == [
        ("initial", "ghost-99", "not-a-target"),
        ("repair", "s-3", "not-rejected"),
    ]
    assert checks_of(read_graph(run_dir)) == {}


@pytest.mark.parametrize("target", [["s-3"], {"id": "s-3"}])
def test_a_correction_whose_target_is_not_a_string_is_refused(tmp_path: Path, target: Any) -> None:
    # A list or dict cannot be looked up in the refused targets; it was raised, ending the run.
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([{**RECOMPUTE, "target": "ghost-99"}], repair=[{**RECOMPUTE, "target": target}]))
    assert [(item["round"], item["node"], item["code"]) for item in check_rejections(run_dir)] == [
        ("initial", "ghost-99", "not-a-target"),
        ("repair", None, "not-rejected"),
    ]


def test_a_labelled_and_an_unknown_target_admit_only_the_labelled_ones_correction(tmp_path: Path) -> None:
    refused = [{**RECOMPUTE, "target": "s-3 (result)", "expected": "350"}, {**PRODUCT, "target": "ghost-99"}]
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with(refused, repair=[{**RECOMPUTE, "expected": "350"}, PRODUCT]))
    assert [(item["round"], item["node"], item["code"]) for item in check_rejections(run_dir)][2:] == [
        ("repair", "e-2", "not-rejected")
    ]
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["s-3"]


@pytest.mark.parametrize("written", ["s-3 (result)", "s-3(result)", "s-3  (result)"])
def test_a_label_is_recognised_however_it_is_spaced(tmp_path: Path, written: str) -> None:
    run_dir = tmp_path / "run"
    make_run(
        run_dir,
        bodies_with([{**RECOMPUTE, "target": written, "expected": "350"}], repair=[{**RECOMPUTE, "expected": "350"}]),
    )
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["s-3"]


def test_a_correction_of_a_proposal_that_named_no_target_may_be_on_any_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([{"type": "recompute-numbers"}], repair=[{**RECOMPUTE, "expected": "350"}]))
    assert [item["round"] for item in check_rejections(run_dir)] == ["initial"]
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["s-3"]


@pytest.mark.parametrize(
    "again",
    [
        {**PRODUCT, "target": "e-1"},
        {**PRODUCT, "target": "e-1", "lhs": "c", "rhs": "N * c / N"},
    ],
    ids=["same encoding", "sides swapped and rewritten"],
)
def test_one_step_proposed_for_two_targets_is_one_check(tmp_path: Path, again: dict[str, Any]) -> None:
    # Checks that pass or fail together are one piece of evidence, not one per target.
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([PRODUCT, again], repair=[]))
    assert [(item["round"], item["node"], item["code"]) for item in check_rejections(run_dir)] == [
        ("initial", "e-1", "duplicate-record")
    ]
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["e-2"]
    prompt = (run_dir / "prompts" / "0003.txt").read_text(encoding="utf-8")
    assert "propose this check once, on the step's own target; 'e-2' already has it" in prompt


X = [{"name": "x", "assumptions": ["real"], "domain": [-2, 2]}]


@pytest.mark.parametrize(
    ("one", "other", "same"),
    [
        # Two true steps about the same polynomial, each written differently, are one step.
        (("(x + 1)**2", "x**2 + 2*x + 1"), ("x**2 + 2*x + 1", "x*(x + 2) + 1"), True),
        (("(x + 1)**2", "x**2 + 2*x + 1"), ("x**2 + 2*x + 1", "(x + 1)**2"), True),
        # A different claim about the same expression is a different step.
        (("(x + 1)**2", "x**2 + 2*x + 1"), ("(x + 1)**2", "x**2 + 1"), False),
        (("(x + 1)**2", "x**2 + 2*x + 1"), ("(x - 1)**2", "x**2 - 2*x + 1"), False),
        # A large power is compared as written rather than expanded.
        (("(x + 1)**40", "(x + 1)**40"), ("(1 + x)**40", "(x + 1)**40"), True),
    ],
)
def test_a_maths_step_is_the_same_step_however_its_sides_are_written(
    one: tuple[str, str], other: tuple[str, str], same: bool
) -> None:
    first = checks._step("rederive-maths", {"lhs": one[0], "rhs": one[1], "symbols": X})
    second = checks._step("rederive-maths", {"lhs": other[0], "rhs": other[1], "symbols": X})
    assert first is not None
    assert (first == second) is same


@pytest.mark.parametrize(
    ("lhs", "names"),
    [
        ("*".join(f"(x{i} + 1)" for i in range(28)), [f"x{i}" for i in range(28)]),
        ("(a + b + c + d + e + f + g + h)**32", list("abcdefgh")),
        ("((x + 1)**16 + 1)**16", ["x"]),
        ("sin(((x + 1)**100 + 1)**100)", ["x"]),
        ("1/((x + 1)**100 + 1)**100", ["x"]),
    ],
    ids=["28 binomials", "power of a sum", "nested powers", "inside a function", "negative power"],
)
def test_a_step_too_large_to_expand_is_keyed_quickly(lhs: str, names: list[str]) -> None:
    # Expanding any of these takes minutes or more; its key must not.
    symbols = [{"name": name, "assumptions": ["real"], "domain": [-1, 1]} for name in names]
    started = time.monotonic()
    assert checks._step("rederive-maths", {"lhs": lhs, "rhs": "1", "symbols": symbols}) is not None
    assert time.monotonic() - started < 1


def test_one_value_reported_with_another_rounding_is_another_step() -> None:
    # Whether an arithmetic check passes depends on the rounding the cited numeral implies: 350 is 350 +- 0.5,
    # 350.0 is 350 +- 0.05, so the same calculation checked against each is two steps, not one.
    graph = Graph()
    graph.add_node(Node("s-1", kind="result", statement="So 350 remain.", attributes={"origin": "document"}))
    graph.add_node(Node("s-2", kind="result", statement="So 350.0 remain.", attributes={"origin": "document"}))
    coarse, fine = checks._rounding(graph, ["s-1"], "350"), checks._rounding(graph, ["s-2"], "350")
    assert (coarse, fine) == (Fraction(1, 2), Fraction(1, 20))
    fields = {"expression": "400 - 50", "expected": "350"}
    assert checks._step("recompute-numbers", fields, coarse) != checks._step("recompute-numbers", fields, fine)
    assert checks._step("recompute-numbers", fields, coarse) == checks._step(
        "recompute-numbers", {"expression": "7 * 50", "expected": "350"}, coarse
    )


@pytest.mark.parametrize("artifact", ["malformed", None, ["s-1"], {"nodes": "s-1"}])
def test_a_recorded_check_of_another_shape_does_not_stop_admission(artifact: object) -> None:
    graph = Graph()
    graph.add_node(Node("s-1", kind="result", statement="So 350 remain.", attributes={"origin": "document"}))
    record = {"type": "recompute-numbers", "target": "s-1", "inputs": {"expression": "1 + 1", "expected": "2"}}
    graph.add_node(
        Node(
            "c-1", kind="recompute-numbers", statement="a check", attributes={"check": {**record, "artifact": artifact}}
        )
    )
    [(target, check_type, _)] = checks._admitted(graph, [])
    assert (target, check_type) == ("s-1", "recompute-numbers")


def test_one_value_written_two_ways_for_two_targets_is_one_check(tmp_path: Path) -> None:
    rewritten = {**RECOMPUTE, "target": "s-2", "expression": "400 - 0.125 * 400", "expected": "350"}
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([{**RECOMPUTE, "expected": "350"}, rewritten], repair=[]))
    assert [(item["node"], item["code"]) for item in check_rejections(run_dir)] == [("s-2", "duplicate-record")]
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["s-3"]


def test_a_citation_naming_the_key_is_told_the_id_and_its_correction_runs(tmp_path: Path) -> None:
    keyed = {"target": "s-7", "type": "citation-support", "reference": "smith2020"}
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([keyed], JUDGEMENT, repair=[{**keyed, "reference": "r-1"}]))
    assert [(item["round"], item["node"], item["code"]) for item in check_rejections(run_dir)] == [
        ("initial", "s-7", "invalid-record")
    ]
    prompt = (run_dir / "prompts" / "0003.txt").read_text(encoding="utf-8")
    # The prompt sends the fix as JSON, so its quotes are escaped there.
    assert json.dumps('\'smith2020\' is the key of r-1: give "reference": "r-1"')[1:-1] in prompt
    assert [node.attributes["check"]["result"] for node in checks_of(read_graph(run_dir)).values()] == ["passed"]


def test_no_repair_call_is_made_when_every_proposal_is_admitted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([{**RECOMPUTE, "expected": "350"}]))
    assert len(read_json(run_dir / "manifest.json")["passes"][2]["calls"]) == 1


RECOMPUTE = {"target": "s-3", "type": "recompute-numbers", "nodes": ["s-3"], "expression": "400 - 400 * 0.125"}


@pytest.mark.parametrize(
    ("proposal", "code"),
    [
        ({**RECOMPUTE, "expected": "349"}, "unfaithful-numeral"),
        ({**RECOMPUTE, "expected": "350", "expression": "400 - 400 * 0.25"}, "unfaithful-numeral"),
        ({**RECOMPUTE, "expected": "350", "nodes": ["s-1"]}, "unfaithful-numeral"),
        ({**RECOMPUTE, "expected": "350", "expression": "len('x')"}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350.0e0 + 0"}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "nodes": ["s-99"]}, "unknown-node"),
        ({**RECOMPUTE, "expected": "350", "nodes": ["c-1"]}, "unknown-node"),
        ({**RECOMPUTE, "expected": "350", "nodes": []}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "credence": 0.9}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "type": "rerun-code"}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "type": ["recompute-numbers"]}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "target": "s-9"}, "not-a-target"),
        ({**PRODUCT, "lhs": "__import__('os').getcwd()"}, "invalid-record"),
        ({**PRODUCT, "symbols": [{"name": "c", "assumptions": ["real"]}]}, "invalid-record"),
        ({**PRODUCT, "nodes": ["e-1", "s-5"]}, "invalid-record"),
        ({"target": "s-7", "type": "citation-support", "reference": "r-2"}, "invalid-record"),
        ({"target": "s-7", "type": "citation-support", "reference": "s-8"}, "invalid-record"),
        ("not an object", "invalid-record"),
    ],
)
def test_a_proposal_the_runner_cannot_run_is_rejected(tmp_path: Path, proposal: Any, code: str) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([proposal], repair=[]))
    assert [item["code"] for item in check_rejections(run_dir)] == [code]
    assert checks_of(read_graph(run_dir)) == {}


def test_a_second_check_of_a_type_on_one_node_is_rejected(tmp_path: Path) -> None:
    proposal = {**RECOMPUTE, "expected": "350"}
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([proposal, {**proposal, "nodes": ["s-1", "s-3"]}, PRODUCT], repair=[]))
    assert [(item["node"], item["code"]) for item in check_rejections(run_dir)] == [("s-3", "duplicate-record")]
    assert [node.attributes["check"]["target"] for node in checks_of(read_graph(run_dir)).values()] == ["s-3", "e-2"]


def run_on_paper(tmp_path: Path, old: str, new: str, proposal: dict[str, Any]) -> Path:
    """Run the fixture answers, with this one proposal, on the fixture paper with ``old`` replaced by ``new``."""
    paper = tmp_path / "paper.tex"
    text = PAPER.read_text(encoding="utf-8")
    assert old in text
    paper.write_text(text.replace(old, new), encoding="utf-8")
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([proposal], repair=[]), paper=paper)
    return run_dir


def test_an_operand_only_a_float_confuses_with_a_written_numeral_is_unfaithful(tmp_path: Path) -> None:
    # 0.10000000000000001 and 0.1 are one float but two decimals: the text does not contain 0.1.
    old = "Of the 400 realisations, 12.5\\% are rejected"
    new = "Of the 400 realisations, a share of 0.12500000000000001 is rejected"
    run_dir = run_on_paper(tmp_path, old, new, {**RECOMPUTE, "expected": "350"})
    assert [item["code"] for item in check_rejections(run_dir)] == ["unfaithful-numeral"]
    assert checks_of(read_graph(run_dir)) == {}


def test_a_reported_value_only_a_float_confuses_with_a_written_numeral_is_unfaithful(tmp_path: Path) -> None:
    run_dir = run_on_paper(
        tmp_path, "which leaves 350.", "which leaves 350.00000000000001.", {**RECOMPUTE, "expected": "350"}
    )
    assert [item["code"] for item in check_rejections(run_dir)] == ["unfaithful-numeral"]


def test_the_reported_value_and_its_rounding_are_the_exactly_written_numeral(tmp_path: Path) -> None:
    # The text writes both 3.5e2 (350 ± 5) and 350.00000000000001; only the second is the reported value.
    old = "which leaves 350."
    new = "which leaves 350.00000000000001, about 3.5e2."
    run_dir = run_on_paper(tmp_path, old, new, {**RECOMPUTE, "expected": "350.00000000000001"})
    [node] = checks_of(read_graph(run_dir)).values()
    assert node.attributes["check"]["result"] == "failed"
    assert node.attributes["check"]["detail"].endswith("outside 350.00000000000001 ± 0.000000000000005")


def test_the_tolerance_is_the_rounding_of_the_reported_numeral(tmp_path: Path) -> None:
    paper = tmp_path / "paper.tex"
    text = PAPER.read_text(encoding="utf-8").replace("which leaves 350.", "which leaves 349.6.")
    paper.write_text(text, encoding="utf-8")
    proposal = {**RECOMPUTE, "expression": "400 - 400 * 0.125", "expected": "349.6"}
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies_with([proposal], repair=[]), paper=paper)
    [node] = checks_of(read_graph(run_dir)).values()
    assert node.attributes["check"]["result"] == "failed"
    text = text.replace("which leaves 349.6.", "which leaves 349.96.")
    paper.write_text(text, encoding="utf-8")
    shutil.rmtree(run_dir)
    make_run(run_dir, bodies_with([{**proposal, "expected": "349.96"}]), paper=paper)
    [node] = checks_of(read_graph(run_dir)).values()
    assert node.attributes["check"]["result"] == "failed"
    text = text.replace("which leaves 349.96.", "which leaves 3.5e2.")
    paper.write_text(text, encoding="utf-8")
    shutil.rmtree(run_dir)
    make_run(run_dir, bodies_with([{**proposal, "expected": "3.5e2"}]), paper=paper)
    [node] = checks_of(read_graph(run_dir)).values()
    assert node.attributes["check"]["result"] == "passed"
    assert node.attributes["check"]["detail"].endswith("within 3.5e2 ± 5")


# Calls, prompts and the schema.


def test_no_call_without_a_node_to_assess(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC.read_text(encoding="utf-8").replace("assess = true\n", ""), encoding="utf-8")
    run_dir = tmp_path / "run"
    client = make_run(run_dir, rubric=rubric)
    assert len(client.sent) == 1
    assert read_json(run_dir / "manifest.json")["passes"][2] == {
        "name": "checks",
        "template_digest": bytes_digest(CheckPass.template.encode("utf-8")),
        "calls": [],
    }
    assert checks_of(read_graph(run_dir)) == {}


def test_no_call_when_no_check_type_applies(tmp_path: Path) -> None:
    rubric = edited_rubric(tmp_path, 'targets = ["result"]', 'targets = ["method"]')
    text = rubric.read_text(encoding="utf-8")
    for old in ('targets = ["derivation"]', 'targets = ["literature_claim"]'):
        text = text.replace(old, 'targets = ["method"]')
    rubric.write_text(text, encoding="utf-8")
    run_dir = tmp_path / "run"
    client = make_run(run_dir, rubric=rubric)
    assert len(client.sent) == 1
    reasons = {
        node.id: node.attributes["assessment"]["reason"]
        for node in read_graph(run_dir)
        if "assessment" in node.attributes
    }
    assert set(reasons.values()) == {"no v1 check applies"}


def test_the_plan_lists_targets_and_chunks_but_no_verdicts(rubric: Rubric, made: Path) -> None:
    prompt = (made / "prompts" / "0002.txt").read_text(encoding="utf-8")
    assert "s-2 (result): In total we analyse 90000 segments.\n  checks: recompute-numbers" in prompt
    assert "e-1 (derivation) [label eq:bias]:" in prompt
    assert "r-2 (reference) [key jones2021]:" in prompt
    assert "s-9 (claim)" not in prompt.split("Chunks of the document")[0]
    assert "s-1 (method)" not in prompt.split("Chunks of the document")[0]
    assert (made / "prompts" / "checks.template.txt").read_text(encoding="utf-8") == CheckPass.template


def test_the_judgement_prompt_carries_the_statement_and_the_cited_text(made: Path) -> None:
    prompt = (made / "prompts" / "0004.txt").read_text(encoding="utf-8")
    assert "Smith reports a bias below 1\\% in comparable settings" in prompt
    assert JUDGEMENT["passage"] in prompt


def test_the_schemas_have_no_credence_or_result_field(rubric: Rubric) -> None:
    text = json.dumps(output_schema(rubric))
    for word in ("credence", "base", "strength", "probability", "result", "verdict", "observed", "passed"):
        assert f'"{word}"' not in text
    branches = output_schema(rubric)["properties"]["checks"]["items"]["anyOf"]
    assert [branch["properties"]["type"]["enum"] for branch in branches] == [
        ["recompute-numbers"],
        ["rederive-maths"],
        ["citation-support"],
    ]


def test_the_plan_offers_only_the_check_types_the_rubric_declares(tmp_path: Path, made: Path) -> None:
    rubric = load_rubric(edited_rubric(tmp_path, "[checks.citation-support]", "[checks.other-judgement]"))
    branches = output_schema(rubric)["properties"]["checks"]["items"]["anyOf"]
    assert [branch["properties"]["type"]["enum"] for branch in branches] == [["recompute-numbers"], ["rederive-maths"]]
    targets = render(read_graph(made), rubric).split("Chunks of the document")[0]
    assert "s-7 (literature_claim)" not in targets
    assert "checks: citation-support" not in targets


def test_an_answer_without_checks_is_an_error(tmp_path: Path) -> None:
    bodies = [BODIES[0], message(2, json.dumps({"records": []}))]
    with pytest.raises(RunError) as raised:
        make_run(tmp_path / "run", bodies)
    assert raised.value.code == "invalid-record"


def test_the_classify_answer_types_every_chunk(made: Path) -> None:
    assert {record["node"] for record in CLASSIFY} | {"r-1", "r-2"} == {
        node.id for node in read_graph(made) if node.attributes.get("origin") == "document"
    }


# Replay.


def test_replay_of_the_recorded_run_is_byte_identical(tmp_path: Path, no_network: None) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    rebuilt = replay(run_dir, passes=passes(cited=cited.directory(CITED), allow_uncalibrated=True), source=PAPER)
    assert rebuilt == (run_dir / "graph.json").read_bytes()


def test_a_fresh_run_replays_byte_identically(made: Path, no_network: None) -> None:
    rebuilt = replay(made, passes=passes(cited=cited.directory(CITED)), source=PAPER)
    assert rebuilt == (made / "graph.json").read_bytes()


def test_the_recorded_check_directories_match_their_digests() -> None:
    for node in checks_of(loads((RUN / "graph.json").read_text(encoding="utf-8"))).values():
        log = node.attributes["check"]["log"]
        files = {path.name: path.read_bytes() for path in (RUN / log["path"]).iterdir()}
        assert log["digest"] == directory_digest(files), node.id


def test_the_command_replays_the_recorded_run(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    assert main(["replay", str(run_dir), "--source", str(PAPER), "--cited", str(CITED)]) == 0
    assert "byte-identical" in capsys.readouterr().out


def test_replay_without_the_cited_works_differs(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    assert main(["replay", str(run_dir), "--source", str(PAPER)]) == 2
    assert "error [unrecorded-call]" in capsys.readouterr().err


def test_replay_with_a_changed_cited_work_differs(
    tmp_path: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    changed = tmp_path / "cited"
    shutil.copytree(CITED, changed)
    path = changed / "smith2020.txt"
    path.write_text(path.read_text(encoding="utf-8") + "A sentence added later.\n", encoding="utf-8")
    assert main(["replay", str(run_dir), "--source", str(PAPER), "--cited", str(changed)]) == 2
    assert "error [unrecorded-call]" in capsys.readouterr().err


def test_the_command_runs_the_checks_from_the_cited_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = fake_client(BODIES)
    monkeypatch.setattr(provider.anthropic, "Anthropic", lambda: client)
    run_dir = tmp_path / "run"
    args = ["run", str(PAPER), str(run_dir), "--rubric", str(RUBRIC), "--model", MODEL, "--cited", str(CITED)]
    assert main(args) == 0
    assert [item["result"] for item in checks_report(run_dir)["checks"]] == [result for *_, result in RESULTS]


def test_fetch_arxiv_needs_a_cited_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", str(PAPER), str(tmp_path / "run"), "--rubric", str(RUBRIC), "--model", MODEL, "--fetch-arxiv"]
    assert main(args) == 2
    assert "error [invalid-argument]" in capsys.readouterr().err


def test_the_graph_does_not_depend_on_where_the_cited_works_are(tmp_path: Path, made: Path) -> None:
    elsewhere = tmp_path / "somewhere" / "else"
    shutil.copytree(CITED, elsewhere)
    rebuilt = replay(made, passes=passes(cited=cited.directory(elsewhere)), source=PAPER)
    assert rebuilt == (made / "graph.json").read_bytes()
    assert str(CITED) not in rebuilt.decode("utf-8")
