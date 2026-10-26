"""The checks on the invented fixture paper: what is accepted, the four runners, check nodes and verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from checks_fixture import CITED, CLASSIFY, PAPER, PASSAGE, PHASE, PLAN, PRODUCT, RESULTS, RUBRIC, RUN, VERDICTS
from credencegraph.core import Graph, Node, loads
from credencegraph.core.relation import OBSERVES
from credencegraph.inference import conditional, marginal
from credencegraph.rubric import Rubric, check_graph, load_rubric
from credencegraph.semantics.compiler import compile_graph

from credencegraph_extract import checks, evaluate, session
from credencegraph_extract.checks import directory_digest, fix
from credencegraph_extract.citation import text_of
from credencegraph_extract.session import start_run
from credencegraph_extract.tools import classify, finish_run, judge_citation, run_checks

MODEL = "test-model-1"
RECOMPUTE = {"target": "s-3", "type": "recompute-numbers", "nodes": ["s-3"], "expression": "400 - 400 * 0.125"}
# The checks of the fixture, in the order that numbers them as RESULTS lists them.
MATHS = [PLAN[2], PLAN[1], PRODUCT, PHASE]


def started(run_dir: Path, rubric: Path = RUBRIC, paper: Path = PAPER) -> Path:
    """A run of the fixture paper, classified."""
    start_run(paper, rubric, run_dir, harness="test-harness", model=MODEL, cited=CITED)
    assert classify(run_dir, CLASSIFY, model=MODEL)["refused"] == []
    return run_dir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def graph_of(run_dir: Path) -> Graph:
    return loads(session.recorded_graph(run_dir).decode("utf-8"))


def checks_of(graph: Graph) -> dict[str, Any]:
    return {node.id: node for node in graph if node.attributes.get("origin") == "evidence"}


def records(run_dir: Path) -> dict[str, Any]:
    """The check record of each check node, as plain JSON."""
    graph = json.loads(session.recorded_graph(run_dir))
    return {node["id"]: node["attributes"]["check"] for node in graph["nodes"] if "check" in node["attributes"]}


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric(RUBRIC)


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finished run of every fixture check: the maths through run_checks, the citations through judge_citation."""
    run_dir = started(tmp_path_factory.mktemp("made") / "run")
    assert run_checks(run_dir, MATHS, model=MODEL)["refused"] == []
    judge_citation(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE, supports=True, model=MODEL)
    # r-2 names no file among the cited works, so a source for it is not tied to the reference.
    judge_citation(run_dir, "s-8", "r-2", "smith2020.txt", PASSAGE, supports=True, model=MODEL)
    finish_run(run_dir, model=MODEL)
    return run_dir


def edited_rubric(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "rubric.toml"
    text = RUBRIC.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


# The acceptance check: the assessed graph passes the core's strictest check.


def test_the_graph_passes_the_assessed_check_with_rubric_credences(made: Path, rubric: Rubric) -> None:
    assert check_graph(graph_of(made), rubric, "assessed", rubric_credences=True) == []
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


def test_each_check_is_numbered_in_the_order_it_ran(made: Path) -> None:
    found = [
        (node.id, node.kind, node.attributes["check"]["target"], node.attributes["check"]["result"])
        for node in checks_of(graph_of(made)).values()
    ]
    assert found == RESULTS


def test_verdicts_are_derived_from_the_checks(made: Path) -> None:
    graph = graph_of(made)
    assessed = {node.id: dict(node.attributes["assessment"]) for node in graph if "assessment" in node.attributes}
    assert assessed == VERDICTS


def test_observed_is_set_for_a_pass_or_fail_and_absent_for_an_error(made: Path) -> None:
    for node in checks_of(graph_of(made)).values():
        result = node.attributes["check"]["result"]
        if result == "error":
            assert "observed" not in node.attributes, node.id
        else:
            assert node.attributes["observed"] is (result == "passed"), node.id


def test_each_check_is_an_evidence_node_joined_by_the_rubric_check_edge(made: Path, rubric: Rubric) -> None:
    graph = graph_of(made)
    run_id = read_json(made / "manifest.json")["run"]
    for node in checks_of(graph).values():
        target = node.attributes["check"]["target"]
        assert node.base is None
        assert node.statement == f"{node.kind} on {target}: {node.attributes['check']['result']}"
        [edge] = [relation for relation in graph.relations.values() if relation.target == node.id]
        assert (edge.type, edge.source) == (OBSERVES, target)
        check = rubric.checks[node.kind]
        assert (edge.sensitivity, edge.false_pass) == (check.sensitivity, check.false_pass)
        tool = "judge_citation" if node.kind == "citation-support" else "run_checks"
        assert node.attributes["extraction"] == {"run": run_id, "tool": tool}


def observations(graph: Graph) -> dict[str, bool]:
    return {node.id: bool(node.attributes["observed"]) for node in graph if "observed" in node.attributes}


def test_an_errored_check_is_not_observed_and_changes_no_credence(made: Path) -> None:
    graph = graph_of(made)
    observed = observations(graph)
    assert set(observed) == {"c-001", "c-002", "c-003", "c-005", "c-006"}
    network = compile_graph(graph)
    assert conditional(network, "s-8", observed, draws=0).point == pytest.approx(
        marginal(network, "s-8", draws=0).point
    )


def test_a_failed_check_lowers_and_a_passed_one_raises_its_target(made: Path) -> None:
    graph = graph_of(made)
    network = compile_graph(graph)
    observed = observations(graph)
    for node_id, direction in (("s-2", -1), ("s-3", 1), ("e-3", 1), ("s-7", 1)):
        before = marginal(network, node_id, draws=0).point
        after = conditional(network, node_id, observed, draws=0).point
        assert (after - before) * direction > 0, node_id


def test_each_check_records_its_tool_model_and_log(made: Path) -> None:
    versions = session.versions()
    for node in checks_of(graph_of(made)).values():
        record = records(made)[node.id]
        tool = record["tool"]
        assert tool["name"] == "credencegraph-extract"
        assert tool["version"] == versions["credencegraph-extract"]
        assert tool["python"] == versions["python"]
        assert ("sympy" in tool) is node.kind.startswith("rederive-maths")
        if node.kind == "citation-support":
            assert record["model"] == {"model": MODEL}
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
    kinds = {(node.attributes["check"]["target"], node.kind) for node in checks_of(graph_of(made)).values()}
    assert ("e-2", "rederive-maths-numeric") not in kinds
    assert ("e-3", "rederive-maths-numeric") in kinds


def test_no_numeric_check_without_its_check_type(tmp_path: Path) -> None:
    run_dir = started(
        tmp_path / "run", edited_rubric(tmp_path, "[checks.rederive-maths-numeric]", "[checks.unused-numeric]")
    )
    result = run_checks(run_dir, [PHASE], model=MODEL)
    assert [item["type"] for item in result["checks"]] == ["rederive-maths"]
    finish_run(run_dir, model=MODEL)
    assert graph_of(run_dir).nodes["e-3"].attributes["assessment"] == {"verdict": "undetermined"}


def test_a_citation_check_anchors_the_passage_it_rests_on(made: Path) -> None:
    node = checks_of(graph_of(made))["c-006"]
    [anchor] = node.sources
    text = text_of((CITED / "smith2020.txt").read_bytes(), ".txt")
    start, end = (int(part) for part in anchor.locator.removeprefix("chars=").split("-"))  # type: ignore[union-attr]
    assert text[start:end] == anchor.quote == PASSAGE
    assert node.attributes["check"]["artifact"]["quote"] == anchor.quote


def test_a_citation_check_whose_source_is_not_tied_errs_and_anchors_nothing(made: Path) -> None:
    node = checks_of(graph_of(made))["c-007"]
    assert node.attributes["check"]["result"] == "error"
    assert node.attributes["check"]["detail"] == (
        "source not tied to the reference: no file among the cited works is named by the reference's key"
    )
    assert node.sources == ()


def test_a_check_type_without_models_needs_no_calibration_binding(made: Path) -> None:
    # Only citation-support lists models in the fixture rubric; the maths checks are not reported.
    assert {item["type"] for item in read_json(made / "report.json")["uncalibrated"]} == {"citation-support"}


# Proposals the runner refuses.


@pytest.mark.parametrize("written", ["s-3 (result)", "s-3(result)", "s-3  (result)"])
def test_a_target_written_with_its_label_is_told_the_id(tmp_path: Path, written: str) -> None:
    # A small model copied the target as a listing showed it, with its type in parentheses.
    run_dir = started(tmp_path / "run")
    [refused] = run_checks(run_dir, [{**RECOMPUTE, "target": written, "expected": "350"}], model=MODEL)["refused"]
    assert refused["code"] == "not-a-target"
    assert refused["fix"] == [
        f"{written!r} is not a target id: write the id alone, 's-3', without the type in parentheses"
    ]


def test_chunk_ids_written_with_their_labels_are_told_the_ids() -> None:
    record = {**RECOMPUTE, "nodes": ["s-3 (result)", "s-2"]}
    assert fix("unknown-node", record) == (
        "['s-3 (result)'] are not chunk ids: write each id alone, as ['s-3'], without the type in parentheses"
    )
    assert fix("unknown-node", RECOMPUTE) == "use only chunk ids as start_run listed them"


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
    run_dir = started(tmp_path / "run")
    result = run_checks(run_dir, [PRODUCT, again], model=MODEL)
    assert [(item["node"], item["code"]) for item in result["refused"]] == [("e-1", "duplicate-record")]
    assert result["refused"][0]["fix"] == ["propose this check once, on the step's own target; 'e-2' already has it"]
    assert [item["target"] for item in result["checks"]] == ["e-2"]


def test_one_value_written_two_ways_for_two_targets_is_one_check(tmp_path: Path) -> None:
    rewritten = {**RECOMPUTE, "target": "s-2", "expression": "400 - 0.125 * 400", "expected": "350"}
    run_dir = started(tmp_path / "run")
    result = run_checks(run_dir, [{**RECOMPUTE, "expected": "350"}, rewritten], model=MODEL)
    assert [(item["node"], item["code"]) for item in result["refused"]] == [("s-2", "duplicate-record")]
    assert [item["target"] for item in result["checks"]] == ["s-3"]


@pytest.mark.parametrize(
    ("proposal", "code"),
    [
        ({**RECOMPUTE, "expected": "349"}, "unfaithful-numeral"),
        ({**RECOMPUTE, "expected": "350", "expression": "400 - 400 * 0.25"}, "unfaithful-numeral"),
        ({**RECOMPUTE, "expected": "350", "nodes": ["s-1"]}, "unfaithful-numeral"),
        ({**RECOMPUTE, "expected": "350", "expression": "len('x')"}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350.0e0 + 0"}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "nodes": ["s-99"]}, "unknown-node"),
        ({**RECOMPUTE, "expected": "350", "nodes": []}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "credence": 0.9}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "type": "rerun-code"}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "type": ["recompute-numbers"]}, "invalid-record"),
        ({**RECOMPUTE, "expected": "350", "target": "s-9"}, "not-a-target"),
        ({**PRODUCT, "lhs": "__import__('os').getcwd()"}, "invalid-record"),
        ({**PRODUCT, "symbols": [{"name": "c", "assumptions": ["real"]}]}, "invalid-record"),
        ({**PRODUCT, "nodes": ["e-1", "s-5"]}, "invalid-record"),
        ("not an object", "invalid-record"),
        ({**RECOMPUTE, "expression": 400, "expected": "350"}, "invalid-record"),
        ({**PRODUCT, "symbols": "c, N"}, "invalid-record"),
    ],
)
def test_a_proposal_the_runner_cannot_run_is_refused(tmp_path: Path, proposal: Any, code: str) -> None:
    run_dir = started(tmp_path / "run")
    result = run_checks(run_dir, [proposal], model=MODEL)
    assert [item["code"] for item in result["refused"]] == [code]
    assert result["checks"] == []


def test_a_proposal_naming_a_check_node_as_a_chunk_is_refused(made: Path, tmp_path: Path) -> None:
    run_dir = started(tmp_path / "run")
    run_checks(run_dir, [{**RECOMPUTE, "expected": "350"}], model=MODEL)
    [refused] = run_checks(run_dir, [{**PLAN[2], "nodes": ["c-001", "s-2"]}], model=MODEL)["refused"]
    assert refused["code"] == "invalid-record"


@pytest.mark.parametrize(
    ("target", "reference", "message"),
    [
        ("s-7", "r-2", "'s-7' does not cite 'r-2'"),
        ("s-7", "s-8", "'s-8' is not a bibliography entry of the document"),
    ],
)
def test_a_citation_the_runner_cannot_run_is_refused(tmp_path: Path, target: str, reference: str, message: str) -> None:
    run_dir = started(tmp_path / "run")
    result = judge_citation(run_dir, target, reference, "smith2020.txt", PASSAGE, supports=True, model=MODEL)
    [refused] = result["refused"]
    assert (refused["code"], refused["message"]) == ("invalid-record", message)


def test_a_second_check_of_a_type_on_one_node_is_refused(tmp_path: Path) -> None:
    proposal = {**RECOMPUTE, "expected": "350"}
    run_dir = started(tmp_path / "run")
    result = run_checks(run_dir, [proposal, {**proposal, "nodes": ["s-1", "s-3"]}, PRODUCT], model=MODEL)
    assert [(item["node"], item["code"]) for item in result["refused"]] == [("s-3", "duplicate-record")]
    assert [item["target"] for item in result["checks"]] == ["s-3", "e-2"]


def run_on_paper(tmp_path: Path, old: str, new: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Run this one proposal on the fixture paper with ``old`` replaced by ``new``."""
    paper = tmp_path / "paper.tex"
    text = PAPER.read_text(encoding="utf-8")
    assert old in text
    paper.write_text(text.replace(old, new), encoding="utf-8")
    run_dir = started(tmp_path / "run", paper=paper)
    return run_checks(run_dir, [proposal], model=MODEL)


def test_an_operand_only_a_float_confuses_with_a_written_numeral_is_unfaithful(tmp_path: Path) -> None:
    # 0.10000000000000001 and 0.1 are one float but two decimals: the text does not contain 0.1.
    old = "Of the 400 realisations, 12.5\\% are rejected"
    new = "Of the 400 realisations, a share of 0.12500000000000001 is rejected"
    result = run_on_paper(tmp_path, old, new, {**RECOMPUTE, "expected": "350"})
    assert [item["code"] for item in result["refused"]] == ["unfaithful-numeral"]
    assert result["checks"] == []


def test_a_reported_value_only_a_float_confuses_with_a_written_numeral_is_unfaithful(tmp_path: Path) -> None:
    result = run_on_paper(
        tmp_path, "which leaves 350.", "which leaves 350.00000000000001.", {**RECOMPUTE, "expected": "350"}
    )
    assert [item["code"] for item in result["refused"]] == ["unfaithful-numeral"]


def test_the_reported_value_and_its_rounding_are_the_exactly_written_numeral(tmp_path: Path) -> None:
    # The text writes both 3.5e2 (350 ± 5) and 350.00000000000001; only the second is the reported value.
    old = "which leaves 350."
    new = "which leaves 350.00000000000001, about 3.5e2."
    [check] = run_on_paper(tmp_path, old, new, {**RECOMPUTE, "expected": "350.00000000000001"})["checks"]
    assert check["result"] == "failed"
    assert check["detail"].endswith("outside 350.00000000000001 ± 0.000000000000005")


@pytest.mark.parametrize(
    ("written", "result", "detail"),
    [
        ("349.6", "failed", "outside 349.6 ± 0.05"),
        ("349.96", "failed", "outside 349.96 ± 0.005"),
        ("3.5e2", "passed", "within 3.5e2 ± 5"),
    ],
)
def test_the_tolerance_is_the_rounding_of_the_reported_numeral(
    tmp_path: Path, written: str, result: str, detail: str
) -> None:
    proposal = {**RECOMPUTE, "expression": "400 - 400 * 0.125", "expected": written}
    [check] = run_on_paper(tmp_path, "which leaves 350.", f"which leaves {written}.", proposal)["checks"]
    assert check["result"] == result
    assert check["detail"].endswith(detail)


def test_a_rubric_with_no_node_to_assess_refuses_every_proposal(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC.read_text(encoding="utf-8").replace("assess = true\n", ""), encoding="utf-8")
    run_dir = started(tmp_path / "run", rubric)
    result = run_checks(run_dir, [{**RECOMPUTE, "expected": "350"}, PRODUCT], model=MODEL)
    assert [item["code"] for item in result["refused"]] == ["not-a-target", "not-a-target"]


def test_a_node_no_check_type_applies_to_is_not_assessed_for_that_reason(tmp_path: Path) -> None:
    rubric = edited_rubric(tmp_path, 'targets = ["result"]', 'targets = ["method"]')
    text = rubric.read_text(encoding="utf-8")
    for old in ('targets = ["derivation"]', 'targets = ["literature_claim"]'):
        text = text.replace(old, 'targets = ["method"]')
    rubric.write_text(text, encoding="utf-8")
    run_dir = started(tmp_path / "run", rubric)
    finish_run(run_dir, model=MODEL)
    reasons = {
        node.id: node.attributes["assessment"]["reason"]
        for node in graph_of(run_dir)
        if "assessment" in node.attributes
    }
    assert set(reasons.values()) == {"no v1 check applies"}


def test_the_graph_does_not_depend_on_where_the_cited_works_are(tmp_path: Path) -> None:
    graphs = []
    for where in (CITED, tmp_path / "somewhere" / "else"):
        if where != CITED:
            where.parent.mkdir(parents=True)
            where.mkdir()
            for path in CITED.iterdir():
                (where / path.name).write_bytes(path.read_bytes())
        run_dir = tmp_path / f"run-{len(graphs)}"
        start_run(PAPER, RUBRIC, run_dir, harness="h", model=MODEL, cited=where)
        classify(run_dir, CLASSIFY, model=MODEL)
        judge_citation(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE, supports=True, model=MODEL)
        run_id = read_json(run_dir / "manifest.json")["run"]
        graphs.append(session.recorded_graph(run_dir).decode("utf-8").replace(run_id, "<run>"))
    assert graphs[0] == graphs[1]
    assert str(CITED) not in graphs[0]


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


RECORDED = {"type": "recompute-numbers", "target": "s-1", "inputs": {"expression": "1 + 1", "expected": "2"}}


def recorded_graph(**changes: object) -> Graph:
    graph = Graph()
    graph.add_node(Node("s-1", kind="result", statement="So 350 remain.", attributes={"origin": "document"}))
    record = {**RECORDED, **changes}
    graph.add_node(Node("c-1", kind="recompute-numbers", statement="a check", attributes={"check": record}))
    return graph


@pytest.mark.parametrize("artifact", ["malformed", None, ["s-1"], {"nodes": "s-1"}])
def test_a_recorded_check_with_an_artifact_of_another_shape_still_counts(artifact: object) -> None:
    [(target, check_type, _)] = checks._admitted(recorded_graph(artifact=artifact), [])
    assert (target, check_type) == ("s-1", "recompute-numbers")


@pytest.mark.parametrize(
    "changes",
    [{"type": {"bad": "type"}}, {"type": ["recompute-numbers"]}, {"target": ["s-1"]}, {"inputs": "1 + 1"}],
)
def test_a_recorded_check_of_another_shape_is_not_one_a_proposal_repeats(changes: dict[str, object]) -> None:
    assert checks._admitted(recorded_graph(**changes), []) == []


def test_cited_nodes_written_as_a_string_give_no_rounding() -> None:
    # A string is not a list of ids: "s1" must not be read as the nodes "s" and "1".
    graph = Graph()
    for node_id, text in (("s", "So 350 remain."), ("1", "So 350.0 remain."), ("s1", "So 350.00 remain.")):
        graph.add_node(Node(node_id, kind="result", statement=text, attributes={"origin": "document"}))
    assert checks._rounding(graph, "s1", "350") is None
    assert checks._rounding(graph, ["s1"], "350") == Fraction(1, 200)


# The recorded run of the earlier passes, which verify still reads.


def test_the_recorded_check_directories_match_their_digests() -> None:
    for node in checks_of(loads((RUN / "graph.json").read_text(encoding="utf-8"))).values():
        log = node.attributes["check"]["log"]
        files = {path.name: path.read_bytes() for path in (RUN / log["path"]).iterdir()}
        assert log["digest"] == directory_digest(files), node.id


@pytest.mark.parametrize(
    "fields",
    [
        {"lhs": "c +", "rhs": "c", "symbols": [{"name": "c", "assumptions": []}]},
        {"lhs": "c", "rhs": "c", "symbols": "c"},
        {"lhs": object(), "rhs": "c", "symbols": [{"name": "c", "assumptions": []}]},
    ],
    ids=["does not parse", "symbols not a list", "a value that is not JSON"],
)
def test_a_step_that_cannot_be_read_has_no_key(fields: dict[str, Any]) -> None:
    # Such a check is refused before it runs; one recorded so is not one a proposal can repeat.
    assert checks._step("rederive-maths", fields) is None


def test_the_rounding_of_a_value_the_cited_chunks_do_not_write_is_none() -> None:
    graph = Graph()
    graph.add_node(Node("s-1", kind="result", statement="So 350 remain.", attributes={"origin": "document"}))
    assert checks._rounding(graph, ["s-1"], "351") is None
    assert checks._rounding(graph, ["s-1"], "not a number") is None


def test_a_recorded_check_without_an_artifact_or_inputs_cannot_be_run_again(made: Path) -> None:
    graph = graph_of(made)
    record = {**dict(graph.nodes["c-002"].attributes["check"]), "artifact": "lost", "inputs": None}
    check = Node("c-099", kind="recompute-numbers", statement="a check", attributes={"check": record})
    rebuilt, _ = checks.rerun(graph, load_rubric(RUBRIC), check)
    assert rebuilt["result"] == "error"
    assert rebuilt["detail"].startswith("the recorded inputs cannot be run")
