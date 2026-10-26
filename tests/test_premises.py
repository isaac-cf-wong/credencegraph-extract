"""Premises on the invented fixture paper: links, analyst nodes, the cap, merging, ids and credences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from classify_fixture import FIRST, PAPER, PREMISES_FIRST, PREMISES_REPAIR, REPAIR, RUBRIC, STATIONARY
from credencegraph.core import Graph, loads
from credencegraph.core.relation import REQUIRES
from credencegraph.diagnostics import unanchored
from credencegraph.inference import marginal
from credencegraph.ingest.chunks import chunks_to_graph
from credencegraph.ingest.latex import read_latex
from credencegraph.rubric import Rubric, check_graph, load_rubric, loads_rubric
from credencegraph.semantics.compiler import compile_graph

from credencegraph_extract import session
from credencegraph_extract.assemble import assemble, assemble_premises
from credencegraph_extract.tools import add_premises, classify, finish_run

MODEL = "test-model-1"

ANALYSTS = {
    "a-1": "The segments are long compared with the noise correlation time.",
    "a-2": STATIONARY,
    "a-3": "The settings of Smith match ours.",
}


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric(RUBRIC)


@pytest.fixture
def classified(rubric: Rubric) -> Graph:
    """The fixture paper after the classify pass, both rounds."""
    graph = chunks_to_graph(read_latex(PAPER), "paper.tex", rubric.initial)[0]
    graph = assemble(graph, rubric, FIRST).graph
    return assemble(graph, rubric, REPAIR).graph


def make_run(run_dir: Path) -> dict[str, Any]:
    """Classify the fixture paper and add its premises through the tools; return the first premise call's result."""
    session.start_run(PAPER, RUBRIC, run_dir, harness="test-harness", model=MODEL)
    classify(run_dir, FIRST, model=MODEL)
    classify(run_dir, REPAIR, model=MODEL)
    first = add_premises(run_dir, PREMISES_FIRST, model=MODEL)
    add_premises(run_dir, PREMISES_REPAIR, model=MODEL)
    finish_run(run_dir, model=MODEL)
    return first


@pytest.fixture(scope="module")
def made_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """A run of the fixture records through the tools, and its first premise call's result."""
    run_dir = tmp_path_factory.mktemp("made") / "run"
    return run_dir, make_run(run_dir)


@pytest.fixture
def made(made_run: tuple[Path, dict[str, Any]]) -> Graph:
    """The graph of that run."""
    return loads((made_run[0] / "graph.json").read_text(encoding="utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def analyst_parents(graph: Graph, conclusion: str) -> list[str]:
    return [
        relation.source
        for relation in graph.relations.values()
        if relation.type == REQUIRES
        and relation.target == conclusion
        and graph.nodes[relation.source].attributes.get("origin") == "analyst"
    ]


def points(graph: Graph) -> dict[str, float]:
    network = compile_graph(graph)
    return {node.id: marginal(network, node.id, draws=0).point for node in graph if node.base is not None}


# The acceptance checks.


def test_the_analyst_nodes_are_reported_unanchored(made: Graph) -> None:
    graph = made
    analysts = {node.id for node in graph if node.attributes.get("origin") == "analyst"}
    assert analysts == set(ANALYSTS)
    flagged = {node_id for finding in unanchored(graph) for node_id in finding.nodes}
    assert flagged == analysts


@pytest.mark.parametrize(
    ("base", "strength"),
    [("beta:2,2", "beta:8,2"), ("beta:99,1", "beta:99,1"), ("beta:1,99", "beta:1,1"), ("0.999", "1")],
    ids=["fixture", "confident-premise", "doubtful-premise", "near-certain-point"],
)
def test_adding_premises_never_raises_a_credence(rubric: Rubric, classified: Graph, base: str, strength: str) -> None:
    text = RUBRIC.read_text(encoding="utf-8").replace('base = "beta:2,2"', f'base = "{base}"')
    text = text.replace('strength = "beta:8,2"', f'strength = "{strength}"')
    varied = loads_rubric(text)
    assert varied.premises is not None
    before = points(classified)
    after = points(
        assemble_premises(assemble_premises(classified, varied, PREMISES_FIRST).graph, varied, PREMISES_REPAIR).graph
    )
    assert set(before) <= set(after)
    for node_id, credence in before.items():
        assert after[node_id] <= credence + 1e-12, node_id
    # The conclusions that gained a premise are strictly lower.
    for node_id in ("s-2b", "s-3", "s-5"):
        assert after[node_id] < before[node_id], node_id


# What the premises add.


def test_a_stated_at_link_reaches_a_chunk_the_classify_records_did_not_link(
    rubric: Rubric, classified: Graph, made: Graph
) -> None:
    assert "s-1-requires-s-2b" not in classified.relations
    graph = made
    relation = graph.relations["s-1-requires-s-2b"]
    rule = rubric.types["claim"].rests_on
    assert rule is not None
    assert relation.type == REQUIRES
    assert relation.strength == rule.strength


def test_an_unstated_premise_is_an_analyst_assumption_with_the_rubric_values(rubric: Rubric, made: Graph) -> None:
    graph = made
    assert rubric.premises is not None
    for node_id, statement in ANALYSTS.items():
        node = graph.nodes[node_id]
        assert node.kind == "assumption"
        assert node.statement == statement
        assert node.sources == ()
        assert node.base == rubric.premises.base
        assert node.attributes["origin"] == "analyst"
        assert node.attributes["extraction"]["tool"] == "add_premises"
    for relation_id in ("a-1-requires-s-2b", "a-2-requires-s-2b", "a-2-requires-s-3", "a-3-requires-s-5"):
        assert graph.relations[relation_id].strength == rubric.premises.strength


def test_the_graph_passes_the_typed_check_with_rubric_credences(rubric: Rubric, made: Graph) -> None:
    assert check_graph(made, rubric, "typed", rubric_credences=True) == []


def test_identical_premises_are_one_node_with_several_requires(made: Graph) -> None:
    graph = made
    stationary = [node.id for node in graph if node.statement == STATIONARY]
    assert stationary == ["a-2"]
    targets = [relation.target for relation in graph.relations.values() if relation.source == "a-2"]
    assert targets == ["s-2b", "s-3"]


def test_overflow_beyond_the_cap_is_reported_not_added(made_run: tuple[Path, dict[str, Any]], made: Graph) -> None:
    assert made_run[1]["overflow"] == [{"conclusion": "s-2b", "statement": "The segments are independent."}]
    graph = made
    assert analyst_parents(graph, "s-2b") == ["a-1", "a-2"]
    assert all(node.statement != "The segments are independent." for node in graph)


def test_the_cap_counts_the_premises_a_conclusion_already_has(rubric: Rubric, classified: Graph) -> None:
    graph = assemble_premises(classified, rubric, PREMISES_FIRST).graph
    later = [{"conclusion": "s-2b", "premises": [{"statement": "The estimator is unbiased."}]}]
    result = assemble_premises(graph, rubric, later)
    assert result.rejections == []
    assert result.overflow == [{"conclusion": "s-2b", "statement": "The estimator is unbiased."}]
    assert result.graph == graph


def test_a_merged_premise_counts_towards_the_cap(rubric: Rubric, classified: Graph) -> None:
    # s-3 has room for two; the first is a node s-2b already uses, the third statement overflows.
    records = [
        PREMISES_FIRST[1],
        {"conclusion": "s-3", "premises": [{"statement": STATIONARY}, {"statement": "X."}, {"statement": "Y."}]},
    ]
    result = assemble_premises(classified, rubric, records)
    assert analyst_parents(result.graph, "s-3") == ["a-2", "a-3"]
    assert result.graph.nodes["a-3"].statement == "X."
    assert result.overflow[-1] == {"conclusion": "s-3", "statement": "Y."}


def test_a_statement_repeated_in_one_record_is_one_premise(rubric: Rubric, classified: Graph) -> None:
    records = [{"conclusion": "s-3", "premises": [{"statement": "X."}, {"statement": " X. "}]}]
    result = assemble_premises(classified, rubric, records)
    assert result.rejections == []
    assert analyst_parents(result.graph, "s-3") == ["a-1"]


# Ids.


def test_a_rerun_gives_the_analyst_nodes_the_same_ids(tmp_path: Path) -> None:
    graphs = []
    for name in ("one", "two"):
        make_run(tmp_path / name)
        run_id = read_json(tmp_path / name / "manifest.json")["run"]
        graphs.append((tmp_path / name / "graph.json").read_text(encoding="utf-8").replace(run_id, "<run>"))
    assert graphs[0] == graphs[1]
    graph = loads(graphs[0])
    assert [node.id for node in graph if node.attributes.get("origin") == "analyst"] == ["a-1", "a-2", "a-3"]


def test_ids_follow_document_order_not_answer_order(rubric: Rubric, classified: Graph) -> None:
    records = [
        {"conclusion": "s-3", "premises": [{"statement": "Later in the document."}]},
        {"conclusion": "s-2b", "premises": [{"statement": "Earlier in the document."}]},
    ]
    graph = assemble_premises(classified, rubric, records).graph
    assert graph.nodes["a-1"].statement == "Earlier in the document."
    assert graph.nodes["a-2"].statement == "Later in the document."


def test_ids_skip_one_the_graph_already_has(rubric: Rubric, classified: Graph) -> None:
    first = assemble_premises(classified, rubric, [{"conclusion": "s-3", "premises": [{"statement": "X."}]}]).graph
    record = {"conclusion": "s-2b", "premises": [{"statement": "Y."}]}
    graph = assemble_premises(first, rubric, [record]).graph
    assert graph.nodes["a-2"].statement == "Y."


@pytest.mark.parametrize(
    ("record", "code"),
    [
        ({"conclusion": "s-1", "premises": []}, "invalid-record"),
        ({"conclusion": "s-9", "premises": []}, "unknown-node"),
        ({"premises": []}, "invalid-record"),
        ({"conclusion": "s-3", "premises": {}}, "invalid-record"),
        ({"conclusion": "s-3", "premises": [], "base": 0.9}, "invalid-record"),
        ({"conclusion": "s-3", "premises": [{"statement": "X.", "stated_at": "s-1"}]}, "invalid-record"),
        ({"conclusion": "s-3", "premises": [{"statement": " "}]}, "invalid-record"),
        ({"conclusion": "s-3", "premises": [{"stated_at": "s-9"}]}, "unknown-node"),
        ({"conclusion": "s-3", "premises": [{"stated_at": "r-1"}]}, "invalid-argument"),
        ({"conclusion": "s-2a", "premises": [{"stated_at": "s-3"}]}, "invalid-record"),
    ],
    ids=[
        "not-a-conclusion",
        "unknown-conclusion",
        "no-conclusion",
        "premises-not-a-list",
        "credence-field",
        "both-kinds",
        "empty-statement",
        "unknown-chunk",
        "chunk-of-a-type-the-rule-refuses",
        "result-is-not-a-conclusion",
    ],
)
def test_a_malformed_record_is_rejected_whole(rubric: Rubric, classified: Graph, record: Any, code: str) -> None:
    result = assemble_premises(classified, rubric, [record])
    assert [rejection["code"] for rejection in result.rejections] == [code]
    assert result.graph == classified


def test_a_second_record_for_one_conclusion_is_rejected(rubric: Rubric, classified: Graph) -> None:
    records = [
        {"conclusion": "s-3", "premises": [{"statement": "X."}]},
        {"conclusion": "s-3", "premises": [{"statement": "Y."}]},
    ]
    result = assemble_premises(classified, rubric, records)
    assert [rejection["code"] for rejection in result.rejections] == ["duplicate-record"]
    assert analyst_parents(result.graph, "s-3") == ["a-1"]


def test_a_link_that_would_close_a_cycle_is_rejected(rubric: Rubric, classified: Graph) -> None:
    linked = assemble_premises(classified, rubric, [{"conclusion": "s-3", "premises": [{"stated_at": "s-1"}]}])
    assert linked.rejections == []
    # s-2b may rest on a method, but not on itself.
    result = assemble_premises(linked.graph, rubric, [{"conclusion": "s-2b", "premises": [{"stated_at": "s-2b"}]}])
    assert [rejection["code"] for rejection in result.rejections] == ["invalid-argument"]
