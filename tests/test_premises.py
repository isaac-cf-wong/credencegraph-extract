"""The premise pass on the invented fixture paper: links, analyst nodes, the cap, merging, ids and credences."""

from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from classify_fixture import (
    BODIES,
    CLASSIFY_BODIES,
    FIRST,
    PAPER,
    PREMISES_FIRST,
    PREMISES_REPAIR,
    REPAIR,
    RUBRIC,
    RUN,
    RUN_ID,
    STATIONARY,
)
from credencegraph.core import Graph, loads
from credencegraph.core.relation import REQUIRES
from credencegraph.diagnostics import unanchored
from credencegraph.inference import marginal
from credencegraph.ingest.chunks import chunks_to_graph
from credencegraph.ingest.latex import read_latex
from credencegraph.rubric import Rubric, check_graph, load_rubric, loads_rubric
from credencegraph.semantics.compiler import compile_graph
from notes_pass import MODEL, fake_client, message

from credencegraph_extract import provider
from credencegraph_extract.assemble import assemble, assemble_premises
from credencegraph_extract.checks import CheckPass
from credencegraph_extract.classify import ClassifyPass
from credencegraph_extract.pipeline import PASSES
from credencegraph_extract.premises import PremisePass, output_schema, render
from credencegraph_extract.run import RunError, replay, run_extraction

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
    return assemble(graph, rubric, REPAIR, only={"s-4", "s-5"}).graph


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Graph:
    """The graph a fresh run of the pipeline makes from the fixture answers."""
    run_dir = tmp_path_factory.mktemp("made") / "run"
    make_run(run_dir, BODIES)
    return loads((run_dir / "graph.json").read_text(encoding="utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def premise_rejections(run_dir: Path) -> list[dict[str, Any]]:
    """The entries of a run's ``rejections.json`` that the premise pass made."""
    # "pass" is the field naming the pipeline stage, not a credential.
    return [item for item in read_json(run_dir / "rejections.json") if item["pass"] == "premises"]  # noqa: S105


def make_run(run_dir: Path, bodies: list[dict[str, Any]], rubric: Path = RUBRIC) -> Any:
    client = fake_client(bodies)
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=client)
    run_extraction(PAPER, run_dir, rubric, call=call, passes=PASSES, run_id=RUN_ID)
    return client


def edited_rubric(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "rubric.toml"
    text = RUBRIC.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


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
        assemble_premises(
            assemble_premises(classified, varied, PREMISES_FIRST).graph, varied, PREMISES_REPAIR, only={"s-5"}
        ).graph
    )
    assert set(before) <= set(after)
    for node_id, credence in before.items():
        assert after[node_id] <= credence + 1e-12, node_id
    # The conclusions that gained a premise are strictly lower.
    for node_id in ("s-2b", "s-3", "s-5"):
        assert after[node_id] < before[node_id], node_id


# The output of the pass.


def test_a_stated_at_link_reaches_a_chunk_the_classify_pass_did_not_link(
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
        assert node.attributes["extraction"] == {"run": RUN_ID, "pass": "premises"}
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


def test_overflow_beyond_the_cap_is_reported_not_added(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    make_run(run_dir, BODIES)
    report = read_json(run_dir / "report.json")
    assert report["passes"]["premises"] == {
        "overflow": [{"conclusion": "s-2b", "statement": "The segments are independent."}]
    }
    graph = loads((run_dir / "graph.json").read_text(encoding="utf-8"))
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
    make_run(tmp_path / "one", BODIES)
    make_run(tmp_path / "two", BODIES)
    one = (tmp_path / "one" / "graph.json").read_bytes()
    assert one == (tmp_path / "two" / "graph.json").read_bytes() == (RUN / "graph.json").read_bytes()
    graph = loads(one.decode("utf-8"))
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


def test_replay_of_the_recorded_run_is_byte_identical(tmp_path: Path, no_network: None) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    rebuilt = replay(run_dir, passes=PASSES, source=PAPER)
    assert rebuilt == (run_dir / "graph.json").read_bytes()
    assert b'"id": "a-3"' in rebuilt


def test_replay_refuses_a_changed_premise_record(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(RUN, run_dir)
    raw = run_dir / "raw" / "0003.json"
    raw.write_text(raw.read_text(encoding="utf-8").replace("independent", "dependent"), encoding="utf-8")
    with pytest.raises(RunError) as caught:
        replay(run_dir, passes=PASSES, source=PAPER)
    assert caught.value.code == "record-mismatch"


# Rejections and the repair round.


def test_a_refused_record_is_repaired_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    client = make_run(run_dir, BODIES)
    assert len(client.sent) == 4
    rejections = premise_rejections(run_dir)
    assert [(item["round"], item["node"], item["code"]) for item in rejections] == [
        ("initial", "s-5", "invalid-argument")
    ]
    assert "'s-4-requires-s-5' already exists" in rejections[0]["message"]
    prompt = (run_dir / "prompts" / "0004.txt").read_text(encoding="utf-8")
    assert "'s-4-requires-s-5' already exists" in prompt
    entry = read_json(run_dir / "manifest.json")["passes"][1]
    assert entry["name"] == "premises"
    assert [call["raw"] for call in entry["calls"]] == ["raw/0003.json", "raw/0004.json"]


def test_the_repair_answer_may_only_name_refused_conclusions(tmp_path: Path) -> None:
    stray = {"conclusion": "s-3", "premises": [{"statement": "The noise is Gaussian."}]}
    bodies = [*CLASSIFY_BODIES, BODIES[2], message(4, json.dumps({"records": [*PREMISES_REPAIR, stray]}))]
    run_dir = tmp_path / "run"
    make_run(run_dir, bodies)
    rejections = premise_rejections(run_dir)
    assert [(item["round"], item["node"], item["code"]) for item in rejections] == [
        ("initial", "s-5", "invalid-argument"),
        ("repair", "s-3", "not-rejected"),
    ]
    graph = loads((run_dir / "graph.json").read_text(encoding="utf-8"))
    assert analyst_parents(graph, "s-3") == ["a-2"]
    assert analyst_parents(graph, "s-5") == ["a-3"]


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


# The rubric and the prompt.


def test_a_rubric_without_premises_needs_no_call(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.toml"
    text = RUBRIC.read_text(encoding="utf-8")
    rubric.write_text(text[: text.index("[premises]")], encoding="utf-8")
    run_dir = tmp_path / "run"
    client = make_run(run_dir, CLASSIFY_BODIES, rubric)
    assert len(client.sent) == 2
    assert read_json(run_dir / "report.json")["passes"]["premises"] == {}
    graph = read_json(run_dir / "graph.json")
    assert all(node["attributes"]["origin"] == "document" for node in graph["nodes"])


def test_a_rubric_without_the_assumption_type_is_refused_before_the_call(tmp_path: Path) -> None:
    rubric = edited_rubric(tmp_path, 'origins = ["analyst"]', 'origins = ["document"]')
    client = fake_client(CLASSIFY_BODIES)
    call = partial(provider.call, model=MODEL, max_tokens=1024, client=client)
    with pytest.raises(RunError) as caught:
        run_extraction(PAPER, tmp_path / "run", rubric, call=call, passes=PASSES, run_id=RUN_ID)
    assert caught.value.code == "invalid-rubric"
    assert "'assumption'" in str(caught.value)
    assert len(client.sent) == 2


def test_an_answer_without_records_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RunError) as caught:
        make_run(tmp_path / "run", [*CLASSIFY_BODIES, message(3, json.dumps({"conclusions": []}))])
    assert caught.value.code == "invalid-record"


def test_the_prompt_carries_no_classify_output_beyond_the_links(rubric: Rubric, classified: Graph) -> None:
    prompt = render(classified, rubric, [])
    for name in [*rubric.types, "unassigned", "compound", "hedge", "form"]:
        assert f"({name})" not in prompt
        assert f'"{name}"' not in prompt
    assert "s-5: We expect the estimator to stay unbiased in practice.\n  rests on s-4: Smith reports" in prompt
    assert "s-1: We simulate 400 noise realisations with a Welch estimator." in prompt
    assert "\ns-2: " not in prompt  # a compound is not a premise; its clauses are listed instead
    assert "s-2b: the estimator is therefore consistent\n  rests on s-2a:" in prompt
    assert prompt.endswith("nothing else:\n[]\n")
    assert (RUN / "prompts" / "premises.template.txt").read_text(encoding="utf-8") == PremisePass.template


def test_the_schema_has_no_credence_field() -> None:
    text = json.dumps(output_schema())
    for word in ("base", "strength", "stated", "credence", "probability"):
        assert f'"{word}"' not in text
    record = output_schema()["properties"]["records"]["items"]
    assert set(record["properties"]) == {"conclusion", "premises"}
    kinds = record["properties"]["premises"]["items"]["anyOf"]
    assert [set(kind["properties"]) for kind in kinds] == [{"stated_at"}, {"statement"}]


def test_the_pipeline_runs_classify_then_premises_then_checks() -> None:
    assert [type(item) for item in PASSES] == [ClassifyPass, PremisePass, CheckPass]
