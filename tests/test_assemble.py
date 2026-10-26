"""The assembler applies classify records through the core's ``split`` and ``annotate``, in document order."""

from __future__ import annotations

from dataclasses import replace

import pytest
from classify_fixture import FIRST, PAPER, RUBRIC
from credencegraph.core import Graph
from credencegraph.core.credence import Beta
from credencegraph.ingest.chunks import chunks_to_graph
from credencegraph.ingest.latex import read_latex
from credencegraph.rubric import Rubric, check_graph, load_rubric, loads_rubric

from credencegraph_extract.assemble import assemble, edge_rules_not_requires, fixes


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric(RUBRIC)


@pytest.fixture
def graph(rubric: Rubric) -> Graph:
    return chunks_to_graph(read_latex(PAPER), "paper.tex", rubric.initial)[0]


def by_node(records: list[dict]) -> dict[str, dict]:
    return {record["node"]: record for record in records}


def test_records_are_applied_in_document_order(graph: Graph, rubric: Rubric) -> None:
    # s-5 comes first in the answer but rests on s-4, which is unassigned until its record is
    # applied; in document order s-4 is applied first, so s-5 is accepted.
    fixed = {**by_node(FIRST)["s-4"], "form": {"statement": "a bias below 1\\% in comparable settings"}}
    records = [fixed if record["node"] == "s-4" else record for record in FIRST]
    assembly = assemble(graph, rubric, records)
    assert assembly.rejections == []
    assert check_graph(assembly.graph, rubric, "typed", rubric_credences=True) == []
    assert assembly.graph.nodes["s-5"].kind == "claim"


def test_a_refused_record_is_returned_with_the_core_error_code(graph: Graph, rubric: Rubric) -> None:
    assembly = assemble(graph, rubric, FIRST)
    rejected = {rejection["node"]: rejection for rejection in assembly.rejections}
    assert list(rejected) == ["s-4", "s-5"]
    assert rejected["s-4"]["code"] == "invalid-argument"
    assert "'a bias below one percent' is not a substring" in rejected["s-4"]["message"]
    assert rejected["s-4"]["record"] == by_node(FIRST)["s-4"]
    assert "has type 'unassigned'" in rejected["s-5"]["message"]
    assert assembly.graph.nodes["s-4"].kind == "unassigned"
    assert assembly.graph.nodes["s-5"].kind == "unassigned"


def test_a_record_is_applied_whole_or_not_at_all(graph: Graph, rubric: Rubric) -> None:
    # The split succeeds and the first child is typed, then the second child's form is refused:
    # nothing of the record may remain.
    record = {
        **by_node(FIRST)["s-2"],
        "children": [
            by_node(FIRST)["s-2"]["children"][0],
            {"type": "claim", "form": {"shape": "universal", "statement": "not in the text"}, "rests_on": ["s-2a"]},
        ],
    }
    assembly = assemble(graph, rubric, [record])
    assert [rejection["node"] for rejection in assembly.rejections] == ["s-2"]
    assert assembly.graph == graph
    assert "s-2a" not in assembly.graph


def test_the_base_and_strength_are_the_rubric_values(graph: Graph, rubric: Rubric) -> None:
    assembly = assemble(graph, rubric, FIRST)
    out = assembly.graph
    assert out.nodes["s-3"].base == rubric.types["claim"].base
    assert out.nodes["s-2a"].base == rubric.types["result"].base
    rule = rubric.types["claim"].rests_on
    assert rule is not None
    assert out.relations["e-1-requires-s-3"].strength == rule.strength
    assert {relation.type for relation in out.relations.values()} == {"cites", "part-of", "requires"}


def test_a_mapped_hedge_sets_stated(graph: Graph, rubric: Rubric) -> None:
    assembly = assemble(graph, rubric, FIRST)
    assert assembly.graph.nodes["s-3"].stated == Beta(6, 4)
    assert all(node.stated is None for node in assembly.graph if node.id != "s-3")


def test_hedges_are_matched_after_normalisation_and_case_folding(graph: Graph, rubric: Rubric) -> None:
    record = {**by_node(FIRST)["s-3"], "hedge": "This suggests"}
    custom = loads_rubric(RUBRIC.read_text(encoding="utf-8") + '"this  SUGGESTS" = "beta:7,3"\n')
    assembly = assemble(graph, custom, [by_node(FIRST)["s-2"], by_node(FIRST)["e-1"], record])
    assert assembly.rejections == []
    assert assembly.graph.nodes["s-3"].stated == Beta(7, 3)


def test_an_unmapped_hedge_is_listed_and_sets_nothing(graph: Graph, rubric: Rubric) -> None:
    record = {**by_node(FIRST)["s-3"], "hedge": "the method may extend"}
    assembly = assemble(graph, rubric, [by_node(FIRST)["s-2"], by_node(FIRST)["e-1"], record])
    assert assembly.rejections == []
    assert assembly.unmapped == [{"node": "s-3", "phrase": "the method may extend"}]
    assert assembly.graph.nodes["s-3"].stated is None


def test_a_hedge_not_in_the_text_rejects_the_record(graph: Graph, rubric: Rubric) -> None:
    record = {**by_node(FIRST)["s-3"], "hedge": "may well"}
    assembly = assemble(graph, rubric, [by_node(FIRST)["s-2"], by_node(FIRST)["e-1"], record])
    assert [(rejection["node"], rejection["code"]) for rejection in assembly.rejections] == [
        ("s-3", "invalid-argument")
    ]
    assert "hedge 'may well' is not a substring" in assembly.rejections[0]["message"]
    assert assembly.graph.nodes["s-3"].kind == "unassigned"


@pytest.mark.parametrize(
    ("record", "code"),
    [
        pytest.param({"type": "claim"}, "invalid-record", id="no-node"),
        pytest.param("s-1", "invalid-record", id="not-an-object"),
        pytest.param({"node": "s-9", "type": "method"}, "unknown-node", id="unknown-node"),
        pytest.param({"node": "s-1", "type": "method", "base": 0.9}, "invalid-record", id="credence-field"),
        pytest.param({"node": "s-1", "type": "method", "stated": 0.9}, "invalid-record", id="stated-field"),
        pytest.param({"node": "s-1"}, "invalid-record", id="no-type"),
        pytest.param({"node": "s-1", "type": "method", "form": []}, "invalid-record", id="form-not-object"),
        pytest.param({"node": "s-1", "type": "method", "rests_on": "s-2"}, "invalid-record", id="rests-on-not-list"),
        pytest.param({"node": "s-1", "type": "method", "hedge": 3}, "invalid-record", id="hedge-not-string"),
        pytest.param({"node": "s-1", "type": "opinion"}, "invalid-argument", id="unknown-type"),
        pytest.param({"node": "s-1", "type": "method", "form": {"kind": "survey"}}, "invalid-argument", id="bad-enum"),
        pytest.param(
            {"node": "s-3", "type": "claim", "form": {"shape": "bound", "statement": "noise"}, "rests_on": ["s-8"]},
            "unknown-node",
            id="rests-on-unknown",
        ),
        pytest.param({"node": "s-1", "split": ["We simulate"], "children": []}, "invalid-record", id="children-short"),
        pytest.param({"node": "s-1", "split": [], "children": []}, "invalid-record", id="split-empty"),
        pytest.param(
            {"node": "s-1", "split": ["We simulate"], "children": [{"type": "method"}], "type": "method"},
            "invalid-record",
            id="split-with-type",
        ),
        pytest.param(
            {"node": "s-1", "split": ["We simulate"], "children": [{"node": "x", "type": "method"}]},
            "invalid-record",
            id="child-with-node",
        ),
        pytest.param(
            {"node": "s-1", "split": ["simulated"], "children": [{"type": "method"}]},
            "invalid-argument",
            id="clause-not-found",
        ),
    ],
)
def test_malformed_and_refused_records(graph: Graph, rubric: Rubric, record: object, code: str) -> None:
    assembly = assemble(graph, rubric, [record])
    assert [rejection["code"] for rejection in assembly.rejections] == [code]
    assert assembly.rejections[0]["record"] == record
    assert assembly.graph == graph


def test_a_record_for_a_node_that_is_not_a_chunk_is_rejected(graph: Graph, rubric: Rubric) -> None:
    analyst = replace(graph.nodes["s-1"], attributes={"origin": "analyst"}, sources=())
    rebuilt = Graph()
    for node in graph:
        rebuilt.add_node(analyst if node.id == "s-1" else node)
    for relation in graph.relations.values():
        rebuilt.add_relation(relation)
    assembly = assemble(rebuilt, rubric, [{"node": "s-1", "type": "method"}])
    assert assembly.rejections[0]["message"] == "node 's-1' is not a chunk of the document"


def test_a_second_record_for_a_node_is_rejected(graph: Graph, rubric: Rubric) -> None:
    first = {"node": "s-1", "type": "method", "form": {"kind": "simulation"}}
    second = {"node": "s-1", "type": "method", "form": {"kind": "procedure"}}
    assembly = assemble(graph, rubric, [first, second])
    assert [(rejection["record"], rejection["code"]) for rejection in assembly.rejections] == [
        (second, "duplicate-record")
    ]
    assert assembly.graph.nodes["s-1"].attributes["form"] == {"kind": "simulation"}


def test_edge_rules_other_than_requires_are_found(rubric: Rubric) -> None:
    assert edge_rules_not_requires(rubric) == []
    text = RUBRIC.read_text(encoding="utf-8").replace('relation = "requires"', 'relation = "supports"')
    assert edge_rules_not_requires(loads_rubric(text)) == ["claim"]


# ``fixes`` says what to change in a record the core refused, for the model's one repair call.


def test_a_clause_id_used_as_a_node_says_how_to_split(rubric: Rubric, graph: Graph) -> None:
    [fix] = fixes(graph, rubric, {"node": "s-2a", "type": "result", "form": {"quantity": "The bias"}})
    assert fix.startswith("'s-2a' is not a chunk id: a clause id such as 's-2a' exists only once a split of 's-2'")
    assert '"node": \'s-2\', "split" and "children"' in fix


def test_an_unknown_node_points_at_the_chunk_list(rubric: Rubric, graph: Graph) -> None:
    assert fixes(graph, rubric, {"node": "x-9", "type": "method"}) == [
        "'x-9' is not a chunk id: use one of the chunk ids start_run listed"
    ]


def test_an_unknown_type_lists_the_types_a_chunk_may_have(rubric: Rubric, graph: Graph) -> None:
    [fix] = fixes(graph, rubric, {"node": "s-1", "type": "assumption"})
    assert fix == (
        "'assumption' is not a type a chunk may have: use one of "
        "['reference', 'method', 'result', 'derivation', 'literature_claim', 'claim']"
    )


def test_rests_on_on_a_type_that_rests_on_nothing_is_to_be_left_out(rubric: Rubric, graph: Graph) -> None:
    record = {"node": "e-1", "type": "derivation", "rests_on": ["s-1"]}
    assert fixes(graph, rubric, record) == ["type 'derivation' rests on nothing: leave out rests_on"]


def test_rests_on_a_node_of_another_type_says_which_types_are_allowed(rubric: Rubric, graph: Graph) -> None:
    record = {"node": "s-5", "type": "claim", "form": {"shape": "bound", "statement": "the estimator"}}
    assert fixes(graph, rubric, {**record, "rests_on": ["s-4", "q-1"]}) == [
        (
            "'s-4' in rests_on is of type 'unassigned', and type 'claim' rests only on "
            "['result', 'method', 'derivation', 'literature_claim']: type 's-4' as one of those in the same answer, "
            "or leave it out of rests_on"
        ),
        "'q-1' in rests_on is not a node id: use node ids as start_run listed them",
    ]
    assert fixes(graph, rubric, record) == [
        "type 'claim' must rest on at least 1 chunk(s) of the types ['result', 'method', 'derivation', 'literature_claim']"
    ]


def test_form_fields_of_another_type_and_missing_ones_are_named(rubric: Rubric, graph: Graph) -> None:
    assert fixes(graph, rubric, {"node": "e-1", "type": "derivation", "form": {"quantity": "b"}}) == [
        "type 'derivation' has no form field ['quantity']: its form fields are ['equation']; leave the others out"
    ]
    assert fixes(graph, rubric, {"node": "s-2", "type": "result"}) == ["type 'result' needs the form field 'quantity'"]


def test_text_values_not_in_the_chunk_are_to_be_copied_exactly(rubric: Rubric, graph: Graph) -> None:
    record = {"node": "s-2", "type": "result", "form": {"quantity": "the Bias"}, "hedge": "perhaps"}
    assert fixes(graph, rubric, record) == [
        "form field 'quantity': 'the Bias' is not a substring of the text; copy it exactly from the chunk's text",
        "hedge 'perhaps' is not in the chunk's text: copy the phrase exactly, or leave hedge out",
    ]
    assert fixes(graph, rubric, {"node": "s-2", "type": "result", "form": {"quantity": 3}}) == [
        "form field 'quantity' of type 'result' must be a verbatim"
    ]


def test_a_split_names_each_clause_and_child_that_needs_a_change(rubric: Rubric, graph: Graph) -> None:
    record = {
        "node": "s-2",
        "split": ["The bias vanishes for long segments", "the estimator is consistent"],
        "children": [{"type": "result", "form": {"quantity": "The bias"}}, {"type": "derivation", "rests_on": ["s-1"]}],
    }
    assert fixes(graph, rubric, record) == [
        "clause 2 occurs 0 times in the text of 's-2': copy a clause that occurs once",
        "children[1]: type 'derivation' rests on nothing: leave out rests_on",
    ]
    assert fixes(graph, rubric, {"node": "s-2", "split": ["a"], "children": []}) == [
        'a split record needs "split", a list of clauses, and "children", one typed record per clause'
    ]


def test_a_record_the_core_accepts_needs_no_fix(rubric: Rubric, graph: Graph) -> None:
    assert fixes(graph, rubric, {"node": "s-1", "type": "method", "form": {"kind": "simulation"}}) == []
    assert fixes(graph, rubric, "not a record") == ["send each record as a JSON object"]


def test_a_rests_on_that_is_not_a_list_is_named_as_such(rubric: Rubric, graph: Graph) -> None:
    graph = assemble(graph, rubric, [{"node": "s-1", "type": "method", "form": {"kind": "simulation"}}]).graph
    record = {
        "node": "s-5",
        "type": "claim",
        "form": {"shape": "bound", "statement": "the estimator"},
        "rests_on": "s-1",
    }
    assert fixes(graph, rubric, record) == ['rests_on must be a list of chunk ids, such as ["s-2"]']


def test_a_form_that_is_not_an_object_lists_the_types_fields(rubric: Rubric, graph: Graph) -> None:
    assert fixes(graph, rubric, {"node": "s-1", "type": "method", "form": "simulation"}) == [
        "form must be an object of field names and values; the form fields of type 'method' are ['kind']"
    ]


def test_an_enum_value_outside_its_values_lists_them(rubric: Rubric, graph: Graph) -> None:
    assert fixes(graph, rubric, {"node": "s-1", "type": "method", "form": {"kind": "survey"}}) == [
        "form field 'kind' of type 'method' must be one of ['simulation', 'dataset', 'procedure']"
    ]


def test_an_incomplete_quantity_says_what_it_needs() -> None:
    rubric = loads_rubric(
        '[rubric]\nformat = 1\nname = "q"\nversion = "1"\n'
        '[types.result]\ndescription = "r"\nform.amount = { type = "quantity", required = true }\n'
    )
    graph = chunks_to_graph(read_latex(PAPER), "paper.tex", rubric.initial)[0]
    assert fixes(graph, rubric, {"node": "s-1", "type": "result", "form": {"amount": {"value": 400}}}) == [
        "form field 'amount' of type 'result' must be an object with a number \"value\" and a string \"unit\""
    ]


def test_a_split_child_is_told_to_copy_from_its_clause(rubric: Rubric, graph: Graph) -> None:
    record = {
        "node": "s-2",
        "split": ["The bias vanishes for long segments", "the estimator is therefore consistent"],
        "children": [
            {"type": "result", "form": {"quantity": "the estimator"}, "hedge": "therefore"},
            {"type": "method", "form": {"kind": "procedure"}},
        ],
    }
    assert fixes(graph, rubric, record) == [
        (
            "children[0]: form field 'quantity': 'the estimator' is not a substring of the text; copy it exactly "
            "from the clause's text"
        ),
        "children[0]: hedge 'therefore' is not in the clause's text: copy the phrase exactly, or leave hedge out",
    ]


def test_a_chunk_id_written_with_its_label_is_told_the_id(rubric: Rubric, graph: Graph) -> None:
    assert fixes(graph, rubric, {"node": "s-1 (unassigned)", "type": "method", "form": {"kind": "simulation"}}) == [
        "'s-1 (unassigned)' is not a chunk id: write the id alone, 's-1', without the type in parentheses"
    ]
