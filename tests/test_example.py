"""The end-to-end example: a live run on an invented paper, committed with its run directory, replays offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from credencegraph.core import loads
from credencegraph.inference import conditional, marginal
from credencegraph.rubric import check_graph, load_rubric
from credencegraph.semantics.compiler import compile_graph

from credencegraph_extract import cited
from credencegraph_extract.pipeline import passes
from credencegraph_extract.run import replay
from credencegraph_extract.verify import verify_run

EXAMPLE = Path(__file__).parent.parent / "examples" / "end-to-end"
RUN = EXAMPLE / "run"


def test_the_example_replays_byte_identically_offline(no_network: None) -> None:
    pipeline = passes(cited=cited.directory(EXAMPLE / "cited"), allow_uncalibrated=True)
    assert replay(RUN, passes=pipeline, source=EXAMPLE / "paper.tex") == (RUN / "graph.json").read_bytes()


def test_every_deterministic_check_of_the_example_verifies() -> None:
    found = verify_run(RUN)
    assert found.rerun == ("c-001", "c-002", "c-003", "c-004", "c-005", "c-006")
    assert found.differences == ()


def test_the_example_passes_the_assessed_check_with_rubric_credences() -> None:
    graph = loads((RUN / "graph.json").read_text(encoding="utf-8"))
    assert check_graph(graph, load_rubric(EXAMPLE / "rubric.toml"), "assessed", rubric_credences=True) == []


def test_the_planted_error_fails_its_check_and_lowers_its_credence() -> None:
    graph = loads((RUN / "graph.json").read_text(encoding="utf-8"))
    record = graph.nodes["c-003"].attributes["check"]
    assert (record["target"], record["inputs"]["expression"], record["result"]) == (
        "s-4",
        "1036800 - 1036800 * 0.05",
        "failed",
    )
    observed = {node.id: node.attributes["observed"] for node in graph if "observed" in node.attributes}
    network = compile_graph(graph)
    assert marginal(network, "s-4", draws=0).point == pytest.approx(0.8, abs=0.0)
    # Bayes by hand: base 0.8, a failed check with E[sensitivity] 0.95 and E[false_pass] 0.1.
    expected = 0.8 * 0.05 / (0.8 * 0.05 + 0.2 * 0.9)
    assert conditional(network, "s-4", observed, draws=0).point == pytest.approx(expected, abs=1e-12)


def test_the_example_names_no_endpoint_of_the_machine_it_ran_on() -> None:
    urls = {url for path in RUN.rglob("*.json") for url in _base_urls(json.loads(path.read_text(encoding="utf-8")))}
    assert urls == {"http://local-openai-compatible-server.invalid/v1/"}


def _base_urls(value: object) -> list[str]:
    if isinstance(value, dict):
        found = [value["base_url"]] if isinstance(value.get("base_url"), str) else []
        return found + [url for item in value.values() for url in _base_urls(item)]
    if isinstance(value, list):
        return [url for item in value for url in _base_urls(item)]
    return []
