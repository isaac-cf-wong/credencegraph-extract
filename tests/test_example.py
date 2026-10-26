"""The end-to-end example: a run an agent made through a harness on the tools, committed whole, checked offline."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from credencegraph.core import Graph, loads
from credencegraph.inference import conditional, marginal
from credencegraph.rubric import check_graph, load_rubric
from credencegraph.semantics.compiler import compile_graph

from credencegraph_extract import session
from credencegraph_extract.verify import verify_run

EXAMPLE = Path(__file__).parent.parent / "examples" / "end-to-end"
RUN = EXAMPLE / "run"
CORE = str(Path(sys.executable).parent / "credencegraph")


@pytest.fixture
def run_copy(tmp_path: Path) -> Path:
    """A copy of the committed run, so that taking its lock leaves the example as committed."""
    target = tmp_path / "run"
    shutil.copytree(RUN, target)
    return target


def graph() -> Graph:
    return loads((RUN / "graph.json").read_text(encoding="utf-8"))


def test_the_example_replays_byte_identically_offline(no_network: None, run_copy: Path) -> None:
    rebuilt = session.replay(run_copy, source=EXAMPLE / "paper.tex")
    assert rebuilt == session.recorded_graph(run_copy) == (RUN / "graph.json").read_bytes()


def test_the_example_was_made_through_a_harness_and_finished() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["version"], manifest["harness"], manifest["finished"]) == (2, "claude-code", True)
    assert manifest["models"] == ["claude-opus-5-5"]
    assert manifest["calls"] == len(session.read_calls(RUN))


def test_every_deterministic_check_of_the_example_verifies(run_copy: Path) -> None:
    found = verify_run(run_copy)
    assert found.rerun == ("c-001", "c-002", "c-003", "c-004", "c-005")
    assert found.not_rerun == ("c-006",)
    assert found.differences == ()


def test_the_example_passes_the_assessed_check_with_rubric_credences() -> None:
    assert check_graph(graph(), load_rubric(EXAMPLE / "rubric.toml"), "assessed", rubric_credences=True) == []


def test_the_planted_error_fails_its_check_and_lowers_its_credence() -> None:
    found = graph()
    record = found.nodes["c-004"].attributes["check"]
    assert (record["target"], record["inputs"]["expression"], record["result"]) == (
        "s-4",
        "1036800 - 1036800 * 0.05",
        "failed",
    )
    assert found.nodes["s-4"].attributes["assessment"] == {"verdict": "fails"}
    observed = {node.id: node.attributes["observed"] for node in found if "observed" in node.attributes}
    assert [node_id for node_id, value in observed.items() if not value] == ["c-004"]
    network = compile_graph(found)
    assert marginal(network, "s-4", draws=0).point == pytest.approx(0.8, abs=0.0)
    # Bayes by hand: base 0.8, a failed check with E[sensitivity] 0.95 and E[false_pass] 0.1.
    expected = 0.8 * 0.05 / (0.8 * 0.05 + 0.2 * 0.9)
    assert conditional(network, "s-4", observed, draws=0).point == pytest.approx(expected, abs=1e-12)


def test_diagnose_observed_names_the_failed_check_as_the_crux_of_the_planted_error() -> None:
    result = subprocess.run(  # noqa: S603 - the core's console script, fixed arguments
        [CORE, "diagnose", str(RUN / "graph.json"), "--target", "s-4", "--observed", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["observed"] == {f"c-00{i}": i != 4 for i in range(1, 7)}
    cruxes = [finding for finding in report["findings"] if finding["diagnostic"] == "crux"]
    assert cruxes[0]["id"] == "crux:sensitivity:s-4-observes-c-004"


# An absolute path (POSIX, home-relative or Windows), a URL, or a user@host address.
ADDRESS = re.compile(r"https?://|(?<![\w.~-])(?:~|[A-Za-z]:)?[\\/](?:[\w.-]+[\\/])+[\w.-]*|[\w.-]+@[\w-]+\.[\w.-]+")


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ('"file": "/home/someone/paper.tex"', True),
        ('"cited": "~/papers/cited"', True),
        ('"file": "C:\\Users\\someone\\paper.tex"', True),
        # The same path as JSON writes it, each backslash escaped.
        ('"file": "C:\\\\Users\\\\someone\\\\paper.tex"', True),
        ("read from https://example.org/paper", True),
        ("by someone@machine.example", True),
        ('"path": "checks/c-001/", "file": "lee2019.txt"', False),
        ('"digest": "sha256:99b53d117dd30e01"', False),
    ],
)
def test_the_address_pattern_finds_paths_urls_and_hosts(text: str, found: bool) -> None:
    assert bool(ADDRESS.search(text)) is found


def test_the_example_names_no_path_or_address_of_the_machine_it_ran_on() -> None:
    files = {path: path.read_text(encoding="utf-8") for path in RUN.rglob("*") if path.is_file()}
    assert {path.suffix for path in files} == {".json", ".jsonl", ".toml", ".txt"}
    assert {
        str(path.relative_to(RUN)): ADDRESS.findall(text) for path, text in files.items() if ADDRESS.search(text)
    } == {}
