"""The check tools on an agent-tool run: run_checks and judge_citation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from checks_fixture import CITED, CLASSIFY, PAPER, PHASE, PLAN, PRODUCT, RESULTS, RUBRIC, VERDICTS

from credencegraph_extract import citation, session
from credencegraph_extract.checks import directory_digest
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.fetch import Fetched, FetchError
from credencegraph_extract.run import RunError
from credencegraph_extract.session import get_state, recorded_graph, replay, start_run
from credencegraph_extract.tools import classify, finish_run, judge_citation, run_checks
from credencegraph_extract.verify import verify_run

MODEL = "test-model-1"
# The maths proposals of the recorded check run: s-3 passes, s-2 fails, e-2 passes, e-3 errs and its
# numeric fallback passes.
MATHS = [PLAN[1], PLAN[2], PRODUCT, PHASE]
RECOMPUTE = PLAN[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run of the check fixture's paper, classified as the recorded run was."""
    target = tmp_path / "run"
    start_run(PAPER, RUBRIC, target, harness="test-harness", model=MODEL, cited=CITED)
    assert classify(target, CLASSIFY, model=MODEL)["refused"] == []
    return target


def test_the_maths_checks_give_what_the_check_pass_gave(run_dir: Path) -> None:
    result = run_checks(run_dir, MATHS, model=MODEL)
    assert result["refused"] == []
    found = [(item["type"], item["target"], item["result"]) for item in result["checks"]]
    recorded = [(check_type, target, outcome) for _, check_type, target, outcome in RESULTS[:5]]
    # The pass numbers checks in document order of their targets; the tool, in the order proposed.
    assert sorted(found) == sorted(recorded)
    assert [item["id"] for item in result["checks"]] == ["c-001", "c-002", "c-003", "c-004", "c-005"]
    assert all(item["detail"] for item in result["checks"])


def test_each_check_is_an_observed_evidence_node_with_its_files(run_dir: Path) -> None:
    run_checks(run_dir, MATHS, model=MODEL)
    graph = json.loads(recorded_graph(run_dir))
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {(edge["source"], edge["target"]) for edge in graph["relations"] if edge["type"] == "observes"}
    assert edges == {("s-3", "c-001"), ("s-2", "c-002"), ("e-2", "c-003"), ("e-3", "c-004"), ("e-3", "c-005")}
    assert [nodes[f"c-00{n}"]["attributes"].get("observed") for n in range(1, 6)] == [True, False, True, None, True]
    for check_id in ("c-001", "c-004"):
        record = nodes[check_id]["attributes"]["check"]
        files = {path.name: path.read_bytes() for path in (run_dir / "checks" / check_id).iterdir()}
        assert set(files) == {"inputs.json", "output.json", "log.txt"}
        assert record["log"] == {"path": f"checks/{check_id}/", "digest": directory_digest(files)}
        assert record["model"] is None
        assert record["tool"]["version"] == session.versions()["credencegraph-extract"]


def test_the_versions_that_ran_the_checks_are_logged_with_the_call(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    assert log(run_dir)[-1]["args"]["versions"] == session.versions()


def test_a_check_is_not_run_twice_on_its_target_in_a_later_call(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    again = run_checks(run_dir, [{**RECOMPUTE, "expression": "400 - 0.125 * 400"}], model=MODEL)
    assert again["checks"] == []
    [refused] = again["refused"]
    assert (refused["node"], refused["code"]) == ("s-3", "duplicate-record")
    assert refused["fix"] == ["propose at most one check of each type on a target"]


def test_one_step_is_not_checked_again_on_another_target_in_a_later_call(run_dir: Path) -> None:
    run_checks(run_dir, [PRODUCT], model=MODEL)
    again = run_checks(run_dir, [{**PRODUCT, "target": "e-1", "lhs": "c", "rhs": "N * c / N"}], model=MODEL)
    assert again["checks"] == []
    [refused] = again["refused"]
    assert (refused["node"], refused["code"]) == ("e-1", "duplicate-record")
    assert refused["fix"] == ["propose this check once, on the step's own target; 'e-2' already has it"]


def test_a_failed_check_stays_observed(run_dir: Path) -> None:
    failed = run_checks(run_dir, [PLAN[2]], model=MODEL)["checks"]
    assert [item["result"] for item in failed] == ["failed"]
    retry = run_checks(run_dir, [{**PLAN[2], "expression": "250 * 400"}], model=MODEL)
    assert [item["code"] for item in retry["refused"]] == ["duplicate-record"]
    finish_run(run_dir, model=MODEL)
    verdicts = {
        node["id"]: node["attributes"]["assessment"]
        for node in read_json(run_dir / "graph.json")["nodes"]
        if "assessment" in node["attributes"]
    }
    assert verdicts["s-2"] == VERDICTS["s-2"]


def test_a_citation_proposal_is_sent_to_the_citation_tool(run_dir: Path) -> None:
    result = run_checks(run_dir, [PLAN[0]], model=MODEL)
    [refused] = result["refused"]
    assert (refused["node"], refused["code"]) == ("s-7", "invalid-record")
    assert refused["fix"] == ["send a citation-support check to the citation tool, with the source and the passage"]


def test_every_refused_proposal_says_what_to_change(run_dir: Path) -> None:
    proposals = [
        {"target": "s-9", "type": "recompute-numbers"},
        {**RECOMPUTE, "expected": "349"},
        {**RECOMPUTE, "nodes": ["x-1"]},
        "not a proposal",
        {**RECOMPUTE, "type": "rerun-code"},
    ]
    result = run_checks(run_dir, proposals, model=MODEL)
    assert [item["code"] for item in result["refused"]] == [
        "not-a-target",
        "unfaithful-numeral",
        "unknown-node",
        "invalid-record",
        "invalid-record",
    ]
    assert all(
        item["fix"] and all(isinstance(line, str) and line for line in item["fix"]) for item in result["refused"]
    )
    assert [item["call"] for item in get_state(run_dir)["rejections"]] == [3] * 5


def test_the_node_a_check_targets_can_no_longer_be_typed_again(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    [refused] = classify(run_dir, [{"node": "s-3", "type": "context"}], model=MODEL)["refused"]
    assert refused["code"] == "checked-node"


def test_proposals_must_be_a_list(run_dir: Path) -> None:
    with pytest.raises(RunError, match="proposals must be a list") as raised:
        run_checks(run_dir, "check s-3", model=MODEL)  # type: ignore[arg-type]
    assert raised.value.code == "invalid-argument"


def test_a_finished_run_with_checks_verifies_and_replays(run_dir: Path, no_network: None) -> None:
    run_checks(run_dir, MATHS, model=MODEL)
    finish_run(run_dir, model=MODEL)
    found = verify_run(run_dir)
    assert found.rerun == ("c-001", "c-002", "c-003", "c-004", "c-005")
    assert found.differences == ()
    assert replay(run_dir, source=PAPER) == recorded_graph(run_dir) == (run_dir / "graph.json").read_bytes()


def test_check_files_lost_or_edited_are_written_again_from_the_log(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    expected = (run_dir / "checks" / "c-001" / "output.json").read_bytes()
    (run_dir / "checks" / "c-001" / "output.json").write_text("{}", encoding="utf-8")
    (run_dir / "checks" / "c-001" / "log.txt").unlink()
    (run_dir / "state.json").unlink()
    session.reconcile(run_dir)
    assert (run_dir / "checks" / "c-001" / "output.json").read_bytes() == expected
    assert (run_dir / "checks" / "c-001" / "log.txt").is_file()


def test_a_state_file_with_other_check_files_is_not_trusted(run_dir: Path) -> None:
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    saved = read_json(run_dir / "state.json")
    path = "checks/c-001/output.json"
    original = saved["files"][path]
    saved["files"][path] = original.replace("passed", "failed")
    (run_dir / "state.json").write_text(json.dumps(saved), encoding="utf-8")
    session.reconcile(run_dir)
    assert read_json(run_dir / "state.json")["files"][path] == original
    assert (run_dir / path).read_text(encoding="utf-8") == original


# judge_citation

PASSAGE = "In comparable settings the bias stays below 1% for all segment lengths we tried"
JONES = "the bias grows with the segment length, as we show for segments of up to a day"


def page(url: str, body: str, content_type: str = "text/plain", final_url: str | None = None) -> Fetched:
    return Fetched(url, final_url or url, (url,), content_type, body.encode("utf-8"))


class Web:
    """A fake fetcher: a page per URL, and a record of what was fetched."""

    def __init__(self, pages: dict[str, Fetched]) -> None:
        """Serve these pages."""
        self.pages = pages
        self.fetched: list[str] = []

    def __call__(self, url: str) -> Fetched:
        """Fetch a page, or fail as a missing page does."""
        self.fetched.append(url)
        if url not in self.pages:
            raise FetchError("fetch-failed", f"{url!r} answered 404")
        return self.pages[url]


ARXIV = Web(
    {
        "https://arxiv.org/abs/2101.01234": page(
            "https://arxiv.org/abs/2101.01234",
            '<html><meta name="citation_pdf_url" content="http://arxiv.org/pdf/2101.01234v2"></html>',
            "text/html",
        ),
        "https://arxiv.org/pdf/2101.01234v2": page("https://arxiv.org/pdf/2101.01234v2", f"Long segments. {JONES}."),
        "https://agent.example/jones.txt": page("https://agent.example/jones.txt", f"My notes: {JONES}."),
    }
)


def judge(run_dir: Path, target: str, reference: str, source: str, passage: str, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("supports", True)
    kwargs.setdefault("fetcher", ARXIV)
    return judge_citation(run_dir, target, reference, source, passage, model=MODEL, **kwargs)


def check_of(run_dir: Path, check_id: str) -> dict[str, Any]:
    graph = json.loads(recorded_graph(run_dir))
    return next(node for node in graph["nodes"] if node["id"] == check_id)


def test_the_cited_copy_named_by_the_key_is_tied(run_dir: Path, no_network: None) -> None:
    result = judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE)
    assert (result["check"], result["result"], result["tied"]) == ("c-001", "passed", True)
    node = check_of(run_dir, "c-001")
    assert node["attributes"]["observed"] is True
    record = node["attributes"]["check"]
    assert record["artifact"]["document"] == "smith2020.txt"
    assert record["artifact"]["tied"] is True
    assert record["model"] == {"model": MODEL}
    assert [source["document"] for source in node["sources"]] == ["smith2020.txt"]


def test_the_cited_copy_given_by_its_path_in_the_run_is_tied(run_dir: Path) -> None:
    result = judge(run_dir, "s-7", "r-1", str(run_dir / "cited" / "smith2020.txt"), PASSAGE)
    assert (result["result"], result["tied"]) == ("passed", True)


def test_a_passage_the_agent_judges_unsupportive_fails(run_dir: Path) -> None:
    result = judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE, supports=False)
    assert (result["result"], result["tied"]) == ("failed", True)
    assert check_of(run_dir, "c-001")["attributes"]["observed"] is False


def test_a_cited_copy_rewritten_after_start_run_is_not_tied(run_dir: Path) -> None:
    (run_dir / "cited" / "smith2020.txt").write_text("The bias is always zero, as we tried.", encoding="utf-8")
    result = judge(run_dir, "s-7", "r-1", "smith2020.txt", "The bias is always zero")
    assert (result["result"], result["tied"]) == ("error", False)
    assert result["detail"] == "source not tied to the reference: the cited copy changed since start_run"
    assert "observed" not in check_of(run_dir, "c-001")["attributes"]


def test_a_cited_file_another_key_names_is_not_tied(run_dir: Path) -> None:
    result = judge(run_dir, "s-8", "r-2", "smith2020.txt", PASSAGE)
    assert (result["result"], result["tied"]) == ("error", False)
    assert "no file among the cited works is named by the reference's key" in result["detail"]


def test_a_local_file_outside_the_cited_copy_is_not_read(run_dir: Path, tmp_path: Path) -> None:
    written = tmp_path / "smith2020.txt"
    written.write_text(f"{PASSAGE}.", encoding="utf-8")
    for source in (str(written), "../smith2020.txt", "cited/../smith2020.txt"):
        result = judge(run_dir, "s-7", "r-1", source, PASSAGE)
        assert result["check"] is None
        [refused] = result["refused"]
        assert refused["code"] == "unreadable-source"
        assert refused["fix"]
    assert not any(node["id"].startswith("c-") for node in json.loads(recorded_graph(run_dir))["nodes"])


def test_a_passage_not_in_the_source_is_an_error_and_may_be_checked_again(run_dir: Path) -> None:
    first = judge(run_dir, "s-7", "r-1", "smith2020.txt", "The bias is always zero")
    assert (first["result"], first["detail"]) == ("error", "the quoted passage is not in the text of the source")
    again = judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE)
    assert (again["check"], again["result"]) == ("c-002", "passed")


def test_a_citation_check_that_passed_or_failed_stays(run_dir: Path) -> None:
    judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE, supports=False)
    result = judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE)
    assert result["check"] is None
    [refused] = result["refused"]
    assert (refused["node"], refused["code"]) == ("s-7", "duplicate-record")
    assert refused["fix"] == ["leave 's-7': a citation check runs again only after one that errs"]


def test_a_reference_given_by_its_key_is_told_the_id(run_dir: Path) -> None:
    result = judge(run_dir, "s-7", "smith2020", "smith2020.txt", PASSAGE)
    [refused] = result["refused"]
    assert refused["code"] == "invalid-record"
    assert refused["fix"] == ['\'smith2020\' is the key of r-1: give "reference": "r-1"']


def test_a_target_that_does_not_cite_the_reference_is_refused(run_dir: Path) -> None:
    [refused] = judge(run_dir, "s-7", "r-2", "smith2020.txt", PASSAGE)["refused"]
    assert (refused["code"], refused["message"]) == ("invalid-record", "'s-7' does not cite 'r-2'")


def test_the_arxiv_paper_of_the_entry_is_read_from_arxiv_at_its_current_version(run_dir: Path) -> None:
    web = Web(ARXIV.pages)
    result = judge(run_dir, "s-8", "r-2", "https://arxiv.org/pdf/2101.01234v1", JONES, fetcher=web)
    assert (result["result"], result["tied"]) == ("passed", True)
    # The version comes from arXiv, not from the agent's URL.
    assert web.fetched == ["https://arxiv.org/abs/2101.01234", "https://arxiv.org/pdf/2101.01234v2"]
    record = check_of(run_dir, result["check"])["attributes"]["check"]
    assert record["artifact"]["document"] == "arXiv:2101.01234v2"
    read = log(run_dir)[-1]["args"]["read"]
    assert read["tie"] == {
        "by": "arxiv",
        "id": "2101.01234",
        "version": "v2",
        "url": "https://arxiv.org/pdf/2101.01234v2",
    }


def test_a_page_the_agent_names_is_read_but_not_tied(run_dir: Path) -> None:
    result = judge(run_dir, "s-8", "r-2", "https://agent.example/jones.txt", JONES)
    assert (result["result"], result["tied"]) == ("error", False)
    assert result["detail"].startswith("source not tied to the reference")
    record = check_of(run_dir, result["check"])["attributes"]["check"]
    assert record["artifact"]["quote"] == JONES
    assert record["artifact"]["source"] == {
        "url": "https://agent.example/jones.txt",
        "final_url": "https://agent.example/jones.txt",
    }
    finish_run(run_dir, model=MODEL)
    report = read_json(run_dir / "report.json")
    assert [(item["check"], item["target"], item["reference"]) for item in report["untied_citations"]] == [
        (result["check"], "s-8", "r-2")
    ]


def test_a_check_made_with_a_model_the_type_was_not_fitted_with_is_reported(run_dir: Path) -> None:
    judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE)
    finish_run(run_dir, model=MODEL)
    assert read_json(run_dir / "report.json")["uncalibrated"] == [
        {"check": "c-001", "type": "citation-support", "model": MODEL}
    ]


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000/v1/models", "http://169.254.169.254/latest/meta-data/", "file:///etc/passwd"]
)
def test_a_url_the_fetch_policy_refuses_is_refused_without_the_network(
    run_dir: Path, url: str, no_network: None
) -> None:
    result = judge_citation(run_dir, "s-8", "r-2", url, JONES, supports=True, model=MODEL)
    [refused] = result["refused"]
    assert refused["code"] == "refused-url"
    assert refused["fix"] == ["give a public http or https URL, or the name of a file of the run's cited works"]


def test_a_source_that_cannot_be_fetched_is_refused(run_dir: Path) -> None:
    [refused] = judge(run_dir, "s-8", "r-2", "https://agent.example/missing", JONES)["refused"]
    assert refused["code"] == "unreadable-source"


@pytest.mark.parametrize("bad", [{"supports": "yes"}, {"passage": ""}, {"source": None}])
def test_arguments_of_the_wrong_type_refuse_the_call(run_dir: Path, bad: dict[str, Any]) -> None:
    args = {"target": "s-7", "reference": "r-1", "source": "smith2020.txt", "passage": PASSAGE, "supports": True}
    with pytest.raises(RunError) as raised:
        judge_citation(run_dir, **{**args, **bad}, model=MODEL)  # type: ignore[arg-type]
    assert raised.value.code == "invalid-argument"


def test_a_rebuild_takes_what_the_log_recorded_and_reads_no_source(run_dir: Path, no_network: None) -> None:
    web = Web(ARXIV.pages)
    judge(run_dir, "s-8", "r-2", "https://arxiv.org/abs/2101.01234", JONES, fetcher=web)
    fetched = list(web.fetched)
    (run_dir / "state.json").unlink()
    session.reconcile(run_dir)
    assert web.fetched == fetched
    assert check_of(run_dir, "c-001")["attributes"]["check"]["result"] == "passed"


def test_replay_needs_the_bytes_of_a_fetched_source_the_run_did_not_store(run_dir: Path, tmp_path: Path) -> None:
    judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE)
    judge(run_dir, "s-8", "r-2", "https://arxiv.org/abs/2101.01234", JONES)
    with pytest.raises(RunError, match="pass them with --sources") as raised:
        replay(run_dir, source=PAPER)
    assert raised.value.code == "missing-source"
    sources = tmp_path / "sources"
    sources.mkdir()
    body = ARXIV.pages["https://arxiv.org/pdf/2101.01234v2"].data
    (sources / bytes_digest(body).removeprefix("sha256:")).write_bytes(body)
    assert replay(run_dir, source=PAPER, sources=sources) == recorded_graph(run_dir)


def test_a_run_that_stores_its_sources_replays_without_them(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    start_run(PAPER, RUBRIC, run_dir, harness="h", model=MODEL, cited=CITED, store_source=True)
    classify(run_dir, CLASSIFY, model=MODEL)
    judge(run_dir, "s-8", "r-2", "https://arxiv.org/abs/2101.01234", JONES)
    assert len(list((run_dir / "sources").iterdir())) == 1
    assert replay(run_dir) == recorded_graph(run_dir)


def test_replay_refuses_a_reading_its_bytes_do_not_give(run_dir: Path, tmp_path: Path) -> None:
    judge(run_dir, "s-8", "r-2", "https://arxiv.org/abs/2101.01234", JONES)
    body = ARXIV.pages["https://arxiv.org/pdf/2101.01234v2"].data
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / bytes_digest(body).removeprefix("sha256:")).write_bytes(body)
    with pytest.MonkeyPatch.context() as patch:
        # The text the replay reads differs from the one the call recorded, as when the reader changed.
        patch.setattr(citation, "normalise_text", lambda text: text.upper())
        with pytest.raises(RunError, match="that its bytes do not give") as raised:
            replay(run_dir, source=PAPER, sources=sources)
    assert raised.value.code == "record-mismatch"


def test_a_finished_run_with_citation_checks_verifies(run_dir: Path, no_network: None) -> None:
    judge(run_dir, "s-7", "r-1", "smith2020.txt", PASSAGE)
    run_checks(run_dir, [RECOMPUTE], model=MODEL)
    finish_run(run_dir, model=MODEL)
    found = verify_run(run_dir)
    assert (found.rerun, found.not_rerun, found.differences) == (("c-002",), ("c-001",), ())
