"""The MCP server: every tool through the SDK's in-process client, and the server over stdio."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
import checks_fixture
import classify_fixture
import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from credencegraph_extract.mcp_server import server
from credencegraph_extract.session import recorded_graph, replay

MODEL = "test-model-1"
TOOLS = ["start_run", "get_state", "classify", "add_premises", "run_checks", "judge_citation", "finish_run"]
PASSAGE = "In comparable settings the bias stays below 1% for all segment lengths we tried"


async def _call(tool: str, args: dict[str, Any]) -> Any:
    async with Client(server) as client:
        result = await client.call_tool(tool, args)
    # A model's mistake is a JSON result, never a tool error or a protocol error.
    assert not result.is_error, [getattr(item, "text", item) for item in result.content]
    return result.structured_content


def call(tool: str, **args: Any) -> Any:
    """Call a tool through a fresh in-process client, as a harness would."""
    return anyio.run(_call, tool, args)


def error_of(result: Any) -> str:
    assert set(result) == {"error"}, result
    return result["error"]["code"]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run of the check fixture's paper, started and classified through the server."""
    target = tmp_path / "run"
    started = call(
        "start_run",
        source=str(checks_fixture.PAPER),
        rubric=str(checks_fixture.RUBRIC),
        run=str(target),
        harness="test-harness",
        model=MODEL,
        cited=str(checks_fixture.CITED),
    )
    assert [chunk["id"] for chunk in started["chunks"]][:3] == ["s-1", "s-2", "s-3"]
    assert call("classify", run=str(target), records=checks_fixture.CLASSIFY, model=MODEL)["refused"] == []
    return target


def test_the_server_lists_one_tool_per_agent_tool_with_its_rules() -> None:
    async def tools() -> Any:
        async with Client(server) as client:
            return (await client.list_tools()).tools

    listed = {tool.name: tool for tool in anyio.run(tools)}
    assert list(listed) == TOOLS
    assert all(tool.description for tool in listed.values())
    assert "fix" in listed["classify"].description
    assert "never a clause id" in listed["classify"].description
    assert "tied to the reference" in listed["judge_citation"].description
    assert "never paste" in listed["judge_citation"].description
    assert "no constant they do not write" in listed["run_checks"].description
    # Every argument is optional to the protocol, so a missing one is answered by the tool, as JSON.
    assert all(not tool.input_schema.get("required") for tool in listed.values())


def test_a_run_made_through_the_server_checks_finishes_and_replays(run_dir: Path, no_network: None) -> None:
    run = str(run_dir)
    checks = call("run_checks", run=run, proposals=[checks_fixture.PLAN[1], checks_fixture.PRODUCT], model=MODEL)
    assert [(item["target"], item["result"]) for item in checks["checks"]] == [("s-3", "passed"), ("e-2", "passed")]
    judged = call(
        "judge_citation",
        run=run,
        target="s-7",
        reference="r-1",
        source="smith2020.txt",
        passage=PASSAGE,
        supports=True,
        model=MODEL,
        agent="citations",
    )
    assert (judged["check"], judged["result"], judged["tied"]) == ("c-003", "passed", True)
    state = call("get_state", run=run)
    assert state["calls"] == 4
    assert state["finished"] is False
    finished = call("finish_run", run=run, model=MODEL)
    assert set(finished) == {"findings", "ok"}
    assert (run_dir / "graph.json").read_bytes() == recorded_graph(run_dir)
    assert replay(run_dir, source=checks_fixture.PAPER) == recorded_graph(run_dir)
    log = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["tool"] for line in log] == [
        "start_run",
        "classify",
        "run_checks",
        "judge_citation",
        "finish_run",
    ]
    assert log[3]["agent"] == "citations"


def test_premises_are_added_through_the_server(tmp_path: Path) -> None:
    run = str(tmp_path / "run")
    call(
        "start_run",
        source=str(classify_fixture.PAPER),
        rubric=str(classify_fixture.RUBRIC),
        run=run,
        harness="h",
        model=MODEL,
    )
    call("classify", run=run, records=classify_fixture.FIRST, model=MODEL)
    call("classify", run=run, records=classify_fixture.REPAIR, model=MODEL)
    first = call("add_premises", run=run, records=classify_fixture.PREMISES_FIRST, model=MODEL)
    assert [item["node"] for item in first["refused"]] == ["s-5"]
    assert all(item["fix"] for item in first["refused"])
    assert call("add_premises", run=run, records=classify_fixture.PREMISES_REPAIR, model=MODEL)["refused"] == []


@pytest.mark.parametrize(
    ("tool", "args", "code"),
    [
        ("get_state", {}, "invalid-argument"),
        ("get_state", {"run": 5}, "invalid-argument"),
        ("get_state", {"run": "no-such-run"}, "missing-run"),
        ("classify", {"records": [], "model": MODEL}, "invalid-argument"),
        ("finish_run", {"run": "no-such-run", "model": MODEL}, "missing-run"),
        (
            "start_run",
            {"source": "missing.tex", "rubric": "missing.toml", "run": "new", "harness": "h", "model": MODEL},
            "invalid-rubric",
        ),
    ],
)
def test_a_call_on_no_run_is_a_json_error(
    tool: str, args: dict[str, Any], code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert error_of(call(tool, **args)) == code


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("classify", {"records": {"node": "s-1"}, "model": MODEL}),
        ("classify", {"records": [], "model": ""}),
        ("run_checks", {"proposals": "check s-3", "model": MODEL}),
        ("add_premises", {"records": "s-9", "model": MODEL}),
        (
            "judge_citation",
            {
                "target": "s-7",
                "reference": "r-1",
                "source": "smith2020.txt",
                "passage": PASSAGE,
                "supports": "yes",
                "model": MODEL,
            },
        ),
        ("finish_run", {"model": 7}),
    ],
)
def test_arguments_the_run_refuses_are_a_json_error(run_dir: Path, tool: str, args: dict[str, Any]) -> None:
    assert error_of(call(tool, run=str(run_dir), **args)) == "invalid-argument"
    # A refused call is not logged.
    assert call("get_state", run=str(run_dir))["calls"] == 2


def test_starting_a_run_that_exists_is_a_json_error(run_dir: Path) -> None:
    args = {"source": str(checks_fixture.PAPER), "rubric": str(checks_fixture.RUBRIC), "harness": "h", "model": MODEL}
    assert error_of(call("start_run", run=str(run_dir), **args)) == "run-exists"
    assert (
        error_of(call("start_run", run=str(run_dir.parent / "other"), store_source="yes", **args)) == "invalid-argument"
    )


def test_records_the_run_refuses_come_back_with_a_fix(run_dir: Path) -> None:
    result = call("classify", run=str(run_dir), records=[{"node": "s-1 (method)", "type": "method"}], model=MODEL)
    [refused] = result["refused"]
    assert refused["fix"]
    citation = call(
        "judge_citation",
        run=str(run_dir),
        target="s-7",
        reference="smith2020",
        source="smith2020.txt",
        passage=PASSAGE,
        supports=True,
        model=MODEL,
    )
    assert citation["refused"][0]["fix"] == ['\'smith2020\' is the key of r-1: give "reference": "r-1"']


def test_a_url_the_fetch_policy_refuses_is_refused_as_a_result(run_dir: Path, no_network: None) -> None:
    result = call(
        "judge_citation",
        run=str(run_dir),
        target="s-8",
        reference="r-2",
        source="http://169.254.169.254/latest/meta-data/",
        passage="the bias grows",
        supports=True,
        model=MODEL,
    )
    assert result["refused"][0]["code"] == "refused-url"


def test_the_command_serves_the_tools_over_stdio(tmp_path: Path) -> None:
    async def session() -> tuple[list[str], Any]:
        command = StdioServerParameters(
            command=sys.executable, args=["-m", "credencegraph_extract", "mcp"], cwd=str(tmp_path)
        )
        with anyio.fail_after(60):
            async with Client(command) as client:
                names = [tool.name for tool in (await client.list_tools()).tools]
                result = await client.call_tool("get_state", {"run": "no-such-run"})
        return names, result.structured_content

    names, result = anyio.run(session)
    assert names == TOOLS
    assert error_of(result) == "missing-run"


# Each tool's arguments, valid for a run of the check fixture; the sweep replaces one at a time.
VALID = {
    "start_run": {
        "source": str(checks_fixture.PAPER),
        "rubric": str(checks_fixture.RUBRIC),
        "harness": "h",
        "model": MODEL,
    },
    "get_state": {},
    "classify": {"records": [], "model": MODEL},
    "add_premises": {"records": [], "model": MODEL},
    "run_checks": {"proposals": [], "model": MODEL},
    "judge_citation": {
        "target": "s-7",
        "reference": "r-1",
        "source": "smith2020.txt",
        "passage": PASSAGE,
        "supports": True,
        "model": MODEL,
    },
    "finish_run": {"model": MODEL},
}
OPTIONAL = {
    "start_run": {
        "agent": "a",
        "note": "n",
        "document": "paper",
        "cited": str(checks_fixture.CITED),
        "store_source": False,
    }
}
WRONG = [7, 1.5, True, None, [], ["x"], {}, {"x": 1}, ""]


async def _sweep(run_dir: Path) -> list[str]:
    """Call every tool with each argument replaced by each wrong JSON value; return the calls that were not JSON."""
    failures = []
    async with Client(server) as client:
        for tool, valid in VALID.items():
            args = {**valid, **OPTIONAL.get(tool, {}), "agent": "a", "note": "n"}
            args["run"] = str(run_dir)
            for name in args:
                for i, wrong in enumerate(WRONG):
                    if tool == "start_run" and name != "run":
                        # A new directory each time, so a call that succeeds does not refuse the next as run-exists.
                        args["run"] = str(run_dir.parent / f"fresh-{name}-{i}")
                    result = await client.call_tool(tool, {**args, name: wrong})
                    if result.is_error or not isinstance(result.structured_content, dict):
                        failures.append(f"{tool}({name}={wrong!r})")
    return failures


def test_no_argument_of_any_wrong_type_makes_a_protocol_error(run_dir: Path, no_network: None) -> None:
    # A model can send any JSON value for any argument; every one must come back as a JSON result.
    assert anyio.run(_sweep, run_dir) == []
