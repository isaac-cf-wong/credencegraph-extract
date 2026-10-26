"""The CLI fallback: ``tool`` makes the agent-tool calls the MCP server makes, and ``unlock`` reports the lock."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import anyio
import checks_fixture
import classify_fixture
import pytest
from mcp.client.client import Client

from credencegraph_extract import session
from credencegraph_extract.cli import main
from credencegraph_extract.mcp_server import server
from credencegraph_extract.run import RunError

MODEL = "test-model-1"


def tool(capsys: pytest.CaptureFixture[str], name: str, run: Path, **args: Any) -> Any:
    """Make a tool call through the CLI; return its JSON result."""
    assert main(["tool", name, "--run", str(run), "--args", json.dumps(args)]) == 0
    out = capsys.readouterr().out
    return json.loads(out)


def refused(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    """Make a CLI call the run refuses; return its one error line."""
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    return lines[0]


# The scenario of the graph tools' tests: classify, correct the refused records, add premises, finish.
SCENARIO: list[tuple[str, dict[str, Any]]] = [
    ("classify", {"records": classify_fixture.FIRST}),
    ("classify", {"records": classify_fixture.REPAIR}),
    ("add_premises", {"records": classify_fixture.PREMISES_FIRST}),
    ("add_premises", {"records": classify_fixture.PREMISES_REPAIR}),
    ("finish_run", {}),
]
START = {"source": str(classify_fixture.PAPER), "rubric": str(classify_fixture.RUBRIC), "harness": "h", "model": MODEL}


def anonymous(run: Path) -> str:
    """A run's graph with its run id replaced, so two runs of one scenario can be compared byte for byte."""
    run_id = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["run"]
    return (run / "graph.json").read_text(encoding="utf-8").replace(run_id, "<run>")


async def _over_mcp(run: Path) -> list[Any]:
    results = []
    async with Client(server) as client:
        for name, args in [("start_run", START), *SCENARIO]:
            result = await client.call_tool(name, {"run": str(run), "model": MODEL, **args})
            results.append(result.structured_content)
    return results


def test_the_scenario_through_the_cli_gives_the_graph_the_mcp_server_gives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_run = tmp_path / "cli"
    results = [tool(capsys, "start_run", cli_run, **START)]
    results += [tool(capsys, name, cli_run, model=MODEL, **args) for name, args in SCENARIO]
    mcp_results = anyio.run(_over_mcp, tmp_path / "mcp")
    assert anonymous(cli_run) == anonymous(tmp_path / "mcp")
    # Each call's result too, but for the run id start_run returns.
    assert results[1:] == mcp_results[1:]
    assert [item["node"] for item in results[3]["refused"]] == ["s-5"]


def test_the_arguments_may_come_from_stdin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"records": classify_fixture.REPAIR, "model": MODEL})))
    assert main(["tool", "classify", "--run", str(run), "--args", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["refused"] == []


def test_refused_records_are_a_result_not_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    result = tool(capsys, "classify", run, records=[{"node": "s-1 (method)", "type": "method"}], model=MODEL)
    assert result["refused"][0]["fix"]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ("{not json", "invalid-argument"),
        ("[1, 2]", "invalid-argument"),
        ('{"recordz": [], "model": "m"}', "invalid-argument"),
        ('{"run": "elsewhere", "records": [], "model": "m"}', "invalid-argument"),
        ('{"records": "s-1", "model": "m"}', "invalid-argument"),
    ],
)
def test_a_call_the_run_refuses_is_one_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: str, code: str
) -> None:
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    line = refused(capsys, ["tool", "classify", "--run", str(run), "--args", args])
    assert line.startswith(f"error [{code}]: ")
    assert len(session.read_calls(run)) == 1


def test_an_unknown_argument_names_the_ones_the_tool_takes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    line = refused(capsys, ["tool", "get_state", "--run", str(tmp_path), "--args", '{"verbose": true}'])
    assert line == "error [invalid-argument]: get_state takes no ['verbose']; it takes ['run']"


def test_a_call_on_no_run_is_one_error_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    line = refused(capsys, ["tool", "get_state", "--run", str(tmp_path / "missing")])
    assert line.startswith("error [missing-run]: ")


def test_unlock_reports_a_free_lock_and_removes_a_leftover_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    # A process that died holding the lock left its record; the operating system released the lock.
    (run / session.LOCK_OWNER).write_text(json.dumps({"pid": 999999, "since": "then"}), encoding="utf-8")
    assert main(["unlock", str(run)]) == 0
    assert capsys.readouterr().out == f"{run}: no tool call holds the lock\n"
    assert not (run / session.LOCK_OWNER).exists()


def test_unlock_names_the_process_holding_the_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    with session.lock(run):
        assert main(["unlock", str(run)]) == 1
        out = capsys.readouterr().out
        assert f"held by process {os.getpid()}" in out
        assert "stop it if it is hung" in out
        with pytest.raises(RunError, match=f"process {os.getpid()}") as busy, session.lock(run, timeout=0.1):
            pass
        assert busy.value.code == "run-busy"
    assert not (run / session.LOCK_OWNER).exists()
    assert main(["unlock", str(run)]) == 0


def test_unlock_refuses_a_directory_that_is_not_a_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert refused(capsys, ["unlock", str(tmp_path)]).startswith("error [missing-run]: ")


def test_a_run_made_through_the_cli_replays(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    run = tmp_path / "run"
    start = {"source": str(checks_fixture.PAPER), "rubric": str(checks_fixture.RUBRIC), "harness": "h", "model": MODEL}
    tool(capsys, "start_run", run, cited=str(checks_fixture.CITED), **start)
    tool(capsys, "classify", run, records=checks_fixture.CLASSIFY, model=MODEL)
    tool(capsys, "run_checks", run, proposals=[checks_fixture.PLAN[1]], model=MODEL)
    tool(capsys, "finish_run", run, model=MODEL)
    assert session.replay(run, source=checks_fixture.PAPER) == (run / "graph.json").read_bytes()


async def _invented_over_mcp(run: Path, name: str) -> Any:
    async with Client(server) as client:
        result = await client.call_tool(name, {"run": str(run), "model": MODEL, "verbose": True})
    assert not result.is_error
    return result.structured_content


@pytest.mark.parametrize(
    "name", ["start_run", "get_state", "classify", "add_premises", "run_checks", "judge_citation", "finish_run"]
)
def test_an_invented_argument_is_refused_alike_over_mcp_and_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    # The protocol would drop it silently; both paths refuse it, with the same message.
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    over_mcp = anyio.run(_invented_over_mcp, run, name)
    line = refused(capsys, ["tool", name, "--run", str(run), "--args", json.dumps({"model": MODEL, "verbose": True})])
    assert line == f"error [{over_mcp['error']['code']}]: {over_mcp['error']['message']}"
    assert "'verbose'" in line.split("; it takes")[0]
    assert len(session.read_calls(run)) == 1


def test_unlock_removes_a_leftover_record_while_it_holds_the_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Removed after the lock is released, the record of a call that took the lock in between would go.
    run = tmp_path / "run"
    tool(capsys, "start_run", run, **START)
    (run / session.LOCK_OWNER).write_text("{}", encoding="utf-8")
    seen = []
    release = session._unlock

    def unlock(handle: Any) -> None:
        seen.append((run / session.LOCK_OWNER).exists())
        release(handle)

    monkeypatch.setattr(session, "_unlock", unlock)
    assert session.lock_holder(run) is None
    assert seen == [False]
