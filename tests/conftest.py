"""Shared fixtures: the recorded run, and a network that refuses every connection."""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import anthropic
import pytest

from credencegraph_extract import provider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def recorded_run(tmp_path: Path) -> Path:
    """A copy of the recorded run, safe to modify."""
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURES / "run", run_dir)
    return run_dir


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every network connection and every provider call fail, and remove the API key."""

    def refuse(*args: object, **kwargs: object) -> None:
        msg = "network access during a test"
        raise AssertionError(msg)

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(anthropic, "Anthropic", refuse)
    monkeypatch.setattr(provider, "call", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
