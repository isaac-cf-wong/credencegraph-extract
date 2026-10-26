"""Shared fixtures: the recorded run, and a network that refuses every connection."""

from __future__ import annotations

import ipaddress
import shutil
import socket
from pathlib import Path
from typing import Any

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


def _loopback(address: object) -> bool:
    """Whether a socket address is on this machine's loopback interface."""
    host = address[0] if isinstance(address, tuple) and address else None
    try:
        return isinstance(host, str) and ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every network connection and every provider call fail, and remove the API key.

    A connection to the loopback interface is let through: an event loop connects a socket pair there
    on Windows, where it has no native one, and it never leaves the machine.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        msg = "network access during a test"
        raise AssertionError(msg)

    def guarded(original: Any) -> Any:
        def connect(sock: socket.socket, address: object, *args: object) -> Any:
            if not _loopback(address):
                refuse()
            return original(sock, address, *args)

        return connect

    monkeypatch.setattr(socket.socket, "connect", guarded(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(socket.socket.connect_ex))

    def addressed(original: Any, address_of: Any) -> Any:
        def send(sock: socket.socket, *args: object) -> Any:
            address = address_of(args)
            if address is not None and not _loopback(address):
                refuse()
            return original(sock, *args)

        return send

    # sendto(data, address) or sendto(data, flags, address); sendmsg(buffers, ancdata, flags, address).
    monkeypatch.setattr(socket.socket, "sendto", addressed(socket.socket.sendto, lambda args: args[-1]))
    if hasattr(socket.socket, "sendmsg"):  # Unix only
        sendmsg_address = lambda args: args[3] if len(args) > 3 else None  # noqa: E731
        monkeypatch.setattr(socket.socket, "sendmsg", addressed(socket.socket.sendmsg, sendmsg_address))
    monkeypatch.setattr(socket, "create_connection", refuse)
    for resolver in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        monkeypatch.setattr(socket, resolver, refuse)
    monkeypatch.setattr(anthropic, "Anthropic", refuse)
    monkeypatch.setattr(provider, "call", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
