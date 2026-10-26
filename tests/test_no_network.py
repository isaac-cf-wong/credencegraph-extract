"""The no_network fixture: it refuses the network and lets loopback through."""

from __future__ import annotations

import socket

import pytest


def test_no_network_refuses_the_network_and_lets_loopback_through(no_network: None) -> None:
    # An event loop's socket pair on Windows connects over loopback; anything else is the network.
    listener = socket.create_server(("127.0.0.1", 0))
    with listener, socket.socket() as client:
        client.connect(listener.getsockname())
    with socket.socket() as outside, pytest.raises(AssertionError, match="network access during a test"):
        outside.connect(("93.184.216.34", 80))
    with socket.socket(socket.AF_INET6) as outside, pytest.raises(AssertionError, match="network access"):
        outside.connect(("2606:4700::1111", 80))
