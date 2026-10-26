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


def test_no_network_refuses_datagrams_to_the_network(no_network: None) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        sends = [
            lambda: udp.sendto(b"x", ("93.184.216.34", 53)),
            lambda: udp.sendto(b"x", 0, ("93.184.216.34", 53)),
        ]
        if hasattr(udp, "sendmsg"):  # Unix only
            sends.append(lambda: udp.sendmsg([b"x"], [], 0, ("93.184.216.34", 53)))
        for send in sends:
            with pytest.raises(AssertionError, match="network access during a test"):
                send()
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with receiver:
            receiver.bind(("127.0.0.1", 0))
            udp.sendto(b"x", receiver.getsockname())
            assert receiver.recv(1) == b"x"


@pytest.mark.parametrize(
    ("resolver", "args"),
    [
        ("getaddrinfo", ("example.org", 80)),
        ("gethostbyname", ("example.org",)),
        ("gethostbyname_ex", ("example.org",)),
        ("gethostbyaddr", ("93.184.216.34",)),
        ("getnameinfo", (("93.184.216.34", 80), 0)),
    ],
)
def test_no_network_refuses_every_name_lookup(no_network: None, resolver: str, args: tuple[object, ...]) -> None:
    with pytest.raises(AssertionError, match="network access during a test"):
        getattr(socket, resolver)(*args)
