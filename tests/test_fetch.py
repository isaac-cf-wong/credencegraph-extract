"""The fetch policy: public addresses only, connected to as checked, on every redirect hop; offline fakes."""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from credencegraph_extract.fetch import FetchError, fetch, forbidden

PUBLIC = "93.184.216.34"
OTHER_PUBLIC = "8.8.4.4"


@dataclass
class Server:
    """A fake network: names, the routes each host answers, and a record of what was asked."""

    names: dict[str, list[str]] = field(default_factory=dict)
    rebinding: dict[str, list[str]] = field(default_factory=dict)
    routes: dict[tuple[str, str], tuple[int, dict[str, str], bytes]] = field(default_factory=dict)
    peers: dict[str, str] = field(default_factory=dict)
    resolved: list[str] = field(default_factory=list)
    connected: list[tuple[str, int, str, bool]] = field(default_factory=list)
    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)

    def resolve(self, host: str, port: int) -> list[str]:
        """Answer every address of a name, or, for a rebinding name, the next address at each call."""
        self.resolved.append(host)
        if host in self.rebinding:
            return [self.rebinding[host].pop(0)]
        return list(self.names[host])

    def connect(self, address: str, port: int, host: str, tls: bool, timeout: float) -> Any:
        """Open a fake connection to an address, answering from the routes of the host name."""
        self.connected.append((address, port, host, tls))
        server = self

        class Connection:
            def peer(self) -> str:
                return server.peers.get(address, address)

            def get(
                self, target: str, headers: Mapping[str, str]
            ) -> tuple[int, Mapping[str, str], Callable[[int], bytes]]:
                server.requests.append((address, target, dict(headers)))
                status, response, body = server.routes[(host, target)]
                return status, response, io.BytesIO(body).read

            def close(self) -> None:
                pass

        return Connection()

    def fetch(self, url: str, **kwargs: Any) -> Any:
        """Fetch through this fake network."""
        return fetch(url, resolve=self.resolve, connect=self.connect, **kwargs)


def ok(body: bytes = b"text", content_type: str = "text/plain; charset=utf-8") -> tuple[int, dict[str, str], bytes]:
    return 200, {"content-type": content_type}, body


def moved(location: str) -> tuple[int, dict[str, str], bytes]:
    return 302, {"location": location}, b""


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "100.64.0.1",
        "169.254.169.254",
        "0.0.0.0",  # noqa: S104 - an address the policy refuses
        "224.0.0.1",
        "239.255.255.250",
        "255.255.255.255",
        "240.0.0.1",
        "::1",
        "::",
        "fe80::1",
        "fd00:ec2::254",
        "ff02::1",
        "ff0e::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "2002:7f00:1::",
        "2001:0:4136:e378:8000:63bf:80ff:fffe",  # Teredo carrying 127.0.0.1
        "not-an-address",
    ],
)
def test_an_address_that_is_not_public_unicast_is_forbidden(address: str) -> None:
    assert forbidden(address) is not None


@pytest.mark.parametrize("address", [PUBLIC, "2606:4700::1111", "64:ff9b::808:808", "::ffff:8.8.8.8"])
def test_a_public_unicast_address_is_allowed(address: str) -> None:
    assert forbidden(address) is None


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.org/paper.pdf",
        "gopher://example.org/",
        "https://user:secret@example.org/paper.pdf",
        "https:///paper.pdf",
        "https://example.org:99999/",
    ],
)
def test_a_url_of_another_form_is_refused_before_any_lookup(url: str) -> None:
    server = Server()
    with pytest.raises(FetchError) as raised:
        server.fetch(url)
    assert raised.value.code == "refused-url"
    assert server.resolved == []
    assert server.connected == []


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/", "http://[::1]:8080/v1", "http://169.254.169.254/latest/meta-data/"]
)
def test_a_literal_private_address_is_refused_without_connecting(url: str) -> None:
    server = Server()
    with pytest.raises(FetchError, match="not a public address") as raised:
        server.fetch(url)
    assert raised.value.code == "refused-url"
    assert server.connected == []


@pytest.mark.parametrize(
    ("addresses", "reason"),
    [
        (["127.0.0.1"], "127.0.0.1"),
        (["10.1.2.3"], "10.1.2.3"),
        (["169.254.169.254"], "169.254.169.254"),
        # A name with one private address among public ones is refused, whichever would be chosen.
        ([PUBLIC, "192.168.0.10"], "192.168.0.10"),
    ],
)
def test_a_name_resolving_to_a_private_address_is_refused_without_connecting(addresses: list[str], reason: str) -> None:
    server = Server(names={"local.example": addresses})
    with pytest.raises(FetchError, match=reason) as raised:
        server.fetch("https://local.example/paper.pdf")
    assert raised.value.code == "refused-url"
    assert server.connected == []


def test_the_connection_goes_to_the_checked_address_with_the_name_kept_for_host_and_tls() -> None:
    server = Server(names={"papers.example": [PUBLIC]}, routes={("papers.example", "/a.pdf?v=2"): ok(b"%PDF")})
    fetched = server.fetch("https://papers.example/a.pdf?v=2")
    assert server.connected == [(PUBLIC, 443, "papers.example", True)]
    assert server.requests[0][1] == "/a.pdf?v=2"
    assert server.requests[0][2]["Host"] == "papers.example"
    assert (fetched.final_url, fetched.content_type, fetched.data) == (
        "https://papers.example/a.pdf?v=2",
        "text/plain",
        b"%PDF",
    )


def test_a_port_other_than_the_default_is_kept_in_the_host_header() -> None:
    server = Server(names={"papers.example": [PUBLIC]}, routes={("papers.example", "/"): ok()})
    server.fetch("http://papers.example:8080")
    assert server.connected == [(PUBLIC, 8080, "papers.example", False)]
    assert server.requests[0][2]["Host"] == "papers.example:8080"


def test_a_rebinding_resolver_cannot_move_the_connection_to_a_private_address() -> None:
    # The name resolves to a public address when checked and to loopback afterwards: the connection is
    # opened to the address checked, and the name is resolved once.
    server = Server(rebinding={"rebind.example": [PUBLIC, "127.0.0.1"]}, routes={("rebind.example", "/"): ok()})
    server.fetch("http://rebind.example/")
    assert server.resolved == ["rebind.example"]
    assert [address for address, *_ in server.connected] == [PUBLIC]


@pytest.mark.parametrize(("peer", "match"), [("127.0.0.1", "not a public address"), (OTHER_PUBLIC, "not the checked")])
def test_a_connection_that_reached_another_address_sends_nothing(peer: str, match: str) -> None:
    server = Server(names={"papers.example": [PUBLIC]}, routes={("papers.example", "/"): ok()}, peers={PUBLIC: peer})
    with pytest.raises(FetchError, match=match) as raised:
        server.fetch("http://papers.example/")
    assert raised.value.code == "refused-url"
    assert server.requests == []


@pytest.mark.parametrize(
    "location",
    [
        "http://10.0.0.5/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://internal.example/",
        "file:///etc/passwd",
    ],
)
def test_a_redirect_into_a_private_address_is_refused(location: str) -> None:
    server = Server(
        names={"papers.example": [PUBLIC], "internal.example": ["10.9.8.7"]},
        routes={("papers.example", "/paper"): moved(location)},
    )
    with pytest.raises(FetchError) as raised:
        server.fetch("https://papers.example/paper")
    assert raised.value.code == "refused-url"
    assert [address for address, *_ in server.connected] == [PUBLIC]


def test_redirects_are_followed_hop_by_hop_and_recorded() -> None:
    server = Server(
        names={"doi.example": [PUBLIC], "publisher.example": ["2606:4700::1111"]},
        routes={
            ("doi.example", "/10.1/x"): moved("https://publisher.example/article/7"),
            ("publisher.example", "/article/7"): moved("/article/7/fulltext"),
            ("publisher.example", "/article/7/fulltext"): ok(b"<p>text</p>", "text/html"),
        },
    )
    fetched = server.fetch("https://doi.example/10.1/x")
    assert fetched.hops == (
        "https://doi.example/10.1/x",
        "https://publisher.example/article/7",
        "https://publisher.example/article/7/fulltext",
    )
    assert fetched.final_url == "https://publisher.example/article/7/fulltext"
    assert server.resolved == ["doi.example", "publisher.example", "publisher.example"]


def test_too_many_redirects_fail() -> None:
    server = Server(names={"loop.example": [PUBLIC]}, routes={("loop.example", "/"): moved("/")})
    with pytest.raises(FetchError, match="redirects more than 2 times") as raised:
        server.fetch("http://loop.example/", max_hops=3)
    assert raised.value.code == "fetch-failed"
    assert len(server.connected) == 3


def test_a_body_over_the_cap_fails() -> None:
    server = Server(names={"big.example": [PUBLIC]}, routes={("big.example", "/"): ok(b"x" * 11)})
    with pytest.raises(FetchError, match="larger than 10 bytes"):
        server.fetch("http://big.example/", max_bytes=10)
    assert server.fetch("http://big.example/", max_bytes=11).data == b"x" * 11


def test_an_answer_other_than_200_fails() -> None:
    server = Server(names={"gone.example": [PUBLIC]}, routes={("gone.example", "/"): (404, {}, b"not found")})
    with pytest.raises(FetchError, match="answered 404") as raised:
        server.fetch("http://gone.example/")
    assert raised.value.code == "fetch-failed"


def test_a_name_that_does_not_resolve_fails() -> None:
    def fail(host: str, port: int) -> list[str]:
        raise OSError("no such name")

    with pytest.raises(FetchError, match="cannot resolve") as raised:
        fetch("http://nowhere.example/", resolve=fail)
    assert raised.value.code == "fetch-failed"


def test_a_body_dripped_slowly_stops_at_the_time_limit() -> None:
    # Each piece arrives within the per-read timeout, but the whole fetch may not take longer than its limit.
    class Drip:
        def peer(self) -> str:
            return PUBLIC

        def get(self, target: str, headers: Mapping[str, str]) -> tuple[int, Mapping[str, str], Callable[[int], bytes]]:
            def read(size: int) -> bytes:
                time.sleep(0.02)
                return b"x"

            return 200, {"content-type": "text/plain"}, read

        def close(self) -> None:
            pass

    started = time.monotonic()
    with pytest.raises(FetchError, match="longer than its time limit") as raised:
        fetch(
            "http://slow.example/", resolve=lambda host, port: [PUBLIC], connect=lambda *args: Drip(), max_seconds=0.2
        )
    assert raised.value.code == "fetch-failed"
    assert time.monotonic() - started < 1


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("http://[2606:4700::1111]/x", "[2606:4700::1111]"),
        ("http://[2606:4700::1111]:8080/x", "[2606:4700::1111]:8080"),
        ("https://例え.テスト/paper", "xn--r8jz45g.xn--zckzah"),
        ("https://Papers.Example/paper", "papers.example"),
    ],
)
def test_the_host_header_is_the_ascii_authority(url: str, host: str) -> None:
    server = Server(names={"xn--r8jz45g.xn--zckzah": [PUBLIC], "papers.example": [PUBLIC]})
    target = "/paper" if "paper" in url else "/x"
    server.routes = {(name, target): ok() for name in ("xn--r8jz45g.xn--zckzah", "papers.example", "2606:4700::1111")}
    server.fetch(url)
    assert server.requests[0][2]["Host"] == host


def test_a_host_that_is_not_a_domain_name_is_refused() -> None:
    server = Server()
    with pytest.raises(FetchError, match="not a valid domain name") as raised:
        server.fetch("https://" + "a" * 64 + ".example/")
    assert raised.value.code == "refused-url"
    assert server.resolved == []
