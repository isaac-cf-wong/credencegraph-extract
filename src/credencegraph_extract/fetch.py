"""Fetching a URL an agent names, without letting it reach the host's own network.

``judge_citation`` reads any ``http`` or ``https`` URL the agent gives it. Fetched naively, such a URL
could reach the machine the tools run on or its network: a local model server, a router, a cloud
metadata service. ``fetch`` therefore:

- accepts ``http`` and ``https`` only, with no user name or password in the URL;
- resolves the host and refuses it when any address it resolves to is not a public unicast address:
  loopback, private, shared, link-local (the cloud metadata address included), unique-local,
  multicast, unspecified, reserved and documentation ranges, in IPv4 and IPv6, and an IPv6 address that
  carries an IPv4 one (mapped, NAT64, 6to4, Teredo) is judged by the IPv4 address it carries;
- connects to the address it checked, never to the name again, keeping the name for the ``Host``
  header and for TLS (the server name sent and the certificate verified against it), and checks the
  address the connection reached before sending the request, so a name that resolves to a public
  address when checked and a private one when connected (DNS rebinding) cannot reach the private one;
- follows redirects itself, applying all of the above to every hop, up to ``MAX_HOPS``;
- gives up after ``TIMEOUT`` seconds per connection and refuses a body larger than ``MAX_BYTES``.

A local file is never named by a URL: ``file:`` and every other scheme are refused.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

MAX_HOPS = 5
TIMEOUT = 30.0
MAX_BYTES = 50 * 1024 * 1024
USER_AGENT = "credencegraph-extract (citation check)"
REDIRECTS = frozenset({301, 302, 303, 307, 308})
SCHEMES = {"http": 80, "https": 443}

# The NAT64 prefix of RFC 6052, which embeds an IPv4 address in its last 32 bits; ipaddress counts it
# as global.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class FetchError(Exception):
    """A URL is refused or cannot be fetched.

    Attributes:
        code: ``refused-url`` for a URL the policy refuses, ``fetch-failed`` for one that could not be read.
    """

    def __init__(self, code: str, message: str) -> None:
        """Make the error.

        Args:
            code: ``refused-url`` or ``fetch-failed``.
            message: What was refused or failed, in words.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Fetched:
    """What a fetch read.

    Attributes:
        url: The URL asked for.
        final_url: The URL the body came from, after redirects.
        hops: Every URL requested, in order, the first and the final included.
        content_type: The media type of the body, without parameters, in lower case.
        data: The body.
    """

    url: str
    final_url: str
    hops: tuple[str, ...]
    content_type: str
    data: bytes


class Connection(Protocol):
    """An HTTP connection opened to one address."""

    def peer(self) -> str:
        """Return the address the connection reached."""
        ...

    def get(self, target: str, headers: Mapping[str, str]) -> tuple[int, Mapping[str, str], Callable[[int], bytes]]:
        """Send a GET request; return the status, the headers by lower-case name, and a reader of the body."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


# Resolves a host name and port to addresses.
Resolve = Callable[[str, int], Sequence[str]]
# Opens a connection to an address: (address, port, host name, TLS?, timeout).
Connect = Callable[[str, int, str, bool, float], Connection]


def forbidden(address: str) -> str | None:
    """Say why an address may not be fetched from, or ``None`` for a public unicast address.

    Args:
        address: An IPv4 or IPv6 address.

    Returns:
        The reason, or ``None``.
    """
    try:
        ip = _ip(address)
    except ValueError:
        return f"{address!r} is not an IP address"
    if isinstance(ip, ipaddress.IPv6Address):
        carried = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if carried is None and ip in _NAT64:
            carried = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if carried is not None:
            reason = forbidden(str(carried))
            return f"{ip} carries {carried}, which {reason.split(' ', 1)[1]}" if reason else None
    if ip.is_multicast:
        return f"{ip} is a multicast address"
    if not ip.is_global:
        return f"{ip} is not a public address (loopback, private, link-local or reserved)"
    return None


def _resolve(host: str, port: int) -> list[str]:
    """Resolve a host name to every address it has."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


class _HTTPConnection(http.client.HTTPConnection):
    """An HTTP connection to a fixed address, with the host name only in the request."""

    def __init__(self, address: str, port: int, timeout: float) -> None:
        super().__init__(address, port, timeout=timeout)


class _HTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection to a fixed address, with the host name kept for the server name and the certificate."""

    def __init__(self, address: str, port: int, host: str, timeout: float) -> None:
        super().__init__(address, port, timeout=timeout, context=ssl.create_default_context())
        self._server_name = host

    def connect(self) -> None:
        """Connect to the address, then wrap the socket for the host name, verifying its certificate."""
        http.client.HTTPConnection.connect(self)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self._server_name)  # type: ignore[attr-defined]


class _Socket:
    """The connection ``fetch`` uses by default: http.client to a fixed address."""

    def __init__(self, address: str, port: int, host: str, tls: bool, timeout: float) -> None:
        self._connection: http.client.HTTPConnection = (
            _HTTPSConnection(address, port, host, timeout) if tls else _HTTPConnection(address, port, timeout)
        )
        self._connection.connect()

    def peer(self) -> str:
        sock = self._connection.sock
        assert sock is not None  # noqa: S101 - connected in __init__
        return str(sock.getpeername()[0])

    def get(self, target: str, headers: Mapping[str, str]) -> tuple[int, Mapping[str, str], Callable[[int], bytes]]:
        self._connection.putrequest("GET", target, skip_host=True, skip_accept_encoding=True)
        for name, value in headers.items():
            self._connection.putheader(name, value)
        self._connection.endheaders()
        response = self._connection.getresponse()
        return response.status, {name.lower(): value for name, value in response.getheaders()}, response.read

    def close(self) -> None:
        self._connection.close()


def _connect(address: str, port: int, host: str, tls: bool, timeout: float) -> Connection:
    """Open the default connection."""
    return _Socket(address, port, host, tls, timeout)


def _split(url: str) -> tuple[str, str, int, str]:
    """Check a URL's form; return its scheme, host, port and request target."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise FetchError("refused-url", f"{url!r} is not a URL: {error}") from None
    if parts.scheme not in SCHEMES:
        raise FetchError("refused-url", f"{url!r} is not an http or https URL")
    if parts.username is not None or parts.password is not None:
        raise FetchError("refused-url", f"{url!r} carries a user name or password")
    if not parts.hostname:
        raise FetchError("refused-url", f"{url!r} names no host")
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return parts.scheme, parts.hostname, port or SCHEMES[parts.scheme], target


def _address(host: str, port: int, resolve: Resolve) -> str:
    """Resolve a host and check every address; return the one to connect to."""
    try:
        addresses = [host] if _literal(host) else list(resolve(host, port))
    except OSError as error:
        raise FetchError("fetch-failed", f"cannot resolve {host!r}: {error}") from None
    if not addresses:
        raise FetchError("fetch-failed", f"{host!r} resolves to no address")
    for address in addresses:
        reason = forbidden(address)
        if reason is not None:
            raise FetchError("refused-url", f"{host!r} resolves to {address}: {reason}")
    return addresses[0]


def _ip(address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """An address as a value, without an IPv6 zone; ``forbidden`` has accepted it."""
    return ipaddress.ip_address(address.split("%", 1)[0])


def _literal(host: str) -> bool:
    """Whether a URL's host is an IP address rather than a name."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _body(read: Callable[[int], bytes], max_bytes: int) -> bytes:
    """Read a body, refusing one larger than ``max_bytes``."""
    data = read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FetchError("fetch-failed", f"the body is larger than {max_bytes} bytes")
    return data


def _hop(
    url: str, resolve: Resolve, connect: Connect, timeout: float, max_bytes: int
) -> tuple[int, Mapping[str, str], bytes]:
    """Make one request under the policy: check, resolve, check, connect to the checked address, check the peer."""
    scheme, host, port, target = _split(url)
    address = _address(host, port, resolve)
    try:
        connection = connect(address, port, host, scheme == "https", timeout)
    except (OSError, ssl.SSLError, http.client.HTTPException) as error:
        raise FetchError("fetch-failed", f"cannot connect to {host!r}: {error}") from None
    try:
        peer = connection.peer()
        reason = forbidden(peer)
        if reason is not None:
            raise FetchError("refused-url", f"the connection to {host!r} reached {peer}: {reason}")
        if _ip(peer) != _ip(address):
            raise FetchError("refused-url", f"the connection to {host!r} reached {peer}, not the checked {address}")
        default = SCHEMES[scheme] == port
        headers = {"Host": host if default else f"{host}:{port}", "User-Agent": USER_AGENT, "Accept": "*/*"}
        status, response_headers, read = connection.get(target, headers)
        data = b"" if status in REDIRECTS else _body(read, max_bytes)
    except (OSError, ssl.SSLError, http.client.HTTPException) as error:
        raise FetchError("fetch-failed", f"cannot read {url!r}: {error}") from None
    finally:
        connection.close()
    return status, response_headers, data


def fetch(  # noqa: PLR0913 - the URL, the policy's limits, and the two seams for tests
    url: str,
    *,
    resolve: Resolve = _resolve,
    connect: Connect = _connect,
    max_hops: int = MAX_HOPS,
    timeout: float = TIMEOUT,
    max_bytes: int = MAX_BYTES,
) -> Fetched:
    """Fetch a URL under the policy of this module, following redirects.

    Args:
        url: An ``http`` or ``https`` URL.
        resolve: Resolves a host name; the system resolver by default.
        connect: Opens a connection to an address; http.client by default.
        max_hops: The most requests made, the first included.
        timeout: Seconds per connection.
        max_bytes: The largest body read.

    Returns:
        What was read.

    Raises:
        FetchError: ``refused-url`` when the URL, a redirect or an address is refused; ``fetch-failed``
            when the body cannot be read, is too large, or the final response is not 200.
    """
    hops: list[str] = []
    current = url
    while True:
        hops.append(current)
        status, headers, data = _hop(current, resolve, connect, timeout, max_bytes)
        if status in REDIRECTS:
            location = headers.get("location")
            if not location:
                raise FetchError("fetch-failed", f"{current!r} redirects without a location")
            if len(hops) >= max_hops:
                raise FetchError("fetch-failed", f"{url!r} redirects more than {max_hops - 1} times")
            current = urljoin(current, location)
            continue
        if status != http.client.OK:
            raise FetchError("fetch-failed", f"{current!r} answered {status}")
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        return Fetched(url, current, tuple(hops), content_type, data)


def describe(fetched: Fetched) -> dict[str, Any]:
    """What a run records of a fetch: the URLs and the content type, never the body.

    Args:
        fetched: What a fetch read.

    Returns:
        ``url``, ``final_url``, ``hops`` and ``content_type``.
    """
    return {
        "url": fetched.url,
        "final_url": fetched.final_url,
        "hops": list(fetched.hops),
        "content_type": fetched.content_type,
    }
