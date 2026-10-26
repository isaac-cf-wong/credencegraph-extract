"""Reading the source a citation check quotes, and deciding whether it is tied to the reference.

``judge_citation`` reads the source itself: it never takes text the agent pastes. What it read is kept
as an **observation**, a JSON record logged with the call: which file or URL, the digest of the bytes,
the digest of their normalised text, where the quoted passage is in that text, and what ties the source
to the reference. The bytes themselves are not logged. The decision the check rests on (``tie``) is made
from the observation, the reference's bibliography entry and the digests ``start_run`` pinned, so the run
can be rebuilt from its log without reading anything again; ``replay`` reads the bytes again
(``verify_observation``) and refuses an observation they do not give.

A source is **tied to the reference** when it is one of:

1. the file named by the entry's key in the run's copy of the cited works, ``cited/``, while its bytes
   still have the digest ``start_run`` recorded;
2. the arXiv paper the entry names, fetched by the tool from the identifier in the entry (never the
   agent's), at the version the entry names or, when it names none, the current one, whose version is
   recorded;
3. the work the entry's DOI resolves to: the agent's URL, after redirects, ends at the URL the tool
   reached by requesting ``https://doi.org/<the entry's DOI>``, or the agent gave that DOI URL itself.

Any other source is read, and its passage located, but its check is an ``error``: it moves no credence.
"""

from __future__ import annotations

import html.parser
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from credencegraph.core import Node
from credencegraph.ingest.latex import LatexError, read_latex
from credencegraph.ingest.pdf import PdfError, read_pdf
from credencegraph.rubric import normalise_text, text_digest

from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.fetch import Fetched, FetchError, describe, fetch

CITED = "cited"
# The suffixes a cited work may have, in the order they are looked for.
SUFFIXES = (".txt", ".tex", ".pdf")
ARXIV_ABS = "https://arxiv.org/abs/{}"
ARXIV_PDF = "https://arxiv.org/pdf/{}"
DOI_URL = "https://doi.org/{}"

# An arXiv identifier in a bibliography entry, new style (2101.01234v2) or old (hep-th/9901001).
_ARXIV = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")
# An arXiv URL: its identifier, without a version, and the version if it names one.
_ARXIV_URL = re.compile(
    r"https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})"
    r"(?P<version>v\d+)?(?:\.pdf)?/?"
)
# The versioned PDF an arXiv abstract page names as its citation PDF.
_ARXIV_PDF_META = re.compile(r'<meta\s+name="citation_pdf_url"\s+content="https?://arxiv\.org/pdf/([^"]+)"')
_VERSION = re.compile(r"v\d+$")
# A DOI in a bibliography entry, bare, after ``doi:``, or in a doi.org URL.
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}]+)", re.IGNORECASE)
_DOI_HOSTS = frozenset({"doi.org", "dx.doi.org", "www.doi.org"})

# Fetches a URL under the fetch policy.
Fetch = Callable[[str], Fetched]


class SourceError(Exception):
    """The source cannot be read.

    Attributes:
        code: ``refused-url`` for a URL the fetch policy refuses, ``unreadable-source`` otherwise.
    """

    def __init__(self, code: str, message: str) -> None:
        """Make the error.

        Args:
            code: ``refused-url`` or ``unreadable-source``.
            message: Why, in words.
        """
        super().__init__(message)
        self.code = code


def arxiv_id(reference: Node) -> str | None:
    """Return the arXiv identifier a bibliography entry names, if it names one.

    Args:
        reference: The reference node.

    Returns:
        The first identifier in its statement, or ``None``.
    """
    found = _ARXIV.search(reference.statement or "")
    return found[1] if found else None


def entry_doi(reference: Node) -> str | None:
    """Return the DOI a bibliography entry names, in lower case, if it names one.

    Args:
        reference: The reference node.

    Returns:
        The first DOI in its statement, without trailing punctuation, or ``None``.
    """
    found = _DOI.search(reference.statement or "")
    return found[1].rstrip(".,;:)]").lower() if found else None


def _split_arxiv(identifier: str) -> tuple[str, str | None]:
    """An arXiv identifier without its version, and the version."""
    version = _VERSION.search(identifier)
    return (identifier[: version.start()], version[0]) if version else (identifier, None)


def _doi_of_url(url: str) -> str | None:
    """The DOI a doi.org URL names, in lower case, or ``None``."""
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"} and parts.hostname in _DOI_HOSTS:
        found = _DOI.fullmatch(parts.path.lstrip("/"))
        return found[1].lower() if found else None
    return None


# The HTML elements that separate text: the text either side of one is two words, not one.
_BLOCKS = frozenset(
    {"address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "figure", "footer"}
    | {"h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section"}
    | {"table", "td", "th", "tr", "ul"}
)


class _Text(html.parser.HTMLParser):
    """The text of an HTML page, without its scripts and styles; inline elements do not split words."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in _BLOCKS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        elif tag in _BLOCKS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def text_of(data: bytes, kind: str) -> str:
    """Return the normalised text of a source's bytes.

    Args:
        data: The bytes.
        kind: A file suffix (``.txt``, ``.tex``, ``.pdf``) or a media type (``text/plain``,
            ``text/html``, ``application/xhtml+xml``, ``application/pdf``).

    Returns:
        The text, normalised as a quoted passage is.

    Raises:
        SourceError: If the kind is not one of those, or the bytes cannot be read as it.
    """
    try:
        if kind in {".txt", "text/plain"}:
            text = data.decode("utf-8")
        elif kind in {"text/html", "application/xhtml+xml"}:
            parser = _Text()
            parser.feed(data.decode("utf-8", errors="replace"))
            parser.close()
            text = "".join(parser.parts)
        elif kind in {".pdf", "application/pdf", ".tex"}:
            suffix = ".tex" if kind == ".tex" else ".pdf"
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"source{suffix}"
                path.write_bytes(data)
                chunks = read_latex(path) if suffix == ".tex" else read_pdf(path)
            text = " ".join(chunk.text for chunk in chunks)
        else:
            raise SourceError("unreadable-source", f"a source of type {kind!r} cannot be read")
    except (UnicodeDecodeError, LatexError, PdfError, OSError) as error:
        raise SourceError("unreadable-source", f"the source cannot be read as {kind}: {type(error).__name__}") from None
    normalised = normalise_text(text)
    if not normalised:
        raise SourceError("unreadable-source", "the source has no text")
    return normalised


def key_file(reference: Node, names: Mapping[str, str] | set[str]) -> str | None:
    """Return the file of the cited works that a reference's key names: the first of its suffixes there.

    Args:
        reference: The reference node.
        names: The names of the files of the run's cited works.

    Returns:
        ``<key>.txt``, ``<key>.tex`` or ``<key>.pdf``, the first there, or ``None``.
    """
    key = reference.attributes.get("key")
    if not isinstance(key, str):
        return None
    return next((f"{key}{suffix}" for suffix in SUFFIXES if f"{key}{suffix}" in names), None)


def _located(observation: dict[str, Any], text: str, passage: str) -> dict[str, Any]:
    """Add the digest of a source's text and where the quoted passage is in it."""
    quote = normalise_text(passage)
    start = text.find(quote) if quote else -1
    return {
        **observation,
        "text_digest": text_digest(text),
        "found": [start, start + len(quote)] if start >= 0 else None,
    }


def read_cited(run_dir: Path, source: str, passage: str) -> tuple[dict[str, Any], bytes]:
    """Read a file of the run's copy of the cited works.

    Args:
        run_dir: The run directory.
        source: The file's name, or a path to it in ``<run>/cited/``.
        passage: The passage the agent quotes.

    Returns:
        The observation (``kind`` ``cited``, ``file``, ``digest``, ``text_digest``, ``found``) and the bytes.

    Raises:
        SourceError: If the path is not a regular file directly in the run's ``cited/``, or cannot be read.
    """
    cited = run_dir / CITED
    path = Path(source)
    name = path.name
    inside = path == Path(name) or (path.is_absolute() and path.parent.resolve() == cited.resolve())
    target = cited / name
    if not inside or not name or target.is_symlink() or not target.is_file():
        msg = f"{source!r} is not a file of the run's copy of the cited works; a local source is read only from there"
        raise SourceError("unreadable-source", msg)
    data = target.read_bytes()
    observation = {"kind": "cited", "file": name, "digest": bytes_digest(data)}
    return _located(observation, text_of(data, target.suffix), passage), data


def _fetched(url: str, fetcher: Fetch) -> Fetched:
    """Fetch a URL, as a source error when it cannot be."""
    try:
        return fetcher(url)
    except FetchError as error:
        code = "refused-url" if error.code == "refused-url" else "unreadable-source"
        raise SourceError(code, str(error)) from None


def _arxiv(identifier: str, fetcher: Fetch) -> tuple[Fetched, dict[str, Any]]:
    """Fetch the arXiv paper an entry names, at its version or the current one; return it and the tie."""
    base, version = _split_arxiv(identifier)
    if version is None:
        page = _fetched(ARXIV_ABS.format(base), fetcher)
        found = _ARXIV_PDF_META.search(page.data.decode("utf-8", errors="replace"))
        current = _split_arxiv(found[1]) if found else (None, None)
        if current[0] != base or current[1] is None:
            raise SourceError("unreadable-source", f"the arXiv page of {base} names no current version")
        version = current[1]
    url = ARXIV_PDF.format(f"{base}{version}")
    return _fetched(url, fetcher), {"by": "arxiv", "id": base, "version": version, "url": url}


def read_url(reference: Node, source: str, passage: str, fetcher: Fetch = fetch) -> tuple[dict[str, Any], bytes]:
    """Fetch a URL source, and what ties it to the reference.

    An arXiv URL of the identifier the entry names is read from arXiv at the entry's version, or the
    current one; a doi.org URL of the entry's DOI is read through doi.org. Any other URL is read as
    given, and, when the entry names a DOI, the DOI is resolved too, so the two final URLs can be compared.

    Args:
        reference: The reference node.
        source: The URL the agent gave.
        passage: The passage the agent quotes.
        fetcher: Fetches a URL under the fetch policy.

    Returns:
        The observation (``kind`` ``url``, the fetch's URLs and content type, ``digest``, ``tie``,
        ``text_digest``, ``found``) and the bytes.

    Raises:
        SourceError: If a URL is refused, cannot be fetched, or the body cannot be read.
    """
    tie: dict[str, Any] | None = None
    entry_arxiv = arxiv_id(reference)
    doi = entry_doi(reference)
    given = _ARXIV_URL.fullmatch(source)
    if entry_arxiv is not None and given is not None and given["id"] == _split_arxiv(entry_arxiv)[0]:
        fetched, tie = _arxiv(entry_arxiv, fetcher)
    elif doi is not None and _doi_of_url(source) == doi:
        fetched = _fetched(DOI_URL.format(doi), fetcher)
        tie = {"by": "doi", "doi": doi, "final_url": fetched.final_url}
    else:
        fetched = _fetched(source, fetcher)
        if doi is not None:
            try:
                resolved = _fetched(DOI_URL.format(doi), fetcher)
            except SourceError:
                tie = None
            else:
                tie = {"by": "doi", "doi": doi, "final_url": resolved.final_url}
    kind = fetched.content_type or ("application/pdf" if fetched.data.startswith(b"%PDF") else "")
    observation = {"kind": "url", **describe(fetched), "digest": bytes_digest(fetched.data), "tie": tie}
    return _located(observation, text_of(fetched.data, kind), passage), fetched.data


_NOT_TIED = "the source is not the reference's arXiv paper, its DOI's work, or its cited copy"


def _cited_tie(reference: Node, pins: Mapping[str, str], observation: Mapping[str, Any]) -> str | None:
    """Why a file of the cited works is not tied to the reference, or ``None``."""
    expected = key_file(reference, set(pins))
    if expected is None:
        return "no file among the cited works is named by the reference's key"
    if observation.get("file") != expected:
        return f"{observation.get('file')!r} is not {expected!r}, the file the reference's key names"
    if observation.get("digest") != pins[expected]:
        return "the cited copy changed since start_run"
    return None


def _arxiv_tie(reference: Node, found: Mapping[str, Any], observation: Mapping[str, Any]) -> str | None:
    """Why a fetched arXiv paper is not tied to the reference, or ``None``."""
    entry = arxiv_id(reference)
    base, version = _split_arxiv(entry) if entry else (None, None)
    if base is None or found.get("id") != base or (version is not None and found.get("version") != version):
        return "the arXiv paper read is not the one the reference names"
    url = ARXIV_PDF.format(f"{base}{found.get('version')}")
    if found.get("url") != url or observation.get("url") != url:
        return "the arXiv paper was not read from arXiv"
    return None


def _doi_tie(reference: Node, found: Mapping[str, Any], observation: Mapping[str, Any]) -> str | None:
    """Why a fetched source is not where the reference's DOI resolves, or ``None``."""
    if found.get("doi") != entry_doi(reference):
        return "the DOI resolved is not the one the reference names"
    if observation.get("final_url") != found.get("final_url"):
        return "the source is not where the reference's DOI resolves"
    return None


def tie(reference: Node, pins: Mapping[str, str], observation: Mapping[str, Any]) -> str | None:
    """Decide whether an observed source is tied to the reference; return why not, or ``None`` when it is.

    Args:
        reference: The reference node.
        pins: The digest of each file of the cited works, as ``start_run`` recorded it.
        observation: What the tool read.

    Returns:
        ``None`` for a tied source, else the reason it is not, in words.
    """
    if observation.get("kind") == "cited":
        return _cited_tie(reference, pins, observation)
    found = observation.get("tie")
    if isinstance(found, Mapping) and found.get("by") == "arxiv":
        return _arxiv_tie(reference, found, observation)
    if isinstance(found, Mapping) and found.get("by") == "doi":
        return _doi_tie(reference, found, observation)
    return _NOT_TIED


def verify_observation(observation: Mapping[str, Any], data: bytes, passage: str) -> list[str]:
    """Return the fields of an observation that the source's bytes do not give again.

    Args:
        observation: What the tool recorded.
        data: The bytes, found by their digest.
        passage: The passage the agent quoted.

    Returns:
        The names of the fields that differ: ``digest``, ``text_digest`` or ``found``; or ``unreadable``.
    """
    if bytes_digest(data) != observation.get("digest"):
        return ["digest"]
    kind = (
        Path(str(observation.get("file"))).suffix
        if observation.get("kind") == "cited"
        else observation.get("content_type")
    )
    if observation.get("kind") == "url" and not kind:
        kind = "application/pdf" if data.startswith(b"%PDF") else ""
    try:
        again = _located({}, text_of(data, str(kind)), passage)
    except SourceError:
        return ["unreadable"]
    return [name for name in ("text_digest", "found") if again[name] != observation.get(name)]
