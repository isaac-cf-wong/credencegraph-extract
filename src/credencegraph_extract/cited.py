"""The texts of cited works, which the citation-support check judges a citing sentence against.

A cited work is found by the key of its bibliography entry, ``attributes.key`` of the reference node,
in a directory the caller names: ``<key>.txt`` is read as plain text, ``<key>.tex`` and ``<key>.pdf``
through the core's LaTeX and PDF readers. When no file is there and the entry names an arXiv
identifier, ``fetch_arxiv`` downloads the PDF from arXiv into the directory first, so a replay finds
it there without the network. The text is normalised with the core's ``normalise_text``, the form in
which a quoted passage is located. An error names files by name only: its message is recorded in the
graph, which must not depend on where the directory is.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from credencegraph.core import Node
from credencegraph.ingest.latex import LatexError, read_latex
from credencegraph.ingest.pdf import PdfError, read_pdf
from credencegraph.rubric import normalise_text

# The suffixes a cited work may have in the directory, in the order they are looked for.
SUFFIXES = (".txt", ".tex", ".pdf")
ARXIV_PDF = "https://arxiv.org/pdf/{}"
ARXIV_TIMEOUT = 60

# An arXiv identifier in a bibliography entry, new style (2101.01234v2) or old (hep-th/9901001).
_ARXIV = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")
# A key that can name a file: no separators, and not a hidden or relative name.
_KEY = re.compile(r"[\w:+-][\w.:+-]*")


class CitedError(Exception):
    """The text of a cited work cannot be had."""


@dataclass(frozen=True)
class CitedSource:
    """The text of a cited work.

    Attributes:
        document: The work's id: ``arXiv:<id>`` when its entry names one, else its file's name.
        text: Its text, normalised.
    """

    document: str
    text: str


# Return the text of the work a reference node names, or raise ``CitedError``.
Cited = Callable[[Node], CitedSource]


def no_sources(reference: Node) -> CitedSource:
    """Refuse every cited work: the caller named no directory of cited works.

    Args:
        reference: The reference node.

    Raises:
        CitedError: Always.
    """
    msg = f"no cited works were given, so the text of {reference.id!r} cannot be read"
    raise CitedError(msg)


def arxiv_id(reference: Node) -> str | None:
    """Return the arXiv identifier a bibliography entry names, if it names one.

    Args:
        reference: The reference node.

    Returns:
        The first identifier in its statement, or ``None``.
    """
    found = _ARXIV.search(reference.statement or "")
    return found[1] if found else None


def download(url: str) -> bytes:
    """Download a URL over HTTPS.

    Args:
        url: The URL.

    Returns:
        The response body.
    """
    with urllib.request.urlopen(url, timeout=ARXIV_TIMEOUT) as response:  # noqa: S310 - a fixed https URL
        return response.read()


def _read(path: Path) -> str:
    """Read a cited work's file into one text."""
    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")
    chunks = read_latex(path) if path.suffix == ".tex" else read_pdf(path)
    return " ".join(chunk.text for chunk in chunks)


def directory(root: Path, *, fetch_arxiv: bool = False, fetch: Callable[[str], bytes] = download) -> Cited:
    """Find cited works in a directory, by the key of their bibliography entries.

    Args:
        root: The directory.
        fetch_arxiv: Download a missing work from arXiv when its entry names an identifier.
        fetch: Downloads a URL; ``download`` by default.

    Returns:
        The resolver.
    """

    def resolve(reference: Node) -> CitedSource:
        key = reference.attributes.get("key")
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            msg = f"reference {reference.id!r} has no bibliography key that can name a file"
            raise CitedError(msg)
        identifier = arxiv_id(reference)
        found = [root / f"{key}{suffix}" for suffix in SUFFIXES if (root / f"{key}{suffix}").is_file()]
        if not found and fetch_arxiv and identifier is not None:
            target = root / f"{key}.pdf"
            try:
                target.write_bytes(fetch(ARXIV_PDF.format(identifier)))
            except OSError as error:
                msg = f"cannot download arXiv:{identifier} for {reference.id!r}: {type(error).__name__}"
                raise CitedError(msg) from None
            found = [target]
        if not found:
            msg = f"no file {key}{{{','.join(SUFFIXES)}}} among the cited works for {reference.id!r}"
            raise CitedError(msg)
        try:
            text = normalise_text(_read(found[0]))
        except (OSError, UnicodeDecodeError, LatexError, PdfError) as error:
            msg = f"cannot read {found[0].name!r} for {reference.id!r}: {type(error).__name__}"
            raise CitedError(msg) from None
        if not text:
            msg = f"{found[0].name!r} for {reference.id!r} has no text"
            raise CitedError(msg)
        return CitedSource(f"arXiv:{identifier}" if identifier else found[0].name, text)

    return resolve
