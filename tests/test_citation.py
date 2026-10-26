"""Whether a citation's source is tied to the reference: the forms the entry's arXiv id and DOI give."""

from __future__ import annotations

from typing import Any

import pytest
from credencegraph.core import Node

from credencegraph_extract.citation import SourceError, arxiv_id, entry_doi, read_url, text_of, tie, verify_observation
from credencegraph_extract.fetch import Fetched, FetchError

PASSAGE = "the estimator is unbiased for every segment length"
BODY = f"Results. We show that {PASSAGE}, as expected."
DOI = "10.1000/xyz.123"
LANDING = "https://publisher.example/article/123"


def entry(statement: str) -> Node:
    return Node("r-9", kind="reference", statement=statement, attributes={"unit": "reference", "key": "lee2019"})


def fetched(url: str, body: str = BODY, final_url: str | None = None, content_type: str = "text/plain") -> Fetched:
    return Fetched(url, final_url or url, (url,), content_type, body.encode("utf-8"))


def web(pages: dict[str, Fetched]) -> Any:
    def fetch(url: str) -> Fetched:
        if url not in pages:
            raise FetchError("fetch-failed", f"{url!r} answered 404")
        return pages[url]

    return fetch


DOI_WEB = web(
    {
        f"https://doi.org/{DOI}": fetched(f"https://doi.org/{DOI}", final_url=LANDING),
        LANDING: fetched(LANDING),
        "https://mirror.example/lee.txt": fetched("https://mirror.example/lee.txt"),
    }
)


@pytest.mark.parametrize(
    ("statement", "doi"),
    [
        (f"C. Lee, Unbiased estimators, J. Est. 4 (2019), doi:{DOI}.", DOI),
        (f"C. Lee, Unbiased estimators, https://doi.org/{DOI.upper()}", DOI),
        ("C. Lee, Unbiased estimators, J. Est. 4 (2019) 1.", None),
    ],
)
def test_the_doi_an_entry_names(statement: str, doi: str | None) -> None:
    assert entry_doi(entry(statement)) == doi


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("arXiv: 2101.01234", "2101.01234"),
        ("A. Smith, Estimators, arXiv:2101.01234v2.", "2101.01234v2"),
        ("arXiv:hep-th/9901001v1", "hep-th/9901001v1"),
        ("J. Stat. 12", None),
    ],
)
def test_the_arxiv_id_an_entry_names(statement: str, expected: str | None) -> None:
    assert arxiv_id(entry(statement)) == expected


def test_the_page_the_doi_of_the_entry_resolves_to_is_tied() -> None:
    reference = entry(f"C. Lee, doi:{DOI}")
    observation, _ = read_url(reference, LANDING, PASSAGE, DOI_WEB)
    assert observation["tie"] == {"by": "doi", "doi": DOI, "final_url": LANDING}
    assert tie(reference, {}, observation) is None


def test_the_doi_url_of_the_entry_is_tied_through_doi_org() -> None:
    reference = entry(f"C. Lee, doi:{DOI}")
    observation, _ = read_url(reference, f"https://dx.doi.org/{DOI}", PASSAGE, DOI_WEB)
    assert (observation["url"], observation["final_url"]) == (f"https://doi.org/{DOI}", LANDING)
    assert tie(reference, {}, observation) is None


def test_a_page_elsewhere_than_the_doi_resolves_is_not_tied() -> None:
    reference = entry(f"C. Lee, doi:{DOI}")
    observation, _ = read_url(reference, "https://mirror.example/lee.txt", PASSAGE, DOI_WEB)
    assert observation["found"] is not None
    assert tie(reference, {}, observation) == "the source is not where the reference's DOI resolves"


def test_the_doi_of_another_entry_is_not_tied() -> None:
    other = "10.1000/other"
    pages = {
        f"https://doi.org/{other}": fetched(f"https://doi.org/{other}", final_url=LANDING),
        LANDING: fetched(LANDING),
    }
    observation, _ = read_url(entry(f"C. Lee, doi:{other}"), LANDING, PASSAGE, web(pages))
    assert tie(entry(f"C. Lee, doi:{DOI}"), {}, observation) == "the DOI resolved is not the one the reference names"


def test_an_entry_naming_an_arxiv_version_is_read_at_that_version() -> None:
    reference = entry("C. Lee, arXiv:1901.00001v3")
    url = "https://arxiv.org/pdf/1901.00001v3"
    seen: list[str] = []
    pages = {url: fetched(url)}

    def fetch(asked: str) -> Fetched:
        seen.append(asked)
        return pages[asked]

    observation, _ = read_url(reference, "https://arxiv.org/abs/1901.00001", PASSAGE, fetch)
    assert seen == [url]
    assert observation["tie"] == {"by": "arxiv", "id": "1901.00001", "version": "v3", "url": url}
    assert tie(reference, {}, observation) is None
    # The same reading is not tied to an entry that names another version.
    assert tie(entry("C. Lee, arXiv:1901.00001v2"), {}, observation) == (
        "the arXiv paper read is not the one the reference names"
    )


def test_an_arxiv_page_naming_no_version_cannot_be_read() -> None:
    abs_url = "https://arxiv.org/abs/1901.00001"
    with pytest.raises(SourceError, match="names no current version"):
        read_url(entry("C. Lee, arXiv:1901.00001"), abs_url, PASSAGE, web({abs_url: fetched(abs_url, "<html></html>")}))


def test_a_reading_that_claims_arxiv_but_was_read_elsewhere_is_not_tied() -> None:
    reference = entry("C. Lee, arXiv:1901.00001v3")
    url = "https://arxiv.org/pdf/1901.00001v3"
    observation = {
        "kind": "url",
        "url": "https://agent.example/x",
        "final_url": url,
        "tie": {"by": "arxiv", "id": "1901.00001", "version": "v3", "url": url},
    }
    assert tie(reference, {}, observation) == "the arXiv paper was not read from arXiv"


def test_a_url_with_no_tie_is_not_tied() -> None:
    observation, _ = read_url(entry("C. Lee, 2019."), "https://mirror.example/lee.txt", PASSAGE, DOI_WEB)
    assert observation["tie"] is None
    assert tie(entry("C. Lee, 2019."), {}, observation).startswith("the source is not the reference's")


def test_html_is_read_without_its_scripts() -> None:
    page = b"<html><script>var x = 1;</script><p>Results: <b>it holds</b> for <i>bias</i>es.</p><p>Next</p></html>"
    assert text_of(page, "text/html") == "Results: it holds for biases. Next"
    with pytest.raises(SourceError, match="cannot be read"):
        text_of(b"\x00\x01", "application/octet-stream")


def test_an_observation_is_checked_against_its_bytes() -> None:
    reference = entry(f"C. Lee, doi:{DOI}")
    observation, data = read_url(reference, LANDING, PASSAGE, DOI_WEB)
    assert verify_observation(observation, data, PASSAGE) == []
    assert verify_observation(observation, data, "another passage") == ["found"]
    assert verify_observation(observation, data + b"!", PASSAGE) == ["digest"]
