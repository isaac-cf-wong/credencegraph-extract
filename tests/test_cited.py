"""Cited works: found by bibliography key in a directory, or downloaded from arXiv into it."""

from __future__ import annotations

from pathlib import Path

import pytest
from checks_fixture import CITED
from credencegraph.core import Node

from credencegraph_extract import cited
from credencegraph_extract.cited import CitedError, arxiv_id


def reference(key: object, statement: str = "A. Smith, Estimators of noise.") -> Node:
    return Node(
        "r-1", kind="reference", statement=statement, attributes={"origin": "document", "unit": "reference", "key": key}
    )


def test_a_text_file_is_read_and_normalised() -> None:
    source = cited.directory(CITED)(reference("smith2020"))
    assert source.document == "smith2020.txt"
    assert source.text.startswith("Estimators of noise. A. Smith. We study estimators")
    assert "\n" not in source.text


def test_a_latex_file_is_read_through_the_latex_reader(tmp_path: Path) -> None:
    (tmp_path / "doe2019.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nThe bias is small.\n% a comment\n\\end{document}\n",
        encoding="utf-8",
    )
    assert cited.directory(tmp_path)(reference("doe2019")).text == "The bias is small."


def test_an_arxiv_entry_is_named_by_its_identifier() -> None:
    source = cited.directory(CITED)(reference("smith2020", "A. Smith, Estimators, arXiv:2101.01234v2."))
    assert source.document == "arXiv:2101.01234v2"


@pytest.mark.parametrize("key", [None, "../smith2020", ".hidden", "a/b", ""])
def test_a_key_that_cannot_name_a_file_is_refused(key: object) -> None:
    with pytest.raises(CitedError):
        cited.directory(CITED)(reference(key))


def test_a_missing_work_is_an_error_without_a_download(tmp_path: Path) -> None:
    def fetch(url: str) -> bytes:
        raise AssertionError(url)

    with pytest.raises(CitedError, match="no file"):
        cited.directory(tmp_path, fetch=fetch)(reference("jones2021", "B. Jones, arXiv:2101.01234."))


def test_fetch_arxiv_downloads_the_pdf_into_the_directory(tmp_path: Path) -> None:
    asked = []

    def fetch(url: str) -> bytes:
        asked.append(url)
        return b"not a PDF"

    resolve = cited.directory(tmp_path, fetch_arxiv=True, fetch=fetch)
    with pytest.raises(CitedError, match="cannot read"):
        resolve(reference("jones2021", "B. Jones, Long segments, arXiv:2101.01234."))
    assert asked == ["https://arxiv.org/pdf/2101.01234"]
    assert (tmp_path / "jones2021.pdf").read_bytes() == b"not a PDF"


def test_a_failed_download_is_an_error(tmp_path: Path) -> None:
    def fetch(url: str) -> bytes:
        raise OSError("no route to host")

    with pytest.raises(CitedError, match="cannot download"):
        cited.directory(tmp_path, fetch_arxiv=True, fetch=fetch)(reference("jones2021", "arXiv:2101.01234"))


def test_an_empty_work_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text(" \n", encoding="utf-8")
    with pytest.raises(CitedError, match="no text"):
        cited.directory(tmp_path)(reference("empty"))


def test_no_sources_refuses_every_work() -> None:
    with pytest.raises(CitedError):
        cited.no_sources(reference("smith2020"))


@pytest.mark.parametrize(
    ("statement", "expected"),
    [("arXiv: 2101.01234", "2101.01234"), ("arXiv:hep-th/9901001v1", "hep-th/9901001v1"), ("J. Stat. 12", None)],
)
def test_arxiv_id(statement: str, expected: str | None) -> None:
    assert arxiv_id(reference("k", statement)) == expected
