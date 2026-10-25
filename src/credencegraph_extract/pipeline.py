"""The pipeline: the passes a run applies after ingestion, in order. Passes are added here as they are built."""

from __future__ import annotations

from credencegraph_extract.checks import CheckPass
from credencegraph_extract.cited import Cited, no_sources
from credencegraph_extract.classify import ClassifyPass
from credencegraph_extract.premises import PremisePass
from credencegraph_extract.run import Pass


def passes(*, cited: Cited = no_sources, allow_uncalibrated: bool = False) -> tuple[Pass, ...]:
    """Return the pipeline: classify, premises, then checks.

    Args:
        cited: Reads the text of a cited work for the citation checks; none can be read by default.
        allow_uncalibrated: Run a model-judged check with a model its check type was not fitted with.

    Returns:
        The passes, in order.
    """
    return (ClassifyPass(), PremisePass(), CheckPass(cited, allow_uncalibrated))


PASSES: tuple[Pass, ...] = passes()
