"""The pipeline: the passes a run applies after ingestion, in order. Passes are added here as they are built."""

from __future__ import annotations

from credencegraph_extract.classify import ClassifyPass
from credencegraph_extract.premises import PremisePass
from credencegraph_extract.run import Pass

PASSES: tuple[Pass, ...] = (ClassifyPass(), PremisePass())
