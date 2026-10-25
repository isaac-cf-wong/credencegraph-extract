---
title: API Reference
description: Complete reference for credencegraph-extract.
icon: material/api
---

This section documents the public API of credencegraph-extract.

## Main Sections

- **[Provider](provider)** – The one module that calls a model provider.
- **[Runs](run)** – Extraction runs, the run directory and replay.
- **[Classify pass](classify)** – Typing, splitting and linking the chunks of a
  document.
- **[Premise pass](premises)** – Linking the premises a conclusion's text states
  and adding those it leaves unsaid.
- **[Assembler](assemble)** – Applying classify and premise records through the
  core's edits.
