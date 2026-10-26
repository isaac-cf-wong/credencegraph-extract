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
- **[Check pass](checks)** – Proposing checks of the assessed nodes, running
  them, and recording each as an observed evidence node.
- **[Evaluators](evaluate)** – Exact arithmetic, symbolic identity and numeric
  sampling, over whitelisted expressions.
- **[Cited works](cited)** – Reading the texts of cited works.
- **[Agent-tool runs](session)** – The run directory an agent's tool calls
  build: the call log, the lock, crash recovery and replay.
- **[Verify](verify)** – Running the deterministic checks of a run again, and
  reporting rubric versions reused with different values.
