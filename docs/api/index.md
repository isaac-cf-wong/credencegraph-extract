---
title: API Reference
description: Complete reference for credencegraph-extract.
icon: material/api
---

This section documents the public API of credencegraph-extract.

## Main Sections

- **[MCP server](mcp_server)** – The agent tools over MCP on stdio, for an agent
  harness to call.
- **[Graph tools](tools)** – Classifying chunks, adding premises, running checks
  and finishing a run, through the assembler and the check runners.
- **[Agent-tool runs](session)** – The run directory an agent's tool calls
  build: the call log, the lock, crash recovery and replay.
- **[Assembler](assemble)** – Applying classify and premise records through the
  core's edits.
- **[Checks](checks)** – What a check proposal must be, the four check types,
  check nodes and verdicts.
- **[Evaluators](evaluate)** – Exact arithmetic, symbolic identity and numeric
  sampling, over whitelisted expressions.
- **[Citation sources](citation)** – Reading a citation's source, and deciding
  whether it is tied to the reference.
- **[Fetch](fetch)** – Fetching a URL an agent names from public addresses only.
- **[Verify](verify)** – Running the deterministic checks of a run again, and
  reporting rubric versions reused with different values.
- **[Earlier runs](run)** – Reading the runs of format 1 that earlier versions
  made by calling a model, for `verify`.
