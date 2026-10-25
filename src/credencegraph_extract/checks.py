"""The check pass: the model proposes checks of the assessed nodes, and the runner runs them.

The pass runs last. Its targets are the document and analyst nodes of a type with ``assess = true``.
For those a rubric check type may be run on, it sends one prompt listing them with the chunks of the
document, and the model proposes checks, each of a type the runner implements:

- ``recompute-numbers``: an arithmetic expression over numerals of the cited chunks, and the
  reported result. Every numeral, the result included, must be one the cited chunks' text contains,
  compared exactly as the decimal each writes, so the model cannot supply a number the document does
  not, not even one a float cannot tell apart from a number it does. The expression is evaluated exactly;
  it passes when it falls within the rounding the reported result's own digits imply.
- ``rederive-maths``: a step between equation chunks encoded as two sympy expressions over declared
  symbols, ``lhs`` and ``rhs``, that the step claims are equal. It passes when sympy simplifies their
  difference to 0, fails when sympy shows the difference is never 0, and is an ``error`` otherwise.
- ``rederive-maths-numeric``: run, as a check of its own, only when ``rederive-maths`` errs and the
  rubric declares it: the same encoding compared at 32 points drawn in the symbols' domains, from a
  generator seeded by the encoding. The seed and the points are recorded with the check.
- ``citation-support``: a cited work's text is read, the model quotes the passage of it the citing
  node rests on and judges whether it supports the node. The passage must be found in the work's
  normalised text, or the check is an ``error``. A rubric check type that lists ``models`` was fitted
  with those models; the pass refuses to run it with another unless ``allow_uncalibrated`` is set,
  and reports every check run that way.

A proposal the runner cannot run, being malformed, of an unknown type, on a node the type does not
target, with a numeral the cited chunks do not contain, or a second of its type on one node, is
rejected, with the code and record, and sent back to the model once with the error and what to change
in it; what it still gets wrong stays rejected. Two checks of one type on one node would fail or pass together,
so they must not count as two observations.

Each check run becomes an evidence node ``c-NNN`` of its check type, numbered in document order of
the targets and then in run order, joined to its target by the check edge ``target --observes-->
c-NNN`` with the check type's ``sensitivity`` and ``false_pass``. Its statement comes from a fixed
template, never from the model. ``attributes.check`` is the check record and ``attributes.observed``
is ``true`` for a pass and ``false`` for a fail; an errored check has no ``observed``, so it is
recorded but changes no credence. The run writes each check's inputs, the tool's output and a log
into ``checks/c-NNN/``. The verdict on each target is derived from its checks, never asked of the
model: ``fails`` if any failed, else ``holds`` if any passed, else ``undetermined`` if any ran, else
``not_assessed``, because no check type the runner implements may run on the node, or none was run.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from itertools import count
from string import Template
from typing import Any

from credencegraph.core import Graph, Node, SourceAnchor
from credencegraph.core.relation import OBSERVES, Relation
from credencegraph.ingest.chunks import EQUATION, REFERENCE
from credencegraph.rubric import Rubric, normalise_text, text_digest
from credencegraph.rubric.model import ANALYST, DOCUMENT, EVIDENCE, Check
from credencegraph.rubric.validate import CITES, FAILS, HOLDS, NOT_ASSESSED, PART_OF, UNDETERMINED, node_text, origin_of

from credencegraph_extract import evaluate
from credencegraph_extract.assemble import unlabelled
from credencegraph_extract.cited import Cited, CitedError, no_sources
from credencegraph_extract.digest import bytes_digest
from credencegraph_extract.evaluate import ERROR, FAILED, PASSED, EvaluationError
from credencegraph_extract.run import CHECKS, Ask, Outcome, RunError

NAME = "checks"

RECOMPUTE = "recompute-numbers"
REDERIVE = "rederive-maths"
NUMERIC = "rederive-maths-numeric"
CITATION = "citation-support"
# The check types the model may propose; the numeric check is run only as the fallback of the symbolic one.
PROPOSED = (RECOMPUTE, REDERIVE, CITATION)
IMPLEMENTED = (*PROPOSED, NUMERIC)
# The check types decided by code alone, which ``rerun`` runs again; a citation check needs a model.
DETERMINISTIC = (RECOMPUTE, REDERIVE, NUMERIC)

# The reasons of a ``not_assessed`` verdict.
NO_CHECK = "no v1 check applies"
NOT_SELECTED = "not selected"

# Rejection codes of the check pass.
INVALID_RECORD = "invalid-record"
DUPLICATE_RECORD = "duplicate-record"
UNFAITHFUL = "unfaithful-numeral"
NOT_A_TARGET = "not-a-target"
UNKNOWN_NODE = "unknown-node"
NOT_REJECTED = "not-rejected"

# The rounds of the plan: the first answer, and the one correcting its refused proposals.
INITIAL = "initial"
REPAIR = "repair"

# The files of a check's directory.
INPUTS = "inputs.json"
OUTPUT = "output.json"
LOG = "log.txt"

PLAN_TEMPLATE = """\
You are planning checks of the statements of a document. Each target below is a statement whose
truth is to be assessed, with the check types that may be run on it. Propose the checks that can be
run on each target from the document itself, at most one of each type per target, and none where
no check fits. Do not judge whether any statement is true: the checks decide.

Check types:
- recompute-numbers: the target reports a number that follows from other numbers the document
  reports. Give "nodes", the ids of the chunks whose numerals you use (the target's own included),
  "expression", an arithmetic expression of those numerals with + - * / ** and parentheses, and
  "expected", the reported result. Write every number exactly as a plain decimal numeral that occurs
  in the text of one of "nodes" (a percentage such as 12.5% may be written 0.125); write no other
  number, not even 100 or 2.
- rederive-maths: the target rests on a step of mathematics from one equation to another. Give
  "nodes", the ids of the equation chunks the step goes between, "lhs" and "rhs", two expressions the
  step claims are equal for all values of the symbols, "symbols", each symbol with its sympy
  assumptions (real, positive, negative, nonnegative, nonpositive, nonzero, integer) and a finite
  domain [low, high] to sample it from, and "assumption", the step's assumptions in words. Use only
  the declared symbols, numerals, + - * / ** and the functions sqrt, exp, log, sin, cos, tan, asin,
  acos, atan, sinh, cosh, tanh, Abs and the constants pi, E and I.
- citation-support: the target states what a cited work found. Give "reference", the id of the
  bibliography entry it cites.

Targets, as id (type): text, then the check types that may be run on it:
$targets

Chunks of the document, in document order, as id (type): text
$chunks

Proposals from your previous answer that were refused, each with the error it got and what to change
("fix"). If this list is not empty, return corrected proposals for these targets only, and nothing
else:
$rejected
"""

JUDGE_TEMPLATE = """\
A statement of a document cites the work below. Decide whether the work supports the statement.

First copy, word for word, the one passage of the work that the decision rests on: "passage" must be
an exact substring of the work's text. Then set "supports" to true if the passage supports the
statement as written and false if it does not.

Statement:
$statement

Text of the cited work:
$source
"""

TEMPLATE = PLAN_TEMPLATE + "\n" + JUDGE_TEMPLATE

_STRING = {"type": "string"}
_ALLOWED = {
    RECOMPUTE: frozenset({"target", "type", "nodes", "expression", "expected"}),
    REDERIVE: frozenset({"target", "type", "nodes", "lhs", "rhs", "symbols", "assumption"}),
    CITATION: frozenset({"target", "type", "reference"}),
}


class _ProposalError(Exception):
    """A proposal is not run, with the code and message of a rejection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Result:
    """What a check found.

    Attributes:
        type: The check type.
        target: The node the check bears on.
        artifact: What the check was run against, as the check record's ``artifact``.
        inputs: The exact thing evaluated.
        result: ``passed``, ``failed`` or ``error``.
        detail: What the tool found, in words.
        output: The tool's output, written to the check's directory.
        sources: Anchors of the external works checked against.
        libraries: The libraries the method used, named as in the run's versions.
        model: The provider, model and request id of the model that took part, or ``None``.
    """

    type: str
    target: str
    artifact: Mapping[str, Any]
    inputs: Mapping[str, Any]
    result: str
    detail: str
    output: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[SourceAnchor, ...] = ()
    libraries: tuple[str, ...] = ()
    model: Mapping[str, Any] | None = None


def targets(graph: Graph, rubric: Rubric) -> list[str]:
    """Return the nodes the pass assesses: document and analyst nodes of a type with ``assess = true``.

    Args:
        graph: The graph.
        rubric: The rubric.

    Returns:
        Their ids, in document order.
    """
    return [
        node.id
        for node in graph
        if origin_of(node) in (DOCUMENT, ANALYST) and (spec := rubric.types.get(node.kind)) is not None and spec.assess
    ]


def applicable(rubric: Rubric, kind: str) -> list[str]:
    """Return the check types the model may propose for a node type: proposed by the runner, declared and targeting it.

    Args:
        rubric: The rubric.
        kind: The node type.

    Returns:
        The check type names, in the order of ``PROPOSED``.
    """
    return [name for name in PROPOSED if name in rubric.checks and kind in rubric.checks[name].targets]


# What to change in a refused proposal, by rejection code, where the code alone says it.
_FIXES = {
    NOT_A_TARGET: "propose a check only on a target listed under Targets, of a check type listed for it",
    UNKNOWN_NODE: "use only chunk ids listed under Chunks",
    DUPLICATE_RECORD: "propose at most one check of each type on a target",
    NOT_REJECTED: "while correcting, send proposals only for the targets listed below",
}


def _label_fix(code: str, record: Any) -> str | None:
    """What to change in a proposal that wrote an id with the label its prompt lists it with, as ``s-2 (result)``."""
    if not isinstance(record, Mapping):
        return None
    target, nodes = record.get("target"), record.get("nodes")
    bare = unlabelled(target)
    if code == NOT_A_TARGET and bare is not None:
        return f"{target!r} is not a target id: write the id alone, {bare!r}, without the type in parentheses"
    labelled = [node for node in nodes if unlabelled(node)] if isinstance(nodes, list) else []
    if code == UNKNOWN_NODE and labelled:
        bare_nodes = [unlabelled(node) for node in labelled]
        return f"{labelled} are not chunk ids: write each id alone, as {bare_nodes}, without the type in parentheses"
    return None


def fix(code: str, record: Any) -> str | None:
    """Say what to change in a refused proposal, as an instruction to the model, or ``None`` beyond the error.

    Args:
        code: The rejection code.
        record: The proposal.

    Returns:
        The instruction.
    """
    nodes = record.get("nodes") if isinstance(record, Mapping) else None
    label_fix = _label_fix(code, record)
    if label_fix is not None:
        return label_fix
    if code == UNFAITHFUL:
        return (
            f"write every number, the expected value included, exactly as a numeral the text of {nodes} writes, "
            "and add to nodes any chunk whose numeral you use; a constant no cited chunk writes, such as the 1 of "
            "1 - p or a 100, is not allowed: rewrite the expression without it, as a - a * p rather than "
            "a * (1 - p), where a and p are numerals the text writes"
        )
    return _FIXES.get(code)


def render(graph: Graph, rubric: Rubric, rejected: Sequence[Mapping[str, Any]] = ()) -> str:
    """Render the plan prompt.

    Args:
        graph: The graph.
        rubric: The rubric.
        rejected: The refused proposals to correct; none for the first call. Each is sent with its
            code, error and, where ``fix`` gives one, what to change.

    Returns:
        The prompt.
    """
    repair = []
    for rejection in rejected:
        item = {key: rejection[key] for key in ("record", "code", "message")}
        found = fix(rejection["code"], rejection["record"])
        if found:
            item["fix"] = found
        repair.append(item)
    lines = []
    for node_id in targets(graph, rubric):
        node = graph.nodes[node_id]
        if applicable(rubric, node.kind):
            lines.append(f"{node_id} ({node.kind}): {node_text(graph, node)}")
            lines.append(f"  checks: {', '.join(applicable(rubric, node.kind))}")
    chunks = []
    for node in graph:
        if origin_of(node) != DOCUMENT or not node.statement:
            continue
        label = node.attributes.get("label")
        key = node.attributes.get("key")
        extra = f" [label {label}]" if label else f" [key {key}]" if key else ""
        chunks.append(f"{node.id} ({node.kind}){extra}: {node.statement}")
    return Template(PLAN_TEMPLATE).substitute(
        targets="\n".join(lines), chunks="\n".join(chunks), rejected=json.dumps(repair, indent=2, ensure_ascii=False)
    )


def output_schema(rubric: Rubric) -> dict[str, Any]:
    """Return the output schema of the plan: one branch per proposed check type the rubric declares.

    Args:
        rubric: The rubric.

    Returns:
        The JSON Schema. No proposal has a field for a credence or a result.
    """
    nodes = {"type": "array", "items": dict(_STRING)}
    symbol = {
        "type": "object",
        "properties": {
            "name": dict(_STRING),
            "assumptions": {"type": "array", "items": {"type": "string", "enum": list(evaluate.ASSUMPTIONS)}},
            "domain": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["name", "assumptions", "domain"],
        "additionalProperties": False,
    }
    fields: dict[str, dict[str, Any]] = {
        RECOMPUTE: {"nodes": nodes, "expression": dict(_STRING), "expected": dict(_STRING)},
        REDERIVE: {
            "nodes": nodes,
            "lhs": dict(_STRING),
            "rhs": dict(_STRING),
            "symbols": {"type": "array", "items": symbol},
            "assumption": dict(_STRING),
        },
        CITATION: {"reference": dict(_STRING)},
    }
    branches = [
        {
            "type": "object",
            "properties": {"target": dict(_STRING), "type": {"type": "string", "enum": [name]}, **fields[name]},
            "required": ["target", "type", *fields[name]],
            "additionalProperties": False,
        }
        for name in PROPOSED
        if name in rubric.checks
    ]
    return {
        "type": "object",
        "properties": {"checks": {"type": "array", "items": {"anyOf": branches}}},
        "required": ["checks"],
        "additionalProperties": False,
    }


def judge_schema() -> dict[str, Any]:
    """Return the output schema of a citation judgement.

    Returns:
        The JSON Schema: the quoted passage and whether it supports the statement.
    """
    return {
        "type": "object",
        "properties": {"passage": dict(_STRING), "supports": {"type": "boolean"}},
        "required": ["passage", "supports"],
        "additionalProperties": False,
    }


def _proposals(output: Any) -> list[Any]:
    """Return the proposals of the model's answer.

    Raises:
        RunError: If the answer is not an object with a list of checks.
    """
    if not isinstance(output, dict) or not isinstance(output.get("checks"), list):
        raise RunError("invalid-record", "the check plan is not an object with a list of checks")
    return output["checks"]


def _string_list(record: Mapping[str, Any], key: str) -> list[str]:
    """Return a non-empty list of strings from a proposal, or refuse it."""
    value = record.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise _ProposalError(INVALID_RECORD, f"{key} must be a non-empty list of strings")
    return value


def _strings(record: Mapping[str, Any], *keys: str) -> None:
    """Refuse a proposal one of whose ``keys`` is not a string."""
    for key in keys:
        if not isinstance(record.get(key), str):
            raise _ProposalError(INVALID_RECORD, f"{key} must be a string")


def _chunks(graph: Graph, ids: Sequence[str]) -> list[Node]:
    """Return the document nodes a proposal cites, or refuse it."""
    missing = [node_id for node_id in ids if node_id not in graph]
    if missing:
        raise _ProposalError(UNKNOWN_NODE, f"nodes not in the graph: {missing}")
    nodes = [graph.nodes[node_id] for node_id in ids]
    others = [node.id for node in nodes if origin_of(node) != DOCUMENT]
    if others:
        raise _ProposalError(INVALID_RECORD, f"nodes that are not chunks of the document: {others}")
    return nodes


def _target(graph: Graph, rubric: Rubric, record: Any) -> tuple[Node, str]:
    """Check a proposal's shape and target; return the target and the check type."""
    if not isinstance(record, Mapping):
        raise _ProposalError(INVALID_RECORD, "the proposal is not an object")
    check_type = record.get("type")
    if not isinstance(check_type, str) or check_type not in _ALLOWED:
        raise _ProposalError(INVALID_RECORD, f"the type {check_type!r} is not one of {list(_ALLOWED)}")
    unknown = sorted(set(record) - _ALLOWED[check_type])
    if unknown:
        raise _ProposalError(INVALID_RECORD, f"a {check_type} proposal has unknown fields {unknown}")
    target_id = record.get("target")
    if not isinstance(target_id, str) or target_id not in targets(graph, rubric):
        raise _ProposalError(NOT_A_TARGET, f"{target_id!r} is not a node of a type to assess")
    node = graph.nodes[target_id]
    if check_type not in applicable(rubric, node.kind):
        raise _ProposalError(NOT_A_TARGET, f"the rubric does not run {check_type} on a {node.kind!r}")
    return node, check_type


@dataclass(frozen=True)
class _Plan:
    """A proposal that passed every check before running."""

    target: Node
    type: str
    record: Mapping[str, Any]
    nodes: tuple[Node, ...] = ()
    reference: Node | None = None


def _plan(graph: Graph, rubric: Rubric, record: Any) -> _Plan:
    """Check a proposal before it is run.

    Raises:
        _ProposalError: If it is malformed, does not target a node of a type the check runs on, cites what
            the graph lacks, or, for ``recompute-numbers``, has a numeral the cited chunks lack.
    """
    target, check_type = _target(graph, rubric, record)
    if check_type == CITATION:
        _strings(record, "reference")
        reference = graph.nodes.get(record["reference"])
        if reference is None or origin_of(reference) != DOCUMENT or reference.attributes.get("unit") != REFERENCE:
            raise _ProposalError(INVALID_RECORD, f"{record['reference']!r} is not a bibliography entry of the document")
        parents = {target.id} | {
            relation.target
            for relation in graph.relations.values()
            if relation.type == PART_OF and relation.source == target.id
        }
        if not any(
            relation.type == CITES and relation.source in parents and relation.target == reference.id
            for relation in graph.relations.values()
        ):
            raise _ProposalError(INVALID_RECORD, f"{target.id!r} does not cite {reference.id!r}")
        return _Plan(target, check_type, record, reference=reference)
    nodes = tuple(_chunks(graph, _string_list(record, "nodes")))
    if check_type == REDERIVE and any(node.attributes.get("unit") != EQUATION for node in nodes):
        raise _ProposalError(
            INVALID_RECORD, f"a step of mathematics goes between equations, not {[n.id for n in nodes]}"
        )
    if check_type == RECOMPUTE:
        _strings(record, "expression", "expected")
        try:
            numerals = evaluate.arithmetic(record["expression"]).numerals
        except EvaluationError as error:
            raise _ProposalError(INVALID_RECORD, str(error)) from None
        if not evaluate.NUMERAL.fullmatch(record["expected"].removeprefix("-")):
            raise _ProposalError(INVALID_RECORD, f"expected {record['expected']!r} is not one decimal numeral")
        written = {value for node in nodes for value, _ in evaluate.text_numerals(node_text(graph, node))}
        for numeral in (*numerals, record["expected"]):
            if Fraction(Decimal(numeral)) not in written:
                raise _ProposalError(UNFAITHFUL, f"no numeral of {[node.id for node in nodes]} equals {numeral}")
    else:
        _strings(record, "lhs", "rhs", "assumption")
        declared = record.get("symbols")
        if not isinstance(declared, list) or not all(
            isinstance(item, Mapping)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("assumptions"), list)
            and all(isinstance(a, str) for a in item["assumptions"])
            for item in declared
        ):
            raise _ProposalError(INVALID_RECORD, "symbols must be a list of {name, assumptions, domain}")
        try:
            names = evaluate.symbols(declared)
            evaluate.symbolic(record["lhs"], names)
            evaluate.symbolic(record["rhs"], names)
        except EvaluationError as error:
            raise _ProposalError(INVALID_RECORD, str(error)) from None
    return _Plan(target, check_type, record, nodes=nodes)


def _recompute(graph: Graph, plan: _Plan) -> Result:
    """Run ``recompute-numbers``."""
    expression, expected = plan.record["expression"], plan.record["expected"]
    value = evaluate.arithmetic(expression).value
    # The reported value and its rounding as the text writes it; of several numerals exactly equal to it, the
    # coarsest. Values are compared as exact fractions of their decimal text, never as floats, which would let
    # 0.10000000000000001 stand for 0.1.
    exact = Fraction(Decimal(expected))
    reported, half = max(
        (
            (number, rounding)
            for node in plan.nodes
            for number, rounding in evaluate.text_numerals(node_text(graph, node))
            if number == exact
        ),
        key=lambda pair: pair[1],
    )
    difference = abs(value - reported)
    result = PASSED if difference <= half else FAILED
    within = "within" if result == PASSED else "outside"
    return Result(
        RECOMPUTE,
        plan.target.id,
        {"kind": "numbers", "nodes": [node.id for node in plan.nodes]},
        {"expression": expression, "expected": expected},
        result,
        f"{expression} = {_exact(value)} (exact, Fraction); {within} {expected} ± {_exact(half)}",
        {
            "value": _exact(value),
            "reported": _exact(reported),
            "rounding": _exact(half),
            "difference": _exact(difference),
        },
    )


def _exact(value: Fraction) -> str:
    """Write a fraction exactly: as a decimal when one of 28 digits holds it, else as ``p/q``."""
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal, "f") if Fraction(decimal) == value else str(value)


def _encoding(record: Mapping[str, Any]) -> dict[str, Any]:
    """The symbolic encoding of a step, as recorded in ``inputs``."""
    return {key: record[key] for key in ("lhs", "rhs", "symbols", "assumption")}


def _rederive(plan: _Plan) -> Result:
    """Run ``rederive-maths``."""
    encoding = _encoding(plan.record)
    artifact = {
        "kind": "equation",
        "nodes": [node.id for node in plan.nodes],
        "labels": [node.attributes.get("label") for node in plan.nodes],
    }
    names = evaluate.symbols(encoding["symbols"])
    lhs, rhs = evaluate.symbolic(encoding["lhs"], names), evaluate.symbolic(encoding["rhs"], names)
    try:
        result, detail = evaluate.identity(lhs, rhs)
    except (ArithmeticError, TypeError, ValueError, AttributeError, NotImplementedError, RecursionError) as error:
        result, detail = ERROR, f"sympy failed: {type(error).__name__}: {error}"
    return Result(
        REDERIVE, plan.target.id, artifact, encoding, result, detail, {"result": result}, libraries=("sympy",)
    )


def _numeric(plan: _Plan, artifact: Mapping[str, Any]) -> Result:
    """Run ``rederive-maths-numeric`` on the encoding of an errored ``rederive-maths``."""
    encoding = _encoding(plan.record)
    seed = evaluate.seed_of(encoding)
    inputs: dict[str, Any] = {**encoding, "seed": seed, "points": [], "rtol": str(evaluate.RTOL)}
    try:
        points = evaluate.sample(encoding["symbols"], seed)
        inputs["points"] = points
        names = evaluate.symbols(encoding["symbols"])
        lhs, rhs = evaluate.symbolic(encoding["lhs"], names), evaluate.symbolic(encoding["rhs"], names)
        result, detail = evaluate.agree_at(lhs, rhs, encoding["symbols"], names, points)
    except EvaluationError as error:
        result, detail = ERROR, str(error)
    except (ArithmeticError, TypeError, ValueError, AttributeError, NotImplementedError, RecursionError) as error:
        result, detail = ERROR, f"sympy failed: {type(error).__name__}: {error}"
    output = {"result": result}
    return Result(NUMERIC, plan.target.id, artifact, inputs, result, detail, output, libraries=("sympy",))


def _citation(graph: Graph, plan: _Plan, cited: Cited, ask: Ask) -> Result:
    """Run ``citation-support``: read the cited work, ask the model, and locate its passage."""
    reference = plan.reference
    assert reference is not None  # noqa: S101 - _plan sets it for a citation
    artifact: dict[str, Any] = {"kind": "citation", "reference": reference.id}
    statement = node_text(graph, plan.target)
    try:
        source = cited(reference)
    except CitedError as error:
        return Result(CITATION, plan.target.id, artifact, {"statement": statement}, ERROR, str(error))
    artifact["document"] = source.document
    inputs: dict[str, Any] = {"statement": statement, "source_digest": text_digest(source.text)}
    prompt = Template(JUDGE_TEMPLATE).substitute(statement=statement, source=source.text)
    answer, request = ask.answer(prompt, judge_schema())
    model = {key: request.get(key) for key in ("provider", "model", "request_id")}
    passage = answer.get("passage") if isinstance(answer, dict) else None
    supports = answer.get("supports") if isinstance(answer, dict) else None
    if not isinstance(passage, str) or not isinstance(supports, bool):
        detail = "the judgement is not a passage and a true or false"
        return Result(CITATION, plan.target.id, artifact, inputs, ERROR, detail, {"answer": answer}, model=model)
    quote = normalise_text(passage)
    start = source.text.find(quote) if quote else -1
    if start < 0:
        detail = "the quoted passage is not in the text of the cited work"
        output = {"answer": answer}
        return Result(CITATION, plan.target.id, artifact, inputs, ERROR, detail, output, model=model)
    locator = f"chars={start}-{start + len(quote)}"
    artifact.update(locator=locator, quote=quote, digest=text_digest(quote))
    anchor = SourceAnchor(source.document, locator, quote, text_digest(quote))
    result = PASSED if supports else FAILED
    detail = f"the passage at {locator} {'supports' if supports else 'does not support'} the statement"
    return Result(
        CITATION, plan.target.id, artifact, inputs, result, detail, {"answer": answer}, (anchor,), model=model
    )


def _json(value: Any) -> bytes:
    """Encode a value as the indented JSON the run writes."""
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _files(result: Result, record: Mapping[str, Any]) -> dict[str, bytes]:
    """The files of a check's directory: its inputs, the tool's output and a log."""
    log = [
        f"{result.type} on {result.target}: {result.result}",
        result.detail,
        f"tool: {json.dumps(record['tool'], sort_keys=True)}",
        f"model: {json.dumps(record['model'], sort_keys=True)}",
    ]
    return {
        INPUTS: _json(
            {"type": result.type, "target": result.target, "artifact": record["artifact"], "inputs": record["inputs"]}
        ),
        OUTPUT: _json({"result": result.result, "detail": result.detail, **result.output}),
        LOG: ("\n".join(log) + "\n").encode("utf-8"),
    }


def _record(check_id: str, result: Result, tool: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The check record of a result, and the files of its directory, whose digest the record's ``log`` holds."""
    record: dict[str, Any] = {
        "type": result.type,
        "target": result.target,
        "artifact": dict(result.artifact),
        "inputs": dict(result.inputs),
        "result": result.result,
        "detail": result.detail,
        "tool": dict(tool),
        "model": dict(result.model) if result.model is not None else None,
    }
    files = _files(result, record)
    record["log"] = {"path": f"{CHECKS}/{check_id}/", "digest": directory_digest(files)}
    return record, files


def plain(value: Any) -> Any:
    """Return a value as JSON gives it back: a graph's read-only mappings and its tuples become dicts and lists.

    Args:
        value: A value made of JSON types, mappings and sequences.

    Returns:
        The value in JSON types only.
    """
    return json.loads(json.dumps(value, default=lambda item: dict(item) if isinstance(item, Mapping) else list(item)))


def rerun(graph: Graph, rubric: Rubric, check: Node) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run a recorded deterministic check again, from the inputs its record holds.

    The recorded inputs are checked as the run checked the proposal, numeral faithfulness included,
    and evaluated by the current code; ``rederive-maths-numeric`` draws its points again from the
    encoding's seed. A record that can no longer be run gives the result ``error``, with the reason as
    its detail.

    Args:
        graph: The graph the check node is in.
        rubric: The rubric of the run.
        check: A check node of a type in ``DETERMINISTIC``.

    Returns:
        The check record the re-run gives, with the recorded ``tool``, and the files of its directory
        by name. When the code and the record agree, the record equals ``check.attributes["check"]``.

    Raises:
        ValueError: If the node is not a check of a type in ``DETERMINISTIC``.
    """
    recorded = plain(check.attributes.get("check"))
    if not isinstance(recorded, dict) or recorded.get("type") not in DETERMINISTIC:
        msg = f"{check.id!r} is not a check of a type in {list(DETERMINISTIC)}"
        raise ValueError(msg)
    check_type, target = recorded["type"], recorded.get("target")
    artifact, inputs = recorded.get("artifact"), recorded.get("inputs")
    if not isinstance(artifact, dict) or not isinstance(inputs, dict):
        artifact, inputs = {}, {}
    # The proposal the record was run from: the symbolic encoding for both maths checks.
    proposed = RECOMPUTE if check_type == RECOMPUTE else REDERIVE
    proposal = {key: inputs.get(key) for key in _ALLOWED[proposed]}
    proposal.update(target=target, type=proposed, nodes=artifact.get("nodes"))
    try:
        plan = _plan(graph, rubric, proposal)
    except _ProposalError as refused:
        detail = f"the recorded inputs cannot be run: {refused}"
        result = Result(check_type, str(target), artifact, inputs, ERROR, detail, {"result": ERROR})
    else:
        if check_type == RECOMPUTE:
            result = _recompute(graph, plan)
        elif check_type == REDERIVE:
            result = _rederive(plan)
        else:
            result = _numeric(plan, artifact)
    return _record(check.id, result, recorded.get("tool") or {})


def directory_digest(files: Mapping[str, bytes]) -> str:
    """Return the digest of a check's directory: of one line ``<name> <digest>`` per file, sorted by name.

    Args:
        files: The files by name.

    Returns:
        The digest.
    """
    listing = "".join(f"{name} {bytes_digest(files[name])}\n" for name in sorted(files))
    return bytes_digest(listing.encode("utf-8"))


def _corrected(refused: Sequence[Mapping[str, Any]]) -> set[str] | None:
    """The targets a repair answer may propose checks on: those of the refused proposals.

    A target written with its label, as ``s-2 (result)``, counts as the id it names, so its correction is
    admitted. When a refused proposal names no target at all (its target is missing or not a string), its
    correction may be on any target: there is no id to hold it to. Returns ``None`` then, for no restriction; duplicates are still refused.
    """
    named = [rejection["node"] for rejection in refused]
    if not all(isinstance(target, str) for target in named):
        return None
    # An unknown name stays in the set as written: it matches no correction, so it opens nothing.
    return {unlabelled(target) or target for target in named}


def _admit(
    graph: Graph, rubric: Rubric, proposals: Sequence[Any], plans: list[_Plan], *, only: Collection[str] | None = None
) -> list[dict[str, Any]]:
    """Check each proposal and add those that pass to ``plans``; return the refused ones as rejections.

    A proposal is refused as ``_plan`` refuses it, when it repeats the target and type of a plan already
    admitted, and, with ``only``, when its target is not one of those.
    """
    rejections = []
    for record in proposals:
        try:
            target = record.get("target") if isinstance(record, Mapping) else None
            if only is not None and not (isinstance(target, str) and target in only):
                raise _ProposalError(NOT_REJECTED, f"{target!r} had no refused proposal to correct")
            plan = _plan(graph, rubric, record)
            if any((plan.target.id, plan.type) == (other.target.id, other.type) for other in plans):
                msg = f"{plan.target.id!r} has an earlier {plan.type} proposal: one check of a type per node"
                raise _ProposalError(DUPLICATE_RECORD, msg)
            plans.append(plan)
        except _ProposalError as refused:
            node = record.get("target") if isinstance(record, Mapping) else None
            node = node if isinstance(node, str) else None
            rejections.append({"node": node, "code": refused.code, "message": str(refused), "record": record})
    return rejections


def assess(graph: Graph, rubric: Rubric) -> Graph:
    """Set the verdict of every node the checks assess, from the checks that observe it.

    A target's verdict comes from the results of the check nodes its ``observes`` relations lead to,
    never from a model: ``fails``, ``holds`` or ``undetermined`` as ``verdict`` gives it, else
    ``not_assessed`` because no implemented check type may run on the node, or none was run.

    Args:
        graph: The graph.
        rubric: The rubric.

    Returns:
        The graph with ``attributes.assessment`` set on every target.
    """
    by_target: dict[str, list[str]] = {}
    for relation in graph.relations.values():
        check = graph.nodes[relation.target].attributes.get("check") if relation.type == OBSERVES else None
        if isinstance(check, Mapping):
            by_target.setdefault(relation.source, []).append(check.get("result"))
    assessed = set(targets(graph, rubric))
    out = Graph()
    for node in graph:
        if node.id in assessed:
            found = verdict(by_target.get(node.id, []))
            if found is not None:
                assessment: dict[str, Any] = {"verdict": found}
            else:
                reason = NOT_SELECTED if applicable(rubric, node.kind) else NO_CHECK
                assessment = {"verdict": NOT_ASSESSED, "reason": reason}
            node = replace(node, attributes={**node.attributes, "assessment": assessment})  # noqa: PLW2901
        out.add_node(node)
    for relation in graph.relations.values():
        out.add_relation(relation)
    return out


def verdict(results: Sequence[str]) -> str | None:
    """Return the verdict the results of a node's checks give, or ``None`` when no check ran.

    Args:
        results: The results.

    Returns:
        ``fails`` if any failed, else ``holds`` if any passed, else ``undetermined``.
    """
    if not results:
        return None
    if FAILED in results:
        return FAILS
    return HOLDS if PASSED in results else UNDETERMINED


@dataclass
class CheckPass:
    """Propose checks of the assessed nodes, run them, and record each as an observed evidence node.

    Attributes:
        cited: Reads the text of a cited work; none can be read by default.
        allow_uncalibrated: Run a check type that lists ``models`` with a model it does not list.
        name: ``checks``.
        template: The prompt templates as shipped: the plan's, then the citation judgement's.
    """

    cited: Cited = no_sources
    allow_uncalibrated: bool = False
    name: str = NAME
    template: str = TEMPLATE

    def run(self, graph: Graph, rubric: Rubric, ask: Ask) -> Outcome:
        """Apply the pass: one plan call, one repair call if any proposal was refused, then one judgement call per citation check.

        A graph with no node to assess needs no call and is returned unchanged; nor does one whose
        nodes no implemented check type of the rubric may run on, whose nodes are then marked
        ``not_assessed``.

        Args:
            graph: The graph after the earlier passes.
            rubric: The run's rubric.
            ask: Sends a prompt to the model.

        Returns:
            The graph with the check nodes, their check edges and the verdicts; the rejected
            proposals of both rounds, each marked with its round; the files of each check's directory; and the report, whose ``checks`` lists
            each check's id, type, target and result and whose ``uncalibrated`` lists every check
            run with a model its check type does not list.

        Raises:
            RunError: If the plan is not a list of checks, or a citation check would be run with a
                model its check type does not list and ``allow_uncalibrated`` is not set.
        """
        assessed = targets(graph, rubric)
        if not assessed:
            return Outcome(graph)
        model: str | None = None
        plans: list[_Plan] = []
        rejections: list[dict[str, Any]] = []
        if any(applicable(rubric, graph.nodes[node_id].kind) for node_id in assessed):
            output, request = ask.answer(render(graph, rubric), output_schema(rubric))
            model = request.get("model")
            refused = _admit(graph, rubric, _proposals(output), plans)
            rejections = [{"round": INITIAL, **rejection} for rejection in refused]
            if refused:
                answer = ask(render(graph, rubric, refused), output_schema(rubric))
                again = _admit(graph, rubric, _proposals(answer), plans, only=_corrected(refused))
                rejections.extend({"round": REPAIR, **rejection} for rejection in again)
        order = {node_id: i for i, node_id in enumerate(graph.nodes)}
        plans.sort(key=lambda plan: order[plan.target.id])
        results: list[Result] = []
        uncalibrated = []
        for plan in plans:
            if plan.type == RECOMPUTE:
                results.append(_recompute(graph, plan))
            elif plan.type == REDERIVE:
                symbolic = _rederive(plan)
                results.append(symbolic)
                numeric = rubric.checks.get(NUMERIC)
                if symbolic.result == ERROR and numeric is not None and plan.target.kind in numeric.targets:
                    results.append(_numeric(plan, symbolic.artifact))
            else:
                check = rubric.checks[CITATION]
                if check.models and model not in check.models:
                    if not self.allow_uncalibrated:
                        msg = (
                            f"check type {CITATION!r} was fitted with {list(check.models)}, not {model!r}: "
                            "pass --allow-uncalibrated to run it anyway"
                        )
                        raise RunError("uncalibrated-model", msg)
                    uncalibrated.append(len(results))
                results.append(_citation(graph, plan, self.cited, ask))
        return self._outcome(graph, rubric, ask, results, rejections, uncalibrated)

    def _outcome(  # noqa: PLR0913, PLR0917 - the parts the outcome is assembled from
        self,
        graph: Graph,
        rubric: Rubric,
        ask: Ask,
        results: list[Result],
        rejections: list[dict[str, Any]],
        uncalibrated: list[int],
    ) -> Outcome:
        """Assemble the check nodes, the check edges, the verdicts, the files and the report."""
        versions = ask.versions
        fresh = (node_id for n in count(1) if (node_id := f"c-{n:03d}") not in graph)
        nodes: list[Node] = []
        relations: list[Relation] = []
        files: dict[str, bytes] = {}
        report = []
        by_target: dict[str, list[str]] = {}
        for result in results:
            check_id = next(fresh)
            check: Check = rubric.checks[result.type]
            tool = {
                "name": "credencegraph-extract",
                "version": versions.get("credencegraph-extract"),
                "python": versions.get("python"),
                **{library: versions.get(library) for library in result.libraries},
            }
            record, check_files = _record(check_id, result, tool)
            files.update({f"{CHECKS}/{check_id}/{name}": data for name, data in check_files.items()})
            attributes: dict[str, Any] = {"origin": EVIDENCE, "check": record}
            if result.result != ERROR:
                attributes["observed"] = result.result == PASSED
            nodes.append(
                Node(
                    check_id,
                    kind=result.type,
                    statement=f"{result.type} on {result.target}: {result.result}",
                    sources=result.sources,
                    attributes=attributes,
                )
            )
            relations.append(
                Relation(
                    f"{result.target}-{OBSERVES}-{check_id}",
                    OBSERVES,
                    result.target,
                    check_id,
                    sensitivity=check.sensitivity,
                    false_pass=check.false_pass,
                )
            )
            by_target.setdefault(result.target, []).append(result.result)
            report.append({"id": check_id, "type": result.type, "target": result.target, "result": result.result})
        out = Graph()
        for node in (*graph, *nodes):
            out.add_node(node)
        for relation in (*graph.relations.values(), *relations):
            out.add_relation(relation)
        out = assess(out, rubric)
        uncalibrated_report = [
            {"check": report[i]["id"], "type": report[i]["type"], "model": results[i].model} for i in uncalibrated
        ]
        return Outcome(out, rejections, {"checks": report, "uncalibrated": uncalibrated_report}, files)
