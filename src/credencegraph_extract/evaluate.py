"""The deterministic evaluators of the checks: exact arithmetic, symbolic identity, and numeric sampling.

A model writes the expressions these evaluate, so none of them is ever passed to ``eval``, ``sympify``
or ``parse_expr``. Each is parsed with ``ast`` and walked against a whitelist: numerals written in
plain decimal or exponent notation, ``+ - * / **``, unary signs and parentheses, and, for the
symbolic evaluator only, declared symbols, a fixed set of functions and the constants ``pi``, ``E``
and ``I``. Anything else is refused before any value is computed.

``arithmetic`` computes exactly, with every numeral read as a ``Fraction`` of its decimal text.
``identity`` asks sympy whether two expressions are equal for every value of their symbols, and
says ``error`` when sympy cannot decide. ``sample`` and ``agree_at`` evaluate the same two
expressions at points drawn from a seeded generator, so the points can be recorded and drawn again.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

import sympy
from credencegraph.rubric.faithfulness import _GROUP, _MINUS
from credencegraph.rubric.faithfulness import _NUMERAL as _TEXT_NUMERAL
from sympy.core.facts import InconsistentAssumptions

# Results of a check, as the core's check records name them.
PASSED = "passed"
FAILED = "failed"
ERROR = "error"

# The longest expression, and the most syntax nodes, the evaluators accept.
MAX_LENGTH = 500
MAX_NODES = 200
# The largest exponent a power of two numbers may have: ``9 ** 9 ** 9`` would not finish.
MAX_EXPONENT = 100

# A numeral as the evaluators accept it: decimal digits, an optional fraction and an optional exponent.
# Python also reads ``0x10``, ``1_000`` and ``1j`` as numbers; those are refused.
NUMERAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

FUNCTIONS: Mapping[str, Callable[..., Any]] = {
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "log": sympy.log,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "Abs": sympy.Abs,
}
CONSTANTS: Mapping[str, Any] = {"pi": sympy.pi, "E": sympy.E, "I": sympy.I}

# The sympy assumptions a symbol may carry, and the test each puts on a sampled value.
ASSUMPTIONS: Mapping[str, Callable[[Fraction], bool]] = {
    "real": lambda value: True,
    "positive": lambda value: value > 0,
    "negative": lambda value: value < 0,
    "nonnegative": lambda value: value >= 0,
    "nonpositive": lambda value: value <= 0,
    "nonzero": lambda value: value != 0,
    "integer": lambda value: value.denominator == 1,
}

# The numeric check: how many points, and the relative tolerance at each.
POINTS = 32
RTOL = Decimal("1e-12")
_DIGITS = 30


class EvaluationError(ValueError):
    """An expression is refused or cannot be evaluated."""


def _parse(expression: str) -> ast.expr:
    """Parse an expression, refusing one that is too long, not an expression, or has too many nodes."""
    if len(expression) > MAX_LENGTH:
        msg = f"the expression is longer than {MAX_LENGTH} characters"
        raise EvaluationError(msg)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        msg = f"{expression!r} is not an expression: {error.msg}"
        raise EvaluationError(msg) from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        msg = f"the expression has more than {MAX_NODES} syntax nodes"
        raise EvaluationError(msg)
    return tree.body


def _numeral(expression: str, node: ast.Constant) -> str:
    """Return the text of a numeric literal, refusing any that is not a plain decimal numeral."""
    text = ast.get_source_segment(expression, node) or ""
    if isinstance(node.value, bool) or not isinstance(node.value, int | float) or not NUMERAL.fullmatch(text):
        msg = f"{text or node.value!r} is not a decimal numeral"
        raise EvaluationError(msg)
    return text


def _check_power(base: Any, exponent: Any) -> None:
    """Refuse a power of two numbers whose exponent is too large to compute."""
    number = isinstance(base, Fraction | sympy.Number) and isinstance(exponent, Fraction | sympy.Number)
    if number and abs(exponent) > MAX_EXPONENT:
        msg = f"the exponent {exponent} is larger than {MAX_EXPONENT}"
        raise EvaluationError(msg)


@dataclass(frozen=True)
class Arithmetic:
    """The exact value of an arithmetic expression.

    Attributes:
        value: The value.
        numerals: The text of every numeral in the expression, in the order written.
    """

    value: Fraction
    numerals: tuple[str, ...]


def arithmetic(expression: str) -> Arithmetic:
    """Evaluate an arithmetic expression exactly.

    Numerals are read as the ``Fraction`` of their decimal text, so ``0.1 + 0.2`` is exactly
    ``3/10``. A power needs an integer exponent, since a root of a fraction is not a fraction.

    Args:
        expression: Numerals joined by ``+ - * / **``, unary signs and parentheses.

    Returns:
        The value and the numerals.

    Raises:
        EvaluationError: If the expression holds anything else, divides by zero, or raises to a
            power that is not an integer or is too large.
    """
    tree = _parse(expression)
    found: list[str] = []

    def walk(node: ast.expr) -> Fraction:
        if isinstance(node, ast.Constant):
            text = _numeral(expression, node)
            found.append(text)
            return Fraction(Decimal(text))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1:
                    msg = f"the exponent {right} is not an integer"
                    raise EvaluationError(msg)
                _check_power(left, right)
            try:
                return _OPERATORS[type(node.op)](left, right)
            except ZeroDivisionError:
                msg = "the expression divides by zero"
                raise EvaluationError(msg) from None
        msg = f"{ast.unparse(node)!r} is not arithmetic on numerals"
        raise EvaluationError(msg)

    return Arithmetic(walk(tree), tuple(found))


def text_numerals(text: str) -> list[tuple[Fraction, Fraction]]:
    """Return every numeral of a text with the rounding its own digits imply.

    The numerals are found with the core's numeral pattern, the one its faithfulness rule uses, and
    each value is computed the same way: groups removed, the exponent applied, the sign kept. The
    rounding is half a unit of the last digit written, so ``30`` is ``30 ± 0.5`` and ``1.20e-3`` is
    ``0.00120 ± 0.000005``. A numeral followed by ``%`` contributes its value and its value divided by
    100, each with its rounding scaled alike, as the core's ``numerals`` does.

    Args:
        text: The text.

    Returns:
        ``(value, half unit)`` pairs, in the order the numerals occur.
    """
    found = []
    for match in _TEXT_NUMERAL.finditer(text):
        digits = Decimal(re.sub(_GROUP, "", match["digits"]))
        exponent = match["e"] or match["braced"] or match["bare"]
        shift = int(exponent.replace(_MINUS, "-")) if exponent is not None else 0
        value = Fraction(digits.scaleb(shift))
        half = Fraction(Decimal(1).scaleb(digits.as_tuple().exponent + shift)) / 2  # type: ignore[operator]
        if match["sign"] in ("-", _MINUS):
            value = -value
        found.append((value, half))
        if match["percent"]:
            found.append((value / 100, half / 100))
    return found


def symbols(declared: Sequence[Mapping[str, Any]]) -> dict[str, sympy.Symbol]:
    """Create the sympy symbols an encoding declares.

    Args:
        declared: Each symbol as ``{"name", "assumptions": [...]}``.

    Returns:
        The symbols by name.

    Raises:
        EvaluationError: If a name is not an identifier, is repeated, or names a function or constant;
            or an assumption is not one of ``ASSUMPTIONS``, or the assumptions contradict each other.
    """
    made: dict[str, sympy.Symbol] = {}
    for item in declared:
        name, assumptions = item["name"], item.get("assumptions", [])
        if not name.isidentifier() or name in FUNCTIONS or name in CONSTANTS or name in made:
            msg = f"{name!r} cannot name a symbol"
            raise EvaluationError(msg)
        unknown = sorted(set(assumptions) - set(ASSUMPTIONS))
        if unknown:
            msg = f"symbol {name!r} has assumptions {unknown}, not among {list(ASSUMPTIONS)}"
            raise EvaluationError(msg)
        try:
            made[name] = sympy.Symbol(name, **dict.fromkeys(assumptions, True))
        except InconsistentAssumptions as error:
            msg = f"symbol {name!r} has contradictory assumptions: {error}"
            raise EvaluationError(msg) from None
    return made


def symbolic(expression: str, names: Mapping[str, sympy.Symbol]) -> sympy.Expr:
    """Build the sympy expression of a text, without ``eval``.

    Args:
        expression: Numerals, the symbols in ``names``, the functions of ``FUNCTIONS`` called with
            positional arguments, the constants of ``CONSTANTS``, ``+ - * / **``, unary signs and
            parentheses.
        names: The declared symbols.

    Returns:
        The expression; numerals are exact rationals.

    Raises:
        EvaluationError: If the text holds anything else, or a power of two numbers is too large.
    """
    tree = _parse(expression)

    def walk(node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return sympy.Rational(Fraction(Decimal(_numeral(expression, node))))
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.Name) and node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                _check_power(left, right)
            return _OPERATORS[type(node.op)](left, right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and not node.keywords
            and node.args
        ):
            return FUNCTIONS[node.func.id](*(walk(arg) for arg in node.args))
        msg = f"{ast.unparse(node)!r} is not an allowed expression: use declared symbols, numerals and {sorted(FUNCTIONS)}"
        raise EvaluationError(msg)

    return walk(tree)


def identity(lhs: sympy.Expr, rhs: sympy.Expr) -> tuple[str, str]:
    """Decide symbolically whether two expressions are equal for every value of their symbols.

    Args:
        lhs: One side.
        rhs: The other.

    Returns:
        The result and what sympy found: ``passed`` when their difference simplifies to zero,
        ``failed`` when sympy shows the difference is never zero, ``error`` when it can show
        neither, since a difference sympy cannot simplify is not evidence either way.
    """
    difference = sympy.simplify(lhs - rhs)
    if difference == 0:
        return PASSED, "the difference simplifies to 0"
    if difference.is_zero is False:
        return FAILED, f"the difference simplifies to {sympy.sstr(difference)}, which is never 0"
    return ERROR, f"sympy cannot decide whether {sympy.sstr(difference)} is 0"


def seed_of(encoding: Mapping[str, Any]) -> int:
    """Return the seed of an encoding's sample points: the first 52 bits of the SHA-256 of its canonical JSON.

    52 bits keep the seed exact in any JSON reader, which reads numbers as doubles.

    Args:
        encoding: The encoding.

    Returns:
        The seed.
    """
    text = json.dumps(encoding, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:13], 16)


def sample(declared: Sequence[Mapping[str, Any]], seed: int, count: int = POINTS) -> list[dict[str, str]]:
    """Draw points in the symbols' domains from a generator seeded with ``seed``.

    A symbol with the ``integer`` assumption takes integers in its domain; any other takes floats,
    written with ``repr`` so the recorded text reads back as the same value.

    Args:
        declared: Each symbol as ``{"name", "assumptions", "domain": [low, high]}``.
        seed: The seed.
        count: How many points.

    Returns:
        The points, each a symbol name to the text of its value.

    Raises:
        EvaluationError: If a symbol has no finite domain ``[low, high]`` with ``low < high``, or an
            integer symbol's domain holds no integer.
    """
    rng = random.Random(seed)  # noqa: S311 - reproducible sample points, not a secret
    bounds = []
    for item in declared:
        domain = item.get("domain")
        if (
            not isinstance(domain, list)
            or len(domain) != 2  # noqa: PLR2004 - a domain is [low, high]
            or not all(isinstance(end, int | float) and not isinstance(end, bool) for end in domain)
            or not float("-inf") < domain[0] < domain[1] < float("inf")
        ):
            msg = f"symbol {item['name']!r} needs a finite domain [low, high] with low < high"
            raise EvaluationError(msg)
        bounds.append((item["name"], domain[0], domain[1], "integer" in item.get("assumptions", [])))
    points = []
    for _ in range(count):
        point = {}
        for name, low, high, integer in bounds:
            if integer:
                start, stop = int(sympy.ceiling(low)), int(sympy.floor(high))
                if start > stop:
                    msg = f"the domain of integer symbol {name!r} holds no integer"
                    raise EvaluationError(msg)
                point[name] = str(rng.randint(start, stop))
            else:
                point[name] = repr(rng.uniform(low, high))
        points.append(point)
    return points


def agree_at(
    lhs: sympy.Expr,
    rhs: sympy.Expr,
    declared: Sequence[Mapping[str, Any]],
    names: Mapping[str, sympy.Symbol],
    points: Sequence[Mapping[str, str]],
) -> tuple[str, str]:
    """Compare two expressions at each point, to a relative tolerance of ``RTOL``.

    The point's values are substituted exactly, as rationals, so a pole gives no finite value rather
    than a large one, and each side is then evaluated to 30 significant digits. Sides agree at a point when
    ``|lhs - rhs| <= RTOL * max(|lhs|, |rhs|)``.

    Args:
        lhs: One side.
        rhs: The other.
        declared: The symbols with their assumptions, which every point must satisfy.
        names: The symbols by name.
        points: The points, as returned by ``sample``.

    Returns:
        The result and what was found: ``passed`` when the sides agree at every point, ``failed`` at
        the first point where they do not, ``error`` when a point breaks a symbol's assumptions or a
        side has no finite value there.
    """
    for i, point in enumerate(points):
        values = {name: Fraction(Decimal(text)) for name, text in point.items()}
        for item in declared:
            for assumption in item.get("assumptions", []):
                if not ASSUMPTIONS[assumption](values[item["name"]]):
                    return ERROR, f"point {i} gives {item['name']} = {point[item['name']]}, which is not {assumption}"
        subs = {names[name]: sympy.Rational(value) for name, value in values.items()}
        sides = []
        for side in (lhs, rhs):
            value = side.xreplace(subs).evalf(_DIGITS)
            real, imaginary = value.as_real_imag()
            if not (real.is_finite and imaginary.is_finite and real.is_number and imaginary.is_number):
                return ERROR, f"at point {i} a side has no finite value: {sympy.sstr(value)}"
            sides.append((Decimal(str(real)), Decimal(str(imaginary))))
        (lr, li), (rr, ri) = sides
        difference = abs(complex(lr - rr, li - ri))
        scale = max(abs(complex(lr, li)), abs(complex(rr, ri)))
        if Decimal(repr(difference)) > RTOL * Decimal(repr(scale)):
            return FAILED, f"at point {i} the sides differ: {lr}{li:+}i against {rr}{ri:+}i"
    return PASSED, f"the sides agree to rtol {RTOL} at all {len(points)} points"
