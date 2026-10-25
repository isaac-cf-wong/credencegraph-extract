"""The evaluators of the checks: exact arithmetic, the whitelist, symbolic identity and numeric sampling."""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy

from credencegraph_extract import evaluate
from credencegraph_extract.evaluate import (
    ERROR,
    FAILED,
    PASSED,
    EvaluationError,
    agree_at,
    arithmetic,
    identity,
    sample,
    seed_of,
    symbolic,
    symbols,
    text_numerals,
)

X = [{"name": "x", "assumptions": ["real"], "domain": [-3, 3]}]


def test_arithmetic_is_exact() -> None:
    result = arithmetic("0.1 + 0.2 - 3 * (1.5e-1) / 4.5")
    assert result.value == Fraction(3, 10) - Fraction(1, 10)
    assert result.numerals == ("0.1", "0.2", "3", "1.5e-1", "4.5")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "x",
        "(1).real",
        "abs(-1)",
        "0x10",
        "1_000",
        "1j",
        "True",
        "[1]",
        "1 if 1 else 2",
        "1 < 2",
        "7 % 3",
        "7 // 3",
        "",
    ],
)
def test_arithmetic_refuses_what_is_not_arithmetic_on_numerals(expression: str) -> None:
    with pytest.raises(EvaluationError):
        arithmetic(expression)


@pytest.mark.parametrize("expression", ["2 ** 1000", "2 ** 0.5", "1 / (2 - 2)", "1" + " + 1" * 150, "1" * 600])
def test_arithmetic_refuses_what_cannot_be_computed_exactly_or_cheaply(expression: str) -> None:
    with pytest.raises(EvaluationError):
        arithmetic(expression)


def test_text_numerals_carry_the_rounding_of_their_digits() -> None:
    assert text_numerals("We keep 350, 1,000 and 1.20e-3 of 12.5\\% and -2") == [
        (Fraction(350), Fraction(1, 2)),
        (Fraction(1000), Fraction(1, 2)),
        (Fraction(3, 2500), Fraction(1, 200000)),
        (Fraction(25, 2), Fraction(1, 20)),
        (Fraction(1, 8), Fraction(1, 2000)),
        (Fraction(-2), Fraction(1, 2)),
    ]


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "y",
        "x.subs(x, 1)",
        "sin(x, evaluate=False)",
        "Symbol('y')",
        "sympify('x')",
        "lambda: x",
        "sin()",
        "2 ** 1000",
    ],
)
def test_symbolic_refuses_what_is_not_whitelisted(expression: str) -> None:
    with pytest.raises(EvaluationError):
        symbolic(expression, symbols(X))


@pytest.mark.parametrize(
    "declared",
    [
        [{"name": "sin"}],
        [{"name": "pi"}],
        [{"name": "x"}, {"name": "x"}],
        [{"name": "x-1"}],
        [{"name": "x", "assumptions": ["prime"]}],
        [{"name": "x", "assumptions": ["positive", "negative"]}],
    ],
)
def test_symbols_refuse_bad_declarations(declared: list[dict[str, object]]) -> None:
    with pytest.raises(EvaluationError):
        symbols(declared)


@pytest.mark.parametrize(
    ("lhs", "rhs", "result"),
    [
        ("sin(x)**2 + cos(x)**2", "1", PASSED),
        ("(x + 1)**2", "x**2 + 2*x + 1", PASSED),
        ("x + 1", "x", FAILED),
        ("x**2 + 1", "0", FAILED),
        ("x", "0", ERROR),
        ("sin(x)**2", "1", ERROR),
    ],
)
def test_identity_passes_only_what_sympy_proves(lhs: str, rhs: str, result: str) -> None:
    names = symbols(X)
    assert identity(symbolic(lhs, names), symbolic(rhs, names))[0] == result


def test_numerals_are_exact_rationals_in_symbolic_expressions() -> None:
    assert symbolic("0.1 + 0.2", {}) == sympy.Rational(3, 10)


def test_sample_is_reproducible_from_its_seed_and_stays_in_the_domain() -> None:
    declared = [
        {"name": "x", "assumptions": ["real"], "domain": [-3, 3]},
        {"name": "n", "assumptions": ["integer", "positive"], "domain": [1.5, 4.5]},
    ]
    points = sample(declared, 7)
    assert len(points) == evaluate.POINTS == 32
    assert points == sample(declared, 7)
    assert points != sample(declared, 8)
    assert all(-3 <= float(point["x"]) <= 3 for point in points)
    assert {point["n"] for point in points} <= {"2", "3", "4"}


@pytest.mark.parametrize(
    "domain", [None, [0], [1, 0], [0, float("inf")], [float("nan"), 1], [True, 2], "0,1"], ids=repr
)
def test_sample_needs_a_finite_domain(domain: object) -> None:
    with pytest.raises(EvaluationError):
        sample([{"name": "x", "assumptions": [], "domain": domain}], 1)


def test_sample_refuses_an_integer_domain_without_integers() -> None:
    with pytest.raises(EvaluationError):
        sample([{"name": "n", "assumptions": ["integer"], "domain": [0.2, 0.8]}], 1)


def test_seed_depends_on_the_whole_encoding_and_is_json_safe() -> None:
    one = {"lhs": "x", "rhs": "x", "symbols": X}
    assert seed_of(one) == seed_of(dict(reversed(list(one.items()))))
    assert seed_of(one) != seed_of({**one, "rhs": "x + 0"})
    assert seed_of(one) < 2**53


def test_agree_at_passes_an_identity_sympy_cannot_decide() -> None:
    declared = [{"name": "x", "assumptions": ["positive"], "domain": [0.01, 100]}]
    names = symbols(declared)
    lhs, rhs = symbolic("atan(x) + atan(1/x)", names), symbolic("pi/2", names)
    assert identity(lhs, rhs)[0] == ERROR
    assert agree_at(lhs, rhs, declared, names, sample(declared, 1))[0] == PASSED


def test_agree_at_fails_where_the_sides_differ() -> None:
    declared = [{"name": "x", "assumptions": ["real"], "domain": [2, 3]}]
    names = symbols(declared)
    result, detail = agree_at(
        symbolic("asin(sin(x))", names), symbolic("x", names), declared, names, sample(declared, 1)
    )
    assert result == FAILED
    assert detail.startswith("at point 0 ")


def test_agree_at_tolerates_only_rtol() -> None:
    declared = [{"name": "x", "assumptions": ["positive"], "domain": [1, 2]}]
    names = symbols(declared)
    points = sample(declared, 3)
    assert agree_at(symbolic("x", names), symbolic("x * (1 + 1e-13)", names), declared, names, points)[0] == PASSED
    assert agree_at(symbolic("x", names), symbolic("x * (1 + 1e-11)", names), declared, names, points)[0] == FAILED


def test_agree_at_errs_on_a_point_outside_the_assumptions_or_without_a_value() -> None:
    declared = [{"name": "x", "assumptions": ["positive"], "domain": [1, 2]}]
    names = symbols(declared)
    assert agree_at(names["x"], names["x"], declared, names, [{"x": "-1.0"}])[0] == ERROR
    assert agree_at(symbolic("1/(x - 1)", names), names["x"], declared, names, [{"x": "1"}])[0] == ERROR
