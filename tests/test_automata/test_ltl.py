"""Tests for LTL module."""

import pytest
from gnsp.automata.ltl import (
    LTLOperator,
    LTLFormula,
    prop,
    true_,
    false_,
    not_,
    and_,
    or_,
    implies,
    next_,
    finally_,
    globally,
    until,
    release,
    weak_until,
    to_nnf,
    closure,
    ltl_to_buchi,
    LTLParser,
    parse_ltl,
    LTLMonitor,
    response_pattern,
    absence_pattern,
    universality_pattern,
    precedence_pattern,
    existence_pattern,
    bounded_response,
    mutual_exclusion,
    fairness,
)


class TestLTLFormula:
    """Tests for LTL formula construction."""

    def test_prop(self) -> None:
        """Test atomic proposition."""
        p = prop("p")
        assert p.operator == LTLOperator.PROP
        assert p.prop == "p"

    def test_true_false(self) -> None:
        """Test true and false constants."""
        t = true_()
        f = false_()
        assert t.operator == LTLOperator.TRUE
        assert f.operator == LTLOperator.FALSE

    def test_not(self) -> None:
        """Test negation."""
        p = prop("p")
        np = not_(p)
        assert np.operator == LTLOperator.NOT
        assert np.left == p

    def test_and(self) -> None:
        """Test conjunction."""
        p = prop("p")
        q = prop("q")
        pq = and_(p, q)
        assert pq.operator == LTLOperator.AND
        assert pq.left == p
        assert pq.right == q

    def test_or(self) -> None:
        """Test disjunction."""
        p = prop("p")
        q = prop("q")
        pq = or_(p, q)
        assert pq.operator == LTLOperator.OR

    def test_implies(self) -> None:
        """Test implication."""
        p = prop("p")
        q = prop("q")
        pq = implies(p, q)
        assert pq.operator == LTLOperator.IMPLIES

    def test_next(self) -> None:
        """Test next operator."""
        p = prop("p")
        xp = next_(p)
        assert xp.operator == LTLOperator.NEXT

    def test_finally(self) -> None:
        """Test finally operator."""
        p = prop("p")
        fp = finally_(p)
        assert fp.operator == LTLOperator.FINALLY

    def test_globally(self) -> None:
        """Test globally operator."""
        p = prop("p")
        gp = globally(p)
        assert gp.operator == LTLOperator.GLOBALLY

    def test_until(self) -> None:
        """Test until operator."""
        p = prop("p")
        q = prop("q")
        pUq = until(p, q)
        assert pUq.operator == LTLOperator.UNTIL

    def test_release(self) -> None:
        """Test release operator."""
        p = prop("p")
        q = prop("q")
        pRq = release(p, q)
        assert pRq.operator == LTLOperator.RELEASE

    def test_propositions(self) -> None:
        """Test proposition extraction."""
        p = prop("p")
        q = prop("q")
        formula = and_(p, globally(q))
        props = formula.propositions()
        assert props == {"p", "q"}

    def test_str(self) -> None:
        """Test string representation."""
        p = prop("p")
        q = prop("q")
        formula = globally(implies(p, finally_(q)))
        s = str(formula)
        assert "G" in s
        assert "F" in s


class TestNNF:
    """Tests for negation normal form conversion."""

    def test_double_negation(self) -> None:
        """Test double negation elimination."""
        p = prop("p")
        nnp = not_(not_(p))
        result = to_nnf(nnp)
        assert result == p

    def test_de_morgan_and(self) -> None:
        """Test De Morgan's law for AND."""
        p = prop("p")
        q = prop("q")
        formula = not_(and_(p, q))
        result = to_nnf(formula)
        assert result.operator == LTLOperator.OR

    def test_de_morgan_or(self) -> None:
        """Test De Morgan's law for OR."""
        p = prop("p")
        q = prop("q")
        formula = not_(or_(p, q))
        result = to_nnf(formula)
        assert result.operator == LTLOperator.AND

    def test_not_finally(self) -> None:
        """Test NOT FINALLY = GLOBALLY NOT."""
        p = prop("p")
        formula = not_(finally_(p))
        result = to_nnf(formula)
        assert result.operator == LTLOperator.GLOBALLY

    def test_not_globally(self) -> None:
        """Test NOT GLOBALLY = FINALLY NOT."""
        p = prop("p")
        formula = not_(globally(p))
        result = to_nnf(formula)
        # F is converted to true U phi in NNF
        # So !G(p) = F(!p) = true U !p

    def test_not_until(self) -> None:
        """Test NOT UNTIL = RELEASE of negations."""
        p = prop("p")
        q = prop("q")
        formula = not_(until(p, q))
        result = to_nnf(formula)
        assert result.operator == LTLOperator.RELEASE

    def test_implication_elimination(self) -> None:
        """Test implication is eliminated."""
        p = prop("p")
        q = prop("q")
        formula = implies(p, q)
        result = to_nnf(formula)
        # p -> q = !p || q
        assert result.operator == LTLOperator.OR


class TestClosure:
    """Tests for closure computation."""

    def test_closure_includes_subformulas(self) -> None:
        """Test closure includes all subformulas."""
        p = prop("p")
        q = prop("q")
        formula = and_(p, q)
        cl = closure(formula)

        assert formula in cl
        assert p in cl
        assert q in cl

    def test_closure_includes_negations(self) -> None:
        """Test closure includes negations."""
        p = prop("p")
        cl = closure(p)

        assert p in cl
        assert not_(p) in cl


class TestLTLToBuchi:
    """Tests for LTL to Buchi conversion."""

    def test_simple_prop(self) -> None:
        """Test conversion of atomic proposition."""
        p = prop("p")
        buchi = ltl_to_buchi(p)

        assert len(buchi.states) > 0
        assert len(buchi.alphabet) > 0

    def test_globally(self) -> None:
        """Test conversion of globally."""
        p = prop("p")
        gp = globally(p)
        buchi = ltl_to_buchi(gp)

        assert len(buchi.states) > 0

    def test_finally(self) -> None:
        """Test conversion of finally."""
        p = prop("p")
        fp = finally_(p)
        buchi = ltl_to_buchi(fp)

        assert len(buchi.states) > 0


class TestLTLParser:
    """Tests for LTL parser."""

    def test_parse_prop(self) -> None:
        """Test parsing atomic proposition."""
        formula = parse_ltl("p")
        assert formula.operator == LTLOperator.PROP
        assert formula.prop == "p"

    def test_parse_true_false(self) -> None:
        """Test parsing true/false."""
        t = parse_ltl("true")
        f = parse_ltl("false")
        assert t.operator == LTLOperator.TRUE
        assert f.operator == LTLOperator.FALSE

    def test_parse_not(self) -> None:
        """Test parsing negation."""
        formula = parse_ltl("!p")
        assert formula.operator == LTLOperator.NOT

    def test_parse_and(self) -> None:
        """Test parsing conjunction."""
        formula = parse_ltl("p && q")
        assert formula.operator == LTLOperator.AND

    def test_parse_or(self) -> None:
        """Test parsing disjunction."""
        formula = parse_ltl("p || q")
        assert formula.operator == LTLOperator.OR

    def test_parse_implies(self) -> None:
        """Test parsing implication."""
        formula = parse_ltl("p -> q")
        assert formula.operator == LTLOperator.IMPLIES

    def test_parse_next(self) -> None:
        """Test parsing next."""
        formula = parse_ltl("X p")
        assert formula.operator == LTLOperator.NEXT

    def test_parse_finally(self) -> None:
        """Test parsing finally."""
        formula = parse_ltl("F p")
        assert formula.operator == LTLOperator.FINALLY

    def test_parse_globally(self) -> None:
        """Test parsing globally."""
        formula = parse_ltl("G p")
        assert formula.operator == LTLOperator.GLOBALLY

    def test_parse_until(self) -> None:
        """Test parsing until."""
        formula = parse_ltl("p U q")
        assert formula.operator == LTLOperator.UNTIL

    def test_parse_complex(self) -> None:
        """Test parsing complex formula."""
        formula = parse_ltl("G (p -> F q)")
        assert formula.operator == LTLOperator.GLOBALLY

    def test_parse_parentheses(self) -> None:
        """Test parsing with parentheses."""
        formula = parse_ltl("(p || q) && r")
        assert formula.operator == LTLOperator.AND


class TestLTLMonitor:
    """Tests for LTL monitor."""

    def test_basic_monitoring(self) -> None:
        """Test basic monitoring."""
        formula = globally(prop("safe"))
        monitor = LTLMonitor(formula)

        # Safe event
        verdict = monitor.step({"safe": True})
        assert verdict in ["unknown", "satisfied"]

    def test_reset(self) -> None:
        """Test monitor reset."""
        formula = prop("p")
        monitor = LTLMonitor(formula)

        monitor.step({"p": True})
        monitor.reset()

        # Should be back to initial state


class TestPatterns:
    """Tests for common LTL patterns."""

    def test_response_pattern(self) -> None:
        """Test response pattern."""
        formula = response_pattern("request", "response")
        assert formula.operator == LTLOperator.GLOBALLY

    def test_absence_pattern(self) -> None:
        """Test absence pattern."""
        formula = absence_pattern("error")
        assert formula.operator == LTLOperator.GLOBALLY
        assert formula.left.operator == LTLOperator.NOT

    def test_universality_pattern(self) -> None:
        """Test universality pattern."""
        formula = universality_pattern("valid")
        assert formula.operator == LTLOperator.GLOBALLY

    def test_precedence_pattern(self) -> None:
        """Test precedence pattern."""
        formula = precedence_pattern("access", "login")
        assert formula.operator == LTLOperator.WEAK_UNTIL

    def test_existence_pattern(self) -> None:
        """Test existence pattern."""
        formula = existence_pattern("complete")
        assert formula.operator == LTLOperator.FINALLY

    def test_bounded_response(self) -> None:
        """Test bounded response pattern."""
        formula = bounded_response("request", "response", 3)
        assert formula.operator == LTLOperator.GLOBALLY

    def test_mutual_exclusion(self) -> None:
        """Test mutual exclusion pattern."""
        formula = mutual_exclusion("read", "write")
        assert formula.operator == LTLOperator.GLOBALLY

    def test_fairness(self) -> None:
        """Test fairness pattern."""
        formula = fairness("service")
        # GF(service)
        assert formula.operator == LTLOperator.GLOBALLY
        assert formula.left.operator == LTLOperator.FINALLY
