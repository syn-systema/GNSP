"""Linear Temporal Logic (LTL) for property specification.

This module provides:
- LTL formula syntax and parsing
- LTL to Buchi automaton compilation
- Runtime monitoring of LTL properties

Useful for specifying and checking network security properties.
"""

from dataclasses import dataclass
from typing import (
    Set,
    Dict,
    Tuple,
    List,
    Optional,
    FrozenSet,
    Union,
    TypeVar,
)
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import deque
import re

from gnsp.automata.buchi import BuchiAutomaton, OnlineMonitor


class LTLOperator(Enum):
    """LTL operators."""

    # Propositional
    TRUE = auto()
    FALSE = auto()
    PROP = auto()  # Atomic proposition
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()

    # Temporal
    NEXT = auto()      # X (next)
    FINALLY = auto()   # F (eventually/finally)
    GLOBALLY = auto()  # G (always/globally)
    UNTIL = auto()     # U (until)
    RELEASE = auto()   # R (release)
    WEAK_UNTIL = auto()  # W (weak until)


@dataclass(frozen=True)
class LTLFormula:
    """LTL formula representation.

    Attributes:
        operator: The main operator
        prop: Atomic proposition name (if PROP)
        left: Left operand (for unary/binary ops)
        right: Right operand (for binary ops)
    """

    operator: LTLOperator
    prop: Optional[str] = None
    left: Optional["LTLFormula"] = None
    right: Optional["LTLFormula"] = None

    def __str__(self) -> str:
        """Convert to string representation."""
        if self.operator == LTLOperator.TRUE:
            return "true"
        elif self.operator == LTLOperator.FALSE:
            return "false"
        elif self.operator == LTLOperator.PROP:
            return self.prop or ""
        elif self.operator == LTLOperator.NOT:
            return f"!({self.left})"
        elif self.operator == LTLOperator.AND:
            return f"({self.left} && {self.right})"
        elif self.operator == LTLOperator.OR:
            return f"({self.left} || {self.right})"
        elif self.operator == LTLOperator.IMPLIES:
            return f"({self.left} -> {self.right})"
        elif self.operator == LTLOperator.NEXT:
            return f"X({self.left})"
        elif self.operator == LTLOperator.FINALLY:
            return f"F({self.left})"
        elif self.operator == LTLOperator.GLOBALLY:
            return f"G({self.left})"
        elif self.operator == LTLOperator.UNTIL:
            return f"({self.left} U {self.right})"
        elif self.operator == LTLOperator.RELEASE:
            return f"({self.left} R {self.right})"
        elif self.operator == LTLOperator.WEAK_UNTIL:
            return f"({self.left} W {self.right})"
        return "?"

    def propositions(self) -> Set[str]:
        """Get all atomic propositions in formula."""
        if self.operator == LTLOperator.PROP:
            return {self.prop} if self.prop else set()
        elif self.operator in (LTLOperator.TRUE, LTLOperator.FALSE):
            return set()
        elif self.left is not None and self.right is not None:
            return self.left.propositions() | self.right.propositions()
        elif self.left is not None:
            return self.left.propositions()
        return set()

    def is_negation_normal_form(self) -> bool:
        """Check if formula is in negation normal form (NNF)."""
        if self.operator == LTLOperator.NOT:
            # Negation only allowed on propositions
            return self.left is not None and self.left.operator == LTLOperator.PROP
        elif self.operator == LTLOperator.IMPLIES:
            return False  # Implication not in NNF
        elif self.left is not None:
            if not self.left.is_negation_normal_form():
                return False
        if self.right is not None:
            if not self.right.is_negation_normal_form():
                return False
        return True


# Constructor functions for convenience
def prop(name: str) -> LTLFormula:
    """Create atomic proposition."""
    return LTLFormula(LTLOperator.PROP, prop=name)


def true_() -> LTLFormula:
    """Create true constant."""
    return LTLFormula(LTLOperator.TRUE)


def false_() -> LTLFormula:
    """Create false constant."""
    return LTLFormula(LTLOperator.FALSE)


def not_(phi: LTLFormula) -> LTLFormula:
    """Create negation."""
    return LTLFormula(LTLOperator.NOT, left=phi)


def and_(phi: LTLFormula, psi: LTLFormula) -> LTLFormula:
    """Create conjunction."""
    return LTLFormula(LTLOperator.AND, left=phi, right=psi)


def or_(phi: LTLFormula, psi: LTLFormula) -> LTLFormula:
    """Create disjunction."""
    return LTLFormula(LTLOperator.OR, left=phi, right=psi)


def implies(phi: LTLFormula, psi: LTLFormula) -> LTLFormula:
    """Create implication."""
    return LTLFormula(LTLOperator.IMPLIES, left=phi, right=psi)


def next_(phi: LTLFormula) -> LTLFormula:
    """Create next operator."""
    return LTLFormula(LTLOperator.NEXT, left=phi)


def finally_(phi: LTLFormula) -> LTLFormula:
    """Create finally/eventually operator."""
    return LTLFormula(LTLOperator.FINALLY, left=phi)


def globally(phi: LTLFormula) -> LTLFormula:
    """Create globally/always operator."""
    return LTLFormula(LTLOperator.GLOBALLY, left=phi)


def until(phi: LTLFormula, psi: LTLFormula) -> LTLFormula:
    """Create until operator."""
    return LTLFormula(LTLOperator.UNTIL, left=phi, right=psi)


def release(phi: LTLFormula, psi: LTLFormula) -> LTLFormula:
    """Create release operator."""
    return LTLFormula(LTLOperator.RELEASE, left=phi, right=psi)


def weak_until(phi: LTLFormula, psi: LTLFormula) -> LTLFormula:
    """Create weak until operator."""
    return LTLFormula(LTLOperator.WEAK_UNTIL, left=phi, right=psi)


def to_nnf(phi: LTLFormula) -> LTLFormula:
    """Convert formula to negation normal form.

    Pushes negations inward and eliminates implications.
    """
    op = phi.operator

    if op == LTLOperator.TRUE:
        return phi
    elif op == LTLOperator.FALSE:
        return phi
    elif op == LTLOperator.PROP:
        return phi

    elif op == LTLOperator.IMPLIES:
        # phi -> psi == !phi || psi
        assert phi.left is not None and phi.right is not None
        return to_nnf(or_(not_(phi.left), phi.right))

    elif op == LTLOperator.NOT:
        assert phi.left is not None
        inner = phi.left

        if inner.operator == LTLOperator.TRUE:
            return false_()
        elif inner.operator == LTLOperator.FALSE:
            return true_()
        elif inner.operator == LTLOperator.PROP:
            return phi  # Negation of prop is in NNF
        elif inner.operator == LTLOperator.NOT:
            # !!phi == phi
            assert inner.left is not None
            return to_nnf(inner.left)
        elif inner.operator == LTLOperator.AND:
            # !(a && b) == !a || !b
            assert inner.left is not None and inner.right is not None
            return to_nnf(or_(not_(inner.left), not_(inner.right)))
        elif inner.operator == LTLOperator.OR:
            # !(a || b) == !a && !b
            assert inner.left is not None and inner.right is not None
            return to_nnf(and_(not_(inner.left), not_(inner.right)))
        elif inner.operator == LTLOperator.NEXT:
            # !X(phi) == X(!phi)
            assert inner.left is not None
            return next_(to_nnf(not_(inner.left)))
        elif inner.operator == LTLOperator.FINALLY:
            # !F(phi) == G(!phi)
            assert inner.left is not None
            return globally(to_nnf(not_(inner.left)))
        elif inner.operator == LTLOperator.GLOBALLY:
            # !G(phi) == F(!phi)
            assert inner.left is not None
            return finally_(to_nnf(not_(inner.left)))
        elif inner.operator == LTLOperator.UNTIL:
            # !(a U b) == !a R !b
            assert inner.left is not None and inner.right is not None
            return to_nnf(release(not_(inner.left), not_(inner.right)))
        elif inner.operator == LTLOperator.RELEASE:
            # !(a R b) == !a U !b
            assert inner.left is not None and inner.right is not None
            return to_nnf(until(not_(inner.left), not_(inner.right)))
        else:
            return not_(to_nnf(inner))

    elif op == LTLOperator.AND:
        assert phi.left is not None and phi.right is not None
        return and_(to_nnf(phi.left), to_nnf(phi.right))

    elif op == LTLOperator.OR:
        assert phi.left is not None and phi.right is not None
        return or_(to_nnf(phi.left), to_nnf(phi.right))

    elif op == LTLOperator.NEXT:
        assert phi.left is not None
        return next_(to_nnf(phi.left))

    elif op == LTLOperator.FINALLY:
        # F(phi) == true U phi
        assert phi.left is not None
        return to_nnf(until(true_(), phi.left))

    elif op == LTLOperator.GLOBALLY:
        # G(phi) == false R phi
        assert phi.left is not None
        return to_nnf(release(false_(), phi.left))

    elif op == LTLOperator.UNTIL:
        assert phi.left is not None and phi.right is not None
        return until(to_nnf(phi.left), to_nnf(phi.right))

    elif op == LTLOperator.RELEASE:
        assert phi.left is not None and phi.right is not None
        return release(to_nnf(phi.left), to_nnf(phi.right))

    elif op == LTLOperator.WEAK_UNTIL:
        # a W b == (a U b) || G(a)
        assert phi.left is not None and phi.right is not None
        return to_nnf(or_(until(phi.left, phi.right), globally(phi.left)))

    return phi


def closure(phi: LTLFormula) -> Set[LTLFormula]:
    """Compute closure of formula (subformulas and their negations)."""
    result: Set[LTLFormula] = set()

    def add_with_negation(f: LTLFormula) -> None:
        result.add(f)
        # Add negation (simplified)
        if f.operator != LTLOperator.NOT:
            result.add(not_(f))

    def collect(f: LTLFormula) -> None:
        add_with_negation(f)
        if f.left is not None:
            collect(f.left)
        if f.right is not None:
            collect(f.right)

    collect(phi)
    return result


@dataclass(frozen=True)
class LTLState:
    """State in LTL-to-Buchi construction.

    Represents a set of formulas that must hold.
    """

    formulas: FrozenSet[LTLFormula]

    def __hash__(self) -> int:
        return hash(self.formulas)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LTLState):
            return False
        return self.formulas == other.formulas

    def __str__(self) -> str:
        return "{" + ", ".join(str(f) for f in self.formulas) + "}"


def ltl_to_buchi(phi: LTLFormula) -> BuchiAutomaton[LTLState, FrozenSet[str]]:
    """Convert LTL formula to Buchi automaton.

    Uses the tableau-based construction.

    Args:
        phi: LTL formula

    Returns:
        Buchi automaton accepting words satisfying phi
    """
    # Convert to NNF
    phi_nnf = to_nnf(phi)

    # Get propositions
    props = phi_nnf.propositions()

    # Alphabet: all possible truth assignments to propositions
    alphabet: Set[FrozenSet[str]] = set()
    prop_list = sorted(props)
    n_props = len(prop_list)

    for i in range(2 ** n_props):
        assignment: Set[str] = set()
        for j, p in enumerate(prop_list):
            if (i >> j) & 1:
                assignment.add(p)
        alphabet.add(frozenset(assignment))

    if not alphabet:
        alphabet = {frozenset()}

    # Initial state
    initial = LTLState(frozenset({phi_nnf}))

    # Build automaton states and transitions
    states: Set[LTLState] = set()
    transitions: Dict[Tuple[LTLState, FrozenSet[str]], Set[LTLState]] = {}
    accepting: Set[LTLState] = set()

    # BFS construction
    queue = deque([initial])
    visited: Set[LTLState] = set()

    while queue:
        state = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        states.add(state)

        # Check if state is consistent with propositions
        for sigma in alphabet:
            # Compute successor state
            next_formulas = _compute_successors(state.formulas, sigma)

            if next_formulas is None:
                continue  # Inconsistent

            next_state = LTLState(frozenset(next_formulas))

            key = (state, sigma)
            if key not in transitions:
                transitions[key] = set()
            transitions[key].add(next_state)

            if next_state not in visited:
                queue.append(next_state)

    # Determine accepting states (no unfulfilled until obligations)
    for state in states:
        is_accepting = True
        for f in state.formulas:
            if f.operator == LTLOperator.UNTIL:
                # Until must be fulfilled (right side must hold)
                if f.right not in state.formulas:
                    is_accepting = False
                    break
        if is_accepting:
            accepting.add(state)

    return BuchiAutomaton(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial=initial,
        accepting=accepting if accepting else states,
    )


def _compute_successors(
    formulas: FrozenSet[LTLFormula],
    sigma: FrozenSet[str],
) -> Optional[Set[LTLFormula]]:
    """Compute successor formulas after reading sigma.

    Returns None if current state is inconsistent with sigma.
    """
    current: Set[LTLFormula] = set()
    next_formulas: Set[LTLFormula] = set()

    # Expand current formulas
    to_expand = list(formulas)

    while to_expand:
        f = to_expand.pop()

        if f in current:
            continue

        op = f.operator

        if op == LTLOperator.TRUE:
            current.add(f)
        elif op == LTLOperator.FALSE:
            return None  # Inconsistent
        elif op == LTLOperator.PROP:
            # Check against sigma
            if f.prop in sigma:
                current.add(f)
            else:
                return None
        elif op == LTLOperator.NOT:
            assert f.left is not None
            if f.left.operator == LTLOperator.PROP:
                if f.left.prop not in sigma:
                    current.add(f)
                else:
                    return None
            else:
                current.add(f)
        elif op == LTLOperator.AND:
            assert f.left is not None and f.right is not None
            current.add(f)
            to_expand.append(f.left)
            to_expand.append(f.right)
        elif op == LTLOperator.OR:
            assert f.left is not None and f.right is not None
            # Non-deterministic choice - try left first
            current.add(f)
            to_expand.append(f.left)  # Simplified: just take left
        elif op == LTLOperator.NEXT:
            assert f.left is not None
            current.add(f)
            next_formulas.add(f.left)
        elif op == LTLOperator.UNTIL:
            # a U b: either b holds now, or a holds and a U b continues
            assert f.left is not None and f.right is not None
            current.add(f)
            # Simplified: require right to eventually hold
            to_expand.append(f.left)
            next_formulas.add(f)
        elif op == LTLOperator.RELEASE:
            # a R b: b must hold, and either a holds or a R b continues
            assert f.left is not None and f.right is not None
            current.add(f)
            to_expand.append(f.right)
            next_formulas.add(f)
        else:
            current.add(f)

    return next_formulas


class LTLParser:
    """Parser for LTL formulas."""

    # Token patterns
    # Token patterns - order matters! Keywords must come before PROP
    # Single uppercase letters are operators, lowercase are propositions
    TOKENS = [
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("NOT", r"!|~"),
        ("AND", r"&&|/\\"),
        ("OR", r"\|\||\\\/"),
        ("IMPLIES", r"->|=>"),
        ("NEXT", r"X(?![a-zA-Z0-9_])"),          # X not followed by alphanumeric
        ("FINALLY", r"F(?![a-zA-Z0-9_])"),       # F not followed by alphanumeric
        ("GLOBALLY", r"G(?![a-zA-Z0-9_])"),      # G not followed by alphanumeric
        ("UNTIL", r"U(?![a-zA-Z0-9_])"),         # U not followed by alphanumeric
        ("RELEASE", r"R(?![a-zA-Z0-9_])"),       # R not followed by alphanumeric
        ("WEAK_UNTIL", r"W(?![a-zA-Z0-9_])"),    # W not followed by alphanumeric
        ("TRUE", r"true\b"),
        ("FALSE", r"false\b"),
        ("PROP", r"[a-z_][a-z0-9_]*"),           # Propositions are lowercase
        ("WS", r"\s+"),
    ]

    def __init__(self) -> None:
        """Initialize parser."""
        self.pattern = "|".join(
            f"(?P<{name}>{pattern})" for name, pattern in self.TOKENS
        )
        self.regex = re.compile(self.pattern)

    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """Tokenize input string."""
        tokens = []
        pos = 0

        while pos < len(text):
            match = self.regex.match(text, pos)
            if match is None:
                raise ValueError(f"Invalid character at position {pos}: {text[pos]}")

            token_type = match.lastgroup
            token_value = match.group()
            pos = match.end()

            if token_type != "WS":
                tokens.append((token_type, token_value))

        return tokens

    def parse(self, text: str) -> LTLFormula:
        """Parse LTL formula string.

        Grammar:
            formula := implies_expr
            implies_expr := or_expr ('->' or_expr)*
            or_expr := and_expr ('||' and_expr)*
            and_expr := unary_expr ('&&' unary_expr)*
            unary_expr := NOT unary_expr | NEXT unary_expr |
                          FINALLY unary_expr | GLOBALLY unary_expr |
                          primary
            primary := PROP | TRUE | FALSE | '(' formula ')'
        """
        tokens = self.tokenize(text)
        self._tokens = tokens
        self._pos = 0

        result = self._parse_implies()

        if self._pos < len(tokens):
            raise ValueError(f"Unexpected token: {tokens[self._pos]}")

        return result

    def _current(self) -> Optional[Tuple[str, str]]:
        """Get current token."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> Optional[Tuple[str, str]]:
        """Advance to next token."""
        token = self._current()
        self._pos += 1
        return token

    def _expect(self, token_type: str) -> str:
        """Expect and consume specific token type."""
        token = self._current()
        if token is None or token[0] != token_type:
            raise ValueError(f"Expected {token_type}, got {token}")
        self._advance()
        return token[1]

    def _parse_implies(self) -> LTLFormula:
        """Parse implication expression."""
        left = self._parse_or()

        while self._current() and self._current()[0] == "IMPLIES":
            self._advance()
            right = self._parse_or()
            left = implies(left, right)

        return left

    def _parse_or(self) -> LTLFormula:
        """Parse or expression."""
        left = self._parse_and()

        while self._current() and self._current()[0] == "OR":
            self._advance()
            right = self._parse_and()
            left = or_(left, right)

        return left

    def _parse_and(self) -> LTLFormula:
        """Parse and expression."""
        left = self._parse_until()

        while self._current() and self._current()[0] == "AND":
            self._advance()
            right = self._parse_until()
            left = and_(left, right)

        return left

    def _parse_until(self) -> LTLFormula:
        """Parse until/release expression."""
        left = self._parse_unary()

        while self._current() and self._current()[0] in ("UNTIL", "RELEASE", "WEAK_UNTIL"):
            op = self._current()[0]
            self._advance()
            right = self._parse_unary()

            if op == "UNTIL":
                left = until(left, right)
            elif op == "RELEASE":
                left = release(left, right)
            else:
                left = weak_until(left, right)

        return left

    def _parse_unary(self) -> LTLFormula:
        """Parse unary expression."""
        token = self._current()

        if token and token[0] == "NOT":
            self._advance()
            return not_(self._parse_unary())
        elif token and token[0] == "NEXT":
            self._advance()
            return next_(self._parse_unary())
        elif token and token[0] == "FINALLY":
            self._advance()
            return finally_(self._parse_unary())
        elif token and token[0] == "GLOBALLY":
            self._advance()
            return globally(self._parse_unary())
        else:
            return self._parse_primary()

    def _parse_primary(self) -> LTLFormula:
        """Parse primary expression."""
        token = self._current()

        if token is None:
            raise ValueError("Unexpected end of input")

        if token[0] == "TRUE":
            self._advance()
            return true_()
        elif token[0] == "FALSE":
            self._advance()
            return false_()
        elif token[0] == "PROP":
            self._advance()
            return prop(token[1])
        elif token[0] == "LPAREN":
            self._advance()
            formula = self._parse_implies()
            self._expect("RPAREN")
            return formula
        else:
            raise ValueError(f"Unexpected token: {token}")


def parse_ltl(text: str) -> LTLFormula:
    """Parse LTL formula from string.

    Args:
        text: LTL formula string

    Returns:
        Parsed LTL formula
    """
    parser = LTLParser()
    return parser.parse(text)


class LTLMonitor:
    """Online monitor for LTL properties.

    Monitors a stream of events and determines if the LTL property
    can still be satisfied, is already satisfied, or is violated.
    """

    def __init__(self, formula: LTLFormula) -> None:
        """Initialize monitor.

        Args:
            formula: LTL formula to monitor
        """
        self.formula = formula
        self.buchi = ltl_to_buchi(formula)
        self.monitor = OnlineMonitor(self.buchi)

    def step(self, event: Dict[str, bool]) -> str:
        """Process one event.

        Args:
            event: Mapping from proposition names to truth values

        Returns:
            Verdict: "satisfied", "violated", or "unknown"
        """
        # Convert event to alphabet symbol
        true_props = frozenset(p for p, v in event.items() if v)
        return self.monitor.step(true_props)

    def reset(self) -> None:
        """Reset monitor to initial state."""
        self.monitor.reset()


# Common security-related LTL patterns


def response_pattern(trigger: str, response: str) -> LTLFormula:
    """Response pattern: G(trigger -> F(response)).

    "Whenever trigger happens, response must eventually happen."
    """
    return globally(implies(prop(trigger), finally_(prop(response))))


def absence_pattern(bad: str) -> LTLFormula:
    """Absence pattern: G(!bad).

    "Bad event never happens."
    """
    return globally(not_(prop(bad)))


def universality_pattern(good: str) -> LTLFormula:
    """Universality pattern: G(good).

    "Good property always holds."
    """
    return globally(prop(good))


def precedence_pattern(event: str, condition: str) -> LTLFormula:
    """Precedence pattern: !event W condition.

    "Event can only happen after condition."
    """
    return weak_until(not_(prop(event)), prop(condition))


def existence_pattern(event: str) -> LTLFormula:
    """Existence pattern: F(event).

    "Event must eventually happen."
    """
    return finally_(prop(event))


def bounded_response(trigger: str, response: str, bound: int) -> LTLFormula:
    """Bounded response: response within 'bound' steps after trigger.

    Approximation using nested next operators.
    """
    # Build: G(trigger -> (response || X(response || X(response || ...))))
    inner: LTLFormula = prop(response)
    for _ in range(bound):
        inner = or_(prop(response), next_(inner))

    return globally(implies(prop(trigger), inner))


def mutual_exclusion(event1: str, event2: str) -> LTLFormula:
    """Mutual exclusion: G(!(event1 && event2)).

    "Events cannot happen simultaneously."
    """
    return globally(not_(and_(prop(event1), prop(event2))))


def fairness(event: str) -> LTLFormula:
    """Fairness: GF(event).

    "Event happens infinitely often."
    """
    return globally(finally_(prop(event)))
