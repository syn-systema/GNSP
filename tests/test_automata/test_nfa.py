"""Tests for NFA module."""

import pytest
from gnsp.automata.nfa import (
    NFA,
    subset_construction,
    nfa_union,
    nfa_concatenation,
    nfa_kleene_star,
    nfa_from_symbol,
    nfa_from_epsilon,
    thompson_construction,
    reverse_nfa,
)


class TestNFA:
    """Tests for NFA class."""

    def test_basic_construction(self) -> None:
        """Test basic NFA construction."""
        states = {"q0", "q1"}
        alphabet = {"a", "b"}
        transitions = {
            ("q0", "a"): {"q0", "q1"},
            ("q0", "b"): {"q0"},
        }

        nfa = NFA(
            states=states,
            alphabet=alphabet,
            transitions=transitions,
            initial="q0",
            accepting={"q1"},
        )

        assert nfa.initial == "q0"
        assert nfa.accepting == {"q1"}

    def test_transition(self) -> None:
        """Test NFA transition function."""
        nfa = NFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q0", "q1"}},
            initial="q0",
            accepting={"q1"},
        )

        result = nfa.transition("q0", "a")
        assert result == {"q0", "q1"}

        # Non-existent transition returns empty set
        assert nfa.transition("q1", "a") == set()

    def test_epsilon_closure(self) -> None:
        """Test epsilon closure computation."""
        nfa = NFA(
            states={"q0", "q1", "q2"},
            alphabet={"a"},
            transitions={
                ("q0", None): {"q1"},
                ("q1", None): {"q2"},
            },
            initial="q0",
            accepting={"q2"},
        )

        closure = nfa.epsilon_closure({"q0"})
        assert closure == {"q0", "q1", "q2"}

    def test_accepts(self) -> None:
        """Test NFA acceptance."""
        # NFA with epsilon transitions
        nfa = NFA(
            states={"q0", "q1", "q2"},
            alphabet={"a"},
            transitions={
                ("q0", "a"): {"q1"},
                ("q1", None): {"q2"},
            },
            initial="q0",
            accepting={"q2"},
        )

        assert nfa.accepts(["a"])
        assert not nfa.accepts([])
        assert not nfa.accepts(["a", "a"])

    def test_nondeterministic_accepts(self) -> None:
        """Test nondeterministic acceptance."""
        # NFA that nondeterministically accepts 'a' or 'aa'
        nfa = NFA(
            states={"q0", "q1", "q2"},
            alphabet={"a"},
            transitions={
                ("q0", "a"): {"q1", "q2"},
                ("q2", "a"): {"q1"},
            },
            initial="q0",
            accepting={"q1"},
        )

        assert nfa.accepts(["a"])
        assert nfa.accepts(["a", "a"])
        assert not nfa.accepts([])

    def test_run(self) -> None:
        """Test NFA run function."""
        nfa = NFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q1"}},
            initial="q0",
            accepting={"q1"},
        )

        path = nfa.run(["a"])
        assert len(path) == 2
        assert path[0] == {"q0"}
        assert path[1] == {"q1"}


class TestSubsetConstruction:
    """Tests for NFA to DFA conversion."""

    def test_basic_subset_construction(self) -> None:
        """Test basic subset construction."""
        nfa = NFA(
            states={"q0", "q1"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): {"q0", "q1"},
                ("q0", "b"): {"q0"},
                ("q1", "b"): {"q1"},
            },
            initial="q0",
            accepting={"q1"},
        )

        dfa = subset_construction(nfa)

        # DFA should accept same language
        assert dfa.accepts(["a"])
        assert dfa.accepts(["a", "b"])
        assert not dfa.accepts(["b"])

    def test_epsilon_nfa_conversion(self) -> None:
        """Test subset construction with epsilon transitions."""
        nfa = NFA(
            states={"q0", "q1", "q2"},
            alphabet={"a"},
            transitions={
                ("q0", None): {"q1"},
                ("q1", "a"): {"q2"},
            },
            initial="q0",
            accepting={"q2"},
        )

        dfa = subset_construction(nfa)

        assert dfa.accepts(["a"])
        assert not dfa.accepts([])


class TestNFAOperations:
    """Tests for NFA operations."""

    def test_nfa_union(self) -> None:
        """Test NFA union."""
        nfa1 = nfa_from_symbol("a", {"a", "b"}, "p")
        nfa2 = nfa_from_symbol("b", {"a", "b"}, "q")

        union_nfa = nfa_union(nfa1, nfa2, "start")

        assert union_nfa.accepts(["a"])
        assert union_nfa.accepts(["b"])
        assert not union_nfa.accepts(["a", "b"])

    def test_nfa_concatenation(self) -> None:
        """Test NFA concatenation."""
        nfa1 = nfa_from_symbol("a", {"a", "b"}, "p")
        nfa2 = nfa_from_symbol("b", {"a", "b"}, "q")

        concat = nfa_concatenation(nfa1, nfa2)

        assert concat.accepts(["a", "b"])
        assert not concat.accepts(["a"])
        assert not concat.accepts(["b"])

    def test_nfa_kleene_star(self) -> None:
        """Test NFA Kleene star."""
        nfa = nfa_from_symbol("a", {"a", "b"}, "q")
        star = nfa_kleene_star(nfa, "start")

        # Kleene star accepts empty word
        assert star.accepts([])
        assert star.accepts(["a"])
        assert star.accepts(["a", "a"])
        assert star.accepts(["a", "a", "a"])
        assert not star.accepts(["b"])

    def test_nfa_from_epsilon(self) -> None:
        """Test epsilon NFA creation."""
        nfa = nfa_from_epsilon({"a", "b"}, "q0")

        assert nfa.accepts([])
        assert not nfa.accepts(["a"])


class TestThompsonConstruction:
    """Tests for Thompson construction."""

    def test_single_symbol(self) -> None:
        """Test single symbol regex."""
        nfa = thompson_construction("a", {"a", "b"})
        assert nfa.accepts(["a"])
        assert not nfa.accepts(["b"])

    def test_concatenation(self) -> None:
        """Test concatenation in regex."""
        nfa = thompson_construction("ab", {"a", "b"})
        assert nfa.accepts(["a", "b"])
        assert not nfa.accepts(["a"])

    def test_union(self) -> None:
        """Test union in regex."""
        nfa = thompson_construction("a|b", {"a", "b"})
        assert nfa.accepts(["a"])
        assert nfa.accepts(["b"])
        assert not nfa.accepts(["a", "b"])

    def test_kleene_star(self) -> None:
        """Test Kleene star in regex."""
        nfa = thompson_construction("a*", {"a", "b"})
        assert nfa.accepts([])
        assert nfa.accepts(["a"])
        assert nfa.accepts(["a", "a"])
        assert not nfa.accepts(["b"])

    def test_complex_regex(self) -> None:
        """Test complex regex."""
        nfa = thompson_construction("(a|b)*", {"a", "b"})
        assert nfa.accepts([])
        assert nfa.accepts(["a"])
        assert nfa.accepts(["b"])
        assert nfa.accepts(["a", "b"])
        assert nfa.accepts(["b", "a", "b"])

    def test_parentheses(self) -> None:
        """Test parentheses in regex."""
        nfa = thompson_construction("(ab)*", {"a", "b"})
        assert nfa.accepts([])
        assert nfa.accepts(["a", "b"])
        assert nfa.accepts(["a", "b", "a", "b"])
        assert not nfa.accepts(["a"])


class TestReverseNFA:
    """Tests for NFA reversal."""

    def test_reverse_simple(self) -> None:
        """Test simple NFA reversal."""
        nfa = NFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q1"}},
            initial="q0",
            accepting={"q1"},
        )

        reversed_nfa = reverse_nfa(nfa)

        # Original accepts "a", reversed should accept "a" too (single char)
        assert reversed_nfa.accepts(["a"])

    def test_reverse_two_symbols(self) -> None:
        """Test NFA reversal with two symbols."""
        # NFA accepting "ab"
        nfa = NFA(
            states={"q0", "q1", "q2"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): {"q1"},
                ("q1", "b"): {"q2"},
            },
            initial="q0",
            accepting={"q2"},
        )

        reversed_nfa = reverse_nfa(nfa)

        # Reversed should accept "ba"
        assert reversed_nfa.accepts(["b", "a"])
        assert not reversed_nfa.accepts(["a", "b"])
