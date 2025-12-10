"""Tests for DFA module."""

import pytest
from gnsp.automata.dfa import (
    DFA,
    DFAState,
    product_dfa,
    intersection,
    union,
    symmetric_difference,
    minimize,
    are_equivalent,
    from_regex_simple,
    create_tcp_state_machine,
)


class TestDFA:
    """Tests for DFA class."""

    def test_basic_construction(self) -> None:
        """Test basic DFA construction."""
        states = {"q0", "q1"}
        alphabet = {"a", "b"}
        transitions = {
            ("q0", "a"): "q1",
            ("q0", "b"): "q0",
            ("q1", "a"): "q1",
            ("q1", "b"): "q0",
        }

        dfa = DFA(
            states=states,
            alphabet=alphabet,
            transitions=transitions,
            initial="q0",
            accepting={"q1"},
        )

        assert dfa.initial == "q0"
        assert dfa.accepting == {"q1"}
        assert len(dfa.states) == 2

    def test_invalid_initial_state(self) -> None:
        """Test that invalid initial state raises error."""
        with pytest.raises(ValueError, match="Initial state must be in states"):
            DFA(
                states={"q0"},
                alphabet={"a"},
                transitions={},
                initial="q_invalid",
                accepting=set(),
            )

    def test_invalid_accepting_states(self) -> None:
        """Test that invalid accepting states raise error."""
        with pytest.raises(ValueError, match="Accepting states must be subset"):
            DFA(
                states={"q0"},
                alphabet={"a"},
                transitions={},
                initial="q0",
                accepting={"q_invalid"},
            )

    def test_transition(self) -> None:
        """Test transition function."""
        dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): "q1"},
            initial="q0",
            accepting={"q1"},
        )

        assert dfa.transition("q0", "a") == "q1"
        assert dfa.transition("q1", "a") is None

    def test_accepts_empty_word(self) -> None:
        """Test acceptance of empty word."""
        # DFA where initial state is accepting
        dfa = DFA(
            states={"q0"},
            alphabet={"a"},
            transitions={("q0", "a"): "q0"},
            initial="q0",
            accepting={"q0"},
        )

        assert dfa.accepts([])
        assert dfa.accepts_empty()

    def test_accepts_word(self) -> None:
        """Test word acceptance."""
        # DFA accepting words ending in 'a'
        dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): "q1",
                ("q0", "b"): "q0",
                ("q1", "a"): "q1",
                ("q1", "b"): "q0",
            },
            initial="q0",
            accepting={"q1"},
        )

        assert dfa.accepts(["a"])
        assert dfa.accepts(["b", "a"])
        assert dfa.accepts(["a", "a"])
        assert not dfa.accepts(["b"])
        assert not dfa.accepts(["a", "b"])

    def test_run(self) -> None:
        """Test run function returns state path."""
        dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): "q1"},
            initial="q0",
            accepting={"q1"},
        )

        path = dfa.run(["a"])
        assert path == ["q0", "q1"]

    def test_is_complete(self) -> None:
        """Test completeness check."""
        # Complete DFA
        complete_dfa = DFA(
            states={"q0"},
            alphabet={"a"},
            transitions={("q0", "a"): "q0"},
            initial="q0",
            accepting=set(),
        )
        assert complete_dfa.is_complete()

        # Incomplete DFA
        incomplete_dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a", "b"},
            transitions={("q0", "a"): "q1"},
            initial="q0",
            accepting=set(),
        )
        assert not incomplete_dfa.is_complete()

    def test_complete(self) -> None:
        """Test completion with sink state."""
        dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a", "b"},
            transitions={("q0", "a"): "q1"},
            initial="q0",
            accepting={"q1"},
        )

        completed = dfa.complete("sink")
        assert completed.is_complete()
        assert "sink" in completed.states

    def test_complement(self) -> None:
        """Test DFA complement."""
        # DFA accepting 'a'
        dfa = DFA(
            states={"q0", "q1", "sink"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): "q1",
                ("q0", "b"): "sink",
                ("q1", "a"): "sink",
                ("q1", "b"): "sink",
                ("sink", "a"): "sink",
                ("sink", "b"): "sink",
            },
            initial="q0",
            accepting={"q1"},
        )

        complement_dfa = dfa.complement()

        # Original accepts 'a'
        assert dfa.accepts(["a"])
        # Complement rejects 'a'
        assert not complement_dfa.accepts(["a"])
        # Complement accepts 'b'
        assert complement_dfa.accepts(["b"])

    def test_reachable_states(self) -> None:
        """Test reachable states computation."""
        # DFA with unreachable state
        dfa = DFA(
            states={"q0", "q1", "unreachable"},
            alphabet={"a"},
            transitions={
                ("q0", "a"): "q1",
                ("q1", "a"): "q0",
                ("unreachable", "a"): "unreachable",
            },
            initial="q0",
            accepting={"q1"},
        )

        reachable = dfa.reachable_states()
        assert "q0" in reachable
        assert "q1" in reachable
        assert "unreachable" not in reachable

    def test_trim(self) -> None:
        """Test DFA trimming."""
        dfa = DFA(
            states={"q0", "q1", "unreachable"},
            alphabet={"a"},
            transitions={
                ("q0", "a"): "q1",
                ("unreachable", "a"): "unreachable",
            },
            initial="q0",
            accepting={"q1"},
        )

        trimmed = dfa.trim()
        assert "unreachable" not in trimmed.states
        assert len(trimmed.states) == 2

    def test_is_empty(self) -> None:
        """Test empty language check."""
        # Non-empty DFA
        dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): "q1"},
            initial="q0",
            accepting={"q1"},
        )
        assert not dfa.is_empty()

        # Empty DFA (no reachable accepting state)
        empty_dfa = DFA(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): "q0"},
            initial="q0",
            accepting={"q1"},
        )
        assert empty_dfa.is_empty()


class TestDFAOperations:
    """Tests for DFA operations."""

    def test_intersection(self) -> None:
        """Test DFA intersection."""
        # DFA accepting words with at least one 'a'
        dfa1 = DFA(
            states={0, 1},
            alphabet={"a", "b"},
            transitions={
                (0, "a"): 1,
                (0, "b"): 0,
                (1, "a"): 1,
                (1, "b"): 1,
            },
            initial=0,
            accepting={1},
        )

        # DFA accepting words with at least one 'b'
        dfa2 = DFA(
            states={0, 1},
            alphabet={"a", "b"},
            transitions={
                (0, "a"): 0,
                (0, "b"): 1,
                (1, "a"): 1,
                (1, "b"): 1,
            },
            initial=0,
            accepting={1},
        )

        inter = intersection(dfa1, dfa2)

        # Intersection should accept words with both 'a' and 'b'
        assert inter.accepts(["a", "b"])
        assert inter.accepts(["b", "a"])
        assert not inter.accepts(["a"])
        assert not inter.accepts(["b"])

    def test_union(self) -> None:
        """Test DFA union."""
        # DFA accepting "a"
        dfa1 = DFA(
            states={0, 1, 2},
            alphabet={"a", "b"},
            transitions={
                (0, "a"): 1,
                (0, "b"): 2,
                (1, "a"): 2,
                (1, "b"): 2,
                (2, "a"): 2,
                (2, "b"): 2,
            },
            initial=0,
            accepting={1},
        )

        # DFA accepting "b"
        dfa2 = DFA(
            states={0, 1, 2},
            alphabet={"a", "b"},
            transitions={
                (0, "a"): 2,
                (0, "b"): 1,
                (1, "a"): 2,
                (1, "b"): 2,
                (2, "a"): 2,
                (2, "b"): 2,
            },
            initial=0,
            accepting={1},
        )

        union_dfa = union(dfa1, dfa2)

        assert union_dfa.accepts(["a"])
        assert union_dfa.accepts(["b"])
        assert not union_dfa.accepts(["a", "b"])

    def test_symmetric_difference(self) -> None:
        """Test DFA symmetric difference."""
        dfa1 = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        sym_diff = symmetric_difference(dfa1, dfa1)

        # Symmetric difference with self is empty
        assert sym_diff.is_empty()

    def test_minimize(self) -> None:
        """Test DFA minimization."""
        # DFA with redundant states
        dfa = DFA(
            states={0, 1, 2},
            alphabet={"a"},
            transitions={
                (0, "a"): 1,
                (1, "a"): 2,
                (2, "a"): 2,
            },
            initial=0,
            accepting={1, 2},
        )

        minimized = minimize(dfa)

        # Should still accept same language
        assert minimized.accepts(["a"])
        assert minimized.accepts(["a", "a"])
        assert not minimized.accepts([])

    def test_are_equivalent(self) -> None:
        """Test DFA equivalence checking."""
        dfa1 = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        # Same language, different structure
        dfa2 = DFA(
            states={0, 1, 2},
            alphabet={"a"},
            transitions={
                (0, "a"): 1,
                (1, "a"): 2,
                (2, "a"): 2,
            },
            initial=0,
            accepting={1, 2},
        )

        assert are_equivalent(dfa1, dfa2)


class TestDFAFactories:
    """Tests for DFA factory functions."""

    def test_from_regex_simple(self) -> None:
        """Test simple pattern DFA."""
        dfa = from_regex_simple("ab", {"a", "b"})

        assert dfa.accepts(["a", "b"])
        assert not dfa.accepts(["a"])
        assert not dfa.accepts(["b", "a"])

    def test_tcp_state_machine(self) -> None:
        """Test TCP state machine."""
        tcp = create_tcp_state_machine()

        assert tcp.initial == "CLOSED"
        assert "ESTABLISHED" in tcp.accepting

        # Valid connection sequence
        path = tcp.run([
            "passive_open",
            "syn",
            "ack",
        ])
        assert "ESTABLISHED" in path

        # Check accepts valid sequence
        assert tcp.accepts(["passive_open", "syn", "ack"])
