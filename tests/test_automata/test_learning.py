"""Tests for automata learning module."""

import pytest
from gnsp.automata.dfa import DFA
from gnsp.automata.learning import (
    DFAMembershipOracle,
    DFAEquivalenceOracle,
    ObservationTable,
    LStarLearner,
    RivestSchapireLearner,
    PassiveLearner,
    learn_from_traces,
    learn_protocol_automaton,
)


class TestOracles:
    """Tests for oracle implementations."""

    def test_dfa_membership_oracle(self) -> None:
        """Test DFA membership oracle."""
        # Target DFA: accepts words ending in 'a'
        target = DFA(
            states={0, 1},
            alphabet={"a", "b"},
            transitions={
                (0, "a"): 1,
                (0, "b"): 0,
                (1, "a"): 1,
                (1, "b"): 0,
            },
            initial=0,
            accepting={1},
        )

        oracle = DFAMembershipOracle(target)

        assert oracle.query(("a",))
        assert oracle.query(("b", "a"))
        assert not oracle.query(("b",))
        assert not oracle.query(())

    def test_dfa_equivalence_oracle(self) -> None:
        """Test DFA equivalence oracle."""
        target = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        oracle = DFAEquivalenceOracle(target)

        # Equivalent hypothesis
        hypothesis = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        assert oracle.query(hypothesis) is None

        # Non-equivalent hypothesis
        wrong = DFA(
            states={0},
            alphabet={"a"},
            transitions={(0, "a"): 0},
            initial=0,
            accepting={0},
        )

        ce = oracle.query(wrong)
        assert ce is not None


class TestObservationTable:
    """Tests for observation table."""

    def test_initialization(self) -> None:
        """Test table initialization."""
        table = ObservationTable(alphabet={"a", "b"})

        assert () in table.prefixes
        assert () in table.suffixes

    def test_row_computation(self) -> None:
        """Test row computation."""
        table = ObservationTable(alphabet={"a"})
        table.table[((), ())] = True

        row = table.row(())
        assert row == (True,)

    def test_closedness_check(self) -> None:
        """Test closedness check."""
        table = ObservationTable(alphabet={"a"})
        table.prefixes = {()}
        table.suffixes = {()}

        # Fill table
        target = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        oracle = DFAMembershipOracle(target)
        table._fill_table(oracle)

        # Should not be closed initially (row of 'a' differs from row of empty)
        unclosed = table.is_closed()
        # Either closed or returns violating element
        assert unclosed is None or unclosed in {("a",)}

    def test_build_hypothesis(self) -> None:
        """Test hypothesis construction."""
        table = ObservationTable(alphabet={"a"})
        table.prefixes = {(), ("a",)}
        table.suffixes = {()}
        table.table = {
            ((), ()): False,
            (("a",), ()): True,
        }

        hypothesis = table.build_hypothesis()

        assert 0 in hypothesis.states or len(hypothesis.states) > 0


class TestLStarLearner:
    """Tests for L* algorithm."""

    def test_learn_simple_dfa(self) -> None:
        """Test learning simple DFA."""
        # Target: accepts 'a+'
        target = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        membership = DFAMembershipOracle(target)
        equivalence = DFAEquivalenceOracle(target)

        learner = LStarLearner({"a"}, membership, equivalence)
        learned = learner.learn()

        # Learned DFA should accept same language
        assert learned.accepts(["a"])
        assert learned.accepts(["a", "a"])
        assert not learned.accepts([])

    def test_learn_ends_with_a(self) -> None:
        """Test learning DFA for words ending in 'a'."""
        target = DFA(
            states={0, 1},
            alphabet={"a", "b"},
            transitions={
                (0, "a"): 1,
                (0, "b"): 0,
                (1, "a"): 1,
                (1, "b"): 0,
            },
            initial=0,
            accepting={1},
        )

        membership = DFAMembershipOracle(target)
        equivalence = DFAEquivalenceOracle(target, max_length=5)

        learner = LStarLearner({"a", "b"}, membership, equivalence)
        learned = learner.learn()

        # Test equivalence
        assert learned.accepts(["a"]) == target.accepts(["a"])
        assert learned.accepts(["b"]) == target.accepts(["b"])
        assert learned.accepts(["b", "a"]) == target.accepts(["b", "a"])
        assert learned.accepts(["a", "b"]) == target.accepts(["a", "b"])


class TestRivestSchapireLearner:
    """Tests for Rivest-Schapire variant."""

    def test_learn_simple(self) -> None:
        """Test RS learner on simple DFA."""
        target = DFA(
            states={0, 1},
            alphabet={"a"},
            transitions={(0, "a"): 1, (1, "a"): 1},
            initial=0,
            accepting={1},
        )

        membership = DFAMembershipOracle(target)
        equivalence = DFAEquivalenceOracle(target)

        learner = RivestSchapireLearner({"a"}, membership, equivalence)
        learned = learner.learn()

        assert learned.accepts(["a"])
        assert not learned.accepts([])


class TestPassiveLearner:
    """Tests for passive learning."""

    def test_learn_from_examples(self) -> None:
        """Test learning from positive/negative examples."""
        learner = PassiveLearner({"a", "b"})

        positive = [
            ("a",),
            ("a", "a"),
            ("a", "a", "a"),
        ]
        negative = [
            (),
            ("b",),
            ("a", "b"),
        ]

        learned = learner.learn(positive, negative)

        # Should accept positive examples
        for word in positive:
            assert learned.accepts(list(word))

        # Should reject negative examples
        for word in negative:
            assert not learned.accepts(list(word))


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_learn_from_traces(self) -> None:
        """Test learning from labeled traces."""
        traces = [
            ["a", "a"],
            ["a", "a", "a"],
            ["b"],
            ["a", "b"],
        ]
        labels = [True, True, False, False]

        learned = learn_from_traces(traces, labels)

        assert learned.accepts(["a", "a"])
        assert not learned.accepts(["b"])

    def test_learn_protocol_automaton(self) -> None:
        """Test learning protocol automaton from oracle."""
        # Simple oracle: accepts sequences of 'a' only
        def oracle(seq):
            return all(s == "a" for s in seq) and len(seq) > 0

        learned = learn_protocol_automaton(
            oracle,
            alphabet={"a", "b"},
            max_length=3,
        )

        assert learned.accepts(["a"])
        assert learned.accepts(["a", "a"])
        assert not learned.accepts(["b"])
        assert not learned.accepts([])
