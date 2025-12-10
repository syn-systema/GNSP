"""Tests for weighted automata module."""

import pytest
import math
from gnsp.automata.weighted import (
    BooleanSemiring,
    ProbabilitySemiring,
    LogProbabilitySemiring,
    TropicalSemiring,
    MaxTropicalSemiring,
    ViterbiSemiring,
    CountingSemiring,
    WeightedAutomaton,
    create_probabilistic_automaton,
    create_tropical_automaton,
    create_counting_automaton,
    AnomalyScoringAutomaton,
)


class TestSemirings:
    """Tests for semiring implementations."""

    def test_boolean_semiring(self) -> None:
        """Test Boolean semiring."""
        sr = BooleanSemiring()

        assert sr.zero() is False
        assert sr.one() is True
        assert sr.add(True, False) is True
        assert sr.add(False, False) is False
        assert sr.multiply(True, True) is True
        assert sr.multiply(True, False) is False

    def test_probability_semiring(self) -> None:
        """Test probability semiring."""
        sr = ProbabilitySemiring()

        assert sr.zero() == 0.0
        assert sr.one() == 1.0
        assert sr.add(0.3, 0.7) == 1.0
        assert sr.multiply(0.5, 0.5) == 0.25

    def test_log_probability_semiring(self) -> None:
        """Test log probability semiring."""
        sr = LogProbabilitySemiring()

        assert sr.zero() == float("-inf")
        assert sr.one() == 0.0

        # Log-sum-exp
        a = math.log(0.3)
        b = math.log(0.7)
        result = sr.add(a, b)
        assert abs(result - math.log(1.0)) < 1e-9

        # Multiplication is addition in log space
        assert sr.multiply(-1.0, -2.0) == -3.0

    def test_tropical_semiring(self) -> None:
        """Test tropical (min-plus) semiring."""
        sr = TropicalSemiring()

        assert sr.zero() == float("inf")
        assert sr.one() == 0.0
        assert sr.add(5.0, 3.0) == 3.0  # min
        assert sr.multiply(2.0, 3.0) == 5.0  # plus

    def test_max_tropical_semiring(self) -> None:
        """Test max-plus tropical semiring."""
        sr = MaxTropicalSemiring()

        assert sr.zero() == float("-inf")
        assert sr.one() == 0.0
        assert sr.add(5.0, 3.0) == 5.0  # max
        assert sr.multiply(2.0, 3.0) == 5.0  # plus

    def test_viterbi_semiring(self) -> None:
        """Test Viterbi semiring."""
        sr = ViterbiSemiring()

        assert sr.zero() == 0.0
        assert sr.one() == 1.0
        assert sr.add(0.3, 0.7) == 0.7  # max
        assert sr.multiply(0.5, 0.5) == 0.25  # times

    def test_counting_semiring(self) -> None:
        """Test counting semiring."""
        sr = CountingSemiring()

        assert sr.zero() == 0
        assert sr.one() == 1
        assert sr.add(2, 3) == 5
        assert sr.multiply(2, 3) == 6


class TestWeightedAutomaton:
    """Tests for weighted automaton."""

    def test_basic_weight_computation(self) -> None:
        """Test basic weight computation."""
        states = {"q0", "q1"}
        alphabet = {"a"}
        transitions = {
            ("q0", "a", "q1"): 0.5,
        }
        initial = {"q0": 1.0}
        final = {"q1": 1.0}

        wfa = create_probabilistic_automaton(
            states, alphabet, transitions, initial, final
        )

        weight = wfa.weight(["a"])
        assert abs(weight - 0.5) < 1e-9

    def test_multiple_paths(self) -> None:
        """Test weight computation with multiple paths."""
        states = {"q0", "q1", "q2"}
        alphabet = {"a"}
        transitions = {
            ("q0", "a", "q1"): 0.3,
            ("q0", "a", "q2"): 0.7,
        }
        initial = {"q0": 1.0}
        final = {"q1": 1.0, "q2": 1.0}

        wfa = create_probabilistic_automaton(
            states, alphabet, transitions, initial, final
        )

        weight = wfa.weight(["a"])
        assert abs(weight - 1.0) < 1e-9  # 0.3 + 0.7

    def test_tropical_shortest_path(self) -> None:
        """Test tropical automaton for shortest path."""
        states = {"q0", "q1", "q2"}
        alphabet = {"a", "b"}
        transitions = {
            ("q0", "a", "q1"): 1.0,
            ("q0", "b", "q2"): 5.0,
            ("q1", "a", "q2"): 2.0,
        }
        initial = {"q0": 0.0}
        final = {"q2": 0.0}

        wfa = create_tropical_automaton(
            states, alphabet, transitions, initial, final
        )

        # Path via q1: 1 + 2 = 3
        # Direct via b: 5
        # Shortest is 3
        assert wfa.weight(["a", "a"]) == 3.0
        assert wfa.weight(["b"]) == 5.0

    def test_counting_automaton(self) -> None:
        """Test counting automaton for path counting."""
        states = {"q0", "q1", "q2"}
        alphabet = {"a"}
        transitions = {
            ("q0", "a", "q1"): 1,
            ("q0", "a", "q2"): 1,
            ("q1", "a", "q2"): 1,
        }
        initial = {"q0": 1}
        final = {"q2": 1}

        wfa = create_counting_automaton(
            states, alphabet, transitions, initial, final
        )

        # Two paths: q0->q2 directly, or q0->q1->q2
        assert wfa.weight(["a"]) == 1  # Direct path
        assert wfa.weight(["a", "a"]) == 1  # Via q1

    def test_viterbi_path(self) -> None:
        """Test Viterbi path computation."""
        states = {"q0", "q1", "q2"}
        alphabet = {"a"}
        transitions = {
            ("q0", "a", "q1"): 0.8,
            ("q0", "a", "q2"): 0.2,
        }
        initial = {"q0": 1.0}
        final = {"q1": 1.0, "q2": 1.0}

        wfa = WeightedAutomaton(
            states=states,
            alphabet=alphabet,
            semiring=ViterbiSemiring(),
            transitions=transitions,
            initial_weights=initial,
            final_weights=final,
        )

        best_weight, best_path = wfa.viterbi_path(["a"])
        assert abs(best_weight - 0.8) < 1e-9
        assert best_path == ["q0", "q1"]


class TestAnomalyScoringAutomaton:
    """Tests for anomaly scoring automaton."""

    def test_basic_anomaly_score(self) -> None:
        """Test basic anomaly scoring."""
        states = {"normal", "anomaly"}
        alphabet = {"good", "bad"}
        transitions = {
            ("normal", "good", "normal"): 0.9,
            ("normal", "bad", "anomaly"): 0.1,
            ("anomaly", "good", "normal"): 0.3,
            ("anomaly", "bad", "anomaly"): 0.7,
        }
        initial = {"normal": 1.0}
        final = {"normal": 1.0, "anomaly": 1.0}

        scorer = AnomalyScoringAutomaton(
            states, alphabet, transitions, initial, final
        )

        # Normal sequence should have lower score
        normal_score = scorer.anomaly_score(["good", "good", "good"])
        # Anomalous sequence should have higher score
        anomaly_score = scorer.anomaly_score(["bad", "bad", "bad"])

        assert anomaly_score > normal_score

    def test_unknown_symbol_penalty(self) -> None:
        """Test unknown symbol handling."""
        states = {"q0"}
        alphabet = {"known"}
        transitions = {("q0", "known", "q0"): 1.0}
        initial = {"q0": 1.0}
        final = {"q0": 1.0}

        scorer = AnomalyScoringAutomaton(
            states, alphabet, transitions, initial, final
        )

        # Known symbols
        known_score = scorer.anomaly_score(["known", "known"])
        # Unknown symbols
        unknown_score = scorer.anomaly_score(["unknown", "unknown"])

        assert unknown_score > known_score

    def test_is_anomalous(self) -> None:
        """Test anomaly threshold."""
        states = {"q0"}
        alphabet = {"a"}
        transitions = {("q0", "a", "q0"): 0.5}
        initial = {"q0": 1.0}
        final = {"q0": 1.0}

        scorer = AnomalyScoringAutomaton(
            states, alphabet, transitions, initial, final
        )

        score = scorer.anomaly_score(["a", "a"])
        # Should be anomalous at low threshold
        assert scorer.is_anomalous(["a", "a"], 0.0)
        # Should not be anomalous at high threshold
        assert not scorer.is_anomalous(["a", "a"], 1000.0)
