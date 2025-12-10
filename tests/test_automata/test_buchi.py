"""Tests for Buchi automata module."""

import pytest
from gnsp.automata.buchi import (
    BuchiAutomaton,
    product_buchi,
    complement_buchi,
    OnlineMonitor,
    create_safety_monitor,
    create_liveness_monitor,
)


class TestBuchiAutomaton:
    """Tests for Buchi automaton class."""

    def test_basic_construction(self) -> None:
        """Test basic Buchi automaton construction."""
        states = {"q0", "q1"}
        alphabet = {"a", "b"}
        transitions = {
            ("q0", "a"): {"q1"},
            ("q1", "b"): {"q0"},
        }

        buchi = BuchiAutomaton(
            states=states,
            alphabet=alphabet,
            transitions=transitions,
            initial="q0",
            accepting={"q1"},
        )

        assert buchi.initial == "q0"
        assert buchi.accepting == {"q1"}

    def test_invalid_construction(self) -> None:
        """Test invalid construction raises errors."""
        with pytest.raises(ValueError):
            BuchiAutomaton(
                states={"q0"},
                alphabet={"a"},
                transitions={},
                initial="invalid",
                accepting=set(),
            )

    def test_transition(self) -> None:
        """Test transition function."""
        buchi = BuchiAutomaton(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q0", "q1"}},
            initial="q0",
            accepting={"q1"},
        )

        result = buchi.transition("q0", "a")
        assert result == {"q0", "q1"}
        assert buchi.transition("q1", "a") == set()

    def test_run_finite(self) -> None:
        """Test running on finite prefix."""
        buchi = BuchiAutomaton(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={
                ("q0", "a"): {"q1"},
                ("q1", "a"): {"q0"},
            },
            initial="q0",
            accepting={"q1"},
        )

        path = buchi.run_finite(["a", "a"])
        assert path[0] == {"q0"}
        assert path[1] == {"q1"}
        assert path[2] == {"q0"}

    def test_has_accepting_run_simple(self) -> None:
        """Test accepting run detection."""
        # Buchi with cycle through accepting state
        buchi = BuchiAutomaton(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={
                ("q0", "a"): {"q1"},
                ("q1", "a"): {"q1"},  # Self-loop on accepting
            },
            initial="q0",
            accepting={"q1"},
        )

        # After reading 'a', we're in q1 which has accepting cycle
        assert buchi.has_accepting_run(["a"])

    def test_is_empty(self) -> None:
        """Test emptiness check."""
        # Non-empty Buchi (has accepting cycle)
        non_empty = BuchiAutomaton(
            states={"q0"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q0"}},
            initial="q0",
            accepting={"q0"},
        )
        assert not non_empty.is_empty()

        # Empty Buchi (no accepting cycle reachable)
        empty = BuchiAutomaton(
            states={"q0", "q1"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q1"}},  # No cycle
            initial="q0",
            accepting={"q1"},
        )
        assert empty.is_empty()


class TestBuchiOperations:
    """Tests for Buchi automaton operations."""

    def test_product_buchi(self) -> None:
        """Test product construction."""
        # Buchi accepting "a infinitely often"
        buchi1 = BuchiAutomaton(
            states={"p0", "p1"},
            alphabet={"a", "b"},
            transitions={
                ("p0", "a"): {"p1"},
                ("p0", "b"): {"p0"},
                ("p1", "a"): {"p1"},
                ("p1", "b"): {"p0"},
            },
            initial="p0",
            accepting={"p1"},
        )

        # Buchi accepting "b infinitely often"
        buchi2 = BuchiAutomaton(
            states={"q0", "q1"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): {"q0"},
                ("q0", "b"): {"q1"},
                ("q1", "a"): {"q0"},
                ("q1", "b"): {"q1"},
            },
            initial="q0",
            accepting={"q1"},
        )

        product = product_buchi(buchi1, buchi2)

        # Product should have states
        assert len(product.states) > 0

    def test_complement_buchi(self) -> None:
        """Test Buchi complement (simplified)."""
        buchi = BuchiAutomaton(
            states={"q0"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q0"}},
            initial="q0",
            accepting={"q0"},
        )

        complement = complement_buchi(buchi)

        # Complement should exist
        assert len(complement.states) > 0


class TestOnlineMonitor:
    """Tests for online Buchi monitor."""

    def test_basic_monitoring(self) -> None:
        """Test basic online monitoring."""
        # Monitor that accepts if we see 'good' infinitely often
        buchi = BuchiAutomaton(
            states={"waiting", "saw_good"},
            alphabet={"good", "bad"},
            transitions={
                ("waiting", "good"): {"saw_good"},
                ("waiting", "bad"): {"waiting"},
                ("saw_good", "good"): {"saw_good"},
                ("saw_good", "bad"): {"waiting"},
            },
            initial="waiting",
            accepting={"saw_good"},
        )

        monitor = OnlineMonitor(buchi)

        # First step with 'good'
        verdict = monitor.step("good")
        assert verdict in ["unknown", "satisfied"]

        # Reset and test violation path
        monitor.reset()

    def test_violation_detection(self) -> None:
        """Test violation detection."""
        # Monitor that accepts only sequences of 'a'
        buchi = BuchiAutomaton(
            states={"ok", "fail"},
            alphabet={"a", "b"},
            transitions={
                ("ok", "a"): {"ok"},
                ("ok", "b"): {"fail"},
                ("fail", "a"): {"fail"},
                ("fail", "b"): {"fail"},
            },
            initial="ok",
            accepting={"ok"},
        )

        monitor = OnlineMonitor(buchi)

        # 'a' is ok
        assert monitor.step("a") == "unknown"

        # 'b' causes violation
        verdict = monitor.step("b")
        assert verdict == "violated"

    def test_reset(self) -> None:
        """Test monitor reset."""
        buchi = BuchiAutomaton(
            states={"q0"},
            alphabet={"a"},
            transitions={("q0", "a"): {"q0"}},
            initial="q0",
            accepting={"q0"},
        )

        monitor = OnlineMonitor(buchi)
        monitor.step("a")
        monitor.step("a")

        monitor.reset()
        assert monitor.step_count == 0
        assert monitor.current_states == {"q0"}


class TestSafetyLiveness:
    """Tests for safety and liveness monitors."""

    def test_safety_monitor(self) -> None:
        """Test safety property monitor."""
        states = {"safe", "unsafe"}
        alphabet = {"ok", "error"}
        transitions = {
            ("safe", "ok"): {"safe"},
            ("safe", "error"): {"unsafe"},
            ("unsafe", "ok"): {"unsafe"},
            ("unsafe", "error"): {"unsafe"},
        }

        buchi = create_safety_monitor(
            bad_states={"unsafe"},
            all_states=states,
            alphabet=alphabet,
            transitions=transitions,
            initial="safe",
        )

        # Accepting states should be good states
        assert buchi.accepting == {"safe"}

    def test_liveness_monitor(self) -> None:
        """Test liveness property monitor."""
        states = {"wait", "done"}
        alphabet = {"tick", "complete"}
        transitions = {
            ("wait", "tick"): {"wait"},
            ("wait", "complete"): {"done"},
            ("done", "tick"): {"wait"},
            ("done", "complete"): {"done"},
        }

        buchi = create_liveness_monitor(
            goal_states={"done"},
            all_states=states,
            alphabet=alphabet,
            transitions=transitions,
            initial="wait",
        )

        # Goal states should be accepting
        assert buchi.accepting == {"done"}
