"""Tests for timed automata module."""

import pytest
from gnsp.automata.timed import (
    ClockConstraintOp,
    ClockConstraint,
    Guard,
    TimedTransition,
    TimedAutomaton,
    TimedConfiguration,
    TimedSimulator,
    Zone,
    create_timeout_automaton,
    create_tcp_timeout_automaton,
    create_rate_limiter_automaton,
)


class TestClockConstraint:
    """Tests for clock constraints."""

    def test_less_than(self) -> None:
        """Test less than constraint."""
        c = ClockConstraint("x", ClockConstraintOp.LT, 5.0)
        assert c.satisfied(4.0)
        assert not c.satisfied(5.0)
        assert not c.satisfied(6.0)

    def test_less_equal(self) -> None:
        """Test less than or equal constraint."""
        c = ClockConstraint("x", ClockConstraintOp.LE, 5.0)
        assert c.satisfied(4.0)
        assert c.satisfied(5.0)
        assert not c.satisfied(5.1)

    def test_equal(self) -> None:
        """Test equality constraint."""
        c = ClockConstraint("x", ClockConstraintOp.EQ, 5.0)
        assert c.satisfied(5.0)
        assert not c.satisfied(4.9)
        assert not c.satisfied(5.1)

    def test_greater_equal(self) -> None:
        """Test greater than or equal constraint."""
        c = ClockConstraint("x", ClockConstraintOp.GE, 5.0)
        assert c.satisfied(5.0)
        assert c.satisfied(6.0)
        assert not c.satisfied(4.0)

    def test_greater_than(self) -> None:
        """Test greater than constraint."""
        c = ClockConstraint("x", ClockConstraintOp.GT, 5.0)
        assert c.satisfied(6.0)
        assert not c.satisfied(5.0)
        assert not c.satisfied(4.0)

    def test_str(self) -> None:
        """Test string representation."""
        c = ClockConstraint("x", ClockConstraintOp.LT, 5.0)
        assert str(c) == "x < 5.0"


class TestGuard:
    """Tests for guard conditions."""

    def test_empty_guard(self) -> None:
        """Test empty guard (always true)."""
        g = Guard(frozenset())
        assert g.satisfied({})
        assert g.satisfied({"x": 10.0})

    def test_single_constraint(self) -> None:
        """Test guard with single constraint."""
        g = Guard(frozenset({
            ClockConstraint("x", ClockConstraintOp.LT, 5.0)
        }))
        assert g.satisfied({"x": 4.0})
        assert not g.satisfied({"x": 6.0})

    def test_multiple_constraints(self) -> None:
        """Test guard with multiple constraints."""
        g = Guard(frozenset({
            ClockConstraint("x", ClockConstraintOp.GE, 2.0),
            ClockConstraint("x", ClockConstraintOp.LT, 5.0),
        }))
        assert g.satisfied({"x": 3.0})
        assert not g.satisfied({"x": 1.0})
        assert not g.satisfied({"x": 6.0})

    def test_missing_clock(self) -> None:
        """Test guard with missing clock returns False."""
        g = Guard(frozenset({
            ClockConstraint("x", ClockConstraintOp.LT, 5.0)
        }))
        assert not g.satisfied({"y": 3.0})


class TestTimedAutomaton:
    """Tests for timed automaton."""

    def test_basic_construction(self) -> None:
        """Test basic timed automaton construction."""
        ta = TimedAutomaton(
            locations={"l0", "l1"},
            initial="l0",
            clocks={"x"},
            actions={"a"},
            transitions=[
                TimedTransition(
                    source="l0",
                    action="a",
                    guard=Guard(frozenset()),
                    resets=frozenset({"x"}),
                    target="l1",
                )
            ],
            accepting={"l1"},
        )

        assert ta.initial == "l0"
        assert "l1" in ta.accepting

    def test_get_transitions(self) -> None:
        """Test getting transitions from location."""
        trans = TimedTransition(
            source="l0",
            action="a",
            guard=Guard(frozenset()),
            resets=frozenset(),
            target="l1",
        )

        ta = TimedAutomaton(
            locations={"l0", "l1"},
            initial="l0",
            clocks=set(),
            actions={"a"},
            transitions=[trans],
        )

        assert ta.get_transitions("l0") == [trans]
        assert ta.get_transitions("l1") == []

    def test_enabled_transitions(self) -> None:
        """Test getting enabled transitions."""
        ta = TimedAutomaton(
            locations={"l0", "l1"},
            initial="l0",
            clocks={"x"},
            actions={"a"},
            transitions=[
                TimedTransition(
                    source="l0",
                    action="a",
                    guard=Guard(frozenset({
                        ClockConstraint("x", ClockConstraintOp.GE, 5.0)
                    })),
                    resets=frozenset(),
                    target="l1",
                )
            ],
        )

        # Not enabled when x < 5
        enabled = ta.enabled_transitions("l0", {"x": 3.0})
        assert len(enabled) == 0

        # Enabled when x >= 5
        enabled = ta.enabled_transitions("l0", {"x": 5.0})
        assert len(enabled) == 1


class TestTimedSimulator:
    """Tests for timed automaton simulator."""

    def test_basic_simulation(self) -> None:
        """Test basic simulation."""
        ta = TimedAutomaton(
            locations={"l0", "l1"},
            initial="l0",
            clocks={"x"},
            actions={"a"},
            transitions=[
                TimedTransition(
                    source="l0",
                    action="a",
                    guard=Guard(frozenset()),
                    resets=frozenset(),
                    target="l1",
                )
            ],
            accepting={"l1"},
        )

        sim = TimedSimulator(ta)
        assert sim.config.location == "l0"

        # Take transition
        assert sim.take_transition("a")
        assert sim.config.location == "l1"
        assert sim.is_accepting()

    def test_delay(self) -> None:
        """Test time delay."""
        ta = TimedAutomaton(
            locations={"l0"},
            initial="l0",
            clocks={"x"},
            actions=set(),
            transitions=[],
        )

        sim = TimedSimulator(ta)
        assert sim.config.clock_values["x"] == 0.0

        sim.delay(5.0)
        assert sim.config.clock_values["x"] == 5.0

    def test_invariant_blocks_delay(self) -> None:
        """Test that invariant blocks delay."""
        ta = TimedAutomaton(
            locations={"l0"},
            initial="l0",
            clocks={"x"},
            actions=set(),
            transitions=[],
            invariants={
                "l0": Guard(frozenset({
                    ClockConstraint("x", ClockConstraintOp.LE, 5.0)
                }))
            },
        )

        sim = TimedSimulator(ta)

        # Can delay up to 5
        assert sim.can_delay(4.0)
        assert sim.can_delay(5.0)
        # Cannot delay past 5
        assert not sim.can_delay(6.0)

    def test_reset_on_transition(self) -> None:
        """Test clock reset on transition."""
        ta = TimedAutomaton(
            locations={"l0", "l1"},
            initial="l0",
            clocks={"x"},
            actions={"a"},
            transitions=[
                TimedTransition(
                    source="l0",
                    action="a",
                    guard=Guard(frozenset()),
                    resets=frozenset({"x"}),
                    target="l1",
                )
            ],
        )

        sim = TimedSimulator(ta)
        sim.delay(10.0)
        assert sim.config.clock_values["x"] == 10.0

        sim.take_transition("a")
        assert sim.config.clock_values["x"] == 0.0

    def test_run_timed_word(self) -> None:
        """Test running timed word."""
        ta = TimedAutomaton(
            locations={"l0", "l1"},
            initial="l0",
            clocks={"x"},
            actions={"a"},
            transitions=[
                TimedTransition(
                    source="l0",
                    action="a",
                    guard=Guard(frozenset({
                        ClockConstraint("x", ClockConstraintOp.GE, 2.0)
                    })),
                    resets=frozenset(),
                    target="l1",
                )
            ],
        )

        sim = TimedSimulator(ta)

        # Valid timed word
        assert sim.run_timed_word([(3.0, "a")])

        # Invalid (guard not satisfied)
        sim.reset()
        assert not sim.run_timed_word([(1.0, "a")])


class TestZone:
    """Tests for zone representation."""

    def test_empty_zone(self) -> None:
        """Test zone initialization."""
        zone = Zone(["x", "y"])
        assert not zone.is_empty()

    def test_constraint_addition(self) -> None:
        """Test adding constraints."""
        zone = Zone(["x"])
        zone.add_constraint(ClockConstraint("x", ClockConstraintOp.LE, 5.0))
        zone.canonicalize()
        assert not zone.is_empty()

    def test_conflicting_constraints(self) -> None:
        """Test conflicting constraints make empty zone."""
        zone = Zone(["x"])
        zone.add_constraint(ClockConstraint("x", ClockConstraintOp.GE, 10.0))
        zone.add_constraint(ClockConstraint("x", ClockConstraintOp.LE, 5.0))
        zone.canonicalize()
        assert zone.is_empty()

    def test_delay_operation(self) -> None:
        """Test delay operation removes upper bounds."""
        zone = Zone(["x"])
        zone.add_constraint(ClockConstraint("x", ClockConstraintOp.LE, 5.0))
        zone.canonicalize()

        zone.delay()
        zone.canonicalize()

        # After delay, upper bound should be removed
        assert not zone.is_empty()

    def test_reset_clock(self) -> None:
        """Test clock reset."""
        zone = Zone(["x"])
        zone.add_constraint(ClockConstraint("x", ClockConstraintOp.GE, 5.0))
        zone.canonicalize()

        zone.reset_clock("x")
        zone.canonicalize()

        # After reset, x should be 0
        assert zone.contains_valuation({"x": 0.0})


class TestFactoryFunctions:
    """Tests for timed automaton factory functions."""

    def test_timeout_automaton(self) -> None:
        """Test timeout automaton creation."""
        ta = create_timeout_automaton(
            timeout=10.0,
            actions={"work"},
        )

        assert "idle" in ta.locations
        assert "active" in ta.locations
        assert "timeout" in ta.locations

    def test_tcp_timeout_automaton(self) -> None:
        """Test TCP timeout automaton creation."""
        ta = create_tcp_timeout_automaton()

        assert "closed" in ta.locations
        assert "syn_sent" in ta.locations
        assert "established" in ta.locations

    def test_rate_limiter_automaton(self) -> None:
        """Test rate limiter automaton creation."""
        ta = create_rate_limiter_automaton(rate=10.0, burst=3)

        assert "tokens_0" in ta.locations
        assert "tokens_3" in ta.locations
        assert ta.initial == "tokens_3"  # Start with full bucket
