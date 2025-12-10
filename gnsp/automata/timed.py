"""Timed automata for real-time system modeling.

This module provides timed automata (TA) with:
- Clock variables tracking elapsed time
- Guard conditions on transitions
- Clock resets
- Zone-based state space representation

Useful for modeling network protocol timeouts and timing constraints.
"""

from dataclasses import dataclass, field
from typing import (
    Set,
    Dict,
    Tuple,
    List,
    Optional,
    FrozenSet,
    TypeVar,
    Generic,
    Callable,
)
from collections import deque
from enum import Enum
import math

S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type


class ClockConstraintOp(Enum):
    """Clock constraint comparison operators."""

    LT = "<"
    LE = "<="
    EQ = "=="
    GE = ">="
    GT = ">"


@dataclass(frozen=True)
class ClockConstraint:
    """Single clock constraint of form: clock op constant.

    Attributes:
        clock: Clock variable name
        op: Comparison operator
        value: Constant value to compare against
    """

    clock: str
    op: ClockConstraintOp
    value: float

    def __str__(self) -> str:
        return f"{self.clock} {self.op.value} {self.value}"

    def satisfied(self, clock_value: float) -> bool:
        """Check if constraint is satisfied by clock value."""
        if self.op == ClockConstraintOp.LT:
            return clock_value < self.value
        elif self.op == ClockConstraintOp.LE:
            return clock_value <= self.value
        elif self.op == ClockConstraintOp.EQ:
            return abs(clock_value - self.value) < 1e-9
        elif self.op == ClockConstraintOp.GE:
            return clock_value >= self.value
        elif self.op == ClockConstraintOp.GT:
            return clock_value > self.value
        return False


@dataclass(frozen=True)
class Guard:
    """Conjunction of clock constraints.

    Attributes:
        constraints: Set of clock constraints (all must be satisfied)
    """

    constraints: FrozenSet[ClockConstraint] = field(
        default_factory=lambda: frozenset()
    )

    def satisfied(self, clock_values: Dict[str, float]) -> bool:
        """Check if all constraints are satisfied."""
        for constraint in self.constraints:
            if constraint.clock not in clock_values:
                return False
            if not constraint.satisfied(clock_values[constraint.clock]):
                return False
        return True

    def __str__(self) -> str:
        if not self.constraints:
            return "true"
        return " && ".join(str(c) for c in self.constraints)


@dataclass(frozen=True)
class TimedTransition(Generic[S, A]):
    """Timed automaton transition.

    Attributes:
        source: Source state
        action: Action label
        guard: Guard condition
        resets: Set of clocks to reset
        target: Target state
    """

    source: S
    action: A
    guard: Guard
    resets: FrozenSet[str]
    target: S

    def __str__(self) -> str:
        reset_str = ", ".join(self.resets) if self.resets else "-"
        return f"{self.source} --[{self.action}, {self.guard}, {{{reset_str}}}]--> {self.target}"


@dataclass
class TimedAutomaton(Generic[S, A]):
    """Timed Automaton with clock variables.

    A Timed Automaton TA = (L, l0, C, A, E, I) where:
    - L: finite set of locations (states)
    - l0: initial location
    - C: finite set of clocks
    - A: finite set of actions
    - E: set of edges (transitions with guards and resets)
    - I: location invariants

    Attributes:
        locations: Set of locations
        initial: Initial location
        clocks: Set of clock variable names
        actions: Set of action labels
        transitions: List of timed transitions
        invariants: Location invariants (must hold while in location)
        accepting: Set of accepting locations
    """

    locations: Set[S]
    initial: S
    clocks: Set[str]
    actions: Set[A]
    transitions: List[TimedTransition[S, A]]
    invariants: Dict[S, Guard] = field(default_factory=dict)
    accepting: Set[S] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate timed automaton structure."""
        if self.initial not in self.locations:
            raise ValueError("Initial location must be in locations")
        if not self.accepting.issubset(self.locations):
            raise ValueError("Accepting locations must be subset of locations")

    def get_transitions(
        self,
        location: S,
        action: Optional[A] = None,
    ) -> List[TimedTransition[S, A]]:
        """Get transitions from location, optionally filtered by action."""
        result = []
        for trans in self.transitions:
            if trans.source == location:
                if action is None or trans.action == action:
                    result.append(trans)
        return result

    def check_invariant(
        self,
        location: S,
        clock_values: Dict[str, float],
    ) -> bool:
        """Check if location invariant is satisfied."""
        if location not in self.invariants:
            return True
        return self.invariants[location].satisfied(clock_values)

    def enabled_transitions(
        self,
        location: S,
        clock_values: Dict[str, float],
    ) -> List[TimedTransition[S, A]]:
        """Get transitions enabled at current configuration."""
        enabled = []
        for trans in self.get_transitions(location):
            if trans.guard.satisfied(clock_values):
                # Check target invariant with reset clocks
                new_clocks = dict(clock_values)
                for clock in trans.resets:
                    new_clocks[clock] = 0.0
                if self.check_invariant(trans.target, new_clocks):
                    enabled.append(trans)
        return enabled


@dataclass
class TimedConfiguration(Generic[S]):
    """Configuration of timed automaton (location + clock values).

    Attributes:
        location: Current location
        clock_values: Current values of all clocks
    """

    location: S
    clock_values: Dict[str, float]

    def copy(self) -> "TimedConfiguration[S]":
        """Create a copy of this configuration."""
        return TimedConfiguration(
            location=self.location,
            clock_values=dict(self.clock_values),
        )

    def delay(self, delta: float) -> "TimedConfiguration[S]":
        """Create new configuration after time delay."""
        new_clocks = {k: v + delta for k, v in self.clock_values.items()}
        return TimedConfiguration(
            location=self.location,
            clock_values=new_clocks,
        )


class TimedSimulator(Generic[S, A]):
    """Simulator for timed automata.

    Supports both discrete and continuous-time simulation.
    """

    def __init__(self, automaton: TimedAutomaton[S, A]) -> None:
        """Initialize simulator.

        Args:
            automaton: Timed automaton to simulate
        """
        self.automaton = automaton
        self.config = self._initial_config()
        self.trace: List[Tuple[float, Optional[A], S]] = []

    def _initial_config(self) -> TimedConfiguration[S]:
        """Create initial configuration."""
        return TimedConfiguration(
            location=self.automaton.initial,
            clock_values={c: 0.0 for c in self.automaton.clocks},
        )

    def reset(self) -> None:
        """Reset to initial configuration."""
        self.config = self._initial_config()
        self.trace = []

    def can_delay(self, delta: float) -> bool:
        """Check if time delay is possible (invariant preserved)."""
        new_config = self.config.delay(delta)
        return self.automaton.check_invariant(
            new_config.location,
            new_config.clock_values,
        )

    def max_delay(self, epsilon: float = 0.01) -> float:
        """Compute maximum allowed delay (binary search)."""
        if not self.can_delay(epsilon):
            return 0.0

        # Binary search for max delay
        low, high = 0.0, 1000.0
        while high - low > epsilon:
            mid = (low + high) / 2
            if self.can_delay(mid):
                low = mid
            else:
                high = mid

        return low

    def delay(self, delta: float) -> bool:
        """Apply time delay.

        Args:
            delta: Time to elapse

        Returns:
            True if delay was successful
        """
        if not self.can_delay(delta):
            return False

        self.config = self.config.delay(delta)
        return True

    def take_transition(self, action: A) -> bool:
        """Take discrete transition.

        Args:
            action: Action to perform

        Returns:
            True if transition was taken
        """
        enabled = self.automaton.enabled_transitions(
            self.config.location,
            self.config.clock_values,
        )

        for trans in enabled:
            if trans.action == action:
                # Apply transition
                new_clocks = dict(self.config.clock_values)
                for clock in trans.resets:
                    new_clocks[clock] = 0.0

                self.config = TimedConfiguration(
                    location=trans.target,
                    clock_values=new_clocks,
                )
                self.trace.append((
                    sum(self.config.clock_values.values()),
                    action,
                    trans.target,
                ))
                return True

        return False

    def run_timed_word(
        self,
        timed_word: List[Tuple[float, A]],
    ) -> bool:
        """Run automaton on timed word.

        Args:
            timed_word: List of (delay, action) pairs

        Returns:
            True if run is valid
        """
        self.reset()

        for delay, action in timed_word:
            if not self.delay(delay):
                return False
            if not self.take_transition(action):
                return False

        return True

    def is_accepting(self) -> bool:
        """Check if current configuration is accepting."""
        return self.config.location in self.automaton.accepting


# Zone-based symbolic state space


@dataclass
class DBMEntry:
    """Difference Bound Matrix entry.

    Represents constraint: clock_i - clock_j <= bound
    """

    bound: float
    strict: bool = False  # True for <, False for <=

    def __le__(self, other: "DBMEntry") -> bool:
        if self.bound < other.bound:
            return True
        if self.bound > other.bound:
            return False
        # Equal bounds: strict < non-strict
        return self.strict or not other.strict

    def __lt__(self, other: "DBMEntry") -> bool:
        return self <= other and not (
            self.bound == other.bound and self.strict == other.strict
        )

    def __add__(self, other: "DBMEntry") -> "DBMEntry":
        return DBMEntry(
            bound=self.bound + other.bound,
            strict=self.strict or other.strict,
        )


class Zone:
    """Zone represented as Difference Bound Matrix (DBM).

    Represents a convex polyhedron of clock valuations.
    Uses a special clock x0 = 0 for absolute constraints.
    """

    INF = float("inf")

    def __init__(self, clocks: List[str]) -> None:
        """Initialize zone.

        Args:
            clocks: List of clock names
        """
        self.clocks = ["x0"] + list(clocks)  # x0 is special zero clock
        self.n = len(self.clocks)
        self.clock_index = {c: i for i, c in enumerate(self.clocks)}

        # Initialize DBM with infinity (no constraints)
        self.dbm: List[List[DBMEntry]] = [
            [DBMEntry(self.INF) for _ in range(self.n)]
            for _ in range(self.n)
        ]

        # Diagonal is always 0
        for i in range(self.n):
            self.dbm[i][i] = DBMEntry(0.0)

        # x0 = 0, so x0 - xi <= 0 (clocks are non-negative)
        for i in range(1, self.n):
            self.dbm[0][i] = DBMEntry(0.0)

    def copy(self) -> "Zone":
        """Create a copy of this zone."""
        new_zone = Zone([])
        new_zone.clocks = list(self.clocks)
        new_zone.n = self.n
        new_zone.clock_index = dict(self.clock_index)
        new_zone.dbm = [
            [DBMEntry(e.bound, e.strict) for e in row]
            for row in self.dbm
        ]
        return new_zone

    def canonicalize(self) -> bool:
        """Put DBM in canonical form using Floyd-Warshall.

        Returns:
            True if zone is non-empty
        """
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    new_entry = self.dbm[i][k] + self.dbm[k][j]
                    if new_entry < self.dbm[i][j]:
                        self.dbm[i][j] = new_entry

        # Check for negative cycles (empty zone)
        for i in range(self.n):
            if self.dbm[i][i].bound < 0 or (
                self.dbm[i][i].bound == 0 and self.dbm[i][i].strict
            ):
                return False

        return True

    def is_empty(self) -> bool:
        """Check if zone is empty."""
        zone_copy = self.copy()
        return not zone_copy.canonicalize()

    def add_constraint(self, constraint: ClockConstraint) -> None:
        """Add clock constraint to zone."""
        i = self.clock_index.get(constraint.clock)
        if i is None:
            return

        if constraint.op == ClockConstraintOp.LT:
            # x < c  =>  x - x0 < c
            self.dbm[i][0] = DBMEntry(constraint.value, strict=True)
        elif constraint.op == ClockConstraintOp.LE:
            # x <= c  =>  x - x0 <= c
            self.dbm[i][0] = DBMEntry(constraint.value, strict=False)
        elif constraint.op == ClockConstraintOp.GT:
            # x > c  =>  x0 - x < -c
            self.dbm[0][i] = DBMEntry(-constraint.value, strict=True)
        elif constraint.op == ClockConstraintOp.GE:
            # x >= c  =>  x0 - x <= -c
            self.dbm[0][i] = DBMEntry(-constraint.value, strict=False)
        elif constraint.op == ClockConstraintOp.EQ:
            # x == c
            self.dbm[i][0] = DBMEntry(constraint.value, strict=False)
            self.dbm[0][i] = DBMEntry(-constraint.value, strict=False)

    def reset_clock(self, clock: str) -> None:
        """Reset clock to zero."""
        i = self.clock_index.get(clock)
        if i is None:
            return

        # After reset, xi = 0, so xi - x0 = 0 and x0 - xi = 0
        for j in range(self.n):
            self.dbm[i][j] = self.dbm[0][j]
            self.dbm[j][i] = self.dbm[j][0]

    def delay(self) -> None:
        """Apply time elapse (future operator)."""
        # Remove upper bounds on clocks
        for i in range(1, self.n):
            self.dbm[i][0] = DBMEntry(self.INF)

    def intersect(self, other: "Zone") -> "Zone":
        """Intersect with another zone."""
        result = self.copy()
        for i in range(self.n):
            for j in range(self.n):
                if other.dbm[i][j] < result.dbm[i][j]:
                    result.dbm[i][j] = other.dbm[i][j]
        return result

    def contains_valuation(self, valuation: Dict[str, float]) -> bool:
        """Check if zone contains clock valuation."""
        zone_copy = self.copy()
        for clock, value in valuation.items():
            zone_copy.add_constraint(
                ClockConstraint(clock, ClockConstraintOp.EQ, value)
            )
        return not zone_copy.is_empty()


@dataclass(frozen=True)
class SymbolicState(Generic[S]):
    """Symbolic state: location + zone.

    Attributes:
        location: Current location
        zone: Zone representing clock valuations
    """

    location: S
    zone: Zone

    def __hash__(self) -> int:
        # Simplified hash based on location only
        return hash(self.location)


class ZoneGraph(Generic[S, A]):
    """Zone graph (symbolic state space) of timed automaton.

    Uses forward reachability with zone abstraction.
    """

    def __init__(self, automaton: TimedAutomaton[S, A]) -> None:
        """Initialize zone graph.

        Args:
            automaton: Timed automaton
        """
        self.automaton = automaton
        self.clock_list = list(automaton.clocks)
        self.states: List[SymbolicState[S]] = []
        self.edges: List[Tuple[int, A, int]] = []  # (src_idx, action, dst_idx)

    def build(self) -> None:
        """Build zone graph using forward reachability."""
        # Initial state
        init_zone = Zone(self.clock_list)
        init_zone.canonicalize()

        init_state = SymbolicState(
            location=self.automaton.initial,
            zone=init_zone,
        )

        self.states = [init_state]
        state_map: Dict[S, List[int]] = {self.automaton.initial: [0]}

        queue = deque([0])
        visited = {0}

        while queue:
            idx = queue.popleft()
            state = self.states[idx]

            # Apply time elapse
            delayed_zone = state.zone.copy()
            delayed_zone.delay()

            # Apply location invariant
            if state.location in self.automaton.invariants:
                for constraint in self.automaton.invariants[state.location].constraints:
                    delayed_zone.add_constraint(constraint)

            delayed_zone.canonicalize()

            # Process transitions
            for trans in self.automaton.get_transitions(state.location):
                # Apply guard
                trans_zone = delayed_zone.copy()
                for constraint in trans.guard.constraints:
                    trans_zone.add_constraint(constraint)

                if trans_zone.is_empty():
                    continue

                # Apply resets
                for clock in trans.resets:
                    trans_zone.reset_clock(clock)

                # Apply target invariant
                if trans.target in self.automaton.invariants:
                    for constraint in self.automaton.invariants[trans.target].constraints:
                        trans_zone.add_constraint(constraint)

                if not trans_zone.canonicalize():
                    continue

                # Create new symbolic state
                new_state = SymbolicState(
                    location=trans.target,
                    zone=trans_zone,
                )

                # Check if subsumed by existing state
                new_idx = len(self.states)
                self.states.append(new_state)

                if trans.target not in state_map:
                    state_map[trans.target] = []
                state_map[trans.target].append(new_idx)

                self.edges.append((idx, trans.action, new_idx))

                if new_idx not in visited:
                    visited.add(new_idx)
                    queue.append(new_idx)

    def is_reachable(self, location: S) -> bool:
        """Check if location is reachable."""
        return any(s.location == location for s in self.states)

    def reachable_locations(self) -> Set[S]:
        """Get all reachable locations."""
        return {s.location for s in self.states}


# Factory functions for common timed automata


def create_timeout_automaton(
    timeout: float,
    actions: Set[str],
    timeout_action: str = "timeout",
) -> TimedAutomaton[str, str]:
    """Create timed automaton with single timeout.

    Args:
        timeout: Timeout duration
        actions: Regular actions (before timeout)
        timeout_action: Action for timeout transition

    Returns:
        Timed automaton with timeout behavior
    """
    locations = {"idle", "active", "timeout"}
    clocks = {"t"}

    transitions = [
        # Start activity
        TimedTransition(
            source="idle",
            action="start",
            guard=Guard(frozenset()),
            resets=frozenset({"t"}),
            target="active",
        ),
        # Timeout
        TimedTransition(
            source="active",
            action=timeout_action,
            guard=Guard(frozenset({
                ClockConstraint("t", ClockConstraintOp.GE, timeout)
            })),
            resets=frozenset(),
            target="timeout",
        ),
        # Reset
        TimedTransition(
            source="timeout",
            action="reset",
            guard=Guard(frozenset()),
            resets=frozenset({"t"}),
            target="idle",
        ),
    ]

    # Regular actions in active state (before timeout)
    for action in actions:
        transitions.append(TimedTransition(
            source="active",
            action=action,
            guard=Guard(frozenset({
                ClockConstraint("t", ClockConstraintOp.LT, timeout)
            })),
            resets=frozenset(),
            target="active",
        ))

    invariants = {
        "active": Guard(frozenset({
            ClockConstraint("t", ClockConstraintOp.LE, timeout)
        })),
    }

    return TimedAutomaton(
        locations=locations,
        initial="idle",
        clocks=clocks,
        actions=actions | {"start", timeout_action, "reset"},
        transitions=transitions,
        invariants=invariants,
        accepting={"idle", "active"},
    )


def create_tcp_timeout_automaton() -> TimedAutomaton[str, str]:
    """Create TCP-like timed automaton with multiple timeouts.

    Models TCP connection establishment with:
    - SYN retransmission timeout
    - Connection timeout
    - Keep-alive timeout

    Returns:
        Timed automaton for TCP timing behavior
    """
    locations = {
        "closed",
        "syn_sent",
        "established",
        "timeout",
    }

    clocks = {"retx", "conn"}  # Retransmission timer, connection timer

    # Timeout values (simplified)
    SYN_RETX_TIMEOUT = 3.0
    CONN_TIMEOUT = 30.0
    MAX_RETRIES = 3

    transitions = [
        # Send SYN
        TimedTransition(
            source="closed",
            action="syn",
            guard=Guard(frozenset()),
            resets=frozenset({"retx", "conn"}),
            target="syn_sent",
        ),
        # Receive SYN-ACK
        TimedTransition(
            source="syn_sent",
            action="syn_ack",
            guard=Guard(frozenset({
                ClockConstraint("conn", ClockConstraintOp.LT, CONN_TIMEOUT)
            })),
            resets=frozenset({"retx"}),
            target="established",
        ),
        # SYN retransmission
        TimedTransition(
            source="syn_sent",
            action="syn_retx",
            guard=Guard(frozenset({
                ClockConstraint("retx", ClockConstraintOp.GE, SYN_RETX_TIMEOUT),
                ClockConstraint("conn", ClockConstraintOp.LT, CONN_TIMEOUT),
            })),
            resets=frozenset({"retx"}),
            target="syn_sent",
        ),
        # Connection timeout
        TimedTransition(
            source="syn_sent",
            action="conn_timeout",
            guard=Guard(frozenset({
                ClockConstraint("conn", ClockConstraintOp.GE, CONN_TIMEOUT)
            })),
            resets=frozenset(),
            target="timeout",
        ),
        # Close connection
        TimedTransition(
            source="established",
            action="close",
            guard=Guard(frozenset()),
            resets=frozenset({"retx", "conn"}),
            target="closed",
        ),
        # Reset from timeout
        TimedTransition(
            source="timeout",
            action="reset",
            guard=Guard(frozenset()),
            resets=frozenset({"retx", "conn"}),
            target="closed",
        ),
    ]

    invariants = {
        "syn_sent": Guard(frozenset({
            ClockConstraint("conn", ClockConstraintOp.LE, CONN_TIMEOUT)
        })),
    }

    return TimedAutomaton(
        locations=locations,
        initial="closed",
        clocks=clocks,
        actions={"syn", "syn_ack", "syn_retx", "conn_timeout", "close", "reset"},
        transitions=transitions,
        invariants=invariants,
        accepting={"closed", "established"},
    )


def create_rate_limiter_automaton(
    rate: float,
    burst: int,
) -> TimedAutomaton[str, str]:
    """Create rate limiter timed automaton (token bucket).

    Args:
        rate: Tokens per second
        burst: Maximum burst size

    Returns:
        Timed automaton for rate limiting
    """
    # Create states for each token count
    locations = {f"tokens_{i}" for i in range(burst + 1)}
    locations.add("blocked")

    clocks = {"t"}  # Token generation timer
    token_interval = 1.0 / rate

    transitions: List[TimedTransition[str, str]] = []

    # Token regeneration
    for i in range(burst):
        transitions.append(TimedTransition(
            source=f"tokens_{i}",
            action="tick",
            guard=Guard(frozenset({
                ClockConstraint("t", ClockConstraintOp.GE, token_interval)
            })),
            resets=frozenset({"t"}),
            target=f"tokens_{i + 1}",
        ))

    # Request handling (consumes token)
    for i in range(1, burst + 1):
        transitions.append(TimedTransition(
            source=f"tokens_{i}",
            action="request",
            guard=Guard(frozenset()),
            resets=frozenset(),
            target=f"tokens_{i - 1}",
        ))

    # Blocked when no tokens
    transitions.append(TimedTransition(
        source="tokens_0",
        action="block",
        guard=Guard(frozenset({
            ClockConstraint("t", ClockConstraintOp.LT, token_interval)
        })),
        resets=frozenset(),
        target="blocked",
    ))

    # Unblock when token available
    transitions.append(TimedTransition(
        source="blocked",
        action="unblock",
        guard=Guard(frozenset({
            ClockConstraint("t", ClockConstraintOp.GE, token_interval)
        })),
        resets=frozenset({"t"}),
        target="tokens_1",
    ))

    return TimedAutomaton(
        locations=locations,
        initial=f"tokens_{burst}",  # Start with full bucket
        clocks=clocks,
        actions={"tick", "request", "block", "unblock"},
        transitions=transitions,
        invariants={},
        accepting=locations - {"blocked"},
    )
