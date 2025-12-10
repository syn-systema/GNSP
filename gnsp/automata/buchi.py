"""Buchi automata for infinite words (omega-regular languages).

This module provides Buchi automata for modeling infinite behaviors
such as continuous network traffic patterns.

Features:
- Buchi automata construction and simulation
- Acceptance checking via cycle detection
- Product construction for LTL monitoring
"""

from dataclasses import dataclass
from typing import (
    Set,
    Dict,
    Tuple,
    List,
    Optional,
    FrozenSet,
    TypeVar,
    Generic,
)
from collections import deque

S = TypeVar("S")  # State type
A = TypeVar("A")  # Alphabet symbol type


@dataclass
class BuchiAutomaton(Generic[S, A]):
    """Buchi automaton for omega-regular languages.

    Accepts infinite words where some accepting state is visited
    infinitely often.

    A Buchi automaton B = (Q, Sigma, delta, q0, F) where:
    - Q: finite set of states
    - Sigma: finite alphabet
    - delta: Q x Sigma -> P(Q) transition relation
    - q0: initial state
    - F: set of accepting states

    An infinite word w is accepted iff there exists a run that visits
    F infinitely often.

    Attributes:
        states: Set of all states
        alphabet: Set of input symbols
        transitions: Transition relation (state, symbol) -> set of states
        initial: Initial state
        accepting: Set of accepting states
    """

    states: Set[S]
    alphabet: Set[A]
    transitions: Dict[Tuple[S, A], Set[S]]
    initial: S
    accepting: Set[S]

    def __post_init__(self) -> None:
        """Validate Buchi automaton."""
        if self.initial not in self.states:
            raise ValueError("Initial state must be in states")
        if not self.accepting.issubset(self.states):
            raise ValueError("Accepting states must be subset of states")

    def transition(self, state: S, symbol: A) -> Set[S]:
        """Get next states for given state and symbol.

        Args:
            state: Current state
            symbol: Input symbol

        Returns:
            Set of possible next states
        """
        return self.transitions.get((state, symbol), set())

    def run_finite(self, word: List[A]) -> List[Set[S]]:
        """Run automaton on finite prefix and return state sets.

        Args:
            word: Finite prefix

        Returns:
            List of state sets at each step
        """
        path = [{self.initial}]
        current = {self.initial}

        for symbol in word:
            next_states: Set[S] = set()
            for state in current:
                next_states.update(self.transition(state, symbol))
            current = next_states
            path.append(current)

            if not current:
                break

        return path

    def has_accepting_run(self, finite_prefix: List[A]) -> bool:
        """Check if automaton can have an accepting run from finite prefix.

        This checks if after reading the prefix, there exists a path
        that can reach an accepting state that has a cycle back to itself.

        Args:
            finite_prefix: Finite prefix of infinite word

        Returns:
            True if accepting run is possible
        """
        # Get states after reading prefix
        current = {self.initial}
        for symbol in finite_prefix:
            next_states: Set[S] = set()
            for state in current:
                next_states.update(self.transition(state, symbol))
            current = next_states
            if not current:
                return False

        # Check if any reachable accepting state has cycle to itself
        for start_state in current:
            if self._can_reach_accepting_cycle(start_state):
                return True

        return False

    def _can_reach_accepting_cycle(self, start: S) -> bool:
        """Check if accepting state reachable with cycle.

        Uses nested DFS to find accepting cycle reachable from start.
        """
        # First: find reachable accepting states
        reachable_accepting: Set[S] = set()
        visited: Set[S] = set()
        queue = deque([start])

        while queue:
            state = queue.popleft()
            if state in visited:
                continue
            visited.add(state)

            if state in self.accepting:
                reachable_accepting.add(state)

            for symbol in self.alphabet:
                for next_state in self.transition(state, symbol):
                    if next_state not in visited:
                        queue.append(next_state)

        # Check if any accepting state can reach itself
        for acc_state in reachable_accepting:
            if self._has_cycle(acc_state, acc_state):
                return True

        return False

    def _has_cycle(self, start: S, target: S) -> bool:
        """Check if there's a path from start back to target.

        Returns True if target is reachable from start (allowing target != start initially).
        """
        visited: Set[S] = set()
        queue = deque([start])
        first = True

        while queue:
            state = queue.popleft()

            # Check if we found target (not on first step)
            if state == target and not first:
                return True

            if state in visited:
                continue
            visited.add(state)
            first = False

            for symbol in self.alphabet:
                for next_state in self.transition(state, symbol):
                    if next_state not in visited or next_state == target:
                        queue.append(next_state)

        return False

    def is_empty(self) -> bool:
        """Check if language is empty (no accepting runs).

        Uses nested DFS algorithm.

        Returns:
            True if no infinite word is accepted
        """
        return not self._has_accepting_cycle_from_initial()

    def _has_accepting_cycle_from_initial(self) -> bool:
        """Check if there's an accepting cycle reachable from initial."""
        # Tarjan's algorithm or nested DFS
        visited_outer: Set[S] = set()
        stack = [self.initial]

        while stack:
            state = stack.pop()
            if state in visited_outer:
                continue
            visited_outer.add(state)

            # If accepting, check for cycle
            if state in self.accepting:
                if self._has_cycle(state, state):
                    return True

            # Add successors
            for symbol in self.alphabet:
                for next_state in self.transition(state, symbol):
                    if next_state not in visited_outer:
                        stack.append(next_state)

        return False


def product_buchi(
    buchi1: BuchiAutomaton[S, A],
    buchi2: BuchiAutomaton[S, A],
) -> BuchiAutomaton[Tuple[S, S, int], A]:
    """Product of two Buchi automata (intersection).

    Uses the standard product construction with acceptance tracking.

    Args:
        buchi1: First Buchi automaton
        buchi2: Second Buchi automaton

    Returns:
        Product Buchi automaton accepting L(buchi1) ∩ L(buchi2)
    """
    if buchi1.alphabet != buchi2.alphabet:
        raise ValueError("Automata must have same alphabet")

    # Product states: (s1, s2, flag) where flag ∈ {0, 1, 2}
    # flag tracks which accepting set we're waiting for
    new_states: Set[Tuple[S, S, int]] = set()
    new_transitions: Dict[Tuple[Tuple[S, S, int], A], Set[Tuple[S, S, int]]] = {}
    new_accepting: Set[Tuple[S, S, int]] = set()

    # BFS construction
    initial = (buchi1.initial, buchi2.initial, 0)
    queue = deque([initial])
    visited: Set[Tuple[S, S, int]] = set()

    while queue:
        state = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        new_states.add(state)

        s1, s2, flag = state

        # Determine accepting
        if flag == 2 and s2 in buchi2.accepting:
            new_accepting.add(state)

        # Compute transitions
        for symbol in buchi1.alphabet:
            next_s1_set = buchi1.transition(s1, symbol)
            next_s2_set = buchi2.transition(s2, symbol)

            for next_s1 in next_s1_set:
                for next_s2 in next_s2_set:
                    # Determine next flag
                    if flag == 0 and next_s1 in buchi1.accepting:
                        next_flag = 1
                    elif flag == 1 and next_s2 in buchi2.accepting:
                        next_flag = 2
                    elif flag == 2 and next_s2 in buchi2.accepting:
                        next_flag = 0  # Cycle complete
                    else:
                        next_flag = flag

                    next_state = (next_s1, next_s2, next_flag)

                    key = (state, symbol)
                    if key not in new_transitions:
                        new_transitions[key] = set()
                    new_transitions[key].add(next_state)

                    if next_state not in visited:
                        queue.append(next_state)

    return BuchiAutomaton(
        states=new_states,
        alphabet=buchi1.alphabet,
        transitions=new_transitions,
        initial=initial,
        accepting=new_accepting,
    )


def complement_buchi(buchi: BuchiAutomaton[S, A]) -> BuchiAutomaton[FrozenSet[S], A]:
    """Complement a Buchi automaton.

    Note: This is exponential in the number of states!

    Uses Safra's determinization followed by complementation.
    This is a simplified version using subset construction.

    Args:
        buchi: Buchi automaton to complement

    Returns:
        Buchi automaton accepting complement language
    """
    # Simplified: subset construction gives DBA, complement accepting states
    # This doesn't work for all Buchi automata but works for deterministic ones

    # First, determinize (approximately)
    new_states: Set[FrozenSet[S]] = set()
    new_transitions: Dict[Tuple[FrozenSet[S], A], Set[FrozenSet[S]]] = {}

    initial = frozenset({buchi.initial})
    queue = deque([initial])
    visited: Set[FrozenSet[S]] = set()

    while queue:
        state_set = queue.popleft()
        if state_set in visited:
            continue
        visited.add(state_set)
        new_states.add(state_set)

        for symbol in buchi.alphabet:
            next_set: Set[S] = set()
            for state in state_set:
                next_set.update(buchi.transition(state, symbol))

            if next_set:
                next_frozen = frozenset(next_set)
                key = (state_set, symbol)
                new_transitions[key] = {next_frozen}

                if next_frozen not in visited:
                    queue.append(next_frozen)

    # Accepting: states that DON'T contain any original accepting state
    # (This is a simplification and may not be correct for all cases)
    new_accepting = {
        state_set for state_set in new_states
        if not (state_set & buchi.accepting)
    }

    return BuchiAutomaton(
        states=new_states,
        alphabet=buchi.alphabet,
        transitions=new_transitions,
        initial=initial,
        accepting=new_accepting,
    )


class OnlineMonitor(Generic[S, A]):
    """Online monitor using Buchi automaton.

    Monitors a stream of events and tracks whether the property
    (or its negation) can still be satisfied.
    """

    def __init__(self, automaton: BuchiAutomaton[S, A]) -> None:
        """Initialize monitor.

        Args:
            automaton: Buchi automaton for property to monitor
        """
        self.automaton = automaton
        self.current_states: Set[S] = {automaton.initial}
        self.seen_accepting = False
        self.step_count = 0

    def step(self, symbol: A) -> str:
        """Process one symbol and return verdict.

        Args:
            symbol: Input symbol

        Returns:
            Verdict: "satisfied", "violated", or "unknown"
        """
        self.step_count += 1

        # Compute next states
        next_states: Set[S] = set()
        for state in self.current_states:
            next_states.update(self.automaton.transition(state, symbol))

        self.current_states = next_states

        # Check if accepting state visited
        if self.current_states & self.automaton.accepting:
            self.seen_accepting = True

        # Determine verdict
        if not self.current_states:
            return "violated"  # No valid continuation

        # Check if accepting cycle still possible
        can_accept = any(
            self.automaton._can_reach_accepting_cycle(s)
            for s in self.current_states
        )

        if not can_accept:
            return "violated"

        return "unknown"  # Still possible either way

    def reset(self) -> None:
        """Reset monitor to initial state."""
        self.current_states = {self.automaton.initial}
        self.seen_accepting = False
        self.step_count = 0


def create_safety_monitor(
    bad_states: Set[S],
    all_states: Set[S],
    alphabet: Set[A],
    transitions: Dict[Tuple[S, A], Set[S]],
    initial: S,
) -> BuchiAutomaton[S, A]:
    """Create Buchi automaton for safety property.

    Safety property: "bad states are never reached"

    Args:
        bad_states: States that should never be reached
        all_states: All states
        alphabet: Input alphabet
        transitions: Transition relation
        initial: Initial state

    Returns:
        Buchi automaton that accepts if property holds
    """
    # Good states are accepting (visited infinitely often if never bad)
    good_states = all_states - bad_states

    return BuchiAutomaton(
        states=all_states,
        alphabet=alphabet,
        transitions=transitions,
        initial=initial,
        accepting=good_states,
    )


def create_liveness_monitor(
    goal_states: Set[S],
    all_states: Set[S],
    alphabet: Set[A],
    transitions: Dict[Tuple[S, A], Set[S]],
    initial: S,
) -> BuchiAutomaton[S, A]:
    """Create Buchi automaton for liveness property.

    Liveness property: "goal is reached infinitely often"

    Args:
        goal_states: States that should be visited infinitely often
        all_states: All states
        alphabet: Input alphabet
        transitions: Transition relation
        initial: Initial state

    Returns:
        Buchi automaton that accepts if property holds
    """
    return BuchiAutomaton(
        states=all_states,
        alphabet=alphabet,
        transitions=transitions,
        initial=initial,
        accepting=goal_states,
    )
