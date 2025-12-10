"""Deterministic Finite Automata (DFA) implementation.

This module provides DFA operations for protocol modeling and
pattern recognition in network traffic analysis.

Features:
- Standard DFA construction and simulation
- Complement, intersection, union operations
- Hopcroft minimization algorithm
- Product construction for language operations
"""

from dataclasses import dataclass, field
from typing import (
    Optional,
    Set,
    Dict,
    FrozenSet,
    Tuple,
    List,
    Iterator,
    Callable,
    TypeVar,
    Generic,
)
from collections import deque

S = TypeVar("S")  # State type
A = TypeVar("A")  # Alphabet symbol type


@dataclass(frozen=True)
class DFAState:
    """Immutable DFA state identifier.

    Attributes:
        name: State name/identifier
    """

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class DFA(Generic[S, A]):
    """Deterministic Finite Automaton.

    A DFA M = (Q, Sigma, delta, q0, F) where:
    - Q: finite set of states
    - Sigma: finite alphabet
    - delta: Q x Sigma -> Q transition function
    - q0: initial state
    - F: set of accepting states

    Attributes:
        states: Set of all states
        alphabet: Set of input symbols
        transitions: Transition function as dict
        initial: Initial state
        accepting: Set of accepting states
    """

    states: Set[S]
    alphabet: Set[A]
    transitions: Dict[Tuple[S, A], S]
    initial: S
    accepting: Set[S]

    def __post_init__(self) -> None:
        """Validate DFA structure."""
        if self.initial not in self.states:
            raise ValueError("Initial state must be in states")
        if not self.accepting.issubset(self.states):
            raise ValueError("Accepting states must be subset of states")

    def transition(self, state: S, symbol: A) -> Optional[S]:
        """Get next state for given state and symbol.

        Args:
            state: Current state
            symbol: Input symbol

        Returns:
            Next state or None if transition undefined
        """
        return self.transitions.get((state, symbol))

    def accepts(self, word: List[A]) -> bool:
        """Check if DFA accepts the given word.

        Args:
            word: Sequence of input symbols

        Returns:
            True if word is accepted
        """
        current = self.initial

        for symbol in word:
            next_state = self.transition(current, symbol)
            if next_state is None:
                return False
            current = next_state

        return current in self.accepting

    def run(self, word: List[A]) -> List[S]:
        """Run DFA on word and return state sequence.

        Args:
            word: Sequence of input symbols

        Returns:
            List of states visited (including initial)
        """
        path = [self.initial]
        current = self.initial

        for symbol in word:
            next_state = self.transition(current, symbol)
            if next_state is None:
                break
            current = next_state
            path.append(current)

        return path

    def is_complete(self) -> bool:
        """Check if DFA is complete (all transitions defined).

        Returns:
            True if every (state, symbol) pair has a transition
        """
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transitions:
                    return False
        return True

    def complete(self, sink_state: S) -> "DFA[S, A]":
        """Return a complete DFA by adding sink state.

        Args:
            sink_state: State to use as sink

        Returns:
            Complete DFA with all transitions defined
        """
        if self.is_complete():
            return self

        new_states = self.states | {sink_state}
        new_transitions = dict(self.transitions)

        # Add missing transitions to sink
        for state in new_states:
            for symbol in self.alphabet:
                if (state, symbol) not in new_transitions:
                    new_transitions[(state, symbol)] = sink_state

        return DFA(
            states=new_states,
            alphabet=self.alphabet,
            transitions=new_transitions,
            initial=self.initial,
            accepting=self.accepting,
        )

    def complement(self) -> "DFA[S, A]":
        """Return DFA accepting complement language.

        Returns:
            DFA that accepts words not accepted by this DFA
        """
        # Must be complete for complement to work correctly
        if not self.is_complete():
            raise ValueError("DFA must be complete for complement")

        # Swap accepting and non-accepting states
        new_accepting = self.states - self.accepting

        return DFA(
            states=self.states,
            alphabet=self.alphabet,
            transitions=self.transitions,
            initial=self.initial,
            accepting=new_accepting,
        )

    def reachable_states(self) -> Set[S]:
        """Find all states reachable from initial state.

        Returns:
            Set of reachable states
        """
        visited: Set[S] = set()
        queue = deque([self.initial])

        while queue:
            state = queue.popleft()
            if state in visited:
                continue
            visited.add(state)

            for symbol in self.alphabet:
                next_state = self.transition(state, symbol)
                if next_state is not None and next_state not in visited:
                    queue.append(next_state)

        return visited

    def trim(self) -> "DFA[S, A]":
        """Remove unreachable states.

        Returns:
            DFA with only reachable states
        """
        reachable = self.reachable_states()

        new_transitions = {
            (s, a): t
            for (s, a), t in self.transitions.items()
            if s in reachable and t in reachable
        }

        return DFA(
            states=reachable,
            alphabet=self.alphabet,
            transitions=new_transitions,
            initial=self.initial,
            accepting=self.accepting & reachable,
        )

    def is_empty(self) -> bool:
        """Check if language is empty.

        Returns:
            True if no words are accepted
        """
        reachable = self.reachable_states()
        return len(reachable & self.accepting) == 0

    def accepts_empty(self) -> bool:
        """Check if empty word is accepted.

        Returns:
            True if initial state is accepting
        """
        return self.initial in self.accepting


def product_dfa(
    dfa1: DFA[S, A],
    dfa2: DFA[S, A],
    accept_condition: Callable[[bool, bool], bool],
) -> DFA[Tuple[S, S], A]:
    """Construct product automaton of two DFAs.

    Args:
        dfa1: First DFA
        dfa2: Second DFA
        accept_condition: Function (acc1, acc2) -> bool for accepting

    Returns:
        Product DFA
    """
    if dfa1.alphabet != dfa2.alphabet:
        raise ValueError("DFAs must have same alphabet")

    # Product states
    new_states: Set[Tuple[S, S]] = set()
    new_transitions: Dict[Tuple[Tuple[S, S], A], Tuple[S, S]] = {}
    new_accepting: Set[Tuple[S, S]] = set()

    # BFS to construct reachable product states
    initial = (dfa1.initial, dfa2.initial)
    queue = deque([initial])
    visited: Set[Tuple[S, S]] = set()

    while queue:
        state = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        new_states.add(state)

        s1, s2 = state

        # Check if accepting
        acc1 = s1 in dfa1.accepting
        acc2 = s2 in dfa2.accepting
        if accept_condition(acc1, acc2):
            new_accepting.add(state)

        # Compute transitions
        for symbol in dfa1.alphabet:
            next1 = dfa1.transition(s1, symbol)
            next2 = dfa2.transition(s2, symbol)

            if next1 is not None and next2 is not None:
                next_state = (next1, next2)
                new_transitions[(state, symbol)] = next_state

                if next_state not in visited:
                    queue.append(next_state)

    return DFA(
        states=new_states,
        alphabet=dfa1.alphabet,
        transitions=new_transitions,
        initial=initial,
        accepting=new_accepting,
    )


def intersection(dfa1: DFA[S, A], dfa2: DFA[S, A]) -> DFA[Tuple[S, S], A]:
    """Compute intersection of two DFA languages.

    Args:
        dfa1: First DFA
        dfa2: Second DFA

    Returns:
        DFA accepting L(dfa1) ∩ L(dfa2)
    """
    return product_dfa(dfa1, dfa2, lambda a, b: a and b)


def union(dfa1: DFA[S, A], dfa2: DFA[S, A]) -> DFA[Tuple[S, S], A]:
    """Compute union of two DFA languages.

    Args:
        dfa1: First DFA
        dfa2: Second DFA

    Returns:
        DFA accepting L(dfa1) ∪ L(dfa2)
    """
    return product_dfa(dfa1, dfa2, lambda a, b: a or b)


def symmetric_difference(dfa1: DFA[S, A], dfa2: DFA[S, A]) -> DFA[Tuple[S, S], A]:
    """Compute symmetric difference of two DFA languages.

    Args:
        dfa1: First DFA
        dfa2: Second DFA

    Returns:
        DFA accepting L(dfa1) △ L(dfa2)
    """
    return product_dfa(dfa1, dfa2, lambda a, b: a != b)


def minimize(dfa: DFA[S, A]) -> DFA[FrozenSet[S], A]:
    """Minimize DFA using Hopcroft's algorithm.

    Args:
        dfa: DFA to minimize

    Returns:
        Minimal equivalent DFA
    """
    # First trim unreachable states
    dfa = dfa.trim()

    if len(dfa.states) == 0:
        # Empty DFA
        empty_state: FrozenSet[S] = frozenset()
        return DFA(
            states={empty_state},
            alphabet=dfa.alphabet,
            transitions={},
            initial=empty_state,
            accepting=set(),
        )

    # Initial partition: accepting vs non-accepting
    accepting = frozenset(dfa.accepting)
    non_accepting = frozenset(dfa.states - dfa.accepting)

    partition: Set[FrozenSet[S]] = set()
    if accepting:
        partition.add(accepting)
    if non_accepting:
        partition.add(non_accepting)

    # Work list for refinement
    work_list: Set[FrozenSet[S]] = set(partition)

    # Hopcroft refinement
    while work_list:
        splitter = work_list.pop()

        for symbol in dfa.alphabet:
            # Find states that transition to splitter on symbol
            predecessors: Set[S] = set()
            for state in dfa.states:
                next_state = dfa.transition(state, symbol)
                if next_state in splitter:
                    predecessors.add(state)

            # Refine each block
            new_partition: Set[FrozenSet[S]] = set()
            for block in partition:
                intersection_set = block & predecessors
                difference = block - predecessors

                if intersection_set and difference:
                    # Block is split
                    new_partition.add(frozenset(intersection_set))
                    new_partition.add(frozenset(difference))

                    # Update work list
                    if block in work_list:
                        work_list.remove(block)
                        work_list.add(frozenset(intersection_set))
                        work_list.add(frozenset(difference))
                    else:
                        # Add smaller block
                        if len(intersection_set) <= len(difference):
                            work_list.add(frozenset(intersection_set))
                        else:
                            work_list.add(frozenset(difference))
                else:
                    new_partition.add(block)

            partition = new_partition

    # Build minimized DFA
    # Map each state to its equivalence class
    state_to_class: Dict[S, FrozenSet[S]] = {}
    for eq_class in partition:
        for state in eq_class:
            state_to_class[state] = eq_class

    # Find initial class and accepting classes
    initial_class = state_to_class[dfa.initial]
    accepting_classes = {
        state_to_class[s] for s in dfa.accepting if s in state_to_class
    }

    # Build transitions between equivalence classes
    new_transitions: Dict[Tuple[FrozenSet[S], A], FrozenSet[S]] = {}
    for eq_class in partition:
        # Pick representative state
        rep = next(iter(eq_class))
        for symbol in dfa.alphabet:
            next_state = dfa.transition(rep, symbol)
            if next_state is not None:
                next_class = state_to_class[next_state]
                new_transitions[(eq_class, symbol)] = next_class

    return DFA(
        states=partition,
        alphabet=dfa.alphabet,
        transitions=new_transitions,
        initial=initial_class,
        accepting=accepting_classes,
    )


def are_equivalent(dfa1: DFA[S, A], dfa2: DFA[S, A]) -> bool:
    """Check if two DFAs accept the same language.

    Args:
        dfa1: First DFA
        dfa2: Second DFA

    Returns:
        True if L(dfa1) = L(dfa2)
    """
    # L1 = L2 iff (L1 △ L2) is empty
    sym_diff = symmetric_difference(dfa1, dfa2)
    return sym_diff.is_empty()


def from_regex_simple(pattern: str, alphabet: Set[str]) -> DFA[int, str]:
    """Create simple DFA from basic pattern (no full regex).

    Supports only literal string matching.

    Args:
        pattern: Literal string pattern
        alphabet: Alphabet set

    Returns:
        DFA accepting exactly the pattern
    """
    n = len(pattern)
    states = set(range(n + 2))  # 0..n are pattern states, n+1 is sink

    transitions: Dict[Tuple[int, str], int] = {}

    # Build pattern transitions
    for i, char in enumerate(pattern):
        for symbol in alphabet:
            if symbol == char:
                transitions[(i, symbol)] = i + 1
            else:
                transitions[(i, symbol)] = n + 1  # Sink

    # Sink state loops
    for symbol in alphabet:
        transitions[(n, symbol)] = n + 1
        transitions[(n + 1, symbol)] = n + 1

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial=0,
        accepting={n},
    )


def create_tcp_state_machine() -> DFA[str, str]:
    """Create DFA for TCP connection state machine.

    Returns:
        DFA modeling TCP connection states
    """
    states = {
        "CLOSED",
        "LISTEN",
        "SYN_SENT",
        "SYN_RECEIVED",
        "ESTABLISHED",
        "FIN_WAIT_1",
        "FIN_WAIT_2",
        "CLOSE_WAIT",
        "CLOSING",
        "LAST_ACK",
        "TIME_WAIT",
    }

    alphabet = {
        "passive_open",
        "active_open",
        "syn",
        "syn_ack",
        "ack",
        "fin",
        "close",
        "timeout",
        "rst",
    }

    transitions: Dict[Tuple[str, str], str] = {
        # From CLOSED
        ("CLOSED", "passive_open"): "LISTEN",
        ("CLOSED", "active_open"): "SYN_SENT",
        # From LISTEN
        ("LISTEN", "syn"): "SYN_RECEIVED",
        ("LISTEN", "close"): "CLOSED",
        # From SYN_SENT
        ("SYN_SENT", "syn_ack"): "ESTABLISHED",
        ("SYN_SENT", "syn"): "SYN_RECEIVED",
        ("SYN_SENT", "close"): "CLOSED",
        # From SYN_RECEIVED
        ("SYN_RECEIVED", "ack"): "ESTABLISHED",
        ("SYN_RECEIVED", "close"): "FIN_WAIT_1",
        # From ESTABLISHED
        ("ESTABLISHED", "fin"): "CLOSE_WAIT",
        ("ESTABLISHED", "close"): "FIN_WAIT_1",
        # From FIN_WAIT_1
        ("FIN_WAIT_1", "ack"): "FIN_WAIT_2",
        ("FIN_WAIT_1", "fin"): "CLOSING",
        # From FIN_WAIT_2
        ("FIN_WAIT_2", "fin"): "TIME_WAIT",
        # From CLOSE_WAIT
        ("CLOSE_WAIT", "close"): "LAST_ACK",
        # From CLOSING
        ("CLOSING", "ack"): "TIME_WAIT",
        # From LAST_ACK
        ("LAST_ACK", "ack"): "CLOSED",
        # From TIME_WAIT
        ("TIME_WAIT", "timeout"): "CLOSED",
    }

    # Add RST transitions to CLOSED from most states
    for state in states - {"CLOSED", "LISTEN"}:
        transitions[(state, "rst")] = "CLOSED"

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial="CLOSED",
        accepting={"ESTABLISHED"},  # Accept when connection established
    )
