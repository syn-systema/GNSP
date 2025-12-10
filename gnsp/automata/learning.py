"""Automata learning algorithms.

This module provides algorithms for learning automata from observations:
- L* algorithm (Angluin) for learning DFAs
- TTT algorithm for efficient DFA learning
- Passive learning from positive/negative examples

Useful for learning protocol state machines from network traces.
"""

from dataclasses import dataclass, field
from typing import (
    Set,
    Dict,
    Tuple,
    List,
    Optional,
    Callable,
    Protocol,
    TypeVar,
    Generic,
    FrozenSet,
)
from abc import ABC, abstractmethod

from gnsp.automata.dfa import DFA

A = TypeVar("A")  # Alphabet symbol type


class MembershipOracle(Protocol[A]):
    """Oracle that answers membership queries.

    A membership query asks: "Is word w in the target language?"
    """

    def query(self, word: Tuple[A, ...]) -> bool:
        """Check if word is in target language.

        Args:
            word: Word to check

        Returns:
            True if word is accepted
        """
        ...


class EquivalenceOracle(Protocol[A]):
    """Oracle that answers equivalence queries.

    An equivalence query asks: "Is hypothesis H equivalent to target L?"
    If not, returns a counterexample.
    """

    def query(self, hypothesis: DFA[int, A]) -> Optional[Tuple[A, ...]]:
        """Check if hypothesis is equivalent to target.

        Args:
            hypothesis: DFA hypothesis

        Returns:
            None if equivalent, counterexample otherwise
        """
        ...


class DFAMembershipOracle(Generic[A]):
    """Membership oracle backed by a DFA."""

    def __init__(self, target: DFA[int, A]) -> None:
        """Initialize oracle.

        Args:
            target: Target DFA
        """
        self.target = target
        self.query_count = 0

    def query(self, word: Tuple[A, ...]) -> bool:
        """Check if word is accepted by target DFA."""
        self.query_count += 1
        return self.target.accepts(list(word))


class DFAEquivalenceOracle(Generic[A]):
    """Equivalence oracle using bounded exhaustive testing."""

    def __init__(
        self,
        target: DFA[int, A],
        max_length: int = 20,
    ) -> None:
        """Initialize oracle.

        Args:
            target: Target DFA
            max_length: Maximum counterexample length to search
        """
        self.target = target
        self.max_length = max_length
        self.query_count = 0

    def query(self, hypothesis: DFA[int, A]) -> Optional[Tuple[A, ...]]:
        """Find counterexample by exhaustive search."""
        self.query_count += 1

        alphabet = list(self.target.alphabet)

        # BFS over words
        queue: List[Tuple[A, ...]] = [()]
        visited: Set[Tuple[A, ...]] = {()}

        while queue:
            word = queue.pop(0)

            if len(word) > self.max_length:
                break

            # Check if classification differs
            target_accepts = self.target.accepts(list(word))
            hyp_accepts = hypothesis.accepts(list(word))

            if target_accepts != hyp_accepts:
                return word

            # Extend with all symbols
            for a in alphabet:
                new_word = word + (a,)
                if new_word not in visited:
                    visited.add(new_word)
                    queue.append(new_word)

        return None


@dataclass
class ObservationTable(Generic[A]):
    """Observation table for L* algorithm.

    Contains:
    - S: set of prefixes (row labels)
    - E: set of suffixes (column labels)
    - T: mapping from S x E to {+, -}

    Attributes:
        alphabet: Input alphabet
        prefixes: Set of row prefixes (S)
        suffixes: Set of column suffixes (E)
        table: Observation table mapping (prefix, suffix) -> bool
    """

    alphabet: Set[A]
    prefixes: Set[Tuple[A, ...]] = field(default_factory=set)
    suffixes: Set[Tuple[A, ...]] = field(default_factory=set)
    table: Dict[Tuple[Tuple[A, ...], Tuple[A, ...]], bool] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Initialize with empty word."""
        if not self.prefixes:
            self.prefixes = {()}
        if not self.suffixes:
            self.suffixes = {()}

    def row(self, prefix: Tuple[A, ...]) -> Tuple[bool, ...]:
        """Get row for prefix."""
        return tuple(
            self.table.get((prefix, suffix), False)
            for suffix in sorted(self.suffixes, key=lambda x: (len(x), x))
        )

    def is_closed(self) -> Optional[Tuple[A, ...]]:
        """Check if table is closed.

        Returns prefix that violates closedness, or None if closed.
        """
        prefix_rows = {self.row(s) for s in self.prefixes}

        for s in self.prefixes:
            for a in self.alphabet:
                sa = s + (a,)
                if self.row(sa) not in prefix_rows:
                    return sa

        return None

    def is_consistent(self) -> Optional[Tuple[Tuple[A, ...], Tuple[A, ...], A]]:
        """Check if table is consistent.

        Returns (s1, s2, a) that violates consistency, or None if consistent.
        """
        prefix_list = sorted(self.prefixes, key=lambda x: (len(x), x))

        for i, s1 in enumerate(prefix_list):
            for s2 in prefix_list[i + 1:]:
                if self.row(s1) == self.row(s2):
                    # Check if extended rows are equal
                    for a in self.alphabet:
                        if self.row(s1 + (a,)) != self.row(s2 + (a,)):
                            # Find distinguishing suffix
                            for e in self.suffixes:
                                s1_ae = (s1 + (a,), e)
                                s2_ae = (s2 + (a,), e)
                                if self.table.get(s1_ae) != self.table.get(s2_ae):
                                    return (s1, s2, a)

        return None

    def make_consistent(
        self,
        membership: MembershipOracle[A],
    ) -> Tuple[A, ...]:
        """Add suffix to make table consistent.

        Returns the added suffix.
        """
        result = self.is_consistent()
        if result is None:
            return ()

        s1, s2, a = result

        # Find distinguishing suffix
        for e in self.suffixes:
            s1_ae = (s1 + (a,), e)
            s2_ae = (s2 + (a,), e)
            if self.table.get(s1_ae) != self.table.get(s2_ae):
                new_suffix = (a,) + e
                self.suffixes.add(new_suffix)
                self._fill_table(membership)
                return new_suffix

        return ()

    def make_closed(
        self,
        membership: MembershipOracle[A],
    ) -> Tuple[A, ...]:
        """Add prefix to make table closed.

        Returns the added prefix.
        """
        sa = self.is_closed()
        if sa is None:
            return ()

        self.prefixes.add(sa)
        self._fill_table(membership)
        return sa

    def _fill_table(self, membership: MembershipOracle[A]) -> None:
        """Fill missing entries in observation table."""
        # All prefixes and their one-symbol extensions
        all_prefixes = set(self.prefixes)
        for s in self.prefixes:
            for a in self.alphabet:
                all_prefixes.add(s + (a,))

        # Query missing entries
        for prefix in all_prefixes:
            for suffix in self.suffixes:
                key = (prefix, suffix)
                if key not in self.table:
                    word = prefix + suffix
                    self.table[key] = membership.query(word)

    def build_hypothesis(self) -> DFA[int, A]:
        """Build DFA hypothesis from observation table."""
        # Group prefixes by their row
        row_to_state: Dict[Tuple[bool, ...], int] = {}
        state_counter = 0

        for prefix in sorted(self.prefixes, key=lambda x: (len(x), x)):
            row = self.row(prefix)
            if row not in row_to_state:
                row_to_state[row] = state_counter
                state_counter += 1

        # Build DFA
        states = set(range(state_counter))
        transitions: Dict[Tuple[int, A], int] = {}
        accepting: Set[int] = set()

        # Find initial state (row of empty word)
        initial = row_to_state[self.row(())]

        # Build transitions and find accepting states
        for prefix in self.prefixes:
            row = self.row(prefix)
            state = row_to_state[row]

            # Check if accepting
            if self.table.get((prefix, ()), False):
                accepting.add(state)

            # Build transitions
            for a in self.alphabet:
                next_row = self.row(prefix + (a,))
                if next_row in row_to_state:
                    next_state = row_to_state[next_row]
                    transitions[(state, a)] = next_state

        return DFA(
            states=states,
            alphabet=self.alphabet,
            transitions=transitions,
            initial=initial,
            accepting=accepting,
        )


class LStarLearner(Generic[A]):
    """L* algorithm for learning DFAs.

    The L* algorithm learns a minimal DFA from membership and
    equivalence queries.

    Reference: Angluin, D. (1987). Learning Regular Sets from Queries
    and Counterexamples.
    """

    def __init__(
        self,
        alphabet: Set[A],
        membership: MembershipOracle[A],
        equivalence: EquivalenceOracle[A],
    ) -> None:
        """Initialize L* learner.

        Args:
            alphabet: Input alphabet
            membership: Membership oracle
            equivalence: Equivalence oracle
        """
        self.alphabet = alphabet
        self.membership = membership
        self.equivalence = equivalence
        self.table = ObservationTable(alphabet=alphabet)

    def learn(self, max_iterations: int = 1000) -> DFA[int, A]:
        """Learn DFA using L* algorithm.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Learned DFA
        """
        # Initialize table
        self.table._fill_table(self.membership)

        for _ in range(max_iterations):
            # Make table closed and consistent
            while True:
                # Check consistency
                if self.table.is_consistent() is not None:
                    self.table.make_consistent(self.membership)
                    continue

                # Check closedness
                if self.table.is_closed() is not None:
                    self.table.make_closed(self.membership)
                    continue

                break

            # Build hypothesis
            hypothesis = self.table.build_hypothesis()

            # Check equivalence
            counterexample = self.equivalence.query(hypothesis)

            if counterexample is None:
                return hypothesis

            # Process counterexample
            self._process_counterexample(counterexample)

        # Return best hypothesis
        return self.table.build_hypothesis()

    def _process_counterexample(self, ce: Tuple[A, ...]) -> None:
        """Process counterexample by adding prefixes.

        Uses all prefixes of counterexample.
        """
        # Add all prefixes of counterexample
        for i in range(len(ce) + 1):
            prefix = ce[:i]
            if prefix not in self.table.prefixes:
                self.table.prefixes.add(prefix)

        self.table._fill_table(self.membership)


class RivestSchapireLearner(Generic[A]):
    """L* with Rivest-Schapire counterexample processing.

    Uses binary search on counterexamples for more efficient learning.
    """

    def __init__(
        self,
        alphabet: Set[A],
        membership: MembershipOracle[A],
        equivalence: EquivalenceOracle[A],
    ) -> None:
        """Initialize learner.

        Args:
            alphabet: Input alphabet
            membership: Membership oracle
            equivalence: Equivalence oracle
        """
        self.alphabet = alphabet
        self.membership = membership
        self.equivalence = equivalence
        self.table = ObservationTable(alphabet=alphabet)

    def learn(self, max_iterations: int = 1000) -> DFA[int, A]:
        """Learn DFA using L* with RS counterexample processing.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Learned DFA
        """
        self.table._fill_table(self.membership)

        for _ in range(max_iterations):
            while True:
                if self.table.is_consistent() is not None:
                    self.table.make_consistent(self.membership)
                    continue
                if self.table.is_closed() is not None:
                    self.table.make_closed(self.membership)
                    continue
                break

            hypothesis = self.table.build_hypothesis()
            counterexample = self.equivalence.query(hypothesis)

            if counterexample is None:
                return hypothesis

            self._process_counterexample_rs(counterexample, hypothesis)

        return self.table.build_hypothesis()

    def _process_counterexample_rs(
        self,
        ce: Tuple[A, ...],
        hypothesis: DFA[int, A],
    ) -> None:
        """Process counterexample using Rivest-Schapire binary search."""
        n = len(ce)

        # Binary search for breakpoint
        low, high = 0, n

        while high - low > 1:
            mid = (low + high) // 2

            # Compute q = delta*(q0, ce[:mid])
            prefix = ce[:mid]
            suffix = ce[mid:]

            # Check if hypothesis and target agree on prefix classification
            # by checking membership of prefix + various suffixes
            prefix_state = hypothesis.initial
            for symbol in prefix:
                next_state = hypothesis.transition(prefix_state, symbol)
                if next_state is None:
                    break
                prefix_state = next_state

            # Get access string for this state
            access_string = self._get_access_string(prefix_state, hypothesis)

            # Check if access_string + suffix classification matches prefix + suffix
            target_accepts_ce = self.membership.query(ce)
            target_accepts_access = self.membership.query(access_string + suffix)

            if target_accepts_access == target_accepts_ce:
                low = mid
            else:
                high = mid

        # Add distinguishing suffix
        suffix = ce[high:]
        if suffix and suffix not in self.table.suffixes:
            self.table.suffixes.add(suffix)

        # Also add prefixes
        for i in range(len(ce) + 1):
            prefix = ce[:i]
            if prefix not in self.table.prefixes:
                self.table.prefixes.add(prefix)

        self.table._fill_table(self.membership)

    def _get_access_string(
        self,
        state: int,
        hypothesis: DFA[int, A],
    ) -> Tuple[A, ...]:
        """Get access string (shortest path from initial to state)."""
        if state == hypothesis.initial:
            return ()

        # BFS to find shortest path
        from collections import deque
        queue: deque = deque([(hypothesis.initial, ())])
        visited = {hypothesis.initial}

        while queue:
            current, path = queue.popleft()

            for symbol in hypothesis.alphabet:
                next_state = hypothesis.transition(current, symbol)
                if next_state is None:
                    continue

                new_path = path + (symbol,)

                if next_state == state:
                    return new_path

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, new_path))

        return ()  # State not reachable


class PassiveLearner(Generic[A]):
    """Passive DFA learning from positive and negative examples.

    Uses state merging approach (RPNI-like algorithm).
    """

    def __init__(self, alphabet: Set[A]) -> None:
        """Initialize learner.

        Args:
            alphabet: Input alphabet
        """
        self.alphabet = alphabet

    def learn(
        self,
        positive: List[Tuple[A, ...]],
        negative: List[Tuple[A, ...]],
    ) -> DFA[int, A]:
        """Learn DFA from labeled examples.

        Args:
            positive: Positive examples (accepted words)
            negative: Negative examples (rejected words)

        Returns:
            Learned DFA
        """
        # Build prefix tree acceptor (PTA)
        pta = self._build_pta(positive)

        # Try to merge states while preserving consistency
        negative_set = set(negative)

        # Get states in breadth-first order
        states = self._bfs_order(pta)

        for i, q in enumerate(states):
            for q_prime in states[:i]:
                # Try to merge q into q_prime
                merged = self._try_merge(pta, q, q_prime, negative_set)
                if merged is not None:
                    pta = merged
                    break

        return pta

    def _build_pta(self, positive: List[Tuple[A, ...]]) -> DFA[int, A]:
        """Build prefix tree acceptor from positive examples."""
        states = {0}
        transitions: Dict[Tuple[int, A], int] = {}
        accepting = set()
        state_counter = 1

        for word in positive:
            current = 0

            for symbol in word:
                key = (current, symbol)
                if key not in transitions:
                    states.add(state_counter)
                    transitions[key] = state_counter
                    state_counter += 1
                current = transitions[key]

            accepting.add(current)

        return DFA(
            states=states,
            alphabet=self.alphabet,
            transitions=transitions,
            initial=0,
            accepting=accepting,
        )

    def _bfs_order(self, dfa: DFA[int, A]) -> List[int]:
        """Get states in BFS order from initial state."""
        from collections import deque
        order = []
        visited = set()
        queue: deque = deque([dfa.initial])

        while queue:
            state = queue.popleft()
            if state in visited:
                continue
            visited.add(state)
            order.append(state)

            for symbol in dfa.alphabet:
                next_state = dfa.transition(state, symbol)
                if next_state is not None and next_state not in visited:
                    queue.append(next_state)

        return order

    def _try_merge(
        self,
        dfa: DFA[int, A],
        q1: int,
        q2: int,
        negative: Set[Tuple[A, ...]],
    ) -> Optional[DFA[int, A]]:
        """Try to merge q1 into q2.

        Returns merged DFA if consistent, None otherwise.
        """
        # Create mapping from old states to new states
        state_map = {s: s for s in dfa.states}
        state_map[q1] = q2

        # Build merged DFA
        new_states = {s for s in dfa.states if s != q1}
        new_transitions: Dict[Tuple[int, A], int] = {}

        for (src, symbol), dst in dfa.transitions.items():
            new_src = state_map[src]
            new_dst = state_map[dst]
            if new_src in new_states:
                key = (new_src, symbol)
                if key in new_transitions and new_transitions[key] != new_dst:
                    # Conflict - need to merge recursively
                    return None
                new_transitions[key] = new_dst

        new_accepting = {state_map[s] for s in dfa.accepting if state_map[s] in new_states}

        merged = DFA(
            states=new_states,
            alphabet=dfa.alphabet,
            transitions=new_transitions,
            initial=state_map[dfa.initial],
            accepting=new_accepting,
        )

        # Check consistency with negative examples
        for word in negative:
            if merged.accepts(list(word)):
                return None

        return merged


def learn_from_traces(
    traces: List[List[str]],
    labels: List[bool],
    alphabet: Optional[Set[str]] = None,
) -> DFA[int, str]:
    """Learn DFA from labeled traces.

    Convenience function for learning from network traces.

    Args:
        traces: List of traces (each trace is list of symbols)
        labels: Label for each trace (True = positive, False = negative)
        alphabet: Alphabet (inferred from traces if not provided)

    Returns:
        Learned DFA
    """
    # Infer alphabet if not provided
    if alphabet is None:
        alphabet = set()
        for trace in traces:
            alphabet.update(trace)

    # Separate positive and negative examples
    positive = [tuple(t) for t, label in zip(traces, labels) if label]
    negative = [tuple(t) for t, label in zip(traces, labels) if not label]

    # Learn using passive learner
    learner = PassiveLearner(alphabet)
    return learner.learn(positive, negative)


def learn_protocol_automaton(
    oracle: Callable[[List[str]], bool],
    alphabet: Set[str],
    max_length: int = 10,
) -> DFA[int, str]:
    """Learn protocol automaton from oracle.

    Uses L* algorithm with the provided oracle.

    Args:
        oracle: Function that returns True if sequence is valid
        alphabet: Protocol alphabet
        max_length: Maximum word length for equivalence testing

    Returns:
        Learned DFA
    """

    class OracleWrapper:
        def __init__(self, oracle: Callable[[List[str]], bool]) -> None:
            self.oracle = oracle
            self.query_count = 0

        def query(self, word: Tuple[str, ...]) -> bool:
            self.query_count += 1
            return self.oracle(list(word))

    class ExhaustiveEquivalence:
        def __init__(
            self,
            oracle: Callable[[List[str]], bool],
            alphabet: Set[str],
            max_length: int,
        ) -> None:
            self.oracle = oracle
            self.alphabet = alphabet
            self.max_length = max_length

        def query(self, hypothesis: DFA[int, str]) -> Optional[Tuple[str, ...]]:
            from collections import deque

            queue: deque = deque([()])
            visited: Set[Tuple[str, ...]] = {()}

            while queue:
                word = queue.popleft()
                if len(word) > self.max_length:
                    continue

                target = self.oracle(list(word))
                hyp = hypothesis.accepts(list(word))

                if target != hyp:
                    return word

                for a in self.alphabet:
                    new_word = word + (a,)
                    if new_word not in visited:
                        visited.add(new_word)
                        queue.append(new_word)

            return None

    membership = OracleWrapper(oracle)
    equivalence = ExhaustiveEquivalence(oracle, alphabet, max_length)

    learner = LStarLearner(alphabet, membership, equivalence)
    return learner.learn()
