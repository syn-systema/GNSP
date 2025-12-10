"""Weighted automata with semiring operations.

This module provides weighted finite automata using various semirings:
- Probability semiring for probabilistic modeling
- Tropical semiring for shortest path computation
- Viterbi semiring for most likely path
- Boolean semiring (standard automata)

Useful for anomaly scoring and path analysis in network traffic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Generic,
    TypeVar,
    Set,
    Dict,
    Tuple,
    List,
    Optional,
)
import math

W = TypeVar("W")  # Weight type
S = TypeVar("S")  # State type
A = TypeVar("A")  # Alphabet symbol type


class Semiring(ABC, Generic[W]):
    """Abstract semiring (K, +, *, 0, 1).

    A semiring provides:
    - Addition (+ or ⊕): combines weights from different paths
    - Multiplication (* or ⊗): combines weights along a path
    - Zero (0): identity for addition, annihilator for multiplication
    - One (1): identity for multiplication
    """

    @abstractmethod
    def zero(self) -> W:
        """Additive identity."""
        pass

    @abstractmethod
    def one(self) -> W:
        """Multiplicative identity."""
        pass

    @abstractmethod
    def add(self, a: W, b: W) -> W:
        """Semiring addition."""
        pass

    @abstractmethod
    def multiply(self, a: W, b: W) -> W:
        """Semiring multiplication."""
        pass

    def is_zero(self, w: W) -> bool:
        """Check if weight is zero."""
        return w == self.zero()


class BooleanSemiring(Semiring[bool]):
    """Boolean semiring (∨, ∧, False, True).

    Standard automata acceptor behavior.
    """

    def zero(self) -> bool:
        return False

    def one(self) -> bool:
        return True

    def add(self, a: bool, b: bool) -> bool:
        return a or b

    def multiply(self, a: bool, b: bool) -> bool:
        return a and b


class ProbabilitySemiring(Semiring[float]):
    """Probability semiring (+, *, 0, 1).

    Sum of probabilities over paths.
    """

    def zero(self) -> float:
        return 0.0

    def one(self) -> float:
        return 1.0

    def add(self, a: float, b: float) -> float:
        return a + b

    def multiply(self, a: float, b: float) -> float:
        return a * b


class LogProbabilitySemiring(Semiring[float]):
    """Log probability semiring (log-sum-exp, +, -inf, 0).

    Numerically stable probability computations.
    """

    def zero(self) -> float:
        return float("-inf")

    def one(self) -> float:
        return 0.0

    def add(self, a: float, b: float) -> float:
        """Log-sum-exp for numerical stability."""
        if a == float("-inf"):
            return b
        if b == float("-inf"):
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    def multiply(self, a: float, b: float) -> float:
        return a + b

    def is_zero(self, w: float) -> bool:
        return w == float("-inf")


class TropicalSemiring(Semiring[float]):
    """Tropical (min-plus) semiring (min, +, inf, 0).

    Shortest path computation.
    """

    def zero(self) -> float:
        return float("inf")

    def one(self) -> float:
        return 0.0

    def add(self, a: float, b: float) -> float:
        return min(a, b)

    def multiply(self, a: float, b: float) -> float:
        return a + b

    def is_zero(self, w: float) -> bool:
        return w == float("inf")


class MaxTropicalSemiring(Semiring[float]):
    """Max-plus tropical semiring (max, +, -inf, 0).

    Longest/widest path computation.
    """

    def zero(self) -> float:
        return float("-inf")

    def one(self) -> float:
        return 0.0

    def add(self, a: float, b: float) -> float:
        return max(a, b)

    def multiply(self, a: float, b: float) -> float:
        return a + b


class ViterbiSemiring(Semiring[float]):
    """Viterbi semiring (max, *, 0, 1).

    Most likely path computation.
    """

    def zero(self) -> float:
        return 0.0

    def one(self) -> float:
        return 1.0

    def add(self, a: float, b: float) -> float:
        return max(a, b)

    def multiply(self, a: float, b: float) -> float:
        return a * b


class CountingSemiring(Semiring[int]):
    """Counting semiring (+, *, 0, 1).

    Counts number of accepting paths.
    """

    def zero(self) -> int:
        return 0

    def one(self) -> int:
        return 1

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b


@dataclass
class WeightedAutomaton(Generic[S, A, W]):
    """Weighted Finite Automaton over a semiring.

    A WFA M = (Q, Sigma, K, delta, lambda, rho) where:
    - Q: finite set of states
    - Sigma: finite alphabet
    - K: semiring
    - delta: Q x Sigma x Q -> K transition weights
    - lambda: Q -> K initial weights
    - rho: Q -> K final weights

    Attributes:
        states: Set of all states
        alphabet: Set of input symbols
        semiring: Semiring for weight operations
        transitions: Transition weights (src, symbol, dst) -> weight
        initial_weights: Initial weight for each state
        final_weights: Final weight for each state
    """

    states: Set[S]
    alphabet: Set[A]
    semiring: Semiring[W]
    transitions: Dict[Tuple[S, A, S], W]
    initial_weights: Dict[S, W]
    final_weights: Dict[S, W]

    def get_transition_weight(self, src: S, symbol: A, dst: S) -> W:
        """Get weight of transition."""
        return self.transitions.get(
            (src, symbol, dst),
            self.semiring.zero()
        )

    def get_initial_weight(self, state: S) -> W:
        """Get initial weight of state."""
        return self.initial_weights.get(state, self.semiring.zero())

    def get_final_weight(self, state: S) -> W:
        """Get final weight of state."""
        return self.final_weights.get(state, self.semiring.zero())

    def weight(self, word: List[A]) -> W:
        """Compute weight of word (sum over all paths).

        Args:
            word: Input word

        Returns:
            Total weight of word
        """
        # Forward algorithm
        # forward[state] = total weight to reach state
        forward: Dict[S, W] = {}

        # Initialize with initial weights
        for state in self.states:
            forward[state] = self.get_initial_weight(state)

        # Process each symbol
        for symbol in word:
            new_forward: Dict[S, W] = {}

            for dst in self.states:
                total = self.semiring.zero()

                for src in self.states:
                    if not self.semiring.is_zero(forward.get(src, self.semiring.zero())):
                        trans_weight = self.get_transition_weight(src, symbol, dst)
                        if not self.semiring.is_zero(trans_weight):
                            path_weight = self.semiring.multiply(
                                forward[src], trans_weight
                            )
                            total = self.semiring.add(total, path_weight)

                new_forward[dst] = total

            forward = new_forward

        # Combine with final weights
        result = self.semiring.zero()
        for state in self.states:
            if not self.semiring.is_zero(forward.get(state, self.semiring.zero())):
                final_weight = self.get_final_weight(state)
                if not self.semiring.is_zero(final_weight):
                    path_weight = self.semiring.multiply(forward[state], final_weight)
                    result = self.semiring.add(result, path_weight)

        return result

    def viterbi_path(self, word: List[A]) -> Tuple[W, List[S]]:
        """Find most likely path using Viterbi algorithm.

        Only works with Viterbi-like semirings (max-based).

        Args:
            word: Input word

        Returns:
            Tuple of (best weight, best path)
        """
        # forward[state] = (best weight to reach state, backpointer)
        forward: Dict[S, Tuple[W, Optional[S]]] = {}

        # Initialize
        for state in self.states:
            init_w = self.get_initial_weight(state)
            forward[state] = (init_w, None)

        backpointers: List[Dict[S, Optional[S]]] = []

        # Process each symbol
        for symbol in word:
            new_forward: Dict[S, Tuple[W, Optional[S]]] = {}
            current_backpointers: Dict[S, Optional[S]] = {}

            for dst in self.states:
                best_weight = self.semiring.zero()
                best_src: Optional[S] = None

                for src in self.states:
                    src_weight, _ = forward.get(src, (self.semiring.zero(), None))
                    if not self.semiring.is_zero(src_weight):
                        trans_weight = self.get_transition_weight(src, symbol, dst)
                        if not self.semiring.is_zero(trans_weight):
                            path_weight = self.semiring.multiply(src_weight, trans_weight)
                            # For Viterbi, add = max, so we track the argmax
                            if self.semiring.is_zero(best_weight) or path_weight > best_weight:
                                best_weight = path_weight
                                best_src = src

                new_forward[dst] = (best_weight, best_src)
                current_backpointers[dst] = best_src

            forward = new_forward
            backpointers.append(current_backpointers)

        # Find best final state
        best_final_weight = self.semiring.zero()
        best_final_state: Optional[S] = None

        for state in self.states:
            state_weight, _ = forward.get(state, (self.semiring.zero(), None))
            if not self.semiring.is_zero(state_weight):
                final_weight = self.get_final_weight(state)
                if not self.semiring.is_zero(final_weight):
                    total = self.semiring.multiply(state_weight, final_weight)
                    if self.semiring.is_zero(best_final_weight) or total > best_final_weight:
                        best_final_weight = total
                        best_final_state = state

        # Reconstruct path
        if best_final_state is None:
            return (self.semiring.zero(), [])

        path = [best_final_state]
        current = best_final_state

        for bp in reversed(backpointers):
            prev = bp.get(current)
            if prev is not None:
                path.append(prev)
                current = prev

        path.reverse()
        return (best_final_weight, path)


def create_probabilistic_automaton(
    states: Set[S],
    alphabet: Set[A],
    transitions: Dict[Tuple[S, A, S], float],
    initial_probs: Dict[S, float],
    final_probs: Dict[S, float],
) -> WeightedAutomaton[S, A, float]:
    """Create weighted automaton with probability semiring.

    Args:
        states: Set of states
        alphabet: Input alphabet
        transitions: Transition probabilities
        initial_probs: Initial state probabilities
        final_probs: Final state probabilities

    Returns:
        Weighted automaton
    """
    return WeightedAutomaton(
        states=states,
        alphabet=alphabet,
        semiring=ProbabilitySemiring(),
        transitions=transitions,
        initial_weights=initial_probs,
        final_weights=final_probs,
    )


def create_tropical_automaton(
    states: Set[S],
    alphabet: Set[A],
    transitions: Dict[Tuple[S, A, S], float],
    initial_costs: Dict[S, float],
    final_costs: Dict[S, float],
) -> WeightedAutomaton[S, A, float]:
    """Create weighted automaton with tropical semiring.

    Weights represent costs; weight() gives shortest path cost.

    Args:
        states: Set of states
        alphabet: Input alphabet
        transitions: Transition costs
        initial_costs: Initial state costs
        final_costs: Final state costs

    Returns:
        Weighted automaton for shortest path
    """
    return WeightedAutomaton(
        states=states,
        alphabet=alphabet,
        semiring=TropicalSemiring(),
        transitions=transitions,
        initial_weights=initial_costs,
        final_weights=final_costs,
    )


def create_counting_automaton(
    states: Set[S],
    alphabet: Set[A],
    transitions: Dict[Tuple[S, A, S], int],
    initial_counts: Dict[S, int],
    final_counts: Dict[S, int],
) -> WeightedAutomaton[S, A, int]:
    """Create weighted automaton with counting semiring.

    weight() gives number of accepting paths.

    Args:
        states: Set of states
        alphabet: Input alphabet
        transitions: Transition multiplicities
        initial_counts: Initial state counts
        final_counts: Final state counts

    Returns:
        Weighted automaton for path counting
    """
    return WeightedAutomaton(
        states=states,
        alphabet=alphabet,
        semiring=CountingSemiring(),
        transitions=transitions,
        initial_weights=initial_counts,
        final_weights=final_counts,
    )


class AnomalyScoringAutomaton:
    """Specialized weighted automaton for anomaly scoring.

    Uses log-probability semiring for numerical stability and
    outputs anomaly score (negative log likelihood).
    """

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transitions: Dict[Tuple[str, str, str], float],
        initial_probs: Dict[str, float],
        final_probs: Dict[str, float],
    ) -> None:
        """Initialize anomaly scoring automaton.

        Args:
            states: Set of states
            alphabet: Input alphabet
            transitions: Transition probabilities (will be converted to log)
            initial_probs: Initial state probabilities
            final_probs: Final state probabilities
        """
        # Convert to log probabilities
        log_transitions = {
            k: math.log(v) if v > 0 else float("-inf")
            for k, v in transitions.items()
        }
        log_initial = {
            k: math.log(v) if v > 0 else float("-inf")
            for k, v in initial_probs.items()
        }
        log_final = {
            k: math.log(v) if v > 0 else float("-inf")
            for k, v in final_probs.items()
        }

        self.wfa = WeightedAutomaton(
            states=states,
            alphabet=alphabet,
            semiring=LogProbabilitySemiring(),
            transitions=log_transitions,
            initial_weights=log_initial,
            final_weights=log_final,
        )

        self._unknown_symbol_penalty = -10.0  # Log prob for unknown symbols

    def anomaly_score(self, sequence: List[str]) -> float:
        """Compute anomaly score for sequence.

        Higher score = more anomalous.

        Args:
            sequence: Sequence of symbols

        Returns:
            Anomaly score (negative log probability)
        """
        # Filter to known symbols
        filtered = []
        unknown_count = 0
        for symbol in sequence:
            if symbol in self.wfa.alphabet:
                filtered.append(symbol)
            else:
                unknown_count += 1

        log_prob = self.wfa.weight(filtered)

        # Add penalty for unknown symbols
        log_prob += unknown_count * self._unknown_symbol_penalty

        # Return negative log prob (higher = more anomalous)
        if log_prob == float("-inf"):
            return float("inf")
        return -log_prob

    def is_anomalous(self, sequence: List[str], threshold: float) -> bool:
        """Check if sequence is anomalous.

        Args:
            sequence: Sequence of symbols
            threshold: Anomaly threshold

        Returns:
            True if anomaly score exceeds threshold
        """
        return self.anomaly_score(sequence) > threshold
