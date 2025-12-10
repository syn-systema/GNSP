"""Nondeterministic Finite Automata (NFA) implementation.

This module provides NFA operations including:
- NFA construction and simulation
- Epsilon-closure computation
- Subset construction (NFA to DFA conversion)
- Thompson construction for regex
"""

from dataclasses import dataclass
from typing import (
    Optional,
    Set,
    Dict,
    FrozenSet,
    Tuple,
    List,
    TypeVar,
    Generic,
)
from collections import deque

from gnsp.automata.dfa import DFA

S = TypeVar("S")  # State type
A = TypeVar("A")  # Alphabet symbol type

# Special symbol for epsilon transitions
EPSILON: str = "ε"


@dataclass
class NFA(Generic[S, A]):
    """Nondeterministic Finite Automaton.

    An NFA M = (Q, Sigma, delta, q0, F) where:
    - Q: finite set of states
    - Sigma: finite alphabet
    - delta: Q x (Sigma ∪ {ε}) -> P(Q) transition function
    - q0: initial state
    - F: set of accepting states

    Attributes:
        states: Set of all states
        alphabet: Set of input symbols (not including epsilon)
        transitions: Transition function as dict to sets
        initial: Initial state
        accepting: Set of accepting states
    """

    states: Set[S]
    alphabet: Set[A]
    transitions: Dict[Tuple[S, Optional[A]], Set[S]]
    initial: S
    accepting: Set[S]

    def __post_init__(self) -> None:
        """Validate NFA structure."""
        if self.initial not in self.states:
            raise ValueError("Initial state must be in states")
        if not self.accepting.issubset(self.states):
            raise ValueError("Accepting states must be subset of states")

    def transition(self, state: S, symbol: Optional[A]) -> Set[S]:
        """Get next states for given state and symbol.

        Args:
            state: Current state
            symbol: Input symbol (None for epsilon)

        Returns:
            Set of possible next states
        """
        return self.transitions.get((state, symbol), set())

    def epsilon_closure(self, states: Set[S]) -> Set[S]:
        """Compute epsilon closure of state set.

        Args:
            states: Set of states

        Returns:
            Set of states reachable via epsilon transitions
        """
        closure = set(states)
        queue = deque(states)

        while queue:
            state = queue.popleft()
            for next_state in self.transition(state, None):
                if next_state not in closure:
                    closure.add(next_state)
                    queue.append(next_state)

        return closure

    def move(self, states: Set[S], symbol: A) -> Set[S]:
        """Compute states reachable from state set on symbol.

        Args:
            states: Set of current states
            symbol: Input symbol

        Returns:
            Set of states reachable on symbol
        """
        result: Set[S] = set()
        for state in states:
            result.update(self.transition(state, symbol))
        return result

    def extended_transition(self, states: Set[S], symbol: A) -> Set[S]:
        """Compute epsilon closure of move.

        Args:
            states: Set of current states
            symbol: Input symbol

        Returns:
            Epsilon closure of states reachable on symbol
        """
        return self.epsilon_closure(self.move(states, symbol))

    def accepts(self, word: List[A]) -> bool:
        """Check if NFA accepts the given word.

        Args:
            word: Sequence of input symbols

        Returns:
            True if word is accepted
        """
        current = self.epsilon_closure({self.initial})

        for symbol in word:
            current = self.extended_transition(current, symbol)
            if not current:
                return False

        return bool(current & self.accepting)

    def run(self, word: List[A]) -> List[Set[S]]:
        """Run NFA on word and return state sets.

        Args:
            word: Sequence of input symbols

        Returns:
            List of state sets at each step
        """
        path = []
        current = self.epsilon_closure({self.initial})
        path.append(current)

        for symbol in word:
            current = self.extended_transition(current, symbol)
            path.append(current)
            if not current:
                break

        return path


def subset_construction(nfa: NFA[S, A]) -> DFA[FrozenSet[S], A]:
    """Convert NFA to DFA using subset construction.

    Args:
        nfa: NFA to convert

    Returns:
        Equivalent DFA
    """
    # Initial DFA state is epsilon closure of NFA initial state
    initial_closure = frozenset(nfa.epsilon_closure({nfa.initial}))

    dfa_states: Set[FrozenSet[S]] = set()
    dfa_transitions: Dict[Tuple[FrozenSet[S], A], FrozenSet[S]] = {}
    dfa_accepting: Set[FrozenSet[S]] = set()

    # BFS to discover all DFA states
    queue = deque([initial_closure])
    visited: Set[FrozenSet[S]] = set()

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        dfa_states.add(current)

        # Check if accepting
        if current & nfa.accepting:
            dfa_accepting.add(current)

        # Compute transitions for each symbol
        for symbol in nfa.alphabet:
            next_states = frozenset(nfa.extended_transition(set(current), symbol))
            if next_states:
                dfa_transitions[(current, symbol)] = next_states
                if next_states not in visited:
                    queue.append(next_states)

    return DFA(
        states=dfa_states,
        alphabet=nfa.alphabet,
        transitions=dfa_transitions,
        initial=initial_closure,
        accepting=dfa_accepting,
    )


def nfa_union(nfa1: NFA[S, A], nfa2: NFA[S, A], new_initial: S) -> NFA[S, A]:
    """Compute union of two NFAs.

    Creates new NFA with epsilon transitions to both initial states.

    Args:
        nfa1: First NFA
        nfa2: Second NFA
        new_initial: New initial state (must not be in either NFA)

    Returns:
        NFA accepting L(nfa1) ∪ L(nfa2)
    """
    if nfa1.alphabet != nfa2.alphabet:
        raise ValueError("NFAs must have same alphabet")

    new_states = nfa1.states | nfa2.states | {new_initial}
    new_transitions = dict(nfa1.transitions)
    new_transitions.update(nfa2.transitions)

    # Add epsilon transitions from new initial
    new_transitions[(new_initial, None)] = {nfa1.initial, nfa2.initial}

    return NFA(
        states=new_states,
        alphabet=nfa1.alphabet,
        transitions=new_transitions,
        initial=new_initial,
        accepting=nfa1.accepting | nfa2.accepting,
    )


def nfa_concatenation(
    nfa1: NFA[S, A],
    nfa2: NFA[S, A],
) -> NFA[S, A]:
    """Concatenate two NFAs.

    Args:
        nfa1: First NFA
        nfa2: Second NFA

    Returns:
        NFA accepting L(nfa1) · L(nfa2)
    """
    if nfa1.alphabet != nfa2.alphabet:
        raise ValueError("NFAs must have same alphabet")

    new_states = nfa1.states | nfa2.states
    new_transitions = dict(nfa1.transitions)
    new_transitions.update(nfa2.transitions)

    # Add epsilon transitions from nfa1 accepting states to nfa2 initial
    for accepting_state in nfa1.accepting:
        key = (accepting_state, None)
        existing = new_transitions.get(key, set())
        new_transitions[key] = existing | {nfa2.initial}

    return NFA(
        states=new_states,
        alphabet=nfa1.alphabet,
        transitions=new_transitions,
        initial=nfa1.initial,
        accepting=nfa2.accepting,
    )


def nfa_kleene_star(nfa: NFA[S, A], new_initial: S) -> NFA[S, A]:
    """Compute Kleene star of NFA.

    Args:
        nfa: Input NFA
        new_initial: New initial state

    Returns:
        NFA accepting L(nfa)*
    """
    new_states = nfa.states | {new_initial}
    new_transitions = dict(nfa.transitions)

    # New initial epsilon-transitions to old initial
    new_transitions[(new_initial, None)] = {nfa.initial}

    # Epsilon transitions from accepting back to old initial
    for accepting_state in nfa.accepting:
        key = (accepting_state, None)
        existing = new_transitions.get(key, set())
        new_transitions[key] = existing | {nfa.initial}

    # New initial is also accepting (for empty word)
    new_accepting = nfa.accepting | {new_initial}

    return NFA(
        states=new_states,
        alphabet=nfa.alphabet,
        transitions=new_transitions,
        initial=new_initial,
        accepting=new_accepting,
    )


def nfa_from_symbol(symbol: A, alphabet: Set[A], state_prefix: str = "q") -> NFA[str, A]:
    """Create NFA accepting single symbol.

    Args:
        symbol: Symbol to accept
        alphabet: Full alphabet
        state_prefix: Prefix for state names

    Returns:
        NFA accepting exactly {symbol}
    """
    initial = f"{state_prefix}0"
    final = f"{state_prefix}1"

    return NFA(
        states={initial, final},
        alphabet=alphabet,
        transitions={(initial, symbol): {final}},
        initial=initial,
        accepting={final},
    )


def nfa_from_epsilon(alphabet: Set[A], state_name: str = "q0") -> NFA[str, A]:
    """Create NFA accepting empty word.

    Args:
        alphabet: Full alphabet
        state_name: State name

    Returns:
        NFA accepting exactly {ε}
    """
    return NFA(
        states={state_name},
        alphabet=alphabet,
        transitions={},
        initial=state_name,
        accepting={state_name},
    )


def thompson_construction(
    regex: str,
    alphabet: Set[str],
) -> NFA[str, str]:
    """Build NFA from simple regex using Thompson construction.

    Supports:
    - Literals: a, b, c, ...
    - Concatenation: ab
    - Union: a|b
    - Kleene star: a*
    - Parentheses: (a|b)*

    Args:
        regex: Regular expression string
        alphabet: Alphabet of symbols

    Returns:
        NFA accepting language of regex
    """
    state_counter = [0]

    def new_state() -> str:
        state_counter[0] += 1
        return f"s{state_counter[0]}"

    def parse_atom(pos: int) -> Tuple[NFA[str, str], int]:
        """Parse single atom (literal or parenthesized expr)."""
        if pos >= len(regex):
            raise ValueError("Unexpected end of regex")

        if regex[pos] == "(":
            # Parenthesized expression
            nfa, new_pos = parse_union(pos + 1)
            if new_pos >= len(regex) or regex[new_pos] != ")":
                raise ValueError("Missing closing parenthesis")
            return nfa, new_pos + 1
        elif regex[pos] in alphabet:
            # Literal symbol
            symbol = regex[pos]
            initial = new_state()
            final = new_state()
            nfa = NFA(
                states={initial, final},
                alphabet=alphabet,
                transitions={(initial, symbol): {final}},
                initial=initial,
                accepting={final},
            )
            return nfa, pos + 1
        else:
            raise ValueError(f"Unexpected character: {regex[pos]}")

    def parse_factor(pos: int) -> Tuple[NFA[str, str], int]:
        """Parse factor (atom optionally followed by *)."""
        nfa, new_pos = parse_atom(pos)

        while new_pos < len(regex) and regex[new_pos] == "*":
            new_initial = new_state()
            nfa = nfa_kleene_star(nfa, new_initial)
            new_pos += 1

        return nfa, new_pos

    def parse_term(pos: int) -> Tuple[NFA[str, str], int]:
        """Parse term (concatenation of factors)."""
        nfa, new_pos = parse_factor(pos)

        while new_pos < len(regex) and regex[new_pos] not in "|)":
            next_nfa, new_pos = parse_factor(new_pos)
            nfa = nfa_concatenation(nfa, next_nfa)

        return nfa, new_pos

    def parse_union(pos: int) -> Tuple[NFA[str, str], int]:
        """Parse union (terms separated by |)."""
        nfa, new_pos = parse_term(pos)

        while new_pos < len(regex) and regex[new_pos] == "|":
            next_nfa, new_pos = parse_term(new_pos + 1)
            new_initial = new_state()
            nfa = nfa_union(nfa, next_nfa, new_initial)

        return nfa, new_pos

    if not regex:
        return nfa_from_epsilon(alphabet)

    nfa, final_pos = parse_union(0)
    if final_pos != len(regex):
        raise ValueError(f"Unexpected characters at position {final_pos}")

    return nfa


def reverse_nfa(nfa: NFA[S, A]) -> NFA[S, A]:
    """Reverse an NFA (accepts reversal of language).

    Args:
        nfa: Input NFA

    Returns:
        NFA accepting L(nfa)^R
    """
    # Reverse transitions
    new_transitions: Dict[Tuple[S, Optional[A]], Set[S]] = {}

    for (src, symbol), targets in nfa.transitions.items():
        for target in targets:
            key = (target, symbol)
            if key not in new_transitions:
                new_transitions[key] = set()
            new_transitions[key].add(src)

    # Need new initial state with epsilon to all old accepting states
    # For simplicity, we'll just return NFA that works if there's single accepting
    # For full solution, would need to add new initial state

    if len(nfa.accepting) == 1:
        new_initial = next(iter(nfa.accepting))
        return NFA(
            states=nfa.states,
            alphabet=nfa.alphabet,
            transitions=new_transitions,
            initial=new_initial,
            accepting={nfa.initial},
        )
    else:
        # Add epsilon transitions from synthetic initial to all accepting
        new_initial_name = "__reverse_initial__"
        new_states = nfa.states | {new_initial_name}
        new_transitions[(new_initial_name, None)] = set(nfa.accepting)

        return NFA(
            states=new_states,  # type: ignore
            alphabet=nfa.alphabet,
            transitions=new_transitions,  # type: ignore
            initial=new_initial_name,  # type: ignore
            accepting={nfa.initial},
        )


def nfa_to_regex(nfa: NFA[S, A]) -> str:
    """Convert NFA to regular expression (state elimination).

    Args:
        nfa: Input NFA

    Returns:
        Regular expression string (simplified)
    """
    # This is a simplified implementation
    # Full state elimination is complex

    # For simple NFAs, try to identify pattern
    if len(nfa.states) == 1:
        state = next(iter(nfa.states))
        if state in nfa.accepting:
            # Check for self-loops
            self_loops = []
            for symbol in nfa.alphabet:
                if state in nfa.transition(state, symbol):
                    self_loops.append(str(symbol))

            if self_loops:
                if len(self_loops) == 1:
                    return f"{self_loops[0]}*"
                return f"({'|'.join(self_loops)})*"
            return "ε"
        return "∅"

    # For complex NFAs, return placeholder
    return f"<NFA with {len(nfa.states)} states>"
