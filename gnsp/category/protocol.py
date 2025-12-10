"""
Protocol categories for network protocol modeling.

Provides categorical representations of common network protocols:
- TCP: Connection-oriented state machine
- HTTP: Request/response protocol
- DNS: Query/response protocol

Each protocol is modeled as a category where:
- Objects are protocol states
- Morphisms are state transitions (with conditions and actions)

This enables compositional reasoning about protocol correctness
and detection of anomalous state transitions.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from gnsp.category.base import Category, Morphism, Object
from gnsp.constants import PHI, PHI_INV, PHI_SQ


class ProtocolType(Enum):
    """Types of network protocols."""

    TCP = auto()
    UDP = auto()
    HTTP = auto()
    HTTPS = auto()
    DNS = auto()
    FTP = auto()
    SSH = auto()
    SMTP = auto()


@dataclass(frozen=True)
class ProtocolState(Object):
    """
    A state in a protocol state machine.

    Extends Object with protocol-specific attributes for
    state classification and security analysis.

    Attributes:
        name: State identifier.
        protocol: Type of protocol this state belongs to.
        is_initial: Whether this is an initial state.
        is_accepting: Whether this is an accepting/final state.
        is_error: Whether this is an error state.
        security_level: Security classification (0.0 = untrusted, 1.0 = trusted).
    """

    protocol: ProtocolType = ProtocolType.TCP
    is_initial: bool = False
    is_accepting: bool = False
    is_error: bool = False
    security_level: float = 0.5

    def __hash__(self) -> int:
        return hash(self.name)

    def is_secure(self) -> bool:
        """Check if state has security level above golden ratio threshold."""
        return self.security_level >= PHI_INV


@dataclass(frozen=True)
class ProtocolTransition(Morphism):
    """
    A transition in a protocol state machine.

    Extends Morphism with protocol-specific attributes for
    transition classification and anomaly detection.

    Attributes:
        name: Transition identifier.
        source: Source protocol state.
        target: Target protocol state.
        trigger: Event or condition triggering the transition.
        action: Action performed during transition.
        timeout: Optional timeout for the transition (ms).
        is_anomalous: Whether this transition is potentially anomalous.
    """

    trigger: str = ""
    action: str = ""
    timeout: Optional[float] = None
    is_anomalous: bool = False

    def __hash__(self) -> int:
        return hash((self.name, self.source, self.target))

    def is_timeout_transition(self) -> bool:
        """Check if this is a timeout-triggered transition."""
        return self.timeout is not None

    def security_delta(self) -> float:
        """
        Compute security level change for this transition.

        Returns:
            Positive for security increase, negative for decrease.
        """
        if isinstance(self.source, ProtocolState) and isinstance(
            self.target, ProtocolState
        ):
            return self.target.security_level - self.source.security_level
        return 0.0


class ProtocolCategory(Category):
    """
    Category representing a network protocol.

    A protocol category has:
    - States as objects (with initial, accepting, error classifications)
    - Transitions as morphisms (with triggers, actions, timeouts)
    - Golden ratio-based anomaly scoring

    Provides methods for:
    - Path analysis (valid/invalid transition sequences)
    - Anomaly detection (unexpected transitions)
    - Security flow analysis
    """

    def __init__(
        self,
        name: str,
        protocol_type: ProtocolType,
    ) -> None:
        """
        Initialize a protocol category.

        Args:
            name: Name of the protocol category.
            protocol_type: Type of protocol being modeled.
        """
        super().__init__(name)
        self.protocol_type = protocol_type
        self._initial_states: Set[str] = set()
        self._accepting_states: Set[str] = set()
        self._error_states: Set[str] = set()

    def add_state(self, state: ProtocolState) -> None:
        """
        Add a protocol state to the category.

        Args:
            state: Protocol state to add.
        """
        self.add_object(state)

        if state.is_initial:
            self._initial_states.add(state.name)
        if state.is_accepting:
            self._accepting_states.add(state.name)
        if state.is_error:
            self._error_states.add(state.name)

    def add_transition(self, transition: ProtocolTransition) -> None:
        """
        Add a protocol transition to the category.

        Args:
            transition: Protocol transition to add.
        """
        self.add_morphism(transition)

    @property
    def initial_states(self) -> Set[ProtocolState]:
        """Get all initial states."""
        return {
            self._objects[name]
            for name in self._initial_states
            if name in self._objects
        }

    @property
    def accepting_states(self) -> Set[ProtocolState]:
        """Get all accepting states."""
        return {
            self._objects[name]
            for name in self._accepting_states
            if name in self._objects
        }

    @property
    def error_states(self) -> Set[ProtocolState]:
        """Get all error states."""
        return {
            self._objects[name]
            for name in self._error_states
            if name in self._objects
        }

    def transitions_by_trigger(self, trigger: str) -> Set[ProtocolTransition]:
        """Get all transitions with a given trigger."""
        result: Set[ProtocolTransition] = set()
        for morphism in self.morphisms:
            if isinstance(morphism, ProtocolTransition) and morphism.trigger == trigger:
                result.add(morphism)
        return result

    def valid_transitions_from(self, state: ProtocolState) -> Set[ProtocolTransition]:
        """Get all valid transitions from a state."""
        result: Set[ProtocolTransition] = set()
        for morphism in self.outgoing(state):
            if isinstance(morphism, ProtocolTransition):
                result.add(morphism)
        return result

    def is_valid_sequence(self, triggers: List[str]) -> Tuple[bool, List[ProtocolState]]:
        """
        Check if a sequence of triggers is valid from initial states.

        Args:
            triggers: Sequence of trigger events.

        Returns:
            Tuple of (is_valid, path of states traversed).
        """
        if not self._initial_states:
            return False, []

        # Try from each initial state
        for initial_name in self._initial_states:
            initial = self._objects.get(initial_name)
            if not initial:
                continue

            path = [initial]
            current = initial
            valid = True

            for trigger in triggers:
                found = False
                for trans in self.valid_transitions_from(current):
                    if trans.trigger == trigger:
                        current = trans.target
                        path.append(current)
                        found = True
                        break

                if not found:
                    valid = False
                    break

            if valid:
                return True, path

        return False, []

    def anomaly_score(self, transition: ProtocolTransition) -> float:
        """
        Compute anomaly score for a transition.

        Score is based on:
        - Whether transition is marked anomalous
        - Security level delta
        - Transition to error state

        Returns:
            Anomaly score in [0, 1], higher means more anomalous.
        """
        score = 0.0

        # Base anomaly flag
        if transition.is_anomalous:
            score += PHI_INV

        # Security decrease is suspicious
        delta = transition.security_delta()
        if delta < 0:
            score += abs(delta) * PHI_INV

        # Transition to error state
        if isinstance(transition.target, ProtocolState) and transition.target.is_error:
            score += PHI_INV_SQ

        # Normalize to [0, 1]
        return min(1.0, score)

    def path_anomaly_score(self, path: List[ProtocolTransition]) -> float:
        """
        Compute cumulative anomaly score for a path.

        Uses golden ratio weighting to emphasize recent transitions.

        Args:
            path: List of transitions forming a path.

        Returns:
            Weighted anomaly score.
        """
        if not path:
            return 0.0

        total = 0.0
        weight = 1.0

        # Weight more recent transitions higher
        for transition in reversed(path):
            total += self.anomaly_score(transition) * weight
            weight *= PHI_INV

        # Normalize by geometric series sum
        normalizer = (1 - PHI_INV ** len(path)) / (1 - PHI_INV)
        return total / normalizer if normalizer > 0 else 0.0


# Pre-computed constant for reuse
PHI_INV_SQ = PHI_INV ** 2


class TCPCategory(ProtocolCategory):
    """
    Category modeling TCP protocol state machine.

    Implements the standard TCP state machine with states:
    CLOSED, LISTEN, SYN_SENT, SYN_RECEIVED, ESTABLISHED,
    FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, CLOSING, LAST_ACK, TIME_WAIT

    Transitions represent TCP segment exchanges with appropriate
    security classifications for anomaly detection.
    """

    def __init__(self) -> None:
        """Initialize TCP protocol category."""
        super().__init__("TCP", ProtocolType.TCP)
        self._build_tcp_states()
        self._build_tcp_transitions()

    def _build_tcp_states(self) -> None:
        """Create TCP states."""
        states = [
            ProtocolState(
                name="CLOSED",
                protocol=ProtocolType.TCP,
                is_initial=True,
                is_accepting=True,
                security_level=0.5,
            ),
            ProtocolState(
                name="LISTEN",
                protocol=ProtocolType.TCP,
                is_accepting=True,
                security_level=0.4,
            ),
            ProtocolState(
                name="SYN_SENT",
                protocol=ProtocolType.TCP,
                security_level=0.3,
            ),
            ProtocolState(
                name="SYN_RECEIVED",
                protocol=ProtocolType.TCP,
                security_level=0.3,
            ),
            ProtocolState(
                name="ESTABLISHED",
                protocol=ProtocolType.TCP,
                is_accepting=True,
                security_level=0.8,
            ),
            ProtocolState(
                name="FIN_WAIT_1",
                protocol=ProtocolType.TCP,
                security_level=0.6,
            ),
            ProtocolState(
                name="FIN_WAIT_2",
                protocol=ProtocolType.TCP,
                security_level=0.6,
            ),
            ProtocolState(
                name="CLOSE_WAIT",
                protocol=ProtocolType.TCP,
                security_level=0.5,
            ),
            ProtocolState(
                name="CLOSING",
                protocol=ProtocolType.TCP,
                security_level=0.5,
            ),
            ProtocolState(
                name="LAST_ACK",
                protocol=ProtocolType.TCP,
                security_level=0.4,
            ),
            ProtocolState(
                name="TIME_WAIT",
                protocol=ProtocolType.TCP,
                security_level=0.5,
            ),
            # Error/anomaly states
            ProtocolState(
                name="INVALID",
                protocol=ProtocolType.TCP,
                is_error=True,
                security_level=0.0,
            ),
        ]

        for state in states:
            self.add_state(state)

    def _build_tcp_transitions(self) -> None:
        """Create TCP state transitions."""
        transitions = [
            # Active open
            ("CLOSED", "SYN_SENT", "active_open", "send SYN"),
            # Passive open
            ("CLOSED", "LISTEN", "passive_open", ""),
            # Server receives SYN
            ("LISTEN", "SYN_RECEIVED", "recv SYN", "send SYN+ACK"),
            # Client receives SYN+ACK
            ("SYN_SENT", "ESTABLISHED", "recv SYN+ACK", "send ACK"),
            # Simultaneous open
            ("SYN_SENT", "SYN_RECEIVED", "recv SYN", "send SYN+ACK"),
            # Server receives ACK
            ("SYN_RECEIVED", "ESTABLISHED", "recv ACK", ""),
            # Active close from ESTABLISHED
            ("ESTABLISHED", "FIN_WAIT_1", "close", "send FIN"),
            # Passive close
            ("ESTABLISHED", "CLOSE_WAIT", "recv FIN", "send ACK"),
            # FIN_WAIT_1 transitions
            ("FIN_WAIT_1", "FIN_WAIT_2", "recv ACK", ""),
            ("FIN_WAIT_1", "CLOSING", "recv FIN", "send ACK"),
            ("FIN_WAIT_1", "TIME_WAIT", "recv FIN+ACK", "send ACK"),
            # FIN_WAIT_2 to TIME_WAIT
            ("FIN_WAIT_2", "TIME_WAIT", "recv FIN", "send ACK"),
            # CLOSE_WAIT to LAST_ACK
            ("CLOSE_WAIT", "LAST_ACK", "close", "send FIN"),
            # CLOSING to TIME_WAIT
            ("CLOSING", "TIME_WAIT", "recv ACK", ""),
            # LAST_ACK to CLOSED
            ("LAST_ACK", "CLOSED", "recv ACK", ""),
            # TIME_WAIT timeout
            ("TIME_WAIT", "CLOSED", "timeout", "", 2 * 60000),  # 2 MSL
            # Listen close
            ("LISTEN", "CLOSED", "close", ""),
            # SYN_RECEIVED close
            ("SYN_RECEIVED", "FIN_WAIT_1", "close", "send FIN"),
            # RST handling (can happen from most states)
            ("SYN_RECEIVED", "LISTEN", "recv RST", ""),
            ("SYN_SENT", "CLOSED", "recv RST", ""),
            ("ESTABLISHED", "CLOSED", "recv RST", ""),
        ]

        for t in transitions:
            src_name, tgt_name, trigger, action = t[:4]
            timeout = t[4] if len(t) > 4 else None

            src = self.get_object(src_name)
            tgt = self.get_object(tgt_name)

            if src and tgt:
                trans = ProtocolTransition(
                    name=f"{src_name}->{tgt_name}:{trigger}",
                    source=src,
                    target=tgt,
                    trigger=trigger,
                    action=action,
                    timeout=timeout,
                    weight=PHI_INV if "RST" in trigger else 1.0,
                )
                self.add_transition(trans)

    def detect_syn_flood(
        self,
        events: List[Tuple[str, str]],
        threshold: int = 100,
    ) -> bool:
        """
        Detect potential SYN flood attack.

        Args:
            events: List of (source_ip, trigger) tuples.
            threshold: Number of SYNs without ACK to trigger alert.

        Returns:
            True if SYN flood detected.
        """
        syn_counts: Dict[str, int] = {}
        ack_counts: Dict[str, int] = {}

        for source_ip, trigger in events:
            if trigger == "recv SYN":
                syn_counts[source_ip] = syn_counts.get(source_ip, 0) + 1
            elif trigger == "recv ACK":
                ack_counts[source_ip] = ack_counts.get(source_ip, 0) + 1

        # Check for sources with many SYNs but few ACKs
        for source_ip, syn_count in syn_counts.items():
            ack_count = ack_counts.get(source_ip, 0)
            if syn_count - ack_count > threshold:
                return True

        return False


class HTTPCategory(ProtocolCategory):
    """
    Category modeling HTTP protocol.

    Models HTTP request/response cycle with states for:
    - Connection management
    - Request processing
    - Response handling
    - Error conditions

    Enables detection of protocol violations and suspicious patterns.
    """

    def __init__(self) -> None:
        """Initialize HTTP protocol category."""
        super().__init__("HTTP", ProtocolType.HTTP)
        self._build_http_states()
        self._build_http_transitions()

    def _build_http_states(self) -> None:
        """Create HTTP states."""
        states = [
            ProtocolState(
                name="IDLE",
                protocol=ProtocolType.HTTP,
                is_initial=True,
                security_level=0.5,
            ),
            ProtocolState(
                name="CONNECTED",
                protocol=ProtocolType.HTTP,
                security_level=0.6,
            ),
            ProtocolState(
                name="REQUEST_SENT",
                protocol=ProtocolType.HTTP,
                security_level=0.5,
            ),
            ProtocolState(
                name="HEADERS_RECEIVED",
                protocol=ProtocolType.HTTP,
                security_level=0.6,
            ),
            ProtocolState(
                name="BODY_RECEIVING",
                protocol=ProtocolType.HTTP,
                security_level=0.6,
            ),
            ProtocolState(
                name="RESPONSE_COMPLETE",
                protocol=ProtocolType.HTTP,
                is_accepting=True,
                security_level=0.7,
            ),
            ProtocolState(
                name="KEEP_ALIVE",
                protocol=ProtocolType.HTTP,
                security_level=0.7,
            ),
            # Authentication states
            ProtocolState(
                name="AUTH_REQUIRED",
                protocol=ProtocolType.HTTP,
                security_level=0.3,
            ),
            ProtocolState(
                name="AUTHENTICATED",
                protocol=ProtocolType.HTTP,
                security_level=0.9,
            ),
            # Error states
            ProtocolState(
                name="CLIENT_ERROR",
                protocol=ProtocolType.HTTP,
                is_error=True,
                security_level=0.2,
            ),
            ProtocolState(
                name="SERVER_ERROR",
                protocol=ProtocolType.HTTP,
                is_error=True,
                security_level=0.3,
            ),
            ProtocolState(
                name="TIMEOUT",
                protocol=ProtocolType.HTTP,
                is_error=True,
                security_level=0.2,
            ),
        ]

        for state in states:
            self.add_state(state)

    def _build_http_transitions(self) -> None:
        """Create HTTP transitions."""
        transitions = [
            # Connection establishment
            ("IDLE", "CONNECTED", "connect", "TCP handshake"),
            # Request methods
            ("CONNECTED", "REQUEST_SENT", "GET", "send request"),
            ("CONNECTED", "REQUEST_SENT", "POST", "send request"),
            ("CONNECTED", "REQUEST_SENT", "PUT", "send request"),
            ("CONNECTED", "REQUEST_SENT", "DELETE", "send request"),
            ("CONNECTED", "REQUEST_SENT", "HEAD", "send request"),
            ("CONNECTED", "REQUEST_SENT", "OPTIONS", "send request"),
            # Response handling
            ("REQUEST_SENT", "HEADERS_RECEIVED", "recv 1xx", "continue"),
            ("REQUEST_SENT", "HEADERS_RECEIVED", "recv 2xx", "success"),
            ("REQUEST_SENT", "AUTH_REQUIRED", "recv 401", "auth needed"),
            ("REQUEST_SENT", "CLIENT_ERROR", "recv 4xx", "client error"),
            ("REQUEST_SENT", "SERVER_ERROR", "recv 5xx", "server error"),
            # Headers to body
            ("HEADERS_RECEIVED", "BODY_RECEIVING", "has body", ""),
            ("HEADERS_RECEIVED", "RESPONSE_COMPLETE", "no body", ""),
            # Body receiving
            ("BODY_RECEIVING", "RESPONSE_COMPLETE", "body complete", ""),
            ("BODY_RECEIVING", "BODY_RECEIVING", "recv chunk", ""),
            # Keep-alive
            ("RESPONSE_COMPLETE", "KEEP_ALIVE", "connection: keep-alive", ""),
            ("RESPONSE_COMPLETE", "IDLE", "connection: close", ""),
            # Keep-alive to new request
            ("KEEP_ALIVE", "REQUEST_SENT", "GET", "send request"),
            ("KEEP_ALIVE", "REQUEST_SENT", "POST", "send request"),
            ("KEEP_ALIVE", "IDLE", "timeout", "", 30000),
            # Authentication flow
            ("AUTH_REQUIRED", "CONNECTED", "provide credentials", ""),
            ("REQUEST_SENT", "AUTHENTICATED", "auth success", ""),
            ("AUTHENTICATED", "REQUEST_SENT", "GET", "send request"),
            ("AUTHENTICATED", "REQUEST_SENT", "POST", "send request"),
            # Error recovery
            ("CLIENT_ERROR", "IDLE", "close", ""),
            ("SERVER_ERROR", "IDLE", "close", ""),
            ("TIMEOUT", "IDLE", "close", ""),
            # Timeout from any waiting state
            ("REQUEST_SENT", "TIMEOUT", "timeout", "", 30000),
            ("BODY_RECEIVING", "TIMEOUT", "timeout", "", 60000),
        ]

        for t in transitions:
            src_name, tgt_name, trigger, action = t[:4]
            timeout = t[4] if len(t) > 4 else None

            src = self.get_object(src_name)
            tgt = self.get_object(tgt_name)

            if src and tgt:
                # Mark potentially anomalous transitions
                is_anomalous = tgt_name in ("CLIENT_ERROR", "SERVER_ERROR", "TIMEOUT")

                trans = ProtocolTransition(
                    name=f"{src_name}->{tgt_name}:{trigger}",
                    source=src,
                    target=tgt,
                    trigger=trigger,
                    action=action,
                    timeout=timeout,
                    is_anomalous=is_anomalous,
                    weight=PHI_INV if is_anomalous else 1.0,
                )
                self.add_transition(trans)

    def classify_response(self, status_code: int) -> str:
        """
        Classify HTTP response status code.

        Args:
            status_code: HTTP status code.

        Returns:
            Classification string.
        """
        if 100 <= status_code < 200:
            return "informational"
        elif 200 <= status_code < 300:
            return "success"
        elif 300 <= status_code < 400:
            return "redirect"
        elif 400 <= status_code < 500:
            return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "unknown"


class DNSCategory(ProtocolCategory):
    """
    Category modeling DNS protocol.

    Models DNS query/response cycle with states for:
    - Query types (A, AAAA, CNAME, MX, etc.)
    - Response handling
    - Caching behavior
    - Error conditions

    Enables detection of DNS tunneling and other anomalies.
    """

    def __init__(self) -> None:
        """Initialize DNS protocol category."""
        super().__init__("DNS", ProtocolType.DNS)
        self._build_dns_states()
        self._build_dns_transitions()

    def _build_dns_states(self) -> None:
        """Create DNS states."""
        states = [
            ProtocolState(
                name="IDLE",
                protocol=ProtocolType.DNS,
                is_initial=True,
                security_level=0.5,
            ),
            ProtocolState(
                name="QUERY_SENT",
                protocol=ProtocolType.DNS,
                security_level=0.4,
            ),
            ProtocolState(
                name="WAITING",
                protocol=ProtocolType.DNS,
                security_level=0.4,
            ),
            ProtocolState(
                name="RESPONSE_RECEIVED",
                protocol=ProtocolType.DNS,
                is_accepting=True,
                security_level=0.6,
            ),
            ProtocolState(
                name="CACHED",
                protocol=ProtocolType.DNS,
                security_level=0.7,
            ),
            # Recursive query states
            ProtocolState(
                name="RECURSIVE_QUERY",
                protocol=ProtocolType.DNS,
                security_level=0.5,
            ),
            ProtocolState(
                name="AUTHORITATIVE_QUERY",
                protocol=ProtocolType.DNS,
                security_level=0.6,
            ),
            # Error states
            ProtocolState(
                name="NXDOMAIN",
                protocol=ProtocolType.DNS,
                is_error=True,
                security_level=0.3,
            ),
            ProtocolState(
                name="SERVFAIL",
                protocol=ProtocolType.DNS,
                is_error=True,
                security_level=0.2,
            ),
            ProtocolState(
                name="REFUSED",
                protocol=ProtocolType.DNS,
                is_error=True,
                security_level=0.2,
            ),
            ProtocolState(
                name="TIMEOUT",
                protocol=ProtocolType.DNS,
                is_error=True,
                security_level=0.2,
            ),
        ]

        for state in states:
            self.add_state(state)

    def _build_dns_transitions(self) -> None:
        """Create DNS transitions."""
        # Query types
        query_types = ["A", "AAAA", "CNAME", "MX", "NS", "PTR", "SOA", "TXT", "SRV"]

        transitions = []

        # Standard query flow
        for qtype in query_types:
            transitions.extend([
                ("IDLE", "QUERY_SENT", f"query {qtype}", "send query"),
                ("QUERY_SENT", "WAITING", "sent", "await response"),
            ])

        # Response handling
        transitions.extend([
            ("WAITING", "RESPONSE_RECEIVED", "recv NOERROR", "process response"),
            ("WAITING", "NXDOMAIN", "recv NXDOMAIN", "no such domain"),
            ("WAITING", "SERVFAIL", "recv SERVFAIL", "server failure"),
            ("WAITING", "REFUSED", "recv REFUSED", "query refused"),
            ("WAITING", "TIMEOUT", "timeout", "", 5000),
            # Caching
            ("RESPONSE_RECEIVED", "CACHED", "cache", "store in cache"),
            ("RESPONSE_RECEIVED", "IDLE", "done", ""),
            ("CACHED", "IDLE", "ttl expired", ""),
            ("CACHED", "RESPONSE_RECEIVED", "cache hit", ""),
            # Recursive resolution
            ("QUERY_SENT", "RECURSIVE_QUERY", "recursion needed", ""),
            ("RECURSIVE_QUERY", "AUTHORITATIVE_QUERY", "found NS", ""),
            ("AUTHORITATIVE_QUERY", "WAITING", "query auth", ""),
            # Error recovery
            ("NXDOMAIN", "IDLE", "done", ""),
            ("SERVFAIL", "IDLE", "retry", ""),
            ("REFUSED", "IDLE", "done", ""),
            ("TIMEOUT", "IDLE", "retry", ""),
            ("TIMEOUT", "QUERY_SENT", "retry query", ""),
        ])

        for t in transitions:
            src_name, tgt_name, trigger, action = t[:4]
            timeout = t[4] if len(t) > 4 else None

            src = self.get_object(src_name)
            tgt = self.get_object(tgt_name)

            if src and tgt:
                is_anomalous = tgt_name in ("NXDOMAIN", "SERVFAIL", "REFUSED", "TIMEOUT")

                trans = ProtocolTransition(
                    name=f"{src_name}->{tgt_name}:{trigger}",
                    source=src,
                    target=tgt,
                    trigger=trigger,
                    action=action,
                    timeout=timeout,
                    is_anomalous=is_anomalous,
                    weight=PHI_INV if is_anomalous else 1.0,
                )
                self.add_transition(trans)

    def detect_dns_tunneling(
        self,
        queries: List[Tuple[str, str, int]],
        entropy_threshold: float = 0.8,
        length_threshold: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Detect potential DNS tunneling based on query patterns.

        Args:
            queries: List of (domain, query_type, response_size) tuples.
            entropy_threshold: Entropy threshold for subdomain detection.
            length_threshold: Length threshold for suspicious subdomains.

        Returns:
            List of (domain, suspicion_score) for flagged domains.
        """
        import math

        suspicious: List[Tuple[str, float]] = []

        for domain, qtype, response_size in queries:
            score = 0.0

            # Check subdomain length
            parts = domain.split(".")
            if len(parts) > 2:
                subdomain = parts[0]
                if len(subdomain) > length_threshold:
                    score += PHI_INV

            # Check for high entropy in subdomain (encoded data)
            if len(parts) > 2:
                subdomain = parts[0]
                if len(subdomain) > 10:
                    # Calculate Shannon entropy
                    freq: Dict[str, int] = {}
                    for c in subdomain:
                        freq[c] = freq.get(c, 0) + 1

                    entropy = 0.0
                    for count in freq.values():
                        p = count / len(subdomain)
                        entropy -= p * math.log2(p)

                    # Normalize entropy
                    max_entropy = math.log2(len(set(subdomain)))
                    if max_entropy > 0:
                        normalized_entropy = entropy / max_entropy
                        if normalized_entropy > entropy_threshold:
                            score += normalized_entropy * PHI_INV

            # TXT queries with large responses are suspicious
            if qtype == "TXT" and response_size > 200:
                score += PHI_INV * (response_size / 1000)

            # NULL queries are very suspicious
            if qtype == "NULL":
                score += PHI

            if score > PHI_INV:
                suspicious.append((domain, min(1.0, score)))

        return suspicious
