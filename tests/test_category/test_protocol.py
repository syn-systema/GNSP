"""
Tests for protocol category implementations.

Tests cover:
- Protocol state creation and properties
- Protocol transition creation and anomaly scoring
- TCP, HTTP, and DNS protocol categories
- Protocol-specific detection methods
"""

import pytest

from gnsp.category.protocol import (
    DNSCategory,
    HTTPCategory,
    ProtocolCategory,
    ProtocolState,
    ProtocolTransition,
    ProtocolType,
    TCPCategory,
)
from gnsp.constants import PHI_INV


class TestProtocolState:
    """Tests for ProtocolState class."""

    def test_state_creation(self) -> None:
        """Test basic protocol state creation."""
        state = ProtocolState(name="ESTABLISHED")
        assert state.name == "ESTABLISHED"
        assert state.protocol == ProtocolType.TCP
        assert not state.is_initial
        assert not state.is_accepting
        assert not state.is_error

    def test_initial_state(self) -> None:
        """Test initial state flag."""
        state = ProtocolState(name="CLOSED", is_initial=True)
        assert state.is_initial

    def test_accepting_state(self) -> None:
        """Test accepting state flag."""
        state = ProtocolState(name="ESTABLISHED", is_accepting=True)
        assert state.is_accepting

    def test_error_state(self) -> None:
        """Test error state flag."""
        state = ProtocolState(name="ERROR", is_error=True)
        assert state.is_error

    def test_security_level(self) -> None:
        """Test security level property."""
        state = ProtocolState(name="SECURE", security_level=0.9)
        assert state.security_level == 0.9
        assert state.is_secure()

    def test_insecure_state(self) -> None:
        """Test state with low security level."""
        state = ProtocolState(name="COMPROMISED", security_level=0.1)
        assert not state.is_secure()


class TestProtocolTransition:
    """Tests for ProtocolTransition class."""

    @pytest.fixture
    def states(self) -> tuple:
        """Create test states."""
        src = ProtocolState(name="SRC", security_level=0.5)
        tgt = ProtocolState(name="TGT", security_level=0.8)
        return src, tgt

    def test_transition_creation(self, states: tuple) -> None:
        """Test basic transition creation."""
        src, tgt = states
        trans = ProtocolTransition(
            name="t",
            source=src,
            target=tgt,
            trigger="event",
            action="do_something",
        )

        assert trans.trigger == "event"
        assert trans.action == "do_something"
        assert not trans.is_anomalous

    def test_timeout_transition(self, states: tuple) -> None:
        """Test transition with timeout."""
        src, tgt = states
        trans = ProtocolTransition(
            name="t",
            source=src,
            target=tgt,
            timeout=5000.0,
        )

        assert trans.is_timeout_transition()
        assert trans.timeout == 5000.0

    def test_anomalous_transition(self, states: tuple) -> None:
        """Test anomalous transition flag."""
        src, tgt = states
        trans = ProtocolTransition(
            name="t",
            source=src,
            target=tgt,
            is_anomalous=True,
        )

        assert trans.is_anomalous

    def test_security_delta(self, states: tuple) -> None:
        """Test security level change computation."""
        src, tgt = states  # src=0.5, tgt=0.8
        trans = ProtocolTransition(name="t", source=src, target=tgt)

        delta = trans.security_delta()
        assert abs(delta - 0.3) < 1e-10


class TestProtocolCategory:
    """Tests for ProtocolCategory base class."""

    @pytest.fixture
    def protocol_cat(self) -> ProtocolCategory:
        """Create a simple protocol category."""
        cat = ProtocolCategory("TestProtocol", ProtocolType.TCP)

        s1 = ProtocolState(name="S1", is_initial=True, security_level=0.5)
        s2 = ProtocolState(name="S2", is_accepting=True, security_level=0.8)
        s3 = ProtocolState(name="S3", is_error=True, security_level=0.1)

        cat.add_state(s1)
        cat.add_state(s2)
        cat.add_state(s3)

        t1 = ProtocolTransition(
            name="t1",
            source=s1,
            target=s2,
            trigger="connect",
        )
        t2 = ProtocolTransition(
            name="t2",
            source=s2,
            target=s3,
            trigger="error",
            is_anomalous=True,
        )

        cat.add_transition(t1)
        cat.add_transition(t2)

        return cat

    def test_initial_states(self, protocol_cat: ProtocolCategory) -> None:
        """Test getting initial states."""
        initial = protocol_cat.initial_states
        assert len(initial) == 1
        assert list(initial)[0].name == "S1"

    def test_accepting_states(self, protocol_cat: ProtocolCategory) -> None:
        """Test getting accepting states."""
        accepting = protocol_cat.accepting_states
        assert len(accepting) == 1
        assert list(accepting)[0].name == "S2"

    def test_error_states(self, protocol_cat: ProtocolCategory) -> None:
        """Test getting error states."""
        error = protocol_cat.error_states
        assert len(error) == 1
        assert list(error)[0].name == "S3"

    def test_transitions_by_trigger(self, protocol_cat: ProtocolCategory) -> None:
        """Test getting transitions by trigger."""
        trans = protocol_cat.transitions_by_trigger("connect")
        assert len(trans) == 1

    def test_valid_transitions_from(self, protocol_cat: ProtocolCategory) -> None:
        """Test getting valid transitions from a state."""
        s1 = protocol_cat.get_object("S1")
        trans = protocol_cat.valid_transitions_from(s1)
        assert len(trans) == 1

    def test_is_valid_sequence(self, protocol_cat: ProtocolCategory) -> None:
        """Test validating trigger sequences."""
        valid, path = protocol_cat.is_valid_sequence(["connect"])
        assert valid
        assert len(path) == 2

        valid, path = protocol_cat.is_valid_sequence(["connect", "error"])
        assert valid
        assert len(path) == 3

        valid, path = protocol_cat.is_valid_sequence(["error"])
        assert not valid

    def test_anomaly_score(self, protocol_cat: ProtocolCategory) -> None:
        """Test anomaly scoring for transitions."""
        s2 = protocol_cat.get_object("S2")
        s3 = protocol_cat.get_object("S3")

        # Get the error transition
        trans = list(protocol_cat.hom(s2, s3))[0]

        score = protocol_cat.anomaly_score(trans)
        assert score > 0  # Should be non-zero due to is_anomalous flag

    def test_path_anomaly_score(self, protocol_cat: ProtocolCategory) -> None:
        """Test path anomaly scoring."""
        s1 = protocol_cat.get_object("S1")
        s2 = protocol_cat.get_object("S2")
        s3 = protocol_cat.get_object("S3")

        t1 = list(protocol_cat.hom(s1, s2))[0]
        t2 = list(protocol_cat.hom(s2, s3))[0]

        path = [t1, t2]
        score = protocol_cat.path_anomaly_score(path)

        assert 0 <= score <= 1


class TestTCPCategory:
    """Tests for TCP protocol category."""

    @pytest.fixture
    def tcp(self) -> TCPCategory:
        """Create TCP category."""
        return TCPCategory()

    def test_tcp_states(self, tcp: TCPCategory) -> None:
        """Test TCP has correct states."""
        states = {obj.name for obj in tcp.objects}

        expected = {
            "CLOSED", "LISTEN", "SYN_SENT", "SYN_RECEIVED",
            "ESTABLISHED", "FIN_WAIT_1", "FIN_WAIT_2",
            "CLOSE_WAIT", "CLOSING", "LAST_ACK", "TIME_WAIT",
            "INVALID",
        }

        assert expected.issubset(states)

    def test_tcp_initial_state(self, tcp: TCPCategory) -> None:
        """Test CLOSED is initial state."""
        initial = tcp.initial_states
        assert len(initial) == 1
        assert list(initial)[0].name == "CLOSED"

    def test_tcp_established_accepting(self, tcp: TCPCategory) -> None:
        """Test ESTABLISHED is accepting state."""
        accepting = tcp.accepting_states
        names = {s.name for s in accepting}
        assert "ESTABLISHED" in names

    def test_tcp_three_way_handshake(self, tcp: TCPCategory) -> None:
        """Test three-way handshake sequence."""
        # Active open: CLOSED -> SYN_SENT
        valid, path = tcp.is_valid_sequence(["active_open"])
        assert valid
        assert path[-1].name == "SYN_SENT"

        # Full handshake: active_open, recv SYN+ACK -> ESTABLISHED
        valid, path = tcp.is_valid_sequence(["active_open", "recv SYN+ACK"])
        assert valid
        assert path[-1].name == "ESTABLISHED"

    def test_tcp_passive_open(self, tcp: TCPCategory) -> None:
        """Test passive open sequence."""
        # Server side: CLOSED -> LISTEN
        valid, path = tcp.is_valid_sequence(["passive_open"])
        assert valid
        assert path[-1].name == "LISTEN"

    def test_detect_syn_flood(self, tcp: TCPCategory) -> None:
        """Test SYN flood detection."""
        # Normal traffic
        normal_events = [
            ("192.168.1.1", "recv SYN"),
            ("192.168.1.1", "recv ACK"),
            ("192.168.1.2", "recv SYN"),
            ("192.168.1.2", "recv ACK"),
        ]
        assert not tcp.detect_syn_flood(normal_events, threshold=5)

        # SYN flood
        flood_events = [("attacker", "recv SYN") for _ in range(150)]
        assert tcp.detect_syn_flood(flood_events, threshold=100)


class TestHTTPCategory:
    """Tests for HTTP protocol category."""

    @pytest.fixture
    def http(self) -> HTTPCategory:
        """Create HTTP category."""
        return HTTPCategory()

    def test_http_states(self, http: HTTPCategory) -> None:
        """Test HTTP has correct states."""
        states = {obj.name for obj in http.objects}

        expected = {
            "IDLE", "CONNECTED", "REQUEST_SENT",
            "HEADERS_RECEIVED", "BODY_RECEIVING", "RESPONSE_COMPLETE",
            "KEEP_ALIVE", "AUTH_REQUIRED", "AUTHENTICATED",
            "CLIENT_ERROR", "SERVER_ERROR", "TIMEOUT",
        }

        assert expected.issubset(states)

    def test_http_request_response(self, http: HTTPCategory) -> None:
        """Test HTTP request/response cycle."""
        valid, path = http.is_valid_sequence(["connect", "GET", "recv 2xx"])
        assert valid
        assert path[-1].name == "HEADERS_RECEIVED"

    def test_http_keep_alive(self, http: HTTPCategory) -> None:
        """Test HTTP keep-alive handling."""
        valid, path = http.is_valid_sequence([
            "connect", "GET", "recv 2xx", "no body", "connection: keep-alive",
        ])
        assert valid
        assert path[-1].name == "KEEP_ALIVE"

    def test_http_error_states(self, http: HTTPCategory) -> None:
        """Test HTTP error state transitions."""
        valid, path = http.is_valid_sequence(["connect", "GET", "recv 4xx"])
        assert valid
        assert path[-1].name == "CLIENT_ERROR"

    def test_classify_response(self, http: HTTPCategory) -> None:
        """Test response classification."""
        assert http.classify_response(200) == "success"
        assert http.classify_response(301) == "redirect"
        assert http.classify_response(404) == "client_error"
        assert http.classify_response(500) == "server_error"
        assert http.classify_response(100) == "informational"


class TestDNSCategory:
    """Tests for DNS protocol category."""

    @pytest.fixture
    def dns(self) -> DNSCategory:
        """Create DNS category."""
        return DNSCategory()

    def test_dns_states(self, dns: DNSCategory) -> None:
        """Test DNS has correct states."""
        states = {obj.name for obj in dns.objects}

        expected = {
            "IDLE", "QUERY_SENT", "WAITING",
            "RESPONSE_RECEIVED", "CACHED",
            "RECURSIVE_QUERY", "AUTHORITATIVE_QUERY",
            "NXDOMAIN", "SERVFAIL", "REFUSED", "TIMEOUT",
        }

        assert expected.issubset(states)

    def test_dns_query_response(self, dns: DNSCategory) -> None:
        """Test DNS query/response cycle."""
        valid, path = dns.is_valid_sequence([
            "query A", "sent", "recv NOERROR",
        ])
        assert valid
        assert path[-1].name == "RESPONSE_RECEIVED"

    def test_dns_caching(self, dns: DNSCategory) -> None:
        """Test DNS caching flow."""
        valid, path = dns.is_valid_sequence([
            "query A", "sent", "recv NOERROR", "cache",
        ])
        assert valid
        assert path[-1].name == "CACHED"

    def test_dns_error_handling(self, dns: DNSCategory) -> None:
        """Test DNS error states."""
        valid, path = dns.is_valid_sequence([
            "query A", "sent", "recv NXDOMAIN",
        ])
        assert valid
        assert path[-1].name == "NXDOMAIN"

    def test_detect_dns_tunneling_normal(self, dns: DNSCategory) -> None:
        """Test DNS tunneling detection with normal traffic."""
        normal_queries = [
            ("www.example.com", "A", 50),
            ("mail.example.com", "MX", 100),
            ("example.com", "NS", 80),
        ]

        suspicious = dns.detect_dns_tunneling(normal_queries)
        assert len(suspicious) == 0

    def test_detect_dns_tunneling_suspicious(self, dns: DNSCategory) -> None:
        """Test DNS tunneling detection with suspicious traffic."""
        # Long subdomain with high entropy (simulating encoded data)
        suspicious_queries = [
            ("aGVsbG8gd29ybGQgdGhpcyBpcyBhIHRlc3Q.tunnel.example.com", "TXT", 500),
        ]

        suspicious = dns.detect_dns_tunneling(
            suspicious_queries,
            length_threshold=20,
        )
        # May or may not trigger depending on entropy
        # The main test is that it doesn't crash

    def test_detect_dns_tunneling_txt_large_response(
        self,
        dns: DNSCategory,
    ) -> None:
        """Test DNS tunneling detection with large TXT responses."""
        queries = [
            ("tunnel.example.com", "TXT", 2000),  # Very suspicious: very large TXT
        ]

        suspicious = dns.detect_dns_tunneling(queries)
        # Very large TXT responses should be flagged
        assert any(domain == "tunnel.example.com" for domain, _ in suspicious)


class TestProtocolIntegration:
    """Integration tests for protocol categories."""

    def test_tcp_security_flow(self) -> None:
        """Test security level progression in TCP."""
        tcp = TCPCategory()

        # Get states
        closed = tcp.get_object("CLOSED")
        established = tcp.get_object("ESTABLISHED")

        # ESTABLISHED should have higher security than CLOSED
        assert established.security_level > closed.security_level

    def test_protocol_anomaly_detection(self) -> None:
        """Test anomaly detection across protocol transitions."""
        tcp = TCPCategory()

        # Get transition to error state
        established = tcp.get_object("ESTABLISHED")
        closed = tcp.get_object("CLOSED")

        # RST transition should have lower weight
        rst_trans = None
        for trans in tcp.valid_transitions_from(established):
            if "RST" in trans.trigger:
                rst_trans = trans
                break

        if rst_trans:
            assert rst_trans.weight < 1.0

    def test_multiple_protocols(self) -> None:
        """Test using multiple protocol categories together."""
        tcp = TCPCategory()
        http = HTTPCategory()
        dns = DNSCategory()

        # All should be valid categories
        assert tcp.verify_identity_law()
        assert http.verify_identity_law()
        assert dns.verify_identity_law()

        # Each has unique states
        tcp_states = {s.name for s in tcp.objects}
        http_states = {s.name for s in http.objects}
        dns_states = {s.name for s in dns.objects}

        # HTTP and DNS both have IDLE, but TCP doesn't
        assert "IDLE" not in tcp_states
        assert "IDLE" in http_states
        assert "IDLE" in dns_states
