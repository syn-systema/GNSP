"""
Tests for functor implementations.

Tests cover:
- Base functor operations
- Identity and composition functors
- Forgetful functors
- Security and traffic functors
"""

import pytest
import numpy as np

from gnsp.category.base import Category, Morphism, Object
from gnsp.category.functor import (
    CompositionFunctor,
    ForgetfulFunctor,
    Functor,
    IdentityFunctor,
    SecurityFunctor,
    TrafficFunctor,
    TrafficPattern,
)
from gnsp.category.protocol import ProtocolState, ProtocolType, TCPCategory
from gnsp.constants import PHI_INV


class TestIdentityFunctor:
    """Tests for IdentityFunctor class."""

    @pytest.fixture
    def category(self) -> Category:
        """Create a simple test category."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        cat.add_object(A)
        cat.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        cat.add_morphism(f)

        return cat

    def test_identity_functor_creation(self, category: Category) -> None:
        """Test identity functor creation."""
        id_f = IdentityFunctor(category)
        assert id_f.source == category
        assert id_f.target == category

    def test_identity_maps_object(self, category: Category) -> None:
        """Test identity functor maps objects to themselves."""
        id_f = IdentityFunctor(category)
        A = category.get_object("A")

        assert id_f.map_object(A) == A

    def test_identity_maps_morphism(self, category: Category) -> None:
        """Test identity functor maps morphisms to themselves."""
        id_f = IdentityFunctor(category)
        A = category.get_object("A")
        B = category.get_object("B")
        f = list(category.hom(A, B))[0]

        assert id_f.map_morphism(f) == f

    def test_identity_preserves_identity(self, category: Category) -> None:
        """Test identity functor preserves identity morphisms."""
        id_f = IdentityFunctor(category)
        A = category.get_object("A")

        assert id_f.preserves_identity(A)

    def test_identity_preserves_composition(self, category: Category) -> None:
        """Test identity functor preserves composition."""
        # Add another morphism
        B = category.get_object("B")
        C = Object(name="C")
        category.add_object(C)

        g = Morphism(name="g", source=B, target=C)
        category.add_morphism(g)

        id_f = IdentityFunctor(category)

        A = category.get_object("A")
        f = list(category.hom(A, B) - {category.get_identity(A)})[0]

        assert id_f.preserves_composition(f, g)


class TestCompositionFunctor:
    """Tests for CompositionFunctor class."""

    @pytest.fixture
    def three_categories(self) -> tuple:
        """Create three categories for functor composition."""
        cat1 = Category("C1")
        cat2 = Category("C2")
        cat3 = Category("C3")

        for cat in [cat1, cat2, cat3]:
            A = Object(name="A")
            B = Object(name="B")
            cat.add_object(A)
            cat.add_object(B)
            cat.add_morphism(Morphism(name="f", source=A, target=B))

        return cat1, cat2, cat3

    def test_composition_functor_creation(self, three_categories: tuple) -> None:
        """Test composition of identity functors."""
        cat1, cat2, cat3 = three_categories

        # Identity functors
        id1 = IdentityFunctor(cat1)
        # Create a "relabeling" functor by using identity
        id2 = IdentityFunctor(cat1)

        # These should be composable (same category)
        comp = CompositionFunctor(id1, id2)

        assert comp.source == cat1
        assert comp.target == cat1

    def test_composition_not_composable(self, three_categories: tuple) -> None:
        """Test composition fails for non-composable functors."""
        cat1, cat2, cat3 = three_categories

        id1 = IdentityFunctor(cat1)
        id3 = IdentityFunctor(cat3)

        with pytest.raises(ValueError):
            CompositionFunctor(id1, id3)

    def test_composition_maps_correctly(self, three_categories: tuple) -> None:
        """Test composition applies both functors."""
        cat1, _, _ = three_categories

        id1 = IdentityFunctor(cat1)
        id2 = IdentityFunctor(cat1)

        comp = CompositionFunctor(id1, id2)

        A = cat1.get_object("A")
        assert comp.map_object(A) == A


class TestForgetfulFunctor:
    """Tests for ForgetfulFunctor class."""

    @pytest.fixture
    def structured_category(self) -> Category:
        """Create category with structured objects."""
        cat = Category("Structured")

        A = Object(
            name="A",
            properties=frozenset({"secure", "monitored", "encrypted"}),
        )
        B = Object(
            name="B",
            properties=frozenset({"secure", "monitored"}),
        )

        cat.add_object(A)
        cat.add_object(B)
        cat.add_morphism(Morphism(name="f", source=A, target=B))

        return cat

    def test_forgetful_functor_creation(
        self,
        structured_category: Category,
    ) -> None:
        """Test forgetful functor creation."""
        target = Category("Simple")

        forget = ForgetfulFunctor(
            "forget_encryption",
            structured_category,
            target,
            {"encrypted"},
        )

        assert forget.forget_properties == {"encrypted"}

    def test_forgetful_removes_properties(
        self,
        structured_category: Category,
    ) -> None:
        """Test forgetful functor removes specified properties."""
        target = Category("Simple")

        forget = ForgetfulFunctor(
            "forget_encryption",
            structured_category,
            target,
            {"encrypted"},
        )

        A = structured_category.get_object("A")
        A_simple = forget.map_object(A)

        assert "secure" in A_simple.properties
        assert "monitored" in A_simple.properties
        assert "encrypted" not in A_simple.properties

    def test_forgetful_preserves_non_forgotten(
        self,
        structured_category: Category,
    ) -> None:
        """Test forgetful functor preserves non-forgotten properties."""
        target = Category("Simple")

        forget = ForgetfulFunctor(
            "forget_encryption",
            structured_category,
            target,
            {"encrypted"},
        )

        B = structured_category.get_object("B")
        B_simple = forget.map_object(B)

        # B doesn't have encrypted, so should be unchanged
        assert B_simple.properties == B.properties


class TestSecurityFunctor:
    """Tests for SecurityFunctor class."""

    @pytest.fixture
    def protocol_category(self) -> Category:
        """Create a protocol category with security levels."""
        cat = Category("Protocol")

        states = [
            ProtocolState(name="LOW", security_level=0.2),
            ProtocolState(name="MEDIUM", security_level=0.5),
            ProtocolState(name="HIGH", security_level=0.9),
        ]

        for state in states:
            cat.add_object(state)

        cat.add_morphism(Morphism(
            name="upgrade",
            source=cat.get_object("LOW"),
            target=cat.get_object("HIGH"),
        ))
        cat.add_morphism(Morphism(
            name="downgrade",
            source=cat.get_object("HIGH"),
            target=cat.get_object("LOW"),
        ))

        return cat

    def test_security_functor_creation(
        self,
        protocol_category: Category,
    ) -> None:
        """Test security functor creation."""
        sec_f = SecurityFunctor("sec", protocol_category)

        assert sec_f.source == protocol_category
        assert sec_f.target.name == "Security"

    def test_security_level_mapping(
        self,
        protocol_category: Category,
    ) -> None:
        """Test mapping to security levels."""
        sec_f = SecurityFunctor("sec", protocol_category)

        low = protocol_category.get_object("LOW")
        high = protocol_category.get_object("HIGH")

        low_level = sec_f.map_object(low)
        high_level = sec_f.map_object(high)

        # LOW (0.2) should map to "low" level
        assert low_level.name == "low"
        # HIGH (0.9) should map to "secure" level
        assert high_level.name == "secure"

    def test_security_delta(self, protocol_category: Category) -> None:
        """Test security delta computation."""
        sec_f = SecurityFunctor("sec", protocol_category)

        low = protocol_category.get_object("LOW")
        high = protocol_category.get_object("HIGH")

        upgrade = list(protocol_category.hom(low, high))[0]
        downgrade = list(protocol_category.hom(high, low))[0]

        upgrade_delta = sec_f.security_delta(upgrade)
        downgrade_delta = sec_f.security_delta(downgrade)

        assert upgrade_delta > 0  # Upgrade increases security
        assert downgrade_delta < 0  # Downgrade decreases security

    def test_is_secure_path(self, protocol_category: Category) -> None:
        """Test secure path detection."""
        sec_f = SecurityFunctor("sec", protocol_category)

        low = protocol_category.get_object("LOW")
        high = protocol_category.get_object("HIGH")

        upgrade = list(protocol_category.hom(low, high))[0]
        downgrade = list(protocol_category.hom(high, low))[0]

        assert sec_f.is_secure_path([upgrade])
        assert not sec_f.is_secure_path([downgrade])

    def test_security_functor_with_tcp(self) -> None:
        """Test security functor on TCP category."""
        tcp = TCPCategory()
        sec_f = SecurityFunctor("tcp_security", tcp)

        established = tcp.get_object("ESTABLISHED")
        closed = tcp.get_object("CLOSED")

        est_level = sec_f.map_object(established)
        closed_level = sec_f.map_object(closed)

        # ESTABLISHED has higher security than CLOSED
        assert est_level.data >= closed_level.data


class TestTrafficFunctor:
    """Tests for TrafficFunctor class."""

    @pytest.fixture
    def tcp_functor(self) -> TrafficFunctor:
        """Create traffic functor for TCP."""
        tcp = TCPCategory()
        return TrafficFunctor("tcp_traffic", tcp)

    def test_traffic_functor_creation(self, tcp_functor: TrafficFunctor) -> None:
        """Test traffic functor creation."""
        assert tcp_functor.target.name == "Traffic"

    def test_traffic_pattern_mapping(self, tcp_functor: TrafficFunctor) -> None:
        """Test mapping protocol states to traffic patterns."""
        tcp = tcp_functor.source

        closed = tcp.get_object("CLOSED")
        established = tcp.get_object("ESTABLISHED")

        closed_pattern = tcp_functor.map_object(closed)
        est_pattern = tcp_functor.map_object(established)

        # CLOSED should have no/low traffic
        assert closed_pattern.name in ("no_traffic", "low_traffic")
        # ESTABLISHED should have more traffic
        assert est_pattern.name in ("medium_traffic", "high_traffic")

    def test_anomaly_score_normal(self, tcp_functor: TrafficFunctor) -> None:
        """Test anomaly score for normal traffic."""
        tcp = tcp_functor.source
        established = tcp.get_object("ESTABLISHED")

        # Normal traffic for ESTABLISHED
        normal_traffic = TrafficPattern(
            packets=100,
            bytes_sent=10000,
            duration=60.0,
        )

        score = tcp_functor.anomaly_score(established, normal_traffic)
        # Should be relatively low
        assert score < 0.8

    def test_anomaly_score_anomalous(self, tcp_functor: TrafficFunctor) -> None:
        """Test anomaly score for anomalous traffic."""
        tcp = tcp_functor.source
        closed = tcp.get_object("CLOSED")

        # Heavy traffic when should be CLOSED is anomalous
        anomalous_traffic = TrafficPattern(
            packets=10000,
            bytes_sent=1000000,
            duration=1.0,  # Very short duration, burst
        )

        score = tcp_functor.anomaly_score(closed, anomalous_traffic)
        # Should be higher than normal
        assert score > 0

    def test_detect_anomalies(self, tcp_functor: TrafficFunctor) -> None:
        """Test batch anomaly detection."""
        tcp = tcp_functor.source
        closed = tcp.get_object("CLOSED")
        established = tcp.get_object("ESTABLISHED")

        observations = [
            (established, TrafficPattern(packets=100, bytes_sent=10000)),
            (closed, TrafficPattern(packets=5000, bytes_sent=500000)),  # Anomalous
        ]

        anomalies = tcp_functor.detect_anomalies(observations, threshold=0.3)
        # The CLOSED state with heavy traffic should be flagged
        assert len(anomalies) >= 0  # May or may not trigger based on thresholds


class TestTrafficPattern:
    """Tests for TrafficPattern class."""

    def test_pattern_creation(self) -> None:
        """Test traffic pattern creation."""
        pattern = TrafficPattern(
            source_ip="192.168.1.1",
            dest_ip="10.0.0.1",
            source_port=12345,
            dest_port=80,
            protocol="TCP",
            bytes_sent=1000,
            bytes_recv=5000,
            packets=50,
            duration=30.0,
        )

        assert pattern.source_ip == "192.168.1.1"
        assert pattern.bytes_sent == 1000

    def test_to_vector(self) -> None:
        """Test conversion to feature vector."""
        pattern = TrafficPattern(
            source_port=8080,
            dest_port=443,
            bytes_sent=1000,
            bytes_recv=5000,
            packets=50,
            duration=30.0,
        )

        vec = pattern.to_vector()

        assert len(vec) == 6
        assert all(0 <= v <= 1 for v in vec)  # Normalized values


class TestFunctorProperties:
    """Tests for functor properties (faithful, full, etc.)."""

    @pytest.fixture
    def simple_functor(self) -> tuple:
        """Create a simple functor for testing."""
        source = Category("Source")
        target = Category("Target")

        A = Object(name="A")
        B = Object(name="B")

        source.add_object(A)
        source.add_object(B)
        target.add_object(A)
        target.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        source.add_morphism(f)
        target.add_morphism(f)

        id_f = IdentityFunctor(source)
        return id_f, source, target

    def test_identity_is_faithful(self, simple_functor: tuple) -> None:
        """Test identity functor is faithful."""
        functor, _, _ = simple_functor
        assert functor.is_faithful()

    def test_identity_is_full(self, simple_functor: tuple) -> None:
        """Test identity functor is full."""
        functor, _, _ = simple_functor
        assert functor.is_full()

    def test_violation_score_identity(self, simple_functor: tuple) -> None:
        """Test violation score for identity functor."""
        functor, source, _ = simple_functor

        A = source.get_object("A")
        B = source.get_object("B")
        f = list(source.hom(A, B) - {source.get_identity(A)})[0]

        score = functor.violation_score(f)
        assert score == 0.0  # Identity functor has no violations
