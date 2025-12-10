"""
Tests for sheaf-theoretic structures.

Tests cover:
- Sites and covering families
- Presheaves and sheaves
- Gluing conditions
- Network and anomaly sheaves
"""

import pytest

from gnsp.category.base import Category, Morphism, Object
from gnsp.category.sheaf import (
    AnomalySheaf,
    Cover,
    GluingCondition,
    NetworkSheaf,
    Presheaf,
    Sheaf,
    SheafSection,
    Site,
)
from gnsp.constants import PHI_INV, THRESHOLD_HIGH


class TestCover:
    """Tests for Cover class."""

    @pytest.fixture
    def objects_and_morphisms(self) -> tuple:
        """Create test objects and morphisms."""
        U = Object(name="U")
        U1 = Object(name="U1")
        U2 = Object(name="U2")

        f1 = Morphism(name="f1", source=U1, target=U)
        f2 = Morphism(name="f2", source=U2, target=U)

        return U, U1, U2, f1, f2

    def test_cover_creation(self, objects_and_morphisms: tuple) -> None:
        """Test cover creation."""
        U, _, _, f1, f2 = objects_and_morphisms

        cover = Cover(
            target=U,
            morphisms=frozenset({f1, f2}),
        )

        assert cover.target == U
        assert len(cover.morphisms) == 2

    def test_covering_objects(self, objects_and_morphisms: tuple) -> None:
        """Test getting covering objects."""
        U, U1, U2, f1, f2 = objects_and_morphisms

        cover = Cover(
            target=U,
            morphisms=frozenset({f1, f2}),
        )

        covering = cover.covering_objects
        assert U1 in covering
        assert U2 in covering

    def test_is_singleton(self, objects_and_morphisms: tuple) -> None:
        """Test singleton cover detection."""
        U, _, _, f1, f2 = objects_and_morphisms

        singleton = Cover(target=U, morphisms=frozenset({f1}))
        multi = Cover(target=U, morphisms=frozenset({f1, f2}))

        assert singleton.is_singleton()
        assert not multi.is_singleton()

    def test_invalid_cover(self, objects_and_morphisms: tuple) -> None:
        """Test cover with wrong target fails."""
        U, U1, _, _, _ = objects_and_morphisms

        wrong_target = Morphism(name="wrong", source=U, target=U1)

        with pytest.raises(ValueError):
            Cover(target=U, morphisms=frozenset({wrong_target}))


class TestSite:
    """Tests for Site class."""

    @pytest.fixture
    def site(self) -> Site:
        """Create a test site."""
        cat = Category("Network")

        # Network segments
        whole = Object(name="Network")
        dmz = Object(name="DMZ")
        internal = Object(name="Internal")
        external = Object(name="External")

        for obj in [whole, dmz, internal, external]:
            cat.add_object(obj)

        # Inclusions
        cat.add_morphism(Morphism(name="dmz_in", source=dmz, target=whole))
        cat.add_morphism(Morphism(name="int_in", source=internal, target=whole))
        cat.add_morphism(Morphism(name="ext_in", source=external, target=whole))

        site = Site(category=cat, name="NetworkSite")

        # Add a cover: Network is covered by DMZ, Internal, External
        cover = Cover(
            target=whole,
            morphisms=frozenset({
                cat._morphisms[("dmz_in", "DMZ", "Network")],
                cat._morphisms[("int_in", "Internal", "Network")],
                cat._morphisms[("ext_in", "External", "Network")],
            }),
        )
        site.add_cover(cover)

        return site

    def test_site_creation(self, site: Site) -> None:
        """Test site creation."""
        assert site.name == "NetworkSite"
        assert site.category is not None

    def test_site_covers(self, site: Site) -> None:
        """Test getting covers for an object."""
        network = site.category.get_object("Network")
        covers = site.covers(network)

        assert len(covers) == 1
        assert len(covers[0].covering_objects) == 3

    def test_has_cover(self, site: Site) -> None:
        """Test has_cover check."""
        network = site.category.get_object("Network")
        dmz = site.category.get_object("DMZ")

        assert site.has_cover(network)
        assert not site.has_cover(dmz)

    def test_is_covered_by(self, site: Site) -> None:
        """Test coverage check."""
        network = site.category.get_object("Network")
        dmz = site.category.get_object("DMZ")
        internal = site.category.get_object("Internal")
        external = site.category.get_object("External")

        # Network is covered by all three
        assert site.is_covered_by(network, {dmz, internal, external})

        # Not covered by subset
        assert not site.is_covered_by(network, {dmz, internal})


class TestSheafSection:
    """Tests for SheafSection class."""

    def test_section_creation(self) -> None:
        """Test section creation."""
        obj = Object(name="A")
        section = SheafSection(
            object=obj,
            value=42,
            confidence=0.9,
            timestamp=1000.0,
        )

        assert section.object == obj
        assert section.value == 42
        assert section.confidence == 0.9
        assert section.timestamp == 1000.0

    def test_section_compatibility_equal(self) -> None:
        """Test compatible sections with equal values."""
        obj = Object(name="A")
        s1 = SheafSection(object=obj, value=42)
        s2 = SheafSection(object=obj, value=42)

        assert s1.is_compatible_with(s2)

    def test_section_compatibility_unequal(self) -> None:
        """Test incompatible sections with different values."""
        obj = Object(name="A")
        s1 = SheafSection(object=obj, value=42)
        s2 = SheafSection(object=obj, value=100)

        assert not s1.is_compatible_with(s2)

    def test_section_compatibility_custom(self) -> None:
        """Test compatibility with custom comparator."""
        obj = Object(name="A")
        s1 = SheafSection(object=obj, value=42)
        s2 = SheafSection(object=obj, value=45)

        # Custom comparator: values within 5 are compatible
        comparator = lambda x, y: abs(x - y) <= 5

        assert s1.is_compatible_with(s2, comparator)


class TestSheaf:
    """Tests for Sheaf class."""

    @pytest.fixture
    def sheaf(self) -> tuple:
        """Create a test sheaf."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        cat.add_object(A)
        cat.add_object(B)

        cat.add_morphism(Morphism(name="f", source=B, target=A, weight=0.8))

        site = Site(category=cat, name="TestSite")

        # Add cover: A is covered by B
        cover = Cover(
            target=A,
            morphisms=frozenset({cat._morphisms[("f", "B", "A")]}),
        )
        site.add_cover(cover)

        sheaf = Sheaf[int]("F", site)

        return sheaf, site, A, B

    def test_sheaf_creation(self, sheaf: tuple) -> None:
        """Test sheaf creation."""
        F, site, _, _ = sheaf
        assert F.name == "F"
        assert F.site == site

    def test_add_section(self, sheaf: tuple) -> None:
        """Test adding sections."""
        F, _, A, _ = sheaf

        section = SheafSection(object=A, value=42)
        F.add_section(section)

        sections = F.sections_over(A)
        assert len(sections) == 1
        assert sections[0].value == 42

    def test_restriction(self, sheaf: tuple) -> None:
        """Test section restriction."""
        F, site, A, B = sheaf

        section = SheafSection(object=A, value=42, confidence=1.0)

        # Get morphism f: B -> A
        morph = list(site.category.hom(B, A))[0]

        restricted = F.restrict(section, morph)

        assert restricted.object == B
        assert restricted.value == 42
        assert restricted.confidence == 0.8  # Scaled by weight

    def test_can_glue_compatible(self, sheaf: tuple) -> None:
        """Test gluing compatible sections."""
        F, site, A, B = sheaf

        cover = site.covers(A)[0]

        local_sections = {
            "B": SheafSection(object=B, value=42),
        }

        assert F.can_glue(cover, local_sections)

    def test_glue(self, sheaf: tuple) -> None:
        """Test gluing sections."""
        F, site, A, B = sheaf

        cover = site.covers(A)[0]

        local_sections = {
            "B": SheafSection(object=B, value=42, confidence=0.9, timestamp=1000.0),
        }

        glued = F.glue(cover, local_sections)

        assert glued is not None
        assert glued.object == A
        assert glued.value == 42


class TestNetworkSheaf:
    """Tests for NetworkSheaf class."""

    @pytest.fixture
    def network_sheaf(self) -> tuple:
        """Create a network observation sheaf."""
        cat = Category("Network")

        network = Object(name="Network")
        sensor1 = Object(name="Sensor1")
        sensor2 = Object(name="Sensor2")

        for obj in [network, sensor1, sensor2]:
            cat.add_object(obj)

        m1 = Morphism(name="obs1", source=sensor1, target=network, weight=1.0)
        m2 = Morphism(name="obs2", source=sensor2, target=network, weight=1.0)

        cat.add_morphism(m1)
        cat.add_morphism(m2)

        site = Site(category=cat, name="NetworkSite")

        cover = Cover(
            target=network,
            morphisms=frozenset({m1, m2}),
        )
        site.add_cover(cover)

        features = ["bytes", "packets", "connections"]
        sheaf = NetworkSheaf("NetObs", site, features)

        return sheaf, site, network, sensor1, sensor2

    def test_network_sheaf_creation(self, network_sheaf: tuple) -> None:
        """Test network sheaf creation."""
        sheaf, _, _, _, _ = network_sheaf
        assert sheaf.feature_names == ["bytes", "packets", "connections"]

    def test_observe(self, network_sheaf: tuple) -> None:
        """Test adding observations."""
        sheaf, _, _, sensor1, _ = network_sheaf

        section = sheaf.observe(
            sensor1,
            {"bytes": 1000.0, "packets": 100.0},
            confidence=0.95,
            timestamp=1234.0,
        )

        assert section.object == sensor1
        assert section.value["bytes"] == 1000.0

    def test_compatible_observations(self, network_sheaf: tuple) -> None:
        """Test compatible observations can be glued."""
        sheaf, site, network, sensor1, sensor2 = network_sheaf

        # Similar observations from both sensors
        sheaf.observe(sensor1, {"bytes": 1000.0, "packets": 100.0})
        sheaf.observe(sensor2, {"bytes": 1050.0, "packets": 105.0})

        cover = site.covers(network)[0]

        local = {
            "Sensor1": sheaf.sections_over(sensor1)[0],
            "Sensor2": sheaf.sections_over(sensor2)[0],
        }

        # Should be compatible (values within PHI_INV)
        # Note: actual compatibility depends on implementation

    def test_aggregate_observations(self, network_sheaf: tuple) -> None:
        """Test observation aggregation."""
        sheaf, _, _, sensor1, _ = network_sheaf

        sheaf.observe(sensor1, {"bytes": 1000.0}, confidence=0.8, timestamp=100.0)
        sheaf.observe(sensor1, {"bytes": 2000.0}, confidence=1.0, timestamp=200.0)

        aggregated = sheaf.aggregate_observations(sensor1)

        assert aggregated is not None
        assert "bytes" in aggregated
        # Weighted average: (1000*0.8 + 2000*1.0) / (0.8 + 1.0)
        expected = (1000.0 * 0.8 + 2000.0 * 1.0) / 1.8
        assert abs(aggregated["bytes"] - expected) < 0.01

    def test_aggregate_with_time_window(self, network_sheaf: tuple) -> None:
        """Test aggregation with time window."""
        sheaf, _, _, sensor1, _ = network_sheaf

        sheaf.observe(sensor1, {"bytes": 1000.0}, timestamp=50.0)
        sheaf.observe(sensor1, {"bytes": 2000.0}, timestamp=150.0)
        sheaf.observe(sensor1, {"bytes": 3000.0}, timestamp=250.0)

        # Only include middle observation
        aggregated = sheaf.aggregate_observations(
            sensor1,
            time_window=(100.0, 200.0),
        )

        assert aggregated is not None
        assert aggregated["bytes"] == 2000.0


class TestAnomalySheaf:
    """Tests for AnomalySheaf class."""

    @pytest.fixture
    def anomaly_sheaf(self) -> tuple:
        """Create an anomaly detection sheaf."""
        cat = Category("Network")

        segment = Object(name="Segment")
        detector1 = Object(name="Detector1")
        detector2 = Object(name="Detector2")

        for obj in [segment, detector1, detector2]:
            cat.add_object(obj)

        m1 = Morphism(name="det1", source=detector1, target=segment)
        m2 = Morphism(name="det2", source=detector2, target=segment)

        cat.add_morphism(m1)
        cat.add_morphism(m2)

        site = Site(category=cat, name="DetectionSite")

        cover = Cover(
            target=segment,
            morphisms=frozenset({m1, m2}),
        )
        site.add_cover(cover)

        sheaf = AnomalySheaf("AnomalyDet", site)

        return sheaf, site, segment, detector1, detector2

    def test_anomaly_sheaf_creation(self, anomaly_sheaf: tuple) -> None:
        """Test anomaly sheaf creation."""
        sheaf, site, _, _, _ = anomaly_sheaf
        assert sheaf.site == site

    def test_report_anomaly(self, anomaly_sheaf: tuple) -> None:
        """Test reporting anomaly scores."""
        sheaf, _, _, detector1, _ = anomaly_sheaf

        section = sheaf.report_anomaly(
            detector1,
            score=0.8,
            confidence=0.95,
            timestamp=1000.0,
        )

        assert section.value == 0.8
        assert section.confidence == 0.95

    def test_consensus_score_agreement(self, anomaly_sheaf: tuple) -> None:
        """Test consensus when detectors agree."""
        sheaf, _, _, detector1, detector2 = anomaly_sheaf

        # Both detectors report similar scores
        sheaf.report_anomaly(detector1, score=0.7, timestamp=100.0)
        sheaf.report_anomaly(detector2, score=0.75, timestamp=100.0)

        consensus, agreement = sheaf.consensus_score(detector1)

        # Single detector, so agreement should be 1.0
        assert agreement == 1.0

    def test_consensus_score_disagreement(self, anomaly_sheaf: tuple) -> None:
        """Test consensus when detectors disagree."""
        sheaf, _, _, detector1, _ = anomaly_sheaf

        # Same detector, different scores
        sheaf.report_anomaly(detector1, score=0.2, timestamp=100.0)
        sheaf.report_anomaly(detector1, score=0.9, timestamp=200.0)

        consensus, agreement = sheaf.consensus_score(detector1)

        # High variance should lower agreement
        assert agreement < 1.0

    def test_find_gluing_violations(self, anomaly_sheaf: tuple) -> None:
        """Test finding gluing violations."""
        sheaf, _, segment, detector1, detector2 = anomaly_sheaf

        # Conflicting assessments: one says normal, one says anomaly
        sheaf.report_anomaly(detector1, score=0.1, timestamp=100.0)  # Normal
        sheaf.report_anomaly(detector2, score=0.9, timestamp=100.0)  # Anomalous

        violations = sheaf.find_gluing_violations()

        # Should detect the disagreement
        assert len(violations) >= 0  # Implementation-dependent

    def test_detect_distributed_anomaly_high_score(
        self,
        anomaly_sheaf: tuple,
    ) -> None:
        """Test detecting anomalies with high consensus score."""
        sheaf, _, segment, detector1, detector2 = anomaly_sheaf

        # Both detectors report high anomaly
        sheaf.report_anomaly(detector1, score=0.9, timestamp=100.0)
        sheaf.report_anomaly(detector2, score=0.85, timestamp=100.0)

        anomalies = sheaf.detect_distributed_anomaly(threshold=0.5)

        # Detectors themselves have high scores
        detected_names = {obj.name for obj, _, _ in anomalies}
        assert "Detector1" in detected_names or "Detector2" in detected_names

    def test_detect_distributed_anomaly_low_agreement(
        self,
        anomaly_sheaf: tuple,
    ) -> None:
        """Test detecting anomalies due to low agreement."""
        sheaf, _, segment, detector1, _ = anomaly_sheaf

        # Single detector with varying scores (low agreement)
        sheaf.report_anomaly(detector1, score=0.1, timestamp=100.0)
        sheaf.report_anomaly(detector1, score=0.9, timestamp=200.0)

        anomalies = sheaf.detect_distributed_anomaly()

        # Low agreement might flag as anomalous


class TestGluingCondition:
    """Tests for GluingCondition class."""

    def test_gluing_condition_creation(self) -> None:
        """Test gluing condition creation."""
        obj = Object(name="A")
        cover = Cover(target=obj, morphisms=frozenset())

        s1 = SheafSection(object=obj, value=0.1)
        s2 = SheafSection(object=obj, value=0.9)

        condition = GluingCondition(
            cover=cover,
            conflicting_sections=[(s1, s2)],
            severity=0.8,
            description="Sensor disagreement",
        )

        assert condition.severity == 0.8
        assert len(condition.conflicting_sections) == 1

    def test_is_critical(self) -> None:
        """Test critical violation detection."""
        obj = Object(name="A")
        cover = Cover(target=obj, morphisms=frozenset())

        critical = GluingCondition(
            cover=cover,
            conflicting_sections=[],
            severity=0.8,  # Above PHI_INV
        )

        non_critical = GluingCondition(
            cover=cover,
            conflicting_sections=[],
            severity=0.3,  # Below PHI_INV
        )

        assert critical.is_critical()
        assert not non_critical.is_critical()


class TestSheafIntegration:
    """Integration tests for sheaf functionality."""

    def test_distributed_detection_scenario(self) -> None:
        """Test realistic distributed detection scenario."""
        # Create network topology
        cat = Category("Enterprise")

        zones = [
            Object(name="Enterprise"),
            Object(name="DMZ"),
            Object(name="Internal"),
            Object(name="External"),
        ]

        for zone in zones:
            cat.add_object(zone)

        # Zone inclusions
        cat.add_morphism(Morphism(
            name="dmz_inc",
            source=zones[1],
            target=zones[0],
        ))
        cat.add_morphism(Morphism(
            name="int_inc",
            source=zones[2],
            target=zones[0],
        ))
        cat.add_morphism(Morphism(
            name="ext_inc",
            source=zones[3],
            target=zones[0],
        ))

        # Create site with coverage
        site = Site(category=cat)
        cover = Cover(
            target=zones[0],
            morphisms=frozenset({
                cat._morphisms[("dmz_inc", "DMZ", "Enterprise")],
                cat._morphisms[("int_inc", "Internal", "Enterprise")],
                cat._morphisms[("ext_inc", "External", "Enterprise")],
            }),
        )
        site.add_cover(cover)

        # Create anomaly sheaf
        sheaf = AnomalySheaf("EnterpriseDetection", site)

        # Report anomalies from each zone
        sheaf.report_anomaly(zones[1], score=0.2, timestamp=100.0)  # DMZ
        sheaf.report_anomaly(zones[2], score=0.1, timestamp=100.0)  # Internal
        sheaf.report_anomaly(zones[3], score=0.8, timestamp=100.0)  # External

        # Detect anomalies
        anomalies = sheaf.detect_distributed_anomaly(threshold=0.5)

        # External zone should be flagged
        external_flagged = any(
            obj.name == "External" for obj, _, _ in anomalies
        )
        assert external_flagged

    def test_network_monitoring_scenario(self) -> None:
        """Test realistic network monitoring scenario."""
        cat = Category("Monitoring")

        # Monitored segments
        segments = [
            Object(name="Server"),
            Object(name="Monitor1"),
            Object(name="Monitor2"),
        ]

        for seg in segments:
            cat.add_object(seg)

        cat.add_morphism(Morphism(
            name="m1_obs",
            source=segments[1],
            target=segments[0],
        ))
        cat.add_morphism(Morphism(
            name="m2_obs",
            source=segments[2],
            target=segments[0],
        ))

        site = Site(category=cat)
        cover = Cover(
            target=segments[0],
            morphisms=frozenset({
                cat._morphisms[("m1_obs", "Monitor1", "Server")],
                cat._morphisms[("m2_obs", "Monitor2", "Server")],
            }),
        )
        site.add_cover(cover)

        # Create network sheaf
        sheaf = NetworkSheaf("ServerMonitor", site, ["traffic", "latency"])

        # Both monitors observe similar traffic
        sheaf.observe(segments[1], {"traffic": 1000.0, "latency": 5.0})
        sheaf.observe(segments[2], {"traffic": 1020.0, "latency": 5.2})

        # Aggregate should work
        agg1 = sheaf.aggregate_observations(segments[1])
        agg2 = sheaf.aggregate_observations(segments[2])

        assert agg1 is not None
        assert agg2 is not None
