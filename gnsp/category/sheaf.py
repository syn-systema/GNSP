"""
Sheaf-theoretic structures for distributed anomaly detection.

Sheaves provide a framework for:
- Gluing local data consistently into global data
- Detecting inconsistencies (anomalies) that prevent gluing
- Distributed detection across network segments

A sheaf F on a site (C, J) assigns:
- To each object U, a set F(U) of "sections over U"
- To each morphism f: V -> U, a restriction map F(U) -> F(V)

Satisfying:
- Locality: If sections agree on a cover, they are equal
- Gluing: Compatible sections on a cover can be glued

Sheaf conditions correspond to consistency requirements for
distributed security data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np

from gnsp.category.base import Category, Morphism, Object
from gnsp.constants import PHI, PHI_INV, THRESHOLD_HIGH, THRESHOLD_MID


T = TypeVar("T")


@dataclass
class Cover:
    """
    A covering family for an object.

    A cover of U consists of morphisms {f_i: U_i -> U} such that
    they "cover" U according to the Grothendieck topology.

    In network security context, a cover represents different
    sensors/views of a network segment.
    """

    target: Object  # U
    morphisms: FrozenSet[Morphism]  # {f_i: U_i -> U}

    def __post_init__(self) -> None:
        """Verify all morphisms target the same object."""
        for m in self.morphisms:
            if m.target != self.target:
                raise ValueError(
                    f"Morphism {m.name} does not target {self.target.name}"
                )

    @property
    def covering_objects(self) -> Set[Object]:
        """Get the objects in the cover."""
        return {m.source for m in self.morphisms}

    def is_singleton(self) -> bool:
        """Check if cover is a single morphism."""
        return len(self.morphisms) == 1


@dataclass
class Site:
    """
    A Grothendieck site (category with coverage).

    A site consists of a category C together with a Grothendieck topology J,
    which assigns to each object U a collection J(U) of covering families.

    For network security:
    - Objects are network segments/zones
    - Morphisms are inclusion/observation relationships
    - Covers represent redundant monitoring
    """

    category: Category
    name: str = ""
    _covers: Dict[str, List[Cover]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"Site({self.category.name})"

    def add_cover(self, cover: Cover) -> None:
        """Add a covering family for an object."""
        obj_name = cover.target.name
        if obj_name not in self._covers:
            self._covers[obj_name] = []
        self._covers[obj_name].append(cover)

    def covers(self, obj: Object) -> List[Cover]:
        """Get all covers of an object."""
        return self._covers.get(obj.name, [])

    def has_cover(self, obj: Object) -> bool:
        """Check if object has at least one cover."""
        return obj.name in self._covers and len(self._covers[obj.name]) > 0

    def is_covered_by(self, obj: Object, covering_objs: Set[Object]) -> bool:
        """Check if obj is covered by the given objects."""
        for cover in self.covers(obj):
            if cover.covering_objects.issubset(covering_objs):
                return True
        return False


@dataclass
class SheafSection(Generic[T]):
    """
    A section of a sheaf over an object.

    Represents local data assigned to a network segment.

    Attributes:
        object: The object this section is over.
        value: The section data.
        confidence: Confidence in this section (0 to 1).
        timestamp: Time the section was observed.
    """

    object: Object
    value: T
    confidence: float = 1.0
    timestamp: float = 0.0

    def is_compatible_with(
        self,
        other: "SheafSection[T]",
        comparator: Optional[Callable[[T, T], bool]] = None,
    ) -> bool:
        """
        Check if this section is compatible with another.

        Two sections are compatible if they can be glued together,
        i.e., they agree on overlapping regions.

        Args:
            other: Another section.
            comparator: Function to compare values (default: equality).

        Returns:
            True if sections are compatible.
        """
        if comparator:
            return comparator(self.value, other.value)
        return self.value == other.value


class Presheaf(ABC, Generic[T]):
    """
    Abstract base class for presheaves.

    A presheaf F on a category C assigns:
    - To each object U, a set F(U)
    - To each morphism f: V -> U, a restriction F(f): F(U) -> F(V)

    Presheaves may not satisfy the sheaf conditions.
    """

    def __init__(self, name: str, site: Site) -> None:
        """
        Initialize a presheaf.

        Args:
            name: Presheaf name.
            site: Underlying site.
        """
        self.name = name
        self.site = site
        self._sections: Dict[str, List[SheafSection[T]]] = {}

    @abstractmethod
    def restrict(
        self,
        section: SheafSection[T],
        morphism: Morphism,
    ) -> SheafSection[T]:
        """
        Restrict a section along a morphism.

        Given a section s in F(U) and f: V -> U,
        compute F(f)(s) in F(V).

        Args:
            section: Section over the target of morphism.
            morphism: Morphism f: V -> U.

        Returns:
            Restricted section over source of morphism.
        """
        pass

    def sections_over(self, obj: Object) -> List[SheafSection[T]]:
        """Get all sections over an object."""
        return self._sections.get(obj.name, [])

    def add_section(self, section: SheafSection[T]) -> None:
        """Add a section to the presheaf."""
        obj_name = section.object.name
        if obj_name not in self._sections:
            self._sections[obj_name] = []
        self._sections[obj_name].append(section)

    def is_functorial(self) -> bool:
        """
        Check if restriction maps are functorial.

        For presheaf to be well-defined:
        - F(id_U) = id_F(U)
        - F(f;g) = F(g);F(f)

        Returns:
            True if functoriality holds.
        """
        # Check identity preservation (sampling)
        for obj in self.site.category.objects:
            identity = self.site.category.get_identity(obj)
            for section in self.sections_over(obj):
                restricted = self.restrict(section, identity)
                if restricted.value != section.value:
                    return False

        return True


class Sheaf(Presheaf[T]):
    """
    A sheaf on a site.

    A presheaf F is a sheaf if it satisfies:
    1. Locality: If two sections agree on a cover, they are equal
    2. Gluing: Compatible sections on a cover can be glued

    These conditions ensure consistency of distributed security data.
    """

    def __init__(
        self,
        name: str,
        site: Site,
        value_comparator: Optional[Callable[[T, T], bool]] = None,
    ) -> None:
        """
        Initialize a sheaf.

        Args:
            name: Sheaf name.
            site: Underlying site.
            value_comparator: Function to compare section values.
        """
        super().__init__(name, site)
        self.value_comparator = value_comparator or (lambda x, y: x == y)

    def restrict(
        self,
        section: SheafSection[T],
        morphism: Morphism,
    ) -> SheafSection[T]:
        """Default restriction (identity on value)."""
        return SheafSection(
            object=morphism.source,
            value=section.value,
            confidence=section.confidence * morphism.weight,
            timestamp=section.timestamp,
        )

    def check_locality(
        self,
        cover: Cover,
        section1: SheafSection[T],
        section2: SheafSection[T],
    ) -> bool:
        """
        Check locality condition.

        If section1 and section2 agree when restricted to every
        object in the cover, then they should be equal.

        Args:
            cover: Covering family.
            section1: First section over cover target.
            section2: Second section over cover target.

        Returns:
            True if locality holds.
        """
        if section1.object != cover.target or section2.object != cover.target:
            return True  # Not applicable

        # Restrict both sections to each covering object
        for morphism in cover.morphisms:
            r1 = self.restrict(section1, morphism)
            r2 = self.restrict(section2, morphism)

            if not self.value_comparator(r1.value, r2.value):
                # Restrictions disagree, so sections can differ
                return True

        # Restrictions agree everywhere, sections should be equal
        return self.value_comparator(section1.value, section2.value)

    def can_glue(
        self,
        cover: Cover,
        local_sections: Dict[str, SheafSection[T]],
    ) -> bool:
        """
        Check if local sections can be glued.

        Local sections are compatible if they agree on overlaps,
        i.e., for any f_i: U_i -> U and f_j: U_j -> U, the sections
        s_i and s_j agree when restricted to U_i x_U U_j (fiber product).

        Args:
            cover: Covering family.
            local_sections: Dict mapping covering object names to sections.

        Returns:
            True if sections can be glued.
        """
        covering_objs = list(cover.covering_objects)

        # Check pairwise compatibility
        for i, obj_i in enumerate(covering_objs):
            for obj_j in covering_objs[i + 1:]:
                if obj_i.name not in local_sections:
                    continue
                if obj_j.name not in local_sections:
                    continue

                s_i = local_sections[obj_i.name]
                s_j = local_sections[obj_j.name]

                # Check if they agree (simplified: direct comparison)
                if not s_i.is_compatible_with(s_j, self.value_comparator):
                    return False

        return True

    def glue(
        self,
        cover: Cover,
        local_sections: Dict[str, SheafSection[T]],
        combiner: Optional[Callable[[List[T]], T]] = None,
    ) -> Optional[SheafSection[T]]:
        """
        Glue local sections into a global section.

        Args:
            cover: Covering family.
            local_sections: Dict mapping covering object names to sections.
            combiner: Function to combine values (default: first value).

        Returns:
            Glued global section, or None if gluing fails.
        """
        if not self.can_glue(cover, local_sections):
            return None

        # Gather values
        values = [
            s.value for name, s in local_sections.items()
            if name in {obj.name for obj in cover.covering_objects}
        ]

        if not values:
            return None

        # Combine values
        if combiner:
            combined_value = combiner(values)
        else:
            combined_value = values[0]

        # Compute combined confidence (golden ratio weighted)
        confidences = [
            s.confidence for s in local_sections.values()
        ]
        combined_confidence = sum(confidences) / len(confidences) * PHI_INV

        # Use latest timestamp
        timestamps = [s.timestamp for s in local_sections.values()]
        latest_timestamp = max(timestamps) if timestamps else 0.0

        return SheafSection(
            object=cover.target,
            value=combined_value,
            confidence=combined_confidence,
            timestamp=latest_timestamp,
        )


@dataclass
class GluingCondition:
    """
    Represents a gluing condition violation.

    When local sections cannot be glued, this captures
    the inconsistency for anomaly detection.
    """

    cover: Cover
    conflicting_sections: List[Tuple[SheafSection, SheafSection]]
    severity: float  # 0 to 1
    description: str = ""

    def is_critical(self) -> bool:
        """Check if violation is critical (above phi threshold)."""
        return self.severity > PHI_INV


class NetworkSheaf(Sheaf[Dict[str, float]]):
    """
    Sheaf of network observations.

    Sections are dictionaries mapping feature names to values.
    Used for distributed network monitoring where different sensors
    observe different (but overlapping) network segments.
    """

    def __init__(
        self,
        name: str,
        site: Site,
        feature_names: List[str],
    ) -> None:
        """
        Initialize network observation sheaf.

        Args:
            name: Sheaf name.
            site: Network topology site.
            feature_names: Names of observable features.
        """
        super().__init__(name, site, self._compare_observations)
        self.feature_names = feature_names

    def _compare_observations(
        self,
        obs1: Dict[str, float],
        obs2: Dict[str, float],
    ) -> bool:
        """
        Compare observations for compatibility.

        Observations are compatible if shared features are similar.
        """
        shared_keys = set(obs1.keys()) & set(obs2.keys())

        if not shared_keys:
            return True  # No overlap, vacuously compatible

        for key in shared_keys:
            diff = abs(obs1[key] - obs2[key])
            # Use golden ratio threshold for similarity
            if diff > PHI_INV:
                return False

        return True

    def restrict(
        self,
        section: SheafSection[Dict[str, float]],
        morphism: Morphism,
    ) -> SheafSection[Dict[str, float]]:
        """
        Restrict observation to sub-segment.

        Applies weight scaling to represent partial observability.
        """
        restricted_value = {
            k: v * morphism.weight
            for k, v in section.value.items()
        }

        return SheafSection(
            object=morphism.source,
            value=restricted_value,
            confidence=section.confidence * morphism.weight,
            timestamp=section.timestamp,
        )

    def observe(
        self,
        obj: Object,
        features: Dict[str, float],
        confidence: float = 1.0,
        timestamp: float = 0.0,
    ) -> SheafSection[Dict[str, float]]:
        """
        Create and add an observation section.

        Args:
            obj: Network segment being observed.
            features: Observed feature values.
            confidence: Observation confidence.
            timestamp: Observation time.

        Returns:
            Created section.
        """
        section = SheafSection(
            object=obj,
            value=features,
            confidence=confidence,
            timestamp=timestamp,
        )
        self.add_section(section)
        return section

    def aggregate_observations(
        self,
        obj: Object,
        time_window: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Aggregate all observations over an object.

        Args:
            obj: Object to aggregate over.
            time_window: Optional (start, end) time range.

        Returns:
            Aggregated feature values, or None if no observations.
        """
        sections = self.sections_over(obj)

        if time_window:
            start, end = time_window
            sections = [
                s for s in sections
                if start <= s.timestamp <= end
            ]

        if not sections:
            return None

        # Confidence-weighted aggregation
        aggregated: Dict[str, float] = {}
        total_confidence: Dict[str, float] = {}

        for section in sections:
            for key, value in section.value.items():
                if key not in aggregated:
                    aggregated[key] = 0.0
                    total_confidence[key] = 0.0

                aggregated[key] += value * section.confidence
                total_confidence[key] += section.confidence

        # Normalize
        for key in aggregated:
            if total_confidence[key] > 0:
                aggregated[key] /= total_confidence[key]

        return aggregated


class AnomalySheaf(Sheaf[float]):
    """
    Sheaf of anomaly scores.

    Sections are anomaly scores from different detectors/sensors.
    Gluing conditions detect disagreements between sensors,
    which may indicate:
    - Sensor malfunction
    - Localized attacks
    - Evasion attempts
    """

    def __init__(self, name: str, site: Site) -> None:
        """
        Initialize anomaly score sheaf.

        Args:
            name: Sheaf name.
            site: Network topology site.
        """
        super().__init__(name, site, self._compare_scores)
        self._threshold = THRESHOLD_MID

    def _compare_scores(self, score1: float, score2: float) -> bool:
        """
        Compare anomaly scores for compatibility.

        Scores are compatible if they agree on anomaly/normal classification.
        """
        # Both above threshold or both below
        above1 = score1 > self._threshold
        above2 = score2 > self._threshold

        if above1 != above2:
            # Classification disagreement
            return False

        # Also check magnitude similarity
        diff = abs(score1 - score2)
        return diff < PHI_INV

    def restrict(
        self,
        section: SheafSection[float],
        morphism: Morphism,
    ) -> SheafSection[float]:
        """Restrict anomaly score with confidence scaling."""
        return SheafSection(
            object=morphism.source,
            value=section.value,
            confidence=section.confidence * morphism.weight,
            timestamp=section.timestamp,
        )

    def report_anomaly(
        self,
        obj: Object,
        score: float,
        confidence: float = 1.0,
        timestamp: float = 0.0,
    ) -> SheafSection[float]:
        """
        Report anomaly score for a network segment.

        Args:
            obj: Network segment.
            score: Anomaly score (0 = normal, 1 = highly anomalous).
            confidence: Detection confidence.
            timestamp: Detection time.

        Returns:
            Created section.
        """
        section = SheafSection(
            object=obj,
            value=score,
            confidence=confidence,
            timestamp=timestamp,
        )
        self.add_section(section)
        return section

    def find_gluing_violations(
        self,
    ) -> List[GluingCondition]:
        """
        Find all gluing condition violations.

        These represent inconsistencies between different sensors'
        anomaly assessments.

        Returns:
            List of gluing violations.
        """
        violations: List[GluingCondition] = []

        for obj in self.site.category.objects:
            for cover in self.site.covers(obj):
                # Collect sections from covering objects
                local_sections: Dict[str, SheafSection[float]] = {}

                for covering_obj in cover.covering_objects:
                    sections = self.sections_over(covering_obj)
                    if sections:
                        # Use most recent section
                        latest = max(sections, key=lambda s: s.timestamp)
                        local_sections[covering_obj.name] = latest

                if len(local_sections) < 2:
                    continue

                # Check for violations
                if not self.can_glue(cover, local_sections):
                    # Find conflicting pairs
                    conflicts: List[Tuple[SheafSection, SheafSection]] = []
                    sections_list = list(local_sections.values())

                    for i, s1 in enumerate(sections_list):
                        for s2 in sections_list[i + 1:]:
                            if not s1.is_compatible_with(s2, self._compare_scores):
                                conflicts.append((s1, s2))

                    # Compute severity
                    if conflicts:
                        max_diff = max(
                            abs(c[0].value - c[1].value)
                            for c in conflicts
                        )
                        severity = min(1.0, max_diff * PHI)

                        violation = GluingCondition(
                            cover=cover,
                            conflicting_sections=conflicts,
                            severity=severity,
                            description=(
                                f"Sensor disagreement over {obj.name}: "
                                f"{len(conflicts)} conflicts"
                            ),
                        )
                        violations.append(violation)

        return violations

    def consensus_score(
        self,
        obj: Object,
        time_window: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Compute consensus anomaly score across all sensors.

        Args:
            obj: Object to compute consensus for.
            time_window: Optional time range.

        Returns:
            Tuple of (consensus_score, agreement_level).
        """
        sections = self.sections_over(obj)

        if time_window:
            start, end = time_window
            sections = [
                s for s in sections
                if start <= s.timestamp <= end
            ]

        if not sections:
            return 0.0, 0.0

        scores = [s.value for s in sections]
        confidences = [s.confidence for s in sections]

        # Weighted average score
        total_conf = sum(confidences)
        if total_conf > 0:
            consensus = sum(
                s * c for s, c in zip(scores, confidences)
            ) / total_conf
        else:
            consensus = sum(scores) / len(scores)

        # Agreement level (inverse of variance)
        if len(scores) > 1:
            variance = np.var(scores)
            agreement = 1.0 / (1.0 + variance * PHI)
        else:
            agreement = 1.0

        return consensus, agreement

    def detect_distributed_anomaly(
        self,
        threshold: float = THRESHOLD_HIGH,
    ) -> List[Tuple[Object, float, float]]:
        """
        Detect anomalies across the network using sheaf structure.

        Returns objects where:
        1. Consensus score exceeds threshold, OR
        2. Significant gluing violations exist (sensor disagreement)

        Args:
            threshold: Anomaly score threshold.

        Returns:
            List of (object, anomaly_score, confidence) tuples.
        """
        anomalies: List[Tuple[Object, float, float]] = []

        for obj in self.site.category.objects:
            consensus, agreement = self.consensus_score(obj)

            if consensus > threshold:
                anomalies.append((obj, consensus, agreement))
            elif agreement < PHI_INV:
                # Low agreement indicates potential evasion/attack
                anomalies.append((obj, consensus, agreement))

        # Check gluing violations
        violations = self.find_gluing_violations()
        for violation in violations:
            if violation.is_critical():
                # Add covered objects with violation severity
                for conflict in violation.conflicting_sections:
                    obj = conflict[0].object
                    score = max(conflict[0].value, conflict[1].value)
                    anomalies.append((obj, score, 1.0 - violation.severity))

        # Deduplicate and sort by score
        seen: Set[str] = set()
        unique_anomalies: List[Tuple[Object, float, float]] = []

        for obj, score, conf in sorted(anomalies, key=lambda x: x[1], reverse=True):
            if obj.name not in seen:
                seen.add(obj.name)
                unique_anomalies.append((obj, score, conf))

        return unique_anomalies
