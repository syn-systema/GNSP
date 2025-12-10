"""
Functors for category-theoretic security analysis.

Functors are structure-preserving maps between categories.
In the security context, functors enable:
- Mapping protocol behaviors to traffic patterns
- Abstracting security properties across domains
- Detecting violations as non-functorial behavior

A functor F: C -> D consists of:
- Object mapping: F(A) for each object A in C
- Morphism mapping: F(f) for each morphism f in C
- Identity preservation: F(id_A) = id_F(A)
- Composition preservation: F(f;g) = F(f);F(g)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

import numpy as np

from gnsp.category.base import Category, Morphism, Object
from gnsp.constants import PHI, PHI_INV


class Functor(ABC):
    """
    Abstract base class for functors between categories.

    A functor F: C -> D maps:
    - Objects of C to objects of D
    - Morphisms of C to morphisms of D

    While preserving:
    - Identity morphisms: F(id_A) = id_F(A)
    - Composition: F(f;g) = F(f);F(g)
    """

    def __init__(
        self,
        name: str,
        source: Category,
        target: Category,
    ) -> None:
        """
        Initialize a functor.

        Args:
            name: Name identifier for the functor.
            source: Source category C.
            target: Target category D.
        """
        self.name = name
        self.source = source
        self.target = target
        self._object_map: Dict[str, Object] = {}
        self._morphism_map: Dict[str, Morphism] = {}

    @abstractmethod
    def map_object(self, obj: Object) -> Object:
        """
        Map an object from source to target category.

        Args:
            obj: Object in source category.

        Returns:
            Corresponding object in target category.
        """
        pass

    @abstractmethod
    def map_morphism(self, morphism: Morphism) -> Morphism:
        """
        Map a morphism from source to target category.

        Args:
            morphism: Morphism in source category.

        Returns:
            Corresponding morphism in target category.
        """
        pass

    def __call__(self, x: Object | Morphism) -> Object | Morphism:
        """
        Apply functor to object or morphism.

        Args:
            x: Object or morphism in source category.

        Returns:
            Mapped object or morphism in target category.
        """
        if isinstance(x, Morphism):
            return self.map_morphism(x)
        elif isinstance(x, Object):
            return self.map_object(x)
        else:
            raise TypeError(f"Cannot map {type(x)}")

    def preserves_identity(self, obj: Object) -> bool:
        """
        Check if functor preserves identity for an object.

        F(id_A) should equal id_F(A).

        Args:
            obj: Object to check.

        Returns:
            True if identity is preserved.
        """
        id_source = self.source.get_identity(obj)
        mapped_id = self.map_morphism(id_source)
        target_obj = self.map_object(obj)

        # Check if mapped identity is the identity in target
        return (
            mapped_id.source == target_obj
            and mapped_id.target == target_obj
            and mapped_id.is_identity()
        )

    def preserves_composition(self, f: Morphism, g: Morphism) -> bool:
        """
        Check if functor preserves composition.

        F(f;g) should equal F(f);F(g).

        Args:
            f: First morphism.
            g: Second morphism (composable with f).

        Returns:
            True if composition is preserved.
        """
        if not f.is_composable_with(g):
            return True  # Vacuously true

        # F(f;g)
        fg = f.compose(g)
        mapped_fg = self.map_morphism(fg)

        # F(f);F(g)
        mapped_f = self.map_morphism(f)
        mapped_g = self.map_morphism(g)

        if not mapped_f.is_composable_with(mapped_g):
            return False

        composed = mapped_f.compose(mapped_g)

        # Check source and target match
        return (
            mapped_fg.source == composed.source
            and mapped_fg.target == composed.target
        )

    def is_faithful(self) -> bool:
        """
        Check if functor is faithful (injective on hom-sets).

        A functor is faithful if for each pair of objects A, B,
        the map Hom(A,B) -> Hom(F(A),F(B)) is injective.

        Returns:
            True if functor is faithful.
        """
        for obj1 in self.source.objects:
            for obj2 in self.source.objects:
                hom = self.source.hom(obj1, obj2)
                if len(hom) <= 1:
                    continue

                # Check injectivity
                mapped: Set[Tuple[str, str]] = set()
                for morphism in hom:
                    m = self.map_morphism(morphism)
                    key = (m.source.name, m.target.name, m.name)
                    if key in mapped:
                        return False
                    mapped.add(key)

        return True

    def is_full(self) -> bool:
        """
        Check if functor is full (surjective on hom-sets).

        A functor is full if for each pair of objects A, B,
        the map Hom(A,B) -> Hom(F(A),F(B)) is surjective.

        Returns:
            True if functor is full.
        """
        for obj1 in self.source.objects:
            for obj2 in self.source.objects:
                target_obj1 = self.map_object(obj1)
                target_obj2 = self.map_object(obj2)

                source_hom = self.source.hom(obj1, obj2)
                target_hom = self.target.hom(target_obj1, target_obj2)

                # Map source morphisms
                mapped = {self.map_morphism(m).name for m in source_hom}

                # Check all target morphisms are covered
                for m in target_hom:
                    if m.name not in mapped:
                        return False

        return True

    def violation_score(self, morphism: Morphism) -> float:
        """
        Compute how much a morphism violates functorial properties.

        Higher score indicates more severe violation.

        Args:
            morphism: Morphism to check.

        Returns:
            Violation score in [0, 1].
        """
        score = 0.0

        # Check composition with all composable morphisms
        for other in self.source.morphisms:
            if morphism.is_composable_with(other):
                if not self.preserves_composition(morphism, other):
                    score += PHI_INV

            if other.is_composable_with(morphism):
                if not self.preserves_composition(other, morphism):
                    score += PHI_INV

        return min(1.0, score)

    def __repr__(self) -> str:
        return f"Functor({self.name}: {self.source.name} -> {self.target.name})"


class IdentityFunctor(Functor):
    """
    Identity functor Id_C: C -> C.

    Maps every object and morphism to itself.
    """

    def __init__(self, category: Category) -> None:
        """
        Create identity functor for a category.

        Args:
            category: Category to create identity functor for.
        """
        super().__init__(f"Id_{category.name}", category, category)

    def map_object(self, obj: Object) -> Object:
        """Identity mapping for objects."""
        return obj

    def map_morphism(self, morphism: Morphism) -> Morphism:
        """Identity mapping for morphisms."""
        return morphism


class CompositionFunctor(Functor):
    """
    Composition of two functors F;G.

    Given F: A -> B and G: B -> C, creates F;G: A -> C.
    """

    def __init__(self, first: Functor, second: Functor) -> None:
        """
        Compose two functors.

        Args:
            first: Functor F: A -> B.
            second: Functor G: B -> C.

        Raises:
            ValueError: If functors are not composable.
        """
        if first.target != second.source:
            raise ValueError(
                f"Cannot compose {first.name} and {second.name}: "
                f"target {first.target.name} != source {second.source.name}"
            )

        super().__init__(
            f"{first.name};{second.name}",
            first.source,
            second.target,
        )
        self.first = first
        self.second = second

    def map_object(self, obj: Object) -> Object:
        """Apply both functors to object."""
        intermediate = self.first.map_object(obj)
        return self.second.map_object(intermediate)

    def map_morphism(self, morphism: Morphism) -> Morphism:
        """Apply both functors to morphism."""
        intermediate = self.first.map_morphism(morphism)
        return self.second.map_morphism(intermediate)


class ForgetfulFunctor(Functor):
    """
    Forgetful functor that strips structure.

    Maps objects to simpler objects by forgetting certain properties,
    while preserving morphism structure.
    """

    def __init__(
        self,
        name: str,
        source: Category,
        target: Category,
        forget_properties: Set[str],
    ) -> None:
        """
        Create a forgetful functor.

        Args:
            name: Functor name.
            source: Source category (with structure).
            target: Target category (simpler).
            forget_properties: Properties to forget from objects.
        """
        super().__init__(name, source, target)
        self.forget_properties = forget_properties
        self._setup_mapping()

    def _setup_mapping(self) -> None:
        """Set up object and morphism mappings."""
        for obj in self.source.objects:
            # Create simplified object
            new_props = obj.properties - self.forget_properties
            simplified = Object(
                name=obj.name,
                data=obj.data,
                properties=new_props,
            )
            self._object_map[obj.name] = simplified

            # Ensure object exists in target
            if not self.target.get_object(obj.name):
                self.target.add_object(simplified)

    def map_object(self, obj: Object) -> Object:
        """Map object by forgetting properties."""
        if obj.name in self._object_map:
            return self._object_map[obj.name]

        # Create new simplified object
        new_props = obj.properties - self.forget_properties
        simplified = Object(
            name=obj.name,
            data=obj.data,
            properties=new_props,
        )
        self._object_map[obj.name] = simplified
        return simplified

    def map_morphism(self, morphism: Morphism) -> Morphism:
        """Map morphism (structure preserved)."""
        return Morphism(
            name=morphism.name,
            source=self.map_object(morphism.source),
            target=self.map_object(morphism.target),
            weight=morphism.weight,
            data=morphism.data,
        )


class SecurityFunctor(Functor):
    """
    Functor mapping protocol states to security levels.

    Maps protocol category to a totally ordered category
    representing security levels (0 = compromised, 1 = secure).

    This enables tracking security degradation through
    protocol transitions.
    """

    def __init__(
        self,
        name: str,
        protocol_category: Category,
        security_extractor: Optional[Callable[[Object], float]] = None,
    ) -> None:
        """
        Create a security functor.

        Args:
            name: Functor name.
            protocol_category: Source protocol category.
            security_extractor: Function to extract security level from object.
        """
        # Create security level category
        security_cat = Category("Security")

        # Create security level objects
        levels = [
            Object("compromised", data=0.0),
            Object("low", data=0.25),
            Object("medium", data=0.5),
            Object("high", data=0.75),
            Object("secure", data=1.0),
        ]
        for level in levels:
            security_cat.add_object(level)

        # Add transitions (security can only increase or stay same in normal operation)
        for i, src in enumerate(levels):
            for j, tgt in enumerate(levels):
                if i <= j:  # Can increase or stay same
                    morph = Morphism(
                        name=f"{src.name}->{tgt.name}",
                        source=src,
                        target=tgt,
                        weight=1.0 if i == j else PHI_INV,
                    )
                    if i != j:  # Don't add identity twice
                        security_cat.add_morphism(morph)

        super().__init__(name, protocol_category, security_cat)

        self.security_extractor = security_extractor or self._default_extractor
        self._levels = levels
        self._setup_mapping()

    def _default_extractor(self, obj: Object) -> float:
        """Default security level extraction."""
        # Check for security_level attribute
        if hasattr(obj, "security_level"):
            return obj.security_level

        # Check object data
        if isinstance(obj.data, dict) and "security_level" in obj.data:
            return obj.data["security_level"]

        # Default to medium
        return 0.5

    def _level_for_score(self, score: float) -> Object:
        """Get security level object for a score."""
        if score <= 0.125:
            return self._levels[0]  # compromised
        elif score <= 0.375:
            return self._levels[1]  # low
        elif score <= 0.625:
            return self._levels[2]  # medium
        elif score <= 0.875:
            return self._levels[3]  # high
        else:
            return self._levels[4]  # secure

    def _setup_mapping(self) -> None:
        """Set up object mappings based on security extraction."""
        for obj in self.source.objects:
            score = self.security_extractor(obj)
            self._object_map[obj.name] = self._level_for_score(score)

    def map_object(self, obj: Object) -> Object:
        """Map object to security level."""
        if obj.name in self._object_map:
            return self._object_map[obj.name]

        score = self.security_extractor(obj)
        level = self._level_for_score(score)
        self._object_map[obj.name] = level
        return level

    def map_morphism(self, morphism: Morphism) -> Morphism:
        """Map morphism to security transition."""
        src_level = self.map_object(morphism.source)
        tgt_level = self.map_object(morphism.target)

        return Morphism(
            name=f"{src_level.name}->{tgt_level.name}",
            source=src_level,
            target=tgt_level,
            weight=morphism.weight,
            data=morphism.data,
        )

    def security_delta(self, morphism: Morphism) -> float:
        """
        Compute security level change for a transition.

        Args:
            morphism: Protocol transition.

        Returns:
            Change in security level (-1 to 1).
        """
        src_score = self.security_extractor(morphism.source)
        tgt_score = self.security_extractor(morphism.target)
        return tgt_score - src_score

    def is_secure_path(self, path: List[Morphism]) -> bool:
        """
        Check if a path maintains or improves security.

        Args:
            path: List of morphisms forming a path.

        Returns:
            True if path never decreases security.
        """
        for morphism in path:
            if self.security_delta(morphism) < 0:
                return False
        return True


@dataclass
class TrafficPattern:
    """
    Represents a network traffic pattern.

    Attributes:
        source_ip: Source IP address.
        dest_ip: Destination IP address.
        source_port: Source port.
        dest_port: Destination port.
        protocol: Protocol type.
        bytes_sent: Bytes sent.
        bytes_recv: Bytes received.
        packets: Number of packets.
        duration: Duration in seconds.
        flags: Protocol-specific flags.
    """

    source_ip: str = ""
    dest_ip: str = ""
    source_port: int = 0
    dest_port: int = 0
    protocol: str = ""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets: int = 0
    duration: float = 0.0
    flags: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML analysis."""
        return np.array([
            self.source_port / 65535,
            self.dest_port / 65535,
            np.log1p(self.bytes_sent) / 20,
            np.log1p(self.bytes_recv) / 20,
            np.log1p(self.packets) / 10,
            min(self.duration / 3600, 1.0),
        ])


class TrafficFunctor(Functor):
    """
    Functor mapping protocol states to traffic patterns.

    Maps protocol category to traffic category where:
    - Objects are traffic pattern types
    - Morphisms are pattern transitions

    Enables detection of anomalies as unexpected traffic patterns
    for given protocol states.
    """

    def __init__(
        self,
        name: str,
        protocol_category: Category,
    ) -> None:
        """
        Create a traffic functor.

        Args:
            name: Functor name.
            protocol_category: Source protocol category.
        """
        # Create traffic pattern category
        traffic_cat = Category("Traffic")

        # Create traffic pattern objects
        patterns = [
            Object("no_traffic", data=TrafficPattern()),
            Object("low_traffic", data=TrafficPattern(packets=10, bytes_sent=1000)),
            Object("medium_traffic", data=TrafficPattern(packets=100, bytes_sent=10000)),
            Object("high_traffic", data=TrafficPattern(packets=1000, bytes_sent=100000)),
            Object("burst_traffic", data=TrafficPattern(packets=5000, duration=1.0)),
            Object("sustained_traffic", data=TrafficPattern(packets=1000, duration=60.0)),
        ]
        for pattern in patterns:
            traffic_cat.add_object(pattern)

        # Add transitions between patterns
        pattern_names = [p.name for p in patterns]
        for src_name in pattern_names:
            for tgt_name in pattern_names:
                src = traffic_cat.get_object(src_name)
                tgt = traffic_cat.get_object(tgt_name)
                if src and tgt and src != tgt:
                    morph = Morphism(
                        name=f"{src_name}->{tgt_name}",
                        source=src,
                        target=tgt,
                        weight=1.0,
                    )
                    traffic_cat.add_morphism(morph)

        super().__init__(name, protocol_category, traffic_cat)

        self._patterns = patterns
        self._expected_patterns: Dict[str, str] = {}
        self._setup_default_mappings()

    def _setup_default_mappings(self) -> None:
        """Set up default protocol state to traffic pattern mappings."""
        # Default mappings based on common protocol states
        defaults = {
            # TCP states
            "CLOSED": "no_traffic",
            "LISTEN": "no_traffic",
            "SYN_SENT": "low_traffic",
            "SYN_RECEIVED": "low_traffic",
            "ESTABLISHED": "medium_traffic",
            "FIN_WAIT_1": "low_traffic",
            "FIN_WAIT_2": "low_traffic",
            "CLOSE_WAIT": "low_traffic",
            "CLOSING": "low_traffic",
            "LAST_ACK": "low_traffic",
            "TIME_WAIT": "no_traffic",
            # HTTP states
            "IDLE": "no_traffic",
            "CONNECTED": "low_traffic",
            "REQUEST_SENT": "low_traffic",
            "HEADERS_RECEIVED": "low_traffic",
            "BODY_RECEIVING": "high_traffic",
            "RESPONSE_COMPLETE": "low_traffic",
            "KEEP_ALIVE": "no_traffic",
            # DNS states
            "QUERY_SENT": "low_traffic",
            "WAITING": "no_traffic",
            "RESPONSE_RECEIVED": "low_traffic",
            "CACHED": "no_traffic",
        }

        for state_name, pattern_name in defaults.items():
            self._expected_patterns[state_name] = pattern_name

    def set_expected_pattern(self, state_name: str, pattern_name: str) -> None:
        """Set expected traffic pattern for a protocol state."""
        self._expected_patterns[state_name] = pattern_name

    def map_object(self, obj: Object) -> Object:
        """Map protocol state to expected traffic pattern."""
        pattern_name = self._expected_patterns.get(obj.name, "medium_traffic")
        return self.target.get_object(pattern_name) or self._patterns[2]

    def map_morphism(self, morphism: Morphism) -> Morphism:
        """Map protocol transition to traffic transition."""
        src_pattern = self.map_object(morphism.source)
        tgt_pattern = self.map_object(morphism.target)

        return Morphism(
            name=f"{src_pattern.name}->{tgt_pattern.name}",
            source=src_pattern,
            target=tgt_pattern,
            weight=morphism.weight,
            data=morphism.data,
        )

    def anomaly_score(
        self,
        state: Object,
        observed_traffic: TrafficPattern,
    ) -> float:
        """
        Compute anomaly score for observed traffic in a protocol state.

        Args:
            state: Current protocol state.
            observed_traffic: Observed traffic pattern.

        Returns:
            Anomaly score in [0, 1], higher means more anomalous.
        """
        expected_pattern = self.map_object(state)
        expected_data = expected_pattern.data

        if not isinstance(expected_data, TrafficPattern):
            return 0.5

        # Compare expected vs observed
        expected_vec = expected_data.to_vector()
        observed_vec = observed_traffic.to_vector()

        # Compute normalized distance
        diff = np.abs(expected_vec - observed_vec)
        distance = np.sqrt(np.sum(diff ** 2))

        # Normalize using golden ratio scaling
        normalized = 1 - np.exp(-distance * PHI)

        return float(normalized)

    def detect_anomalies(
        self,
        state_traffic_pairs: List[Tuple[Object, TrafficPattern]],
        threshold: float = PHI_INV,
    ) -> List[Tuple[Object, TrafficPattern, float]]:
        """
        Detect anomalous traffic patterns.

        Args:
            state_traffic_pairs: List of (state, observed_traffic) pairs.
            threshold: Anomaly score threshold.

        Returns:
            List of (state, traffic, score) for anomalous observations.
        """
        anomalies: List[Tuple[Object, TrafficPattern, float]] = []

        for state, traffic in state_traffic_pairs:
            score = self.anomaly_score(state, traffic)
            if score > threshold:
                anomalies.append((state, traffic, score))

        return anomalies
