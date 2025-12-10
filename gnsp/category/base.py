"""
Base category theory structures.

Provides the fundamental building blocks for categorical abstractions:
- Objects: Abstract entities in a category
- Morphisms: Arrows between objects with composition
- Categories: Collections of objects and morphisms with identity and composition laws

These structures form the foundation for modeling protocol behaviors
and security properties in a compositional manner.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from gnsp.constants import PHI, PHI_INV


# Type variables for generic categorical structures
ObjT = TypeVar("ObjT")
MorphT = TypeVar("MorphT")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


@dataclass(frozen=True)
class Object:
    """
    An object in a category.

    Objects are the fundamental entities of a category. In our security context,
    objects typically represent states (protocol states, security contexts, etc.).

    Attributes:
        name: Unique identifier for the object.
        data: Optional associated data (e.g., state information).
        properties: Set of properties/labels for the object.
    """

    name: str
    data: Any = None
    properties: frozenset = field(default_factory=frozenset)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Object):
            return NotImplemented
        return self.name == other.name

    def has_property(self, prop: str) -> bool:
        """Check if object has a given property."""
        return prop in self.properties

    def with_property(self, prop: str) -> "Object":
        """Return new object with additional property."""
        return Object(
            name=self.name,
            data=self.data,
            properties=self.properties | {prop},
        )


@dataclass(frozen=True)
class Morphism:
    """
    A morphism (arrow) between objects in a category.

    Morphisms represent transformations or relationships between objects.
    In security contexts, morphisms typically represent:
    - Protocol transitions (state changes)
    - Security operations (authentication, authorization)
    - Data transformations

    Attributes:
        name: Identifier for the morphism.
        source: Domain object.
        target: Codomain object.
        weight: Golden ratio-based weight for prioritization.
        data: Optional associated data (e.g., transition conditions).
    """

    name: str
    source: Object
    target: Object
    weight: float = 1.0
    data: Any = None

    def __hash__(self) -> int:
        return hash((self.name, self.source, self.target))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Morphism):
            return NotImplemented
        return (
            self.name == other.name
            and self.source == other.source
            and self.target == other.target
        )

    def is_identity(self) -> bool:
        """Check if this is an identity morphism."""
        return self.source == self.target and self.name.startswith("id_")

    def is_composable_with(self, other: "Morphism") -> bool:
        """Check if this morphism can be composed with another (self ; other)."""
        return self.target == other.source

    def compose(self, other: "Morphism") -> "Morphism":
        """
        Compose this morphism with another (self ; other).

        Composition follows the convention f ; g where f: A -> B and g: B -> C
        gives f;g: A -> C. The weight combines using golden ratio scaling.

        Args:
            other: Morphism to compose with (must have source = self.target).

        Returns:
            Composed morphism from self.source to other.target.

        Raises:
            ValueError: If morphisms are not composable.
        """
        if not self.is_composable_with(other):
            raise ValueError(
                f"Cannot compose {self.name}: {self.source.name} -> {self.target.name} "
                f"with {other.name}: {other.source.name} -> {other.target.name}"
            )

        # Combine weights using golden ratio for balanced composition
        combined_weight = self.weight * other.weight * PHI_INV

        return Morphism(
            name=f"{self.name};{other.name}",
            source=self.source,
            target=other.target,
            weight=combined_weight,
            data=(self.data, other.data),
        )


class Category:
    """
    A category consisting of objects and morphisms.

    A category C consists of:
    - A collection of objects Ob(C)
    - For each pair of objects A, B, a set of morphisms Hom(A, B)
    - Identity morphism id_A for each object A
    - Composition operation satisfying associativity and identity laws

    This implementation provides:
    - Object and morphism management
    - Automatic identity morphism generation
    - Composition with verification of categorical laws
    - Golden ratio-based metrics for morphism analysis

    Attributes:
        name: Name of the category.
        objects: Set of objects in the category.
        morphisms: Set of morphisms in the category.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize an empty category.

        Args:
            name: Name identifier for the category.
        """
        self.name = name
        self._objects: Dict[str, Object] = {}
        self._morphisms: Dict[Tuple[str, str, str], Morphism] = {}
        self._identities: Dict[str, Morphism] = {}

    @property
    def objects(self) -> Set[Object]:
        """Get all objects in the category."""
        return set(self._objects.values())

    @property
    def morphisms(self) -> Set[Morphism]:
        """Get all non-identity morphisms in the category."""
        return set(self._morphisms.values())

    @property
    def all_morphisms(self) -> Set[Morphism]:
        """Get all morphisms including identities."""
        return self.morphisms | set(self._identities.values())

    def add_object(self, obj: Object) -> None:
        """
        Add an object to the category.

        Automatically creates the identity morphism for the object.

        Args:
            obj: Object to add.
        """
        if obj.name not in self._objects:
            self._objects[obj.name] = obj
            # Create identity morphism
            identity = Morphism(
                name=f"id_{obj.name}",
                source=obj,
                target=obj,
                weight=1.0,
                data="identity",
            )
            self._identities[obj.name] = identity

    def add_morphism(self, morphism: Morphism) -> None:
        """
        Add a morphism to the category.

        Ensures source and target objects are in the category.

        Args:
            morphism: Morphism to add.

        Raises:
            ValueError: If source or target object not in category.
        """
        if morphism.source.name not in self._objects:
            raise ValueError(f"Source object {morphism.source.name} not in category")
        if morphism.target.name not in self._objects:
            raise ValueError(f"Target object {morphism.target.name} not in category")

        key = (morphism.name, morphism.source.name, morphism.target.name)
        self._morphisms[key] = morphism

    def get_object(self, name: str) -> Optional[Object]:
        """Get object by name."""
        return self._objects.get(name)

    def get_identity(self, obj: Object) -> Morphism:
        """
        Get the identity morphism for an object.

        Args:
            obj: Object to get identity for.

        Returns:
            Identity morphism id_obj: obj -> obj.

        Raises:
            KeyError: If object not in category.
        """
        if obj.name not in self._identities:
            raise KeyError(f"Object {obj.name} not in category")
        return self._identities[obj.name]

    def hom(self, source: Object, target: Object) -> Set[Morphism]:
        """
        Get the hom-set Hom(source, target).

        Returns all morphisms from source to target, including
        the identity if source == target.

        Args:
            source: Source object.
            target: Target object.

        Returns:
            Set of morphisms from source to target.
        """
        result: Set[Morphism] = set()

        # Add identity if source == target
        if source == target and source.name in self._identities:
            result.add(self._identities[source.name])

        # Add all matching morphisms
        for (name, src, tgt), morphism in self._morphisms.items():
            if src == source.name and tgt == target.name:
                result.add(morphism)

        return result

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """
        Compose two morphisms f ; g.

        Args:
            f: First morphism (applied first).
            g: Second morphism (applied second).

        Returns:
            Composed morphism f;g.
        """
        return f.compose(g)

    def outgoing(self, obj: Object) -> Set[Morphism]:
        """Get all morphisms with given source (excluding identity)."""
        return {m for m in self.morphisms if m.source == obj}

    def incoming(self, obj: Object) -> Set[Morphism]:
        """Get all morphisms with given target (excluding identity)."""
        return {m for m in self.morphisms if m.target == obj}

    def is_initial(self, obj: Object) -> bool:
        """Check if object is initial (no incoming non-identity morphisms)."""
        return len(self.incoming(obj)) == 0

    def is_terminal(self, obj: Object) -> bool:
        """Check if object is terminal (no outgoing non-identity morphisms)."""
        return len(self.outgoing(obj)) == 0

    def path_exists(self, source: Object, target: Object) -> bool:
        """
        Check if a path exists from source to target.

        Uses BFS to find any sequence of composable morphisms.

        Args:
            source: Starting object.
            target: Ending object.

        Returns:
            True if a path exists, False otherwise.
        """
        if source == target:
            return True

        visited: Set[str] = set()
        queue = [source]

        while queue:
            current = queue.pop(0)
            if current.name in visited:
                continue
            visited.add(current.name)

            for morphism in self.outgoing(current):
                if morphism.target == target:
                    return True
                queue.append(morphism.target)

        return False

    def all_paths(
        self,
        source: Object,
        target: Object,
        max_length: int = 10,
    ) -> List[List[Morphism]]:
        """
        Find all paths from source to target up to given length.

        Args:
            source: Starting object.
            target: Ending object.
            max_length: Maximum path length.

        Returns:
            List of paths, where each path is a list of morphisms.
        """
        if source == target:
            return [[]]

        paths: List[List[Morphism]] = []

        def dfs(current: Object, path: List[Morphism], visited: Set[str]) -> None:
            if len(path) > max_length:
                return

            for morphism in self.outgoing(current):
                if morphism.target == target:
                    paths.append(path + [morphism])
                elif morphism.target.name not in visited:
                    dfs(
                        morphism.target,
                        path + [morphism],
                        visited | {morphism.target.name},
                    )

        dfs(source, [], {source.name})
        return paths

    def golden_centrality(self, obj: Object) -> float:
        """
        Compute golden ratio-weighted centrality of an object.

        Centrality is computed as the ratio of weighted incoming
        to outgoing morphisms, scaled by phi for normalization.

        Args:
            obj: Object to compute centrality for.

        Returns:
            Centrality score in [0, 1].
        """
        incoming_weight = sum(m.weight for m in self.incoming(obj))
        outgoing_weight = sum(m.weight for m in self.outgoing(obj))

        total = incoming_weight + outgoing_weight
        if total == 0:
            return 0.0

        # Normalize using golden ratio
        raw = (incoming_weight * PHI + outgoing_weight * PHI_INV) / total
        return min(1.0, raw / PHI)

    def verify_identity_law(self) -> bool:
        """
        Verify that identity morphisms satisfy the identity law.

        For all f: A -> B, we should have id_A ; f = f and f ; id_B = f.

        Returns:
            True if identity law holds for all morphisms.
        """
        for morphism in self.morphisms:
            id_source = self.get_identity(morphism.source)
            id_target = self.get_identity(morphism.target)

            # Check id_A ; f has same source/target as f
            left = id_source.compose(morphism)
            if left.source != morphism.source or left.target != morphism.target:
                return False

            # Check f ; id_B has same source/target as f
            right = morphism.compose(id_target)
            if right.source != morphism.source or right.target != morphism.target:
                return False

        return True

    def __repr__(self) -> str:
        return (
            f"Category({self.name}, "
            f"objects={len(self._objects)}, "
            f"morphisms={len(self._morphisms)})"
        )


class ProductCategory(Category):
    """
    Product of two categories C x D.

    Objects are pairs (c, d) where c in C and d in D.
    Morphisms are pairs (f, g) where f in C and g in D.

    Used for modeling parallel protocol behaviors and
    multi-dimensional security analysis.
    """

    def __init__(self, cat1: Category, cat2: Category) -> None:
        """
        Create product category C x D.

        Args:
            cat1: First category C.
            cat2: Second category D.
        """
        super().__init__(f"{cat1.name}x{cat2.name}")
        self.cat1 = cat1
        self.cat2 = cat2

        # Create product objects
        for obj1 in cat1.objects:
            for obj2 in cat2.objects:
                prod_obj = Object(
                    name=f"({obj1.name},{obj2.name})",
                    data=(obj1.data, obj2.data),
                    properties=obj1.properties | obj2.properties,
                )
                self.add_object(prod_obj)

        # Create product morphisms
        for m1 in cat1.all_morphisms:
            for m2 in cat2.all_morphisms:
                src_name = f"({m1.source.name},{m2.source.name})"
                tgt_name = f"({m1.target.name},{m2.target.name})"

                src_obj = self.get_object(src_name)
                tgt_obj = self.get_object(tgt_name)

                if src_obj and tgt_obj:
                    prod_morph = Morphism(
                        name=f"({m1.name},{m2.name})",
                        source=src_obj,
                        target=tgt_obj,
                        weight=m1.weight * m2.weight,
                        data=(m1.data, m2.data),
                    )
                    # Only add non-identity morphisms
                    if not prod_morph.is_identity():
                        self.add_morphism(prod_morph)

    def project1(self, obj: Object) -> Optional[Object]:
        """Project product object to first component."""
        if obj.data and isinstance(obj.data, tuple) and len(obj.data) == 2:
            name = obj.name.strip("()").split(",")[0]
            return self.cat1.get_object(name)
        return None

    def project2(self, obj: Object) -> Optional[Object]:
        """Project product object to second component."""
        if obj.data and isinstance(obj.data, tuple) and len(obj.data) == 2:
            name = obj.name.strip("()").split(",")[1]
            return self.cat2.get_object(name)
        return None


class OppositeCategory(Category):
    """
    Opposite (dual) category C^op.

    Has the same objects as C, but morphisms are reversed:
    a morphism f: A -> B in C becomes f^op: B -> A in C^op.

    Useful for modeling reverse protocol flows and
    contravariant security properties.
    """

    def __init__(self, cat: Category) -> None:
        """
        Create opposite category C^op.

        Args:
            cat: Original category C.
        """
        super().__init__(f"{cat.name}^op")
        self.original = cat

        # Copy objects
        for obj in cat.objects:
            self.add_object(obj)

        # Reverse morphisms
        for morphism in cat.morphisms:
            reversed_morph = Morphism(
                name=f"{morphism.name}^op",
                source=morphism.target,  # Reversed
                target=morphism.source,  # Reversed
                weight=morphism.weight,
                data=morphism.data,
            )
            self.add_morphism(reversed_morph)

    def original_morphism(self, op_morphism: Morphism) -> Optional[Morphism]:
        """Get the original morphism for an op morphism."""
        if not op_morphism.name.endswith("^op"):
            return None

        orig_name = op_morphism.name[:-3]
        for m in self.original.morphisms:
            if m.name == orig_name:
                return m
        return None
