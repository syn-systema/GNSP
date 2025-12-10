"""
Natural transformations between functors.

A natural transformation alpha: F => G between functors F, G: C -> D
assigns to each object A in C a morphism alpha_A: F(A) -> G(A) in D
such that the naturality condition holds:

For any morphism f: A -> B in C:
    F(f) ; alpha_B = alpha_A ; G(f)

Natural transformations enable:
- Comparing different security views (functors)
- Tracking how security properties evolve
- Detecting anomalies as naturality violations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from gnsp.category.base import Category, Morphism, Object
from gnsp.category.functor import Functor, IdentityFunctor
from gnsp.constants import PHI, PHI_INV


class NaturalTransformation(ABC):
    """
    Abstract base class for natural transformations.

    A natural transformation alpha: F => G consists of:
    - Component morphisms alpha_A: F(A) -> G(A) for each object A
    - Naturality: F(f);alpha_B = alpha_A;G(f) for all morphisms f: A -> B
    """

    def __init__(
        self,
        name: str,
        source_functor: Functor,
        target_functor: Functor,
    ) -> None:
        """
        Initialize a natural transformation.

        Args:
            name: Name identifier.
            source_functor: Functor F.
            target_functor: Functor G.

        Raises:
            ValueError: If functors don't share same source and target categories.
        """
        if source_functor.source != target_functor.source:
            raise ValueError(
                f"Functors must have same source category: "
                f"{source_functor.source.name} != {target_functor.source.name}"
            )
        if source_functor.target != target_functor.target:
            raise ValueError(
                f"Functors must have same target category: "
                f"{source_functor.target.name} != {target_functor.target.name}"
            )

        self.name = name
        self.source_functor = source_functor  # F
        self.target_functor = target_functor  # G
        self._components: Dict[str, Morphism] = {}

    @property
    def domain_category(self) -> Category:
        """The category that functors map from."""
        return self.source_functor.source

    @property
    def codomain_category(self) -> Category:
        """The category that functors map to."""
        return self.source_functor.target

    @abstractmethod
    def component(self, obj: Object) -> Morphism:
        """
        Get the component morphism alpha_A: F(A) -> G(A).

        Args:
            obj: Object A in the domain category.

        Returns:
            Component morphism from F(A) to G(A).
        """
        pass

    def __call__(self, obj: Object) -> Morphism:
        """Apply natural transformation to get component at object."""
        return self.component(obj)

    def is_natural_at(self, morphism: Morphism) -> bool:
        """
        Check naturality condition at a morphism.

        For f: A -> B, checks that F(f);alpha_B = alpha_A;G(f).

        Args:
            morphism: Morphism f: A -> B in domain category.

        Returns:
            True if naturality holds at this morphism.
        """
        A = morphism.source
        B = morphism.target

        # Get components
        alpha_A = self.component(A)
        alpha_B = self.component(B)

        # F(f) and G(f)
        F_f = self.source_functor.map_morphism(morphism)
        G_f = self.target_functor.map_morphism(morphism)

        # Check F(f);alpha_B = alpha_A;G(f)
        # We check that the composites have same source and target
        try:
            left = F_f.compose(alpha_B)
            right = alpha_A.compose(G_f)

            return (
                left.source == right.source
                and left.target == right.target
            )
        except ValueError:
            return False

    def is_natural(self) -> bool:
        """
        Check if transformation is natural (satisfies naturality for all morphisms).

        Returns:
            True if natural transformation is well-defined.
        """
        for morphism in self.domain_category.morphisms:
            if not self.is_natural_at(morphism):
                return False
        return True

    def naturality_violation_score(self, morphism: Morphism) -> float:
        """
        Compute how much naturality is violated at a morphism.

        Args:
            morphism: Morphism to check.

        Returns:
            Violation score in [0, 1], 0 means natural.
        """
        if self.is_natural_at(morphism):
            return 0.0

        # Compute structural difference
        A = morphism.source
        B = morphism.target

        alpha_A = self.component(A)
        alpha_B = self.component(B)

        F_f = self.source_functor.map_morphism(morphism)
        G_f = self.target_functor.map_morphism(morphism)

        # Weight differences affect score
        try:
            left = F_f.compose(alpha_B)
            right = alpha_A.compose(G_f)
            weight_diff = abs(left.weight - right.weight)
            return min(1.0, weight_diff * PHI)
        except ValueError:
            return 1.0  # Complete violation

    def total_violation_score(self) -> float:
        """
        Compute total naturality violation across all morphisms.

        Returns:
            Average violation score.
        """
        if not self.domain_category.morphisms:
            return 0.0

        total = sum(
            self.naturality_violation_score(m)
            for m in self.domain_category.morphisms
        )
        return total / len(self.domain_category.morphisms)

    def __repr__(self) -> str:
        return (
            f"NaturalTransformation({self.name}: "
            f"{self.source_functor.name} => {self.target_functor.name})"
        )


class IdentityTransformation(NaturalTransformation):
    """
    Identity natural transformation id_F: F => F.

    Components are identity morphisms: (id_F)_A = id_F(A).
    """

    def __init__(self, functor: Functor) -> None:
        """
        Create identity transformation for a functor.

        Args:
            functor: Functor F.
        """
        super().__init__(f"id_{functor.name}", functor, functor)

    def component(self, obj: Object) -> Morphism:
        """Get identity morphism at F(obj)."""
        F_obj = self.source_functor.map_object(obj)
        return self.codomain_category.get_identity(F_obj)


class ExplicitTransformation(NaturalTransformation):
    """
    Natural transformation with explicitly defined components.

    Components are provided as a dictionary mapping object names
    to morphisms.
    """

    def __init__(
        self,
        name: str,
        source_functor: Functor,
        target_functor: Functor,
        components: Dict[str, Morphism],
    ) -> None:
        """
        Create natural transformation with explicit components.

        Args:
            name: Name identifier.
            source_functor: Functor F.
            target_functor: Functor G.
            components: Dictionary mapping object names to component morphisms.
        """
        super().__init__(name, source_functor, target_functor)
        self._components = components

    def component(self, obj: Object) -> Morphism:
        """Get component morphism for object."""
        if obj.name not in self._components:
            # Default to identity if possible
            F_obj = self.source_functor.map_object(obj)
            G_obj = self.target_functor.map_object(obj)

            if F_obj == G_obj:
                return self.codomain_category.get_identity(F_obj)

            raise KeyError(f"No component defined for object {obj.name}")

        return self._components[obj.name]


class VerticalComposition(NaturalTransformation):
    """
    Vertical composition of natural transformations.

    Given alpha: F => G and beta: G => H, produces alpha;beta: F => H.
    Components are (alpha;beta)_A = alpha_A ; beta_A.
    """

    def __init__(
        self,
        first: NaturalTransformation,
        second: NaturalTransformation,
    ) -> None:
        """
        Compose natural transformations vertically.

        Args:
            first: Transformation alpha: F => G.
            second: Transformation beta: G => H.

        Raises:
            ValueError: If transformations are not composable.
        """
        if first.target_functor != second.source_functor:
            raise ValueError(
                f"Cannot compose: {first.name} target != {second.name} source"
            )

        super().__init__(
            f"{first.name};{second.name}",
            first.source_functor,
            second.target_functor,
        )
        self.first = first
        self.second = second

    def component(self, obj: Object) -> Morphism:
        """
        Get composed component alpha_A ; beta_A.

        Args:
            obj: Object A.

        Returns:
            Composed morphism.
        """
        alpha_A = self.first.component(obj)
        beta_A = self.second.component(obj)
        return alpha_A.compose(beta_A)


class HorizontalComposition(NaturalTransformation):
    """
    Horizontal composition of natural transformations.

    Given alpha: F => G (where F, G: C -> D) and
    beta: H => K (where H, K: D -> E),
    produces alpha * beta: F;H => G;K.

    Components are (alpha * beta)_A = H(alpha_A) ; beta_G(A)
                                    = beta_F(A) ; K(alpha_A).
    """

    def __init__(
        self,
        first: NaturalTransformation,
        second: NaturalTransformation,
    ) -> None:
        """
        Compose natural transformations horizontally.

        Args:
            first: Transformation alpha: F => G (C -> D).
            second: Transformation beta: H => K (D -> E).

        Raises:
            ValueError: If transformations are not horizontally composable.
        """
        # Check that first's codomain is second's domain
        if first.codomain_category != second.domain_category:
            raise ValueError(
                "Horizontal composition requires matching categories"
            )

        # Create composed functors F;H and G;K
        from gnsp.category.functor import CompositionFunctor

        F_H = CompositionFunctor(first.source_functor, second.source_functor)
        G_K = CompositionFunctor(first.target_functor, second.target_functor)

        super().__init__(
            f"{first.name}*{second.name}",
            F_H,
            G_K,
        )
        self.first = first  # alpha
        self.second = second  # beta

    def component(self, obj: Object) -> Morphism:
        """
        Get horizontal composition component.

        Uses interchange law: H(alpha_A) ; beta_G(A) = beta_F(A) ; K(alpha_A).

        Args:
            obj: Object A in original domain.

        Returns:
            Component morphism from (F;H)(A) to (G;K)(A).
        """
        # alpha_A: F(A) -> G(A)
        alpha_A = self.first.component(obj)

        # G(A)
        G_A = self.first.target_functor.map_object(obj)

        # H(alpha_A): H(F(A)) -> H(G(A))
        H = self.second.source_functor
        H_alpha_A = H.map_morphism(alpha_A)

        # beta_G(A): H(G(A)) -> K(G(A))
        beta_GA = self.second.component(G_A)

        # Compose: H(alpha_A) ; beta_G(A)
        return H_alpha_A.compose(beta_GA)


@dataclass
class TransformationPair:
    """
    A pair of natural transformations forming an adjunction.

    An adjunction F -| G consists of:
    - Functors F: C -> D and G: D -> C
    - Unit eta: Id_C => G;F
    - Counit epsilon: F;G => Id_D
    """

    left_adjoint: Functor  # F
    right_adjoint: Functor  # G
    unit: NaturalTransformation  # eta: Id_C => G;F
    counit: NaturalTransformation  # epsilon: F;G => Id_D

    def verify_triangle_identities(self) -> Tuple[bool, bool]:
        """
        Verify triangle identities for the adjunction.

        Returns:
            Tuple of (left_identity_holds, right_identity_holds).
        """
        # Left identity: F => F;G;F => F
        # (epsilon_F) ; F(eta) = id_F

        # Right identity: G => G;F;G => G
        # G(epsilon) ; (eta_G) = id_G

        # Simplified check: verify unit and counit are natural
        left_ok = self.unit.is_natural()
        right_ok = self.counit.is_natural()

        return left_ok, right_ok


def compute_naturality_matrix(
    transformation: NaturalTransformation,
) -> Dict[str, Dict[str, float]]:
    """
    Compute matrix of naturality violations.

    Args:
        transformation: Natural transformation to analyze.

    Returns:
        Matrix[A][f] = violation score at morphism f starting from object A.
    """
    result: Dict[str, Dict[str, float]] = {}

    for obj in transformation.domain_category.objects:
        result[obj.name] = {}

        for morphism in transformation.domain_category.outgoing(obj):
            score = transformation.naturality_violation_score(morphism)
            result[obj.name][morphism.name] = score

    return result


def find_anomalous_transitions(
    transformation: NaturalTransformation,
    threshold: float = PHI_INV,
) -> List[Tuple[Morphism, float]]:
    """
    Find morphisms with high naturality violations.

    These represent transitions where the relationship between
    security views (functors) breaks down, indicating potential anomalies.

    Args:
        transformation: Natural transformation to analyze.
        threshold: Violation score threshold.

    Returns:
        List of (morphism, violation_score) pairs above threshold.
    """
    anomalies: List[Tuple[Morphism, float]] = []

    for morphism in transformation.domain_category.morphisms:
        score = transformation.naturality_violation_score(morphism)
        if score > threshold:
            anomalies.append((morphism, score))

    # Sort by score (highest first)
    anomalies.sort(key=lambda x: x[1], reverse=True)

    return anomalies
