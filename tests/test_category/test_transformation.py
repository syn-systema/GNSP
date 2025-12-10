"""
Tests for natural transformation implementations.

Tests cover:
- Natural transformation basics
- Identity transformations
- Vertical and horizontal composition
- Naturality verification
"""

import pytest

from gnsp.category.base import Category, Morphism, Object
from gnsp.category.functor import CompositionFunctor, IdentityFunctor
from gnsp.category.transformation import (
    ExplicitTransformation,
    HorizontalComposition,
    IdentityTransformation,
    NaturalTransformation,
    VerticalComposition,
    compute_naturality_matrix,
    find_anomalous_transitions,
)
from gnsp.constants import PHI_INV


class TestIdentityTransformation:
    """Tests for IdentityTransformation class."""

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

    def test_identity_transformation_creation(self, category: Category) -> None:
        """Test identity transformation creation."""
        functor = IdentityFunctor(category)
        id_trans = IdentityTransformation(functor)

        assert id_trans.source_functor == functor
        assert id_trans.target_functor == functor

    def test_identity_component(self, category: Category) -> None:
        """Test identity transformation components are identities."""
        functor = IdentityFunctor(category)
        id_trans = IdentityTransformation(functor)

        A = category.get_object("A")
        component = id_trans.component(A)

        assert component.is_identity()
        assert component.source == A
        assert component.target == A

    def test_identity_is_natural(self, category: Category) -> None:
        """Test identity transformation is natural."""
        functor = IdentityFunctor(category)
        id_trans = IdentityTransformation(functor)

        assert id_trans.is_natural()

    def test_identity_naturality_at_morphism(self, category: Category) -> None:
        """Test naturality at specific morphism."""
        functor = IdentityFunctor(category)
        id_trans = IdentityTransformation(functor)

        A = category.get_object("A")
        B = category.get_object("B")
        f = list(category.hom(A, B) - {category.get_identity(A)})[0]

        assert id_trans.is_natural_at(f)


class TestExplicitTransformation:
    """Tests for ExplicitTransformation class."""

    @pytest.fixture
    def two_functors(self) -> tuple:
        """Create two functors with same source and target."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        cat.add_object(A)
        cat.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        cat.add_morphism(f)

        F = IdentityFunctor(cat)
        G = IdentityFunctor(cat)

        return F, G, cat

    def test_explicit_transformation_creation(self, two_functors: tuple) -> None:
        """Test explicit transformation creation."""
        F, G, cat = two_functors

        A = cat.get_object("A")
        B = cat.get_object("B")

        components = {
            "A": cat.get_identity(A),
            "B": cat.get_identity(B),
        }

        trans = ExplicitTransformation("alpha", F, G, components)

        assert trans.source_functor == F
        assert trans.target_functor == G

    def test_explicit_component_retrieval(self, two_functors: tuple) -> None:
        """Test component retrieval."""
        F, G, cat = two_functors

        A = cat.get_object("A")
        B = cat.get_object("B")

        id_A = cat.get_identity(A)
        id_B = cat.get_identity(B)

        components = {"A": id_A, "B": id_B}
        trans = ExplicitTransformation("alpha", F, G, components)

        assert trans.component(A) == id_A
        assert trans.component(B) == id_B


class TestVerticalComposition:
    """Tests for VerticalComposition class."""

    @pytest.fixture
    def three_functors(self) -> tuple:
        """Create three functors for vertical composition."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        cat.add_object(A)
        cat.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        cat.add_morphism(f)

        F = IdentityFunctor(cat)
        G = IdentityFunctor(cat)
        H = IdentityFunctor(cat)

        return F, G, H, cat

    def test_vertical_composition_creation(self, three_functors: tuple) -> None:
        """Test vertical composition of identity transformations."""
        F, G, H, cat = three_functors

        alpha = IdentityTransformation(F)  # F => F
        # Since G is also identity on same category, we can compose
        # But for true vertical composition, we need alpha: F => G and beta: G => H

        # Create alpha: F => G using explicit transformation
        A = cat.get_object("A")
        B = cat.get_object("B")

        # Components that make alpha: F => G well-defined
        # Since F = G = identity, use identity components
        alpha_comps = {
            "A": cat.get_identity(A),
            "B": cat.get_identity(B),
        }
        alpha = ExplicitTransformation("alpha", F, G, alpha_comps)

        beta_comps = {
            "A": cat.get_identity(A),
            "B": cat.get_identity(B),
        }
        beta = ExplicitTransformation("beta", G, H, beta_comps)

        composed = VerticalComposition(alpha, beta)

        assert composed.source_functor == F
        assert composed.target_functor == H

    def test_vertical_composition_component(self, three_functors: tuple) -> None:
        """Test vertical composition component is composition of components."""
        F, G, H, cat = three_functors

        A = cat.get_object("A")
        B = cat.get_object("B")

        alpha_comps = {
            "A": cat.get_identity(A),
            "B": cat.get_identity(B),
        }
        beta_comps = {
            "A": cat.get_identity(A),
            "B": cat.get_identity(B),
        }

        alpha = ExplicitTransformation("alpha", F, G, alpha_comps)
        beta = ExplicitTransformation("beta", G, H, beta_comps)

        composed = VerticalComposition(alpha, beta)

        # Component at A should be alpha_A ; beta_A
        comp_A = composed.component(A)

        # Since both are identities, composition should also be identity-like
        assert comp_A.source == A
        assert comp_A.target == A

    def test_vertical_not_composable(self, three_functors: tuple) -> None:
        """Test vertical composition fails for non-composable transformations."""
        F, G, H, cat = three_functors

        A = cat.get_object("A")
        B = cat.get_object("B")

        # alpha: F => G
        alpha = ExplicitTransformation(
            "alpha",
            F,
            G,
            {"A": cat.get_identity(A), "B": cat.get_identity(B)},
        )

        # gamma: F => H (not starting from G)
        gamma = ExplicitTransformation(
            "gamma",
            F,
            H,
            {"A": cat.get_identity(A), "B": cat.get_identity(B)},
        )

        with pytest.raises(ValueError):
            VerticalComposition(alpha, gamma)


class TestHorizontalComposition:
    """Tests for HorizontalComposition class."""

    @pytest.fixture
    def horizontal_setup(self) -> tuple:
        """Create setup for horizontal composition."""
        C = Category("C")
        D = Category("D")

        A = Object(name="A")
        B = Object(name="B")

        C.add_object(A)
        C.add_object(B)
        D.add_object(A)
        D.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        C.add_morphism(f)
        D.add_morphism(f)

        F = IdentityFunctor(C)
        G = IdentityFunctor(C)

        return F, G, C, D

    def test_horizontal_composition_creation(self, horizontal_setup: tuple) -> None:
        """Test horizontal composition creation."""
        F, G, C, D = horizontal_setup

        # alpha: F => G where F, G: C -> C
        A = C.get_object("A")
        B = C.get_object("B")

        alpha = ExplicitTransformation(
            "alpha",
            F,
            G,
            {"A": C.get_identity(A), "B": C.get_identity(B)},
        )

        # H, K: C -> C (using identity functors again)
        H = IdentityFunctor(C)
        K = IdentityFunctor(C)

        beta = ExplicitTransformation(
            "beta",
            H,
            K,
            {"A": C.get_identity(A), "B": C.get_identity(B)},
        )

        # alpha * beta: F;H => G;K
        horiz = HorizontalComposition(alpha, beta)

        # Composed functors
        assert horiz.source_functor.source == C
        assert horiz.target_functor.target == C


class TestNaturalityVerification:
    """Tests for naturality checking and violation detection."""

    @pytest.fixture
    def test_transformation(self) -> tuple:
        """Create transformation for naturality testing."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        C = Object(name="C")

        cat.add_object(A)
        cat.add_object(B)
        cat.add_object(C)

        f = Morphism(name="f", source=A, target=B)
        g = Morphism(name="g", source=B, target=C)

        cat.add_morphism(f)
        cat.add_morphism(g)

        F = IdentityFunctor(cat)
        G = IdentityFunctor(cat)

        alpha = ExplicitTransformation(
            "alpha",
            F,
            G,
            {
                "A": cat.get_identity(A),
                "B": cat.get_identity(B),
                "C": cat.get_identity(C),
            },
        )

        return alpha, cat

    def test_naturality_verification(self, test_transformation: tuple) -> None:
        """Test naturality is verified correctly."""
        alpha, cat = test_transformation

        assert alpha.is_natural()

    def test_naturality_at_each_morphism(self, test_transformation: tuple) -> None:
        """Test naturality at each morphism."""
        alpha, cat = test_transformation

        for morphism in cat.morphisms:
            assert alpha.is_natural_at(morphism)

    def test_naturality_violation_score(self, test_transformation: tuple) -> None:
        """Test violation score for natural transformation."""
        alpha, cat = test_transformation

        for morphism in cat.morphisms:
            score = alpha.naturality_violation_score(morphism)
            assert score == 0.0  # Natural transformation has no violations

    def test_total_violation_score(self, test_transformation: tuple) -> None:
        """Test total violation score."""
        alpha, _ = test_transformation

        total = alpha.total_violation_score()
        assert total == 0.0


class TestNaturalityAnalysis:
    """Tests for naturality analysis utilities."""

    @pytest.fixture
    def analyzable_transformation(self) -> tuple:
        """Create transformation for analysis."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")

        cat.add_object(A)
        cat.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        cat.add_morphism(f)

        F = IdentityFunctor(cat)
        G = IdentityFunctor(cat)

        alpha = ExplicitTransformation(
            "alpha",
            F,
            G,
            {
                "A": cat.get_identity(A),
                "B": cat.get_identity(B),
            },
        )

        return alpha, cat

    def test_compute_naturality_matrix(
        self,
        analyzable_transformation: tuple,
    ) -> None:
        """Test naturality matrix computation."""
        alpha, cat = analyzable_transformation

        matrix = compute_naturality_matrix(alpha)

        # Should have entries for each object
        assert "A" in matrix

        # A has outgoing morphism f
        A = cat.get_object("A")
        outgoing = cat.outgoing(A)
        for m in outgoing:
            assert m.name in matrix["A"]

    def test_find_anomalous_transitions(
        self,
        analyzable_transformation: tuple,
    ) -> None:
        """Test finding anomalous transitions."""
        alpha, _ = analyzable_transformation

        anomalies = find_anomalous_transitions(alpha)

        # Natural transformation should have no anomalies
        assert len(anomalies) == 0

    def test_find_anomalous_with_threshold(
        self,
        analyzable_transformation: tuple,
    ) -> None:
        """Test anomaly detection with custom threshold."""
        alpha, _ = analyzable_transformation

        # Even with threshold 0, natural transformation has no violations
        anomalies = find_anomalous_transitions(alpha, threshold=0.0)

        assert len(anomalies) == 0


class TestTransformationProperties:
    """Tests for transformation properties."""

    def test_domain_category(self) -> None:
        """Test domain category property."""
        cat = Category("Test")
        cat.add_object(Object(name="A"))

        F = IdentityFunctor(cat)
        alpha = IdentityTransformation(F)

        assert alpha.domain_category == cat

    def test_codomain_category(self) -> None:
        """Test codomain category property."""
        cat = Category("Test")
        cat.add_object(Object(name="A"))

        F = IdentityFunctor(cat)
        alpha = IdentityTransformation(F)

        assert alpha.codomain_category == cat

    def test_callable_interface(self) -> None:
        """Test transformation callable interface."""
        cat = Category("Test")
        A = Object(name="A")
        cat.add_object(A)

        F = IdentityFunctor(cat)
        alpha = IdentityTransformation(F)

        # Should be callable
        component = alpha(A)

        assert component.is_identity()

    def test_repr(self) -> None:
        """Test transformation string representation."""
        cat = Category("Test")
        cat.add_object(Object(name="A"))

        F = IdentityFunctor(cat)
        alpha = IdentityTransformation(F)

        repr_str = repr(alpha)

        assert "NaturalTransformation" in repr_str
        assert "=>" in repr_str
