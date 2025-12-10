"""
Tests for base category structures.

Tests cover:
- Object creation and properties
- Morphism creation and composition
- Category operations and laws
- Product and opposite categories
"""

import pytest

from gnsp.category.base import (
    Category,
    Morphism,
    Object,
    OppositeCategory,
    ProductCategory,
)
from gnsp.constants import PHI, PHI_INV


class TestObject:
    """Tests for Object class."""

    def test_object_creation(self) -> None:
        """Test basic object creation."""
        obj = Object(name="A")
        assert obj.name == "A"
        assert obj.data is None
        assert obj.properties == frozenset()

    def test_object_with_data(self) -> None:
        """Test object with associated data."""
        data = {"value": 42}
        obj = Object(name="A", data=data)
        assert obj.data == data

    def test_object_with_properties(self) -> None:
        """Test object with properties."""
        props = frozenset({"secure", "monitored"})
        obj = Object(name="A", properties=props)
        assert obj.properties == props
        assert obj.has_property("secure")
        assert not obj.has_property("compromised")

    def test_object_equality(self) -> None:
        """Test object equality based on name."""
        obj1 = Object(name="A", data=1)
        obj2 = Object(name="A", data=2)
        obj3 = Object(name="B")

        assert obj1 == obj2
        assert obj1 != obj3

    def test_object_hash(self) -> None:
        """Test object hashing."""
        obj1 = Object(name="A")
        obj2 = Object(name="A")

        assert hash(obj1) == hash(obj2)
        assert {obj1, obj2} == {obj1}

    def test_with_property(self) -> None:
        """Test adding property to object."""
        obj = Object(name="A", properties=frozenset({"prop1"}))
        new_obj = obj.with_property("prop2")

        assert obj.has_property("prop1")
        assert not obj.has_property("prop2")
        assert new_obj.has_property("prop1")
        assert new_obj.has_property("prop2")


class TestMorphism:
    """Tests for Morphism class."""

    @pytest.fixture
    def objects(self) -> tuple:
        """Create test objects."""
        return (
            Object(name="A"),
            Object(name="B"),
            Object(name="C"),
        )

    def test_morphism_creation(self, objects: tuple) -> None:
        """Test basic morphism creation."""
        A, B, _ = objects
        f = Morphism(name="f", source=A, target=B)

        assert f.name == "f"
        assert f.source == A
        assert f.target == B
        assert f.weight == 1.0

    def test_morphism_with_weight(self, objects: tuple) -> None:
        """Test morphism with custom weight."""
        A, B, _ = objects
        f = Morphism(name="f", source=A, target=B, weight=PHI)

        assert f.weight == PHI

    def test_identity_morphism(self, objects: tuple) -> None:
        """Test identity morphism detection."""
        A, B, _ = objects
        id_A = Morphism(name="id_A", source=A, target=A)
        f = Morphism(name="f", source=A, target=B)

        assert id_A.is_identity()
        assert not f.is_identity()

    def test_morphism_composable(self, objects: tuple) -> None:
        """Test morphism composability check."""
        A, B, C = objects
        f = Morphism(name="f", source=A, target=B)
        g = Morphism(name="g", source=B, target=C)
        h = Morphism(name="h", source=A, target=C)

        assert f.is_composable_with(g)
        assert not g.is_composable_with(f)
        assert not f.is_composable_with(h)

    def test_morphism_composition(self, objects: tuple) -> None:
        """Test morphism composition."""
        A, B, C = objects
        f = Morphism(name="f", source=A, target=B)
        g = Morphism(name="g", source=B, target=C)

        fg = f.compose(g)

        assert fg.source == A
        assert fg.target == C
        assert fg.name == "f;g"

    def test_composition_weight(self, objects: tuple) -> None:
        """Test weight combination in composition."""
        A, B, C = objects
        f = Morphism(name="f", source=A, target=B, weight=2.0)
        g = Morphism(name="g", source=B, target=C, weight=3.0)

        fg = f.compose(g)

        expected_weight = 2.0 * 3.0 * PHI_INV
        assert abs(fg.weight - expected_weight) < 1e-10

    def test_composition_not_composable(self, objects: tuple) -> None:
        """Test composition fails for non-composable morphisms."""
        A, B, C = objects
        f = Morphism(name="f", source=A, target=B)
        h = Morphism(name="h", source=A, target=C)

        with pytest.raises(ValueError):
            f.compose(h)

    def test_morphism_equality(self, objects: tuple) -> None:
        """Test morphism equality."""
        A, B, _ = objects
        f1 = Morphism(name="f", source=A, target=B)
        f2 = Morphism(name="f", source=A, target=B)
        g = Morphism(name="g", source=A, target=B)

        assert f1 == f2
        assert f1 != g


class TestCategory:
    """Tests for Category class."""

    @pytest.fixture
    def simple_category(self) -> Category:
        """Create a simple test category."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        C = Object(name="C")

        cat.add_object(A)
        cat.add_object(B)
        cat.add_object(C)

        f = Morphism(name="f", source=A, target=B)
        g = Morphism(name="g", source=B, target=C)
        h = Morphism(name="h", source=A, target=C)

        cat.add_morphism(f)
        cat.add_morphism(g)
        cat.add_morphism(h)

        return cat

    def test_category_creation(self) -> None:
        """Test basic category creation."""
        cat = Category("Test")
        assert cat.name == "Test"
        assert len(cat.objects) == 0
        assert len(cat.morphisms) == 0

    def test_add_object(self) -> None:
        """Test adding objects to category."""
        cat = Category("Test")
        A = Object(name="A")
        cat.add_object(A)

        assert A in cat.objects
        assert cat.get_object("A") == A

    def test_identity_morphism_created(self) -> None:
        """Test identity morphism is created for each object."""
        cat = Category("Test")
        A = Object(name="A")
        cat.add_object(A)

        id_A = cat.get_identity(A)

        assert id_A is not None
        assert id_A.source == A
        assert id_A.target == A
        assert id_A.is_identity()

    def test_add_morphism(self, simple_category: Category) -> None:
        """Test adding morphisms to category."""
        assert len(simple_category.morphisms) == 3

    def test_add_morphism_invalid_source(self) -> None:
        """Test adding morphism with invalid source fails."""
        cat = Category("Test")
        B = Object(name="B")
        cat.add_object(B)

        A = Object(name="A")  # Not in category
        f = Morphism(name="f", source=A, target=B)

        with pytest.raises(ValueError):
            cat.add_morphism(f)

    def test_hom_set(self, simple_category: Category) -> None:
        """Test getting hom-sets."""
        A = simple_category.get_object("A")
        B = simple_category.get_object("B")
        C = simple_category.get_object("C")

        hom_AB = simple_category.hom(A, B)
        hom_AA = simple_category.hom(A, A)
        hom_BC = simple_category.hom(B, C)

        assert len(hom_AB) == 1
        assert len(hom_AA) == 1  # Identity
        assert len(hom_BC) == 1

    def test_outgoing_morphisms(self, simple_category: Category) -> None:
        """Test getting outgoing morphisms."""
        A = simple_category.get_object("A")
        outgoing = simple_category.outgoing(A)

        assert len(outgoing) == 2  # f: A->B and h: A->C

    def test_incoming_morphisms(self, simple_category: Category) -> None:
        """Test getting incoming morphisms."""
        C = simple_category.get_object("C")
        incoming = simple_category.incoming(C)

        assert len(incoming) == 2  # g: B->C and h: A->C

    def test_is_initial(self, simple_category: Category) -> None:
        """Test initial object detection."""
        A = simple_category.get_object("A")
        C = simple_category.get_object("C")

        assert simple_category.is_initial(A)
        assert not simple_category.is_initial(C)

    def test_is_terminal(self, simple_category: Category) -> None:
        """Test terminal object detection."""
        A = simple_category.get_object("A")
        C = simple_category.get_object("C")

        assert not simple_category.is_terminal(A)
        assert simple_category.is_terminal(C)

    def test_path_exists(self, simple_category: Category) -> None:
        """Test path existence."""
        A = simple_category.get_object("A")
        B = simple_category.get_object("B")
        C = simple_category.get_object("C")

        assert simple_category.path_exists(A, C)
        assert simple_category.path_exists(A, B)
        assert not simple_category.path_exists(C, A)

    def test_all_paths(self, simple_category: Category) -> None:
        """Test finding all paths."""
        A = simple_category.get_object("A")
        C = simple_category.get_object("C")

        paths = simple_category.all_paths(A, C)

        assert len(paths) == 2  # A->C direct and A->B->C

    def test_composition(self, simple_category: Category) -> None:
        """Test morphism composition through category."""
        A = simple_category.get_object("A")
        B = simple_category.get_object("B")

        f = list(simple_category.hom(A, B) - {simple_category.get_identity(A)})[0]
        g = list(simple_category.hom(B, simple_category.get_object("C")))[0]

        fg = simple_category.compose(f, g)

        assert fg.source == A
        assert fg.target == simple_category.get_object("C")

    def test_verify_identity_law(self, simple_category: Category) -> None:
        """Test identity law verification."""
        assert simple_category.verify_identity_law()

    def test_golden_centrality(self, simple_category: Category) -> None:
        """Test golden ratio centrality computation."""
        B = simple_category.get_object("B")
        centrality = simple_category.golden_centrality(B)

        assert 0 <= centrality <= 1


class TestProductCategory:
    """Tests for ProductCategory class."""

    @pytest.fixture
    def product_category(self) -> ProductCategory:
        """Create a product of two simple categories."""
        cat1 = Category("C1")
        cat1.add_object(Object(name="A"))
        cat1.add_object(Object(name="B"))
        cat1.add_morphism(Morphism(
            name="f",
            source=cat1.get_object("A"),
            target=cat1.get_object("B"),
        ))

        cat2 = Category("C2")
        cat2.add_object(Object(name="X"))
        cat2.add_object(Object(name="Y"))
        cat2.add_morphism(Morphism(
            name="g",
            source=cat2.get_object("X"),
            target=cat2.get_object("Y"),
        ))

        return ProductCategory(cat1, cat2)

    def test_product_objects(self, product_category: ProductCategory) -> None:
        """Test product category has product objects."""
        assert len(product_category.objects) == 4  # 2 x 2

        assert product_category.get_object("(A,X)") is not None
        assert product_category.get_object("(A,Y)") is not None
        assert product_category.get_object("(B,X)") is not None
        assert product_category.get_object("(B,Y)") is not None

    def test_product_morphisms(self, product_category: ProductCategory) -> None:
        """Test product category has product morphisms."""
        # Should have (f,g), (f,id_X), (f,id_Y), (id_A,g), (id_B,g)
        # Non-identity morphisms in product
        assert len(product_category.morphisms) >= 1

    def test_projection(self, product_category: ProductCategory) -> None:
        """Test projection functions."""
        obj = product_category.get_object("(A,X)")

        proj1 = product_category.project1(obj)
        proj2 = product_category.project2(obj)

        assert proj1.name == "A"
        assert proj2.name == "X"


class TestOppositeCategory:
    """Tests for OppositeCategory class."""

    @pytest.fixture
    def opposite_category(self) -> OppositeCategory:
        """Create opposite of a simple category."""
        cat = Category("C")
        cat.add_object(Object(name="A"))
        cat.add_object(Object(name="B"))
        cat.add_morphism(Morphism(
            name="f",
            source=cat.get_object("A"),
            target=cat.get_object("B"),
        ))

        return OppositeCategory(cat)

    def test_opposite_objects(self, opposite_category: OppositeCategory) -> None:
        """Test opposite category has same objects."""
        assert len(opposite_category.objects) == 2
        assert opposite_category.get_object("A") is not None
        assert opposite_category.get_object("B") is not None

    def test_opposite_morphisms_reversed(
        self,
        opposite_category: OppositeCategory,
    ) -> None:
        """Test morphisms are reversed in opposite category."""
        A = opposite_category.get_object("A")
        B = opposite_category.get_object("B")

        # In original: f: A -> B
        # In opposite: f^op: B -> A
        hom_BA = opposite_category.hom(B, A)

        # Should have f^op (plus possibly identity if B == A, which it's not)
        non_id = [m for m in hom_BA if not m.is_identity()]
        assert len(non_id) == 1

        f_op = non_id[0]
        assert f_op.source == B
        assert f_op.target == A
        assert f_op.name == "f^op"

    def test_original_morphism(self, opposite_category: OppositeCategory) -> None:
        """Test getting original morphism from op morphism."""
        B = opposite_category.get_object("B")
        A = opposite_category.get_object("A")

        hom_BA = opposite_category.hom(B, A)
        f_op = [m for m in hom_BA if not m.is_identity()][0]

        original = opposite_category.original_morphism(f_op)

        assert original is not None
        assert original.name == "f"
        assert original.source.name == "A"
        assert original.target.name == "B"


class TestCategoryComposition:
    """Tests for compositionality in categories."""

    def test_associativity(self) -> None:
        """Test composition is associative."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")
        C = Object(name="C")
        D = Object(name="D")

        for obj in [A, B, C, D]:
            cat.add_object(obj)

        f = Morphism(name="f", source=A, target=B)
        g = Morphism(name="g", source=B, target=C)
        h = Morphism(name="h", source=C, target=D)

        # (f;g);h
        fg = f.compose(g)
        fg_h = fg.compose(h)

        # f;(g;h)
        gh = g.compose(h)
        f_gh = f.compose(gh)

        # Should have same source and target
        assert fg_h.source == f_gh.source
        assert fg_h.target == f_gh.target

    def test_identity_composition(self) -> None:
        """Test identity composition laws."""
        cat = Category("Test")

        A = Object(name="A")
        B = Object(name="B")

        cat.add_object(A)
        cat.add_object(B)

        f = Morphism(name="f", source=A, target=B)
        cat.add_morphism(f)

        id_A = cat.get_identity(A)
        id_B = cat.get_identity(B)

        # id_A ; f = f (same endpoints)
        id_f = id_A.compose(f)
        assert id_f.source == f.source
        assert id_f.target == f.target

        # f ; id_B = f (same endpoints)
        f_id = f.compose(id_B)
        assert f_id.source == f.source
        assert f_id.target == f.target
