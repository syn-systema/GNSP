"""Tests for gnsp.core.golden module."""

import numpy as np
import pytest

from gnsp.core.golden import (
    golden_decay,
    golden_growth,
    golden_threshold,
    fibonacci_sequence,
    fibonacci_generator,
    nearest_fibonacci,
    golden_spiral_points,
    golden_spiral_points_3d,
    golden_ratio_weights,
    is_near_golden,
    golden_power_decomposition,
    golden_matrix,
    lucas_sequence,
    verify_golden_eigenvalues,
    phi_continued_fraction,
    golden_interpolate,
    golden_section_search,
)
from gnsp.constants import PHI, PHI_INV, PHI_SQ


class TestGoldenDecay:
    """Test golden_decay function."""

    def test_single_step(self):
        """Test single step decay."""
        result = golden_decay(1.0, steps=1)
        assert result == pytest.approx(PHI_INV, rel=1e-10)

    def test_double_step(self):
        """Test double step decay."""
        result = golden_decay(1.0, steps=2)
        assert result == pytest.approx(PHI_INV ** 2, rel=1e-10)

    def test_zero_steps(self):
        """Test zero steps returns input."""
        result = golden_decay(5.0, steps=0)
        assert result == 5.0

    def test_preserves_zero(self):
        """Test that zero input stays zero."""
        assert golden_decay(0.0, steps=5) == 0.0

    def test_negative_steps_error(self):
        """Test negative steps raises error."""
        with pytest.raises(ValueError):
            golden_decay(1.0, steps=-1)

    def test_scaling(self):
        """Test scaling property."""
        value = 10.0
        result = golden_decay(value, steps=3)
        assert result == pytest.approx(value * PHI_INV ** 3, rel=1e-10)


class TestGoldenGrowth:
    """Test golden_growth function."""

    def test_single_step(self):
        """Test single step growth."""
        result = golden_growth(1.0, steps=1)
        assert result == pytest.approx(PHI, rel=1e-10)

    def test_inverse_of_decay(self):
        """Test growth is inverse of decay."""
        value = 5.0
        grown = golden_growth(value, steps=3)
        decayed = golden_decay(grown, steps=3)
        assert decayed == pytest.approx(value, rel=1e-10)


class TestGoldenThreshold:
    """Test golden_threshold function."""

    def test_level_zero(self):
        """Test level 0 returns base."""
        result = golden_threshold(1.0, level=0)
        assert result == 1.0

    def test_level_positive(self):
        """Test positive level."""
        result = golden_threshold(1.0, level=1)
        assert result == pytest.approx(PHI, rel=1e-10)

    def test_level_negative(self):
        """Test negative level."""
        result = golden_threshold(1.0, level=-1)
        assert result == pytest.approx(PHI_INV, rel=1e-10)

    def test_scaling(self):
        """Test with non-unit base."""
        result = golden_threshold(2.0, level=1)
        assert result == pytest.approx(2.0 * PHI, rel=1e-10)


class TestFibonacciSequence:
    """Test fibonacci_sequence function."""

    def test_first_ten(self):
        """Test first 10 Fibonacci numbers."""
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        assert fibonacci_sequence(10) == expected

    def test_empty(self):
        """Test n=0 returns empty list."""
        assert fibonacci_sequence(0) == []

    def test_single(self):
        """Test n=1 returns [1]."""
        assert fibonacci_sequence(1) == [1]

    def test_ratio_converges_to_phi(self):
        """Test ratio of consecutive Fibonacci converges to phi."""
        fibs = fibonacci_sequence(20)
        ratios = [fibs[i + 1] / fibs[i] for i in range(len(fibs) - 1)]
        assert ratios[-1] == pytest.approx(PHI, rel=1e-6)


class TestFibonacciGenerator:
    """Test fibonacci_generator function."""

    def test_first_ten(self):
        """Test first 10 values from generator."""
        gen = fibonacci_generator()
        values = [next(gen) for _ in range(10)]
        assert values == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    def test_is_infinite(self):
        """Test generator can produce many values."""
        gen = fibonacci_generator()
        values = [next(gen) for _ in range(100)]
        assert len(values) == 100
        assert values[-1] > 0


class TestNearestFibonacci:
    """Test nearest_fibonacci function."""

    def test_exact_match(self):
        """Test exact Fibonacci number."""
        assert nearest_fibonacci(8) == 8
        assert nearest_fibonacci(13) == 13

    def test_closer_to_lower(self):
        """Test number closer to lower Fibonacci."""
        assert nearest_fibonacci(6) == 5  # 6 is closer to 5 than 8

    def test_closer_to_higher(self):
        """Test number closer to higher Fibonacci."""
        assert nearest_fibonacci(7) == 8  # 7 is closer to 8 than 5

    def test_zero_or_negative(self):
        """Test zero or negative returns 1."""
        assert nearest_fibonacci(0) == 1
        assert nearest_fibonacci(-5) == 1


class TestGoldenSpiralPoints:
    """Test golden_spiral_points function."""

    def test_correct_shape(self):
        """Test output shape."""
        points = golden_spiral_points(100)
        assert points.shape == (100, 2)

    def test_origin_is_first(self):
        """Test first point is at origin."""
        points = golden_spiral_points(10)
        np.testing.assert_array_almost_equal(points[0], [0, 0])

    def test_spiral_grows(self):
        """Test points move outward."""
        points = golden_spiral_points(100)
        distances = np.linalg.norm(points, axis=1)
        # Later points should generally be farther from origin
        assert distances[-1] > distances[10]

    def test_scaling(self):
        """Test scale parameter."""
        points1 = golden_spiral_points(50, scale=1.0)
        points2 = golden_spiral_points(50, scale=2.0)
        # Scaled points should be 2x farther from origin
        ratio = np.linalg.norm(points2[-1]) / np.linalg.norm(points1[-1])
        assert ratio == pytest.approx(2.0, rel=0.01)


class TestGoldenSpiralPoints3D:
    """Test golden_spiral_points_3d function."""

    def test_correct_shape(self):
        """Test output shape."""
        points = golden_spiral_points_3d(100)
        assert points.shape == (100, 3)

    def test_spherical_on_sphere(self):
        """Test spherical mode places points on sphere."""
        points = golden_spiral_points_3d(100, scale=1.0, use_spherical=True)
        distances = np.linalg.norm(points, axis=1)
        np.testing.assert_array_almost_equal(distances, np.ones(100), decimal=10)


class TestGoldenRatioWeights:
    """Test golden_ratio_weights function."""

    def test_first_weight_is_one(self):
        """Test first weight is 1 (before normalization)."""
        weights = golden_ratio_weights(5, normalize=False)
        assert weights[0] == 1.0

    def test_decay_by_phi_inv(self):
        """Test weights decay by phi^-1."""
        weights = golden_ratio_weights(5, normalize=False)
        for i in range(1, len(weights)):
            expected = PHI_INV ** i
            assert weights[i] == pytest.approx(expected, rel=1e-10)

    def test_normalized_sum_to_one(self):
        """Test normalized weights sum to 1."""
        weights = golden_ratio_weights(10, normalize=True)
        assert np.sum(weights) == pytest.approx(1.0, rel=1e-10)


class TestIsNearGolden:
    """Test is_near_golden function."""

    def test_phi_itself(self):
        """Test phi is near golden."""
        assert is_near_golden(PHI)

    def test_phi_inv(self):
        """Test 1/phi is near golden."""
        assert is_near_golden(PHI_INV)

    def test_phi_squared(self):
        """Test phi^2 is near golden."""
        assert is_near_golden(PHI_SQ)

    def test_arbitrary_number(self):
        """Test arbitrary number is not near golden."""
        assert not is_near_golden(1.5)
        assert not is_near_golden(2.0)

    def test_negative_phi(self):
        """Test negative phi is near golden."""
        assert is_near_golden(-PHI)


class TestGoldenMatrix:
    """Test golden_matrix function."""

    def test_2x2_structure(self):
        """Test 2x2 matrix structure."""
        M = golden_matrix(2)
        expected = np.array([[0, 1], [1, 1]])
        np.testing.assert_array_equal(M, expected)

    def test_2x2_eigenvalues(self):
        """Test 2x2 matrix has phi eigenvalues."""
        M = golden_matrix(2)
        eigenvalues = np.linalg.eigvals(M)
        eigenvalues = sorted(eigenvalues, reverse=True)
        assert eigenvalues[0] == pytest.approx(PHI, rel=1e-10)
        assert eigenvalues[1] == pytest.approx(-PHI_INV, rel=1e-10)


class TestLucasSequence:
    """Test lucas_sequence function."""

    def test_first_values(self):
        """Test first Lucas numbers."""
        lucas = lucas_sequence(10)
        expected = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
        assert lucas == expected

    def test_recurrence(self):
        """Test Lucas recurrence."""
        lucas = lucas_sequence(15)
        for i in range(2, len(lucas)):
            assert lucas[i] == lucas[i - 1] + lucas[i - 2]


class TestVerifyGoldenEigenvalues:
    """Test verify_golden_eigenvalues function."""

    def test_golden_matrix(self):
        """Test with known golden matrix."""
        M = golden_matrix(2)
        result = verify_golden_eigenvalues(M)
        # Eigenvalues are phi and -1/phi
        # phi matches power 1, -1/phi matches -target where target=PHI_INV (power -1)
        assert len(result['golden_eigenvalues']) == 2
        assert 1 in result['phi_powers_found']

    def test_non_golden_matrix(self):
        """Test with non-golden matrix."""
        M = np.array([[1, 0], [0, 2]])
        result = verify_golden_eigenvalues(M)
        # Eigenvalue 1 = phi^0
        assert 0 in result['phi_powers_found']

    def test_non_square_error(self):
        """Test non-square matrix raises error."""
        M = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            verify_golden_eigenvalues(M)


class TestPhiContinuedFraction:
    """Test phi_continued_fraction function."""

    def test_convergence(self):
        """Test continued fraction converges to phi."""
        approx = phi_continued_fraction(50)
        assert approx == pytest.approx(PHI, rel=1e-10)

    def test_low_iterations(self):
        """Test low iterations gives rough approximation."""
        approx = phi_continued_fraction(5)
        assert approx == pytest.approx(PHI, rel=0.01)


class TestGoldenInterpolate:
    """Test golden_interpolate function."""

    def test_endpoints(self):
        """Test interpolation at endpoints."""
        assert golden_interpolate(0, 10, 0) == 0
        assert golden_interpolate(0, 10, 1) == 10

    def test_midpoint(self):
        """Test interpolation at midpoint."""
        result = golden_interpolate(0, 10, 0.5)
        assert result == 5.0


class TestGoldenSectionSearch:
    """Test golden_section_search function."""

    def test_quadratic_minimum(self):
        """Test finding minimum of quadratic."""
        f = lambda x: (x - 2.0) ** 2
        x_min, f_min = golden_section_search(f, 0, 4)
        assert x_min == pytest.approx(2.0, rel=1e-5)
        assert f_min == pytest.approx(0.0, abs=1e-10)

    def test_phi_minimum(self):
        """Test finding minimum near phi."""
        f = lambda x: (x - PHI) ** 2
        x_min, _ = golden_section_search(f, 1, 2)
        assert x_min == pytest.approx(PHI, rel=1e-5)
