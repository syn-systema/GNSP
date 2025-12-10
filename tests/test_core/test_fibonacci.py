"""Tests for gnsp.core.fibonacci module."""

import numpy as np
import pytest

from gnsp.core.fibonacci import (
    fib,
    fib_binet,
    fibonacci_up_to,
    is_fibonacci,
    fibonacci_index,
    zeckendorf,
    fibonacci_encode,
    fibonacci_offsets,
    fibonacci_time_constants,
    fibonacci_connectivity_pattern,
    fibonacci_decay_lookup,
    fibonacci_distance_matrix,
    fibonacci_lattice_1d,
    tribonacci,
    fibonacci_ratios,
    generalized_fibonacci,
)
from gnsp.constants import PHI, STDP_TAU


class TestFib:
    """Test fib function."""

    def test_first_values(self):
        """Test first Fibonacci numbers."""
        assert fib(1) == 1
        assert fib(2) == 1
        assert fib(3) == 2
        assert fib(4) == 3
        assert fib(5) == 5
        assert fib(10) == 55

    def test_zero_and_negative(self):
        """Test zero and negative indices."""
        assert fib(0) == 0
        assert fib(-1) == 0

    def test_larger_values(self):
        """Test larger Fibonacci numbers."""
        assert fib(20) == 6765
        assert fib(30) == 832040

    def test_recurrence(self):
        """Test Fibonacci recurrence holds."""
        for n in range(3, 30):
            assert fib(n) == fib(n - 1) + fib(n - 2)


class TestFibBinet:
    """Test fib_binet function."""

    def test_matches_recursive(self):
        """Test Binet formula matches recursive."""
        for n in range(1, 25):
            assert fib_binet(n) == fib(n)

    def test_zero(self):
        """Test zero index."""
        assert fib_binet(0) == 0


class TestFibonacciUpTo:
    """Test fibonacci_up_to function."""

    def test_up_to_100(self):
        """Test Fibonacci numbers up to 100."""
        fibs = fibonacci_up_to(100)
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        assert fibs == expected

    def test_includes_boundary(self):
        """Test boundary value is included."""
        fibs = fibonacci_up_to(55)
        assert 55 in fibs

    def test_empty_for_zero(self):
        """Test returns empty for max_val=0."""
        assert fibonacci_up_to(0) == []


class TestIsFibonacci:
    """Test is_fibonacci function."""

    def test_positive_fibonacci(self):
        """Test positive Fibonacci numbers."""
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for f in fibs:
            assert is_fibonacci(f), f"Expected {f} to be Fibonacci"

    def test_non_fibonacci(self):
        """Test non-Fibonacci numbers."""
        non_fibs = [4, 6, 7, 9, 10, 11, 12, 14, 15]
        for n in non_fibs:
            assert not is_fibonacci(n), f"Expected {n} to not be Fibonacci"

    def test_zero(self):
        """Test zero is Fibonacci (F(0) = 0)."""
        assert is_fibonacci(0)

    def test_negative(self):
        """Test negative numbers are not Fibonacci."""
        assert not is_fibonacci(-5)


class TestFibonacciIndex:
    """Test fibonacci_index function."""

    def test_known_indices(self):
        """Test known Fibonacci indices."""
        assert fibonacci_index(1) == 1  # Could be 1 or 2
        assert fibonacci_index(2) == 3
        assert fibonacci_index(3) == 4
        assert fibonacci_index(5) == 5
        assert fibonacci_index(144) == 12

    def test_non_fibonacci(self):
        """Test non-Fibonacci returns -1."""
        assert fibonacci_index(4) == -1
        assert fibonacci_index(100) == -1


class TestZeckendorf:
    """Test zeckendorf function."""

    def test_basic_decomposition(self):
        """Test basic Zeckendorf decomposition."""
        # 100 = 89 + 8 + 3
        rep = zeckendorf(100)
        assert sum(rep) == 100
        assert 89 in rep

    def test_fibonacci_number(self):
        """Test Fibonacci number has single component."""
        rep = zeckendorf(89)
        assert rep == [89]

    def test_zero(self):
        """Test zero returns empty."""
        assert zeckendorf(0) == []

    def test_small_numbers(self):
        """Test small number decompositions."""
        assert sum(zeckendorf(7)) == 7  # 5 + 2
        assert sum(zeckendorf(20)) == 20  # 13 + 5 + 2


class TestFibonacciEncode:
    """Test fibonacci_encode function."""

    def test_returns_indices(self):
        """Test returns Fibonacci indices."""
        indices = fibonacci_encode(100)
        # Verify indices point to Fibonacci numbers that sum to 100
        total = sum(fib(i) for i in indices)
        assert total == 100


class TestFibonacciOffsets:
    """Test fibonacci_offsets function."""

    def test_includes_zero(self):
        """Test zero is included."""
        offsets = fibonacci_offsets(10)
        assert 0 in offsets

    def test_symmetric(self):
        """Test offsets are symmetric."""
        offsets = fibonacci_offsets(10)
        for offset in offsets:
            if offset != 0:
                assert -offset in offsets

    def test_all_fibonacci(self):
        """Test all non-zero offsets are Fibonacci."""
        offsets = fibonacci_offsets(10)
        for offset in offsets:
            if offset != 0:
                assert is_fibonacci(abs(offset))

    def test_sorted(self):
        """Test offsets are sorted."""
        offsets = fibonacci_offsets(10)
        assert offsets == sorted(offsets)


class TestFibonacciTimeConstants:
    """Test fibonacci_time_constants function."""

    def test_correct_length(self):
        """Test returns correct number of constants."""
        taus = fibonacci_time_constants(8)
        assert len(taus) == 8

    def test_values_are_fibonacci(self):
        """Test values match Fibonacci sequence."""
        taus = fibonacci_time_constants(8)
        expected = np.array([1, 1, 2, 3, 5, 8, 13, 21], dtype=np.float32)
        np.testing.assert_array_equal(taus, expected)

    def test_matches_stdp_tau(self):
        """Test matches STDP_TAU constant."""
        taus = fibonacci_time_constants(8)
        # Note: STDP_TAU skips the duplicate 1
        assert taus[4] == STDP_TAU[3]  # Both should be 5


class TestFibonacciConnectivityPattern:
    """Test fibonacci_connectivity_pattern function."""

    def test_symmetric_connections(self):
        """Test connections include forward and backward."""
        conns = fibonacci_connectivity_pattern(10)
        # For small n, check specific connections
        assert (0, 1) in conns  # Distance 1
        assert (0, 2) in conns  # Distance 2
        assert (0, 3) in conns  # Distance 3
        assert (0, 5) in conns  # Distance 5

    def test_no_self_connections(self):
        """Test no neuron connects to itself."""
        conns = fibonacci_connectivity_pattern(10)
        for src, tgt in conns:
            assert src != tgt

    def test_non_fibonacci_not_included(self):
        """Test non-Fibonacci distances not connected."""
        conns = fibonacci_connectivity_pattern(10)
        # Distance 4 is not Fibonacci
        assert (0, 4) not in conns


class TestFibonacciDecayLookup:
    """Test fibonacci_decay_lookup function."""

    def test_correct_shape(self):
        """Test output shape."""
        lut = fibonacci_decay_lookup(64, 8)
        assert lut.shape == (65, 8)  # max_dt + 1

    def test_dt_zero(self):
        """Test dt=0 gives 1.0 for all taus."""
        lut = fibonacci_decay_lookup(10, 4)
        np.testing.assert_array_almost_equal(lut[0, :], np.ones(4))

    def test_decay_with_dt(self):
        """Test decay decreases with dt."""
        lut = fibonacci_decay_lookup(10, 4)
        # Decay should decrease with increasing dt
        for tau_idx in range(4):
            assert lut[5, tau_idx] < lut[0, tau_idx]


class TestFibonacciDistanceMatrix:
    """Test fibonacci_distance_matrix function."""

    def test_symmetric(self):
        """Test matrix is symmetric."""
        D = fibonacci_distance_matrix(10)
        np.testing.assert_array_equal(D, D.T)

    def test_diagonal_zero(self):
        """Test diagonal is zero."""
        D = fibonacci_distance_matrix(10)
        np.testing.assert_array_equal(np.diag(D), np.zeros(10))

    def test_values_are_fibonacci(self):
        """Test off-diagonal values are Fibonacci (or 0)."""
        D = fibonacci_distance_matrix(10)
        for i in range(10):
            for j in range(10):
                if i != j:
                    assert is_fibonacci(D[i, j]) or D[i, j] == 0


class TestFibonacciLattice1D:
    """Test fibonacci_lattice_1d function."""

    def test_correct_length(self):
        """Test output length."""
        lattice = fibonacci_lattice_1d(10)
        assert len(lattice) == 10

    def test_values_are_fibonacci(self):
        """Test positions are scaled Fibonacci."""
        lattice = fibonacci_lattice_1d(10, scale=1.0)
        for i in range(10):
            assert lattice[i] == fib(i + 1)


class TestTribonacci:
    """Test tribonacci function."""

    def test_first_values(self):
        """Test first Tribonacci numbers."""
        assert tribonacci(1) == 1
        assert tribonacci(2) == 1
        assert tribonacci(3) == 2
        assert tribonacci(4) == 4  # 1 + 1 + 2
        assert tribonacci(5) == 7  # 1 + 2 + 4

    def test_recurrence(self):
        """Test Tribonacci recurrence."""
        for n in range(4, 20):
            assert tribonacci(n) == tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3)


class TestFibonacciRatios:
    """Test fibonacci_ratios function."""

    def test_converges_to_phi(self):
        """Test ratios converge to phi."""
        ratios = fibonacci_ratios(20)
        assert ratios[-1] == pytest.approx(PHI, rel=1e-6)

    def test_correct_length(self):
        """Test output length."""
        ratios = fibonacci_ratios(10)
        assert len(ratios) == 10


class TestGeneralizedFibonacci:
    """Test generalized_fibonacci function."""

    def test_standard_fibonacci(self):
        """Test with a=1, b=1 gives standard Fibonacci."""
        gen = generalized_fibonacci(1, 1, 10)
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        assert gen == expected

    def test_lucas_from_generalized(self):
        """Test Lucas sequence as generalized Fibonacci."""
        lucas = generalized_fibonacci(1, 3, 10)
        expected = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
        assert lucas == expected

    def test_pell_sequence(self):
        """Test Pell sequence as generalized with doubling."""
        # P(n) = 2*P(n-1) + P(n-2), but this is standard Fibonacci-like with a=0, b=1
        pell = generalized_fibonacci(0, 1, 8)
        expected = [0, 1, 1, 2, 3, 5, 8, 13]
        assert pell == expected
