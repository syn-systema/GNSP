"""Tests for gnsp.constants module."""

import math
import pytest

from gnsp.constants import (
    PHI,
    PHI_INV,
    PHI_SQ,
    PHI_INV_SQ,
    PHI_CUBE,
    PHI_INV_CUBE,
    FIBONACCI,
    FIBONACCI_EXTENDED,
    GOLDEN_ANGLE,
    WEIGHT_LEVELS,
    STDP_TAU,
    DEFAULT_THRESHOLD,
    DEFAULT_RESET,
    DEFAULT_LEAK,
    THRESHOLD_LOW,
    THRESHOLD_MID,
    THRESHOLD_HIGH,
    N_WEIGHT_LEVELS,
)


class TestGoldenRatioIdentities:
    """Test mathematical identities of the golden ratio."""

    def test_phi_squared_equals_phi_plus_one(self):
        """Verify phi^2 = phi + 1."""
        assert abs(PHI_SQ - (PHI + 1)) < 1e-10

    def test_phi_times_phi_inv_equals_one(self):
        """Verify phi * (1/phi) = 1."""
        assert abs(PHI * PHI_INV - 1) < 1e-10

    def test_phi_inv_equals_phi_minus_one(self):
        """Verify 1/phi = phi - 1."""
        assert abs(PHI_INV - (PHI - 1)) < 1e-10

    def test_phi_inv_sq_value(self):
        """Verify PHI_INV_SQ = 1/phi^2."""
        assert abs(PHI_INV_SQ - (1 / PHI_SQ)) < 1e-10

    def test_phi_cube_value(self):
        """Verify PHI_CUBE = phi^3."""
        assert abs(PHI_CUBE - (PHI ** 3)) < 1e-10

    def test_phi_inv_cube_value(self):
        """Verify PHI_INV_CUBE = 1/phi^3."""
        assert abs(PHI_INV_CUBE - (PHI_INV ** 3)) < 1e-10

    def test_phi_value(self):
        """Verify phi = (1 + sqrt(5)) / 2."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-14


class TestFibonacciConstants:
    """Test Fibonacci sequence constants."""

    def test_fibonacci_starts_correctly(self):
        """Verify Fibonacci starts with 1, 1."""
        assert FIBONACCI[0] == 1
        assert FIBONACCI[1] == 1

    def test_fibonacci_recurrence(self):
        """Verify Fibonacci recurrence relation."""
        for i in range(2, len(FIBONACCI)):
            assert FIBONACCI[i] == FIBONACCI[i - 1] + FIBONACCI[i - 2]

    def test_fibonacci_extended_continues(self):
        """Verify extended Fibonacci continues from base."""
        assert FIBONACCI_EXTENDED[:len(FIBONACCI)] == FIBONACCI

    def test_fibonacci_extended_recurrence(self):
        """Verify extended Fibonacci recurrence."""
        for i in range(2, len(FIBONACCI_EXTENDED)):
            assert FIBONACCI_EXTENDED[i] == FIBONACCI_EXTENDED[i - 1] + FIBONACCI_EXTENDED[i - 2]

    def test_fibonacci_ratio_converges_to_phi(self):
        """Verify F(n+1)/F(n) converges to phi."""
        ratio = FIBONACCI[-1] / FIBONACCI[-2]
        assert abs(ratio - PHI) < 0.0001


class TestGoldenAngle:
    """Test golden angle constant."""

    def test_golden_angle_value(self):
        """Verify golden angle = 2*pi/phi^2."""
        expected = 2 * math.pi / (PHI ** 2)
        assert abs(GOLDEN_ANGLE - expected) < 1e-10

    def test_golden_angle_degrees(self):
        """Golden angle should be approximately 137.5 degrees."""
        degrees = GOLDEN_ANGLE * 180 / math.pi
        assert 137 < degrees < 138


class TestWeightLevels:
    """Test weight quantization levels."""

    def test_nine_weight_levels(self):
        """Verify 9 weight levels."""
        assert len(WEIGHT_LEVELS) == 9
        assert N_WEIGHT_LEVELS == 9

    def test_weight_levels_symmetric(self):
        """Verify weights are symmetric around 0."""
        for i in range(4):
            assert abs(WEIGHT_LEVELS[i] + WEIGHT_LEVELS[8 - i]) < 1e-10

    def test_weight_levels_include_zero(self):
        """Verify zero is in the middle."""
        assert WEIGHT_LEVELS[4] == 0.0

    def test_weight_levels_sorted(self):
        """Verify weights are sorted ascending."""
        for i in range(len(WEIGHT_LEVELS) - 1):
            assert WEIGHT_LEVELS[i] < WEIGHT_LEVELS[i + 1]

    def test_weight_levels_golden_values(self):
        """Verify weights match golden ratio powers."""
        assert abs(WEIGHT_LEVELS[0] - (-PHI_SQ)) < 1e-10
        assert abs(WEIGHT_LEVELS[1] - (-PHI)) < 1e-10
        assert abs(WEIGHT_LEVELS[3] - (-PHI_INV)) < 1e-10
        assert abs(WEIGHT_LEVELS[5] - PHI_INV) < 1e-10
        assert abs(WEIGHT_LEVELS[7] - PHI) < 1e-10
        assert abs(WEIGHT_LEVELS[8] - PHI_SQ) < 1e-10


class TestSTDPTau:
    """Test STDP time constants."""

    def test_stdp_tau_is_fibonacci(self):
        """Verify STDP tau values are Fibonacci."""
        expected = (1, 2, 3, 5, 8, 13, 21, 34)
        assert STDP_TAU == expected

    def test_stdp_tau_recurrence(self):
        """Verify Fibonacci recurrence in STDP tau."""
        for i in range(2, len(STDP_TAU)):
            assert STDP_TAU[i] == STDP_TAU[i - 1] + STDP_TAU[i - 2]


class TestSNNDefaults:
    """Test SNN default parameters."""

    def test_default_threshold_is_phi(self):
        """Verify default threshold equals phi."""
        assert abs(DEFAULT_THRESHOLD - PHI) < 1e-10

    def test_default_reset_is_phi_inv(self):
        """Verify default reset equals 1/phi."""
        assert abs(DEFAULT_RESET - PHI_INV) < 1e-10

    def test_default_leak_is_phi_inv_sq(self):
        """Verify default leak equals 1/phi^2."""
        assert abs(DEFAULT_LEAK - PHI_INV_SQ) < 1e-10

    def test_threshold_greater_than_reset(self):
        """Verify threshold > reset for proper spiking."""
        assert DEFAULT_THRESHOLD > DEFAULT_RESET

    def test_leak_in_valid_range(self):
        """Verify leak is in (0, 1)."""
        assert 0 < DEFAULT_LEAK < 1


class TestDetectionThresholds:
    """Test detection threshold values."""

    def test_thresholds_ascending(self):
        """Verify thresholds are in ascending order."""
        assert THRESHOLD_LOW < THRESHOLD_MID < THRESHOLD_HIGH < 1.0

    def test_threshold_low_is_phi_inv_cube(self):
        """Verify low threshold equals 1/phi^3."""
        assert abs(THRESHOLD_LOW - PHI_INV_CUBE) < 1e-10

    def test_threshold_mid_is_phi_inv_sq(self):
        """Verify mid threshold equals 1/phi^2."""
        assert abs(THRESHOLD_MID - PHI_INV_SQ) < 1e-10

    def test_threshold_high_is_phi_inv(self):
        """Verify high threshold equals 1/phi."""
        assert abs(THRESHOLD_HIGH - PHI_INV) < 1e-10
