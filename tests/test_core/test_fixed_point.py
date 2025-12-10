"""Tests for gnsp.core.fixed_point module."""

import numpy as np
import pytest

from gnsp.core.fixed_point import (
    QFormat,
    FixedPoint,
    Q8_8,
    Q16_16,
    Q1_15,
    Q4_12,
    float_to_fixed,
    fixed_to_float,
    fixed_multiply,
    fixed_mac,
    quantize_to_fixed,
)
from gnsp.constants import PHI, PHI_INV


class TestQFormat:
    """Test Q-format specification."""

    def test_q8_8_properties(self):
        """Test Q8.8 format properties."""
        assert Q8_8.integer_bits == 8
        assert Q8_8.fractional_bits == 8
        assert Q8_8.total_bits == 16
        assert Q8_8.scale == 256

    def test_q16_16_properties(self):
        """Test Q16.16 format properties."""
        assert Q16_16.integer_bits == 16
        assert Q16_16.fractional_bits == 16
        assert Q16_16.total_bits == 32
        assert Q16_16.scale == 65536

    def test_q1_15_properties(self):
        """Test Q1.15 format properties."""
        assert Q1_15.integer_bits == 1
        assert Q1_15.fractional_bits == 15
        assert Q1_15.total_bits == 16

    def test_max_value(self):
        """Test maximum representable value."""
        # Q8.8: max = 127 + (255/256) = 127.99609375
        assert Q8_8.max_value == pytest.approx(127.99609375, rel=1e-6)

    def test_min_value(self):
        """Test minimum representable value."""
        # Q8.8: min = -128
        assert Q8_8.min_value == -128

    def test_resolution(self):
        """Test resolution (smallest step)."""
        assert Q8_8.resolution == pytest.approx(1 / 256, rel=1e-10)
        assert Q16_16.resolution == pytest.approx(1 / 65536, rel=1e-10)

    def test_invalid_integer_bits(self):
        """Test that integer_bits < 1 raises error."""
        with pytest.raises(ValueError):
            QFormat(0, 8)

    def test_invalid_fractional_bits(self):
        """Test that negative fractional_bits raises error."""
        with pytest.raises(ValueError):
            QFormat(8, -1)

    def test_repr(self):
        """Test string representation."""
        assert str(Q8_8) == "Q8.8"
        assert str(Q16_16) == "Q16.16"


class TestFixedPoint:
    """Test FixedPoint class."""

    def test_create_from_float(self):
        """Test creating FixedPoint from float."""
        fp = FixedPoint(1.5, Q8_8)
        assert fp.value == pytest.approx(1.5, rel=1e-3)
        assert fp.raw == 384  # 1.5 * 256

    def test_create_from_int(self):
        """Test creating FixedPoint from int."""
        fp = FixedPoint(2, Q8_8)
        assert fp.value == pytest.approx(2.0, rel=1e-6)
        assert fp.raw == 512  # 2 * 256

    def test_create_from_raw(self):
        """Test creating FixedPoint from raw value."""
        fp = FixedPoint(256, Q8_8, from_raw=True)
        assert fp.value == pytest.approx(1.0, rel=1e-6)
        assert fp.raw == 256

    def test_phi_representation(self):
        """Test golden ratio representation."""
        fp = FixedPoint(PHI, Q8_8)
        assert fp.value == pytest.approx(PHI, rel=0.01)

    def test_phi_inv_representation(self):
        """Test 1/phi representation."""
        fp = FixedPoint(PHI_INV, Q8_8)
        assert fp.value == pytest.approx(PHI_INV, rel=0.01)

    def test_saturation_max(self):
        """Test saturation at maximum value."""
        fp = FixedPoint(200.0, Q8_8)
        assert fp.value == pytest.approx(Q8_8.max_value, rel=1e-3)

    def test_saturation_min(self):
        """Test saturation at minimum value."""
        fp = FixedPoint(-200.0, Q8_8)
        assert fp.value == pytest.approx(Q8_8.min_value, rel=1e-3)

    def test_addition(self):
        """Test fixed-point addition."""
        a = FixedPoint(1.5, Q8_8)
        b = FixedPoint(0.5, Q8_8)
        c = a + b
        assert c.value == pytest.approx(2.0, rel=1e-3)

    def test_subtraction(self):
        """Test fixed-point subtraction."""
        a = FixedPoint(2.0, Q8_8)
        b = FixedPoint(0.5, Q8_8)
        c = a - b
        assert c.value == pytest.approx(1.5, rel=1e-3)

    def test_multiplication(self):
        """Test fixed-point multiplication."""
        a = FixedPoint(2.0, Q8_8)
        b = FixedPoint(3.0, Q8_8)
        c = a * b
        assert c.value == pytest.approx(6.0, rel=0.01)

    def test_multiplication_fractional(self):
        """Test multiplication of fractional values."""
        a = FixedPoint(1.5, Q8_8)
        b = FixedPoint(2.0, Q8_8)
        c = a * b
        assert c.value == pytest.approx(3.0, rel=0.01)

    def test_division(self):
        """Test fixed-point division."""
        a = FixedPoint(6.0, Q8_8)
        b = FixedPoint(2.0, Q8_8)
        c = a / b
        assert c.value == pytest.approx(3.0, rel=0.01)

    def test_division_by_zero(self):
        """Test division by zero raises error."""
        a = FixedPoint(1.0, Q8_8)
        b = FixedPoint(0.0, Q8_8)
        with pytest.raises(ZeroDivisionError):
            _ = a / b

    def test_negation(self):
        """Test negation."""
        a = FixedPoint(1.5, Q8_8)
        b = -a
        assert b.value == pytest.approx(-1.5, rel=1e-3)

    def test_absolute(self):
        """Test absolute value."""
        a = FixedPoint(-1.5, Q8_8)
        b = abs(a)
        assert b.value == pytest.approx(1.5, rel=1e-3)

    def test_comparison_lt(self):
        """Test less than comparison."""
        a = FixedPoint(1.0, Q8_8)
        b = FixedPoint(2.0, Q8_8)
        assert a < b
        assert not b < a

    def test_comparison_le(self):
        """Test less than or equal comparison."""
        a = FixedPoint(1.0, Q8_8)
        b = FixedPoint(1.0, Q8_8)
        assert a <= b
        assert b <= a

    def test_comparison_gt(self):
        """Test greater than comparison."""
        a = FixedPoint(2.0, Q8_8)
        b = FixedPoint(1.0, Q8_8)
        assert a > b

    def test_comparison_eq(self):
        """Test equality comparison."""
        a = FixedPoint(1.5, Q8_8)
        b = FixedPoint(1.5, Q8_8)
        assert a == b

    def test_format_mismatch_error(self):
        """Test that format mismatch raises error."""
        a = FixedPoint(1.0, Q8_8)
        b = FixedPoint(1.0, Q16_16)
        with pytest.raises(ValueError):
            _ = a + b

    def test_to_format(self):
        """Test format conversion."""
        a = FixedPoint(1.5, Q8_8)
        b = a.to_format(Q16_16)
        assert b.value == pytest.approx(1.5, rel=1e-3)
        assert b.fmt == Q16_16

    def test_hash(self):
        """Test hashing for use in sets/dicts."""
        a = FixedPoint(1.5, Q8_8)
        b = FixedPoint(1.5, Q8_8)
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_repr(self):
        """Test string representation."""
        a = FixedPoint(1.5, Q8_8)
        assert "1.5" in repr(a)
        assert "Q8.8" in repr(a)


class TestArrayConversion:
    """Test array conversion functions."""

    def test_float_to_fixed_basic(self):
        """Test basic float to fixed conversion."""
        arr = np.array([1.0, 2.0, 0.5])
        fixed = float_to_fixed(arr, Q8_8)
        assert fixed[0] == 256
        assert fixed[1] == 512
        assert fixed[2] == 128

    def test_fixed_to_float_basic(self):
        """Test basic fixed to float conversion."""
        fixed = np.array([256, 512, 128], dtype=np.int32)
        floats = fixed_to_float(fixed, Q8_8)
        np.testing.assert_array_almost_equal(floats, [1.0, 2.0, 0.5], decimal=6)

    def test_roundtrip_conversion(self):
        """Test float -> fixed -> float roundtrip."""
        original = np.array([1.5, -0.25, 3.75])
        fixed = float_to_fixed(original, Q8_8)
        recovered = fixed_to_float(fixed, Q8_8)
        np.testing.assert_array_almost_equal(original, recovered, decimal=2)

    def test_saturation_in_conversion(self):
        """Test saturation during conversion."""
        arr = np.array([200.0, -200.0])
        fixed = float_to_fixed(arr, Q8_8)
        assert fixed[0] == Q8_8.max_raw
        assert fixed[1] == Q8_8.min_raw

    def test_phi_roundtrip(self):
        """Test golden ratio roundtrip conversion."""
        arr = np.array([PHI, PHI_INV])
        fixed = float_to_fixed(arr, Q8_8)
        recovered = fixed_to_float(fixed, Q8_8)
        assert recovered[0] == pytest.approx(PHI, rel=0.01)
        assert recovered[1] == pytest.approx(PHI_INV, rel=0.01)


class TestFixedMultiply:
    """Test fixed-point array multiplication."""

    def test_basic_multiply(self):
        """Test basic array multiplication."""
        a = float_to_fixed(np.array([2.0, 3.0]), Q8_8)
        b = float_to_fixed(np.array([3.0, 2.0]), Q8_8)
        c = fixed_multiply(a, b, Q8_8)
        result = fixed_to_float(c, Q8_8)
        np.testing.assert_array_almost_equal(result, [6.0, 6.0], decimal=1)

    def test_fractional_multiply(self):
        """Test fractional multiplication."""
        a = float_to_fixed(np.array([1.5]), Q8_8)
        b = float_to_fixed(np.array([2.0]), Q8_8)
        c = fixed_multiply(a, b, Q8_8)
        result = fixed_to_float(c, Q8_8)
        assert result[0] == pytest.approx(3.0, rel=0.01)


class TestFixedMAC:
    """Test multiply-accumulate operation."""

    def test_basic_mac(self):
        """Test basic MAC operation."""
        acc = float_to_fixed(np.array([1.0]), Q8_8)
        a = float_to_fixed(np.array([2.0]), Q8_8)
        b = float_to_fixed(np.array([3.0]), Q8_8)
        result = fixed_mac(acc, a, b, Q8_8)
        value = fixed_to_float(result, Q8_8)
        assert value[0] == pytest.approx(7.0, rel=0.01)  # 1 + 2*3


class TestQuantizeToFixed:
    """Test quantization function."""

    def test_quantize_basic(self):
        """Test basic quantization."""
        arr = np.array([1.6180339])
        quantized = quantize_to_fixed(arr, Q8_8)
        # Should be close but slightly different due to quantization
        assert quantized[0] != PHI
        assert quantized[0] == pytest.approx(PHI, rel=0.01)

    def test_quantize_preserves_shape(self):
        """Test that quantization preserves array shape."""
        arr = np.random.randn(10, 20)
        quantized = quantize_to_fixed(arr, Q8_8)
        assert quantized.shape == arr.shape
