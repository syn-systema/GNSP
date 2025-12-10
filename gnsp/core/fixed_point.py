"""
Fixed-point arithmetic for hardware compatibility.

Uses Q-format notation: Qm.n means m integer bits, n fractional bits.
Default is Q8.8 (16-bit total) but configurable.

This module provides fixed-point number representations that map directly
to integer arithmetic, suitable for FPGA synthesis.

Example:
    >>> from gnsp.core.fixed_point import FixedPoint, Q8_8
    >>> x = FixedPoint(1.618, Q8_8)
    >>> y = FixedPoint(0.618, Q8_8)
    >>> z = x * y
    >>> print(z.value)  # ~1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class QFormat:
    """
    Q-format specification for fixed-point numbers.

    Args:
        integer_bits: Number of bits for integer part (including sign bit)
        fractional_bits: Number of bits for fractional part

    Example:
        Q8.8 means 8 integer bits (including sign), 8 fractional bits = 16 bits total
        Range: [-128, 127.99609375]
        Resolution: 1/256 = 0.00390625
    """

    integer_bits: int
    fractional_bits: int

    def __post_init__(self) -> None:
        """Validate Q-format parameters."""
        if self.integer_bits < 1:
            raise ValueError("integer_bits must be at least 1 (for sign bit)")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must be non-negative")

    @property
    def total_bits(self) -> int:
        """Total number of bits in the representation."""
        return self.integer_bits + self.fractional_bits

    @property
    def scale(self) -> int:
        """Scaling factor (2^fractional_bits)."""
        return 1 << self.fractional_bits

    @property
    def max_value(self) -> float:
        """Maximum representable value."""
        return (1 << (self.integer_bits - 1)) - (1 / self.scale)

    @property
    def min_value(self) -> float:
        """Minimum representable value."""
        return -(1 << (self.integer_bits - 1))

    @property
    def resolution(self) -> float:
        """Smallest representable step size."""
        return 1.0 / self.scale

    @property
    def max_raw(self) -> int:
        """Maximum raw integer value."""
        return (1 << (self.total_bits - 1)) - 1

    @property
    def min_raw(self) -> int:
        """Minimum raw integer value."""
        return -(1 << (self.total_bits - 1))

    def __repr__(self) -> str:
        return f"Q{self.integer_bits}.{self.fractional_bits}"


# Pre-defined Q-format specifications
Q8_8 = QFormat(8, 8)       # 16-bit, range [-128, 127.996], resolution 1/256
Q16_16 = QFormat(16, 16)   # 32-bit, range [-32768, 32767.9999], resolution 1/65536
Q1_15 = QFormat(1, 15)     # 16-bit, range [-1, 0.999969], resolution 1/32768
Q4_12 = QFormat(4, 12)     # 16-bit, range [-8, 7.9998], resolution 1/4096
Q8_24 = QFormat(8, 24)     # 32-bit, high precision for accumulators


class FixedPoint:
    """
    Fixed-point number representation.

    Designed for eventual hardware synthesis - all operations
    map directly to integer arithmetic.

    Args:
        value: Float value to convert, or raw integer if from_raw=True
        fmt: Q-format specification (default Q8.8)
        from_raw: If True, value is treated as raw integer representation

    Example:
        >>> x = FixedPoint(1.5, Q8_8)
        >>> x.value
        1.5
        >>> x.raw
        384  # 1.5 * 256
    """

    __slots__ = ('fmt', '_raw')

    def __init__(
        self,
        value: Union[float, int],
        fmt: QFormat = Q8_8,
        from_raw: bool = False
    ):
        self.fmt = fmt

        if from_raw:
            self._raw = int(value)
        elif isinstance(value, float) or isinstance(value, int):
            self._raw = int(round(value * fmt.scale))
        else:
            raise TypeError(f"Expected float or int, got {type(value)}")

        # Clamp to valid range (saturation arithmetic)
        self._raw = max(fmt.min_raw, min(fmt.max_raw, self._raw))

    @property
    def value(self) -> float:
        """Convert to floating point."""
        return self._raw / self.fmt.scale

    @property
    def raw(self) -> int:
        """Get raw integer representation."""
        return self._raw

    def to_format(self, new_fmt: QFormat) -> FixedPoint:
        """
        Convert to a different Q-format.

        Args:
            new_fmt: Target Q-format

        Returns:
            New FixedPoint in target format
        """
        if new_fmt.fractional_bits > self.fmt.fractional_bits:
            # Shifting up (more precision)
            shift = new_fmt.fractional_bits - self.fmt.fractional_bits
            new_raw = self._raw << shift
        else:
            # Shifting down (less precision) - round
            shift = self.fmt.fractional_bits - new_fmt.fractional_bits
            new_raw = (self._raw + (1 << (shift - 1))) >> shift if shift > 0 else self._raw
        return FixedPoint(new_raw, new_fmt, from_raw=True)

    def __add__(self, other: FixedPoint) -> FixedPoint:
        """Add two fixed-point numbers."""
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        result_raw = self._raw + other._raw
        return FixedPoint(result_raw, self.fmt, from_raw=True)

    def __sub__(self, other: FixedPoint) -> FixedPoint:
        """Subtract two fixed-point numbers."""
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        result_raw = self._raw - other._raw
        return FixedPoint(result_raw, self.fmt, from_raw=True)

    def __mul__(self, other: FixedPoint) -> FixedPoint:
        """
        Multiply two fixed-point numbers.

        Uses full-precision intermediate result then scales back.
        """
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        # Full precision multiply
        full_result = self._raw * other._raw
        # Scale back by fractional bits (with rounding)
        half = 1 << (self.fmt.fractional_bits - 1) if self.fmt.fractional_bits > 0 else 0
        result_raw = (full_result + half) >> self.fmt.fractional_bits
        return FixedPoint(result_raw, self.fmt, from_raw=True)

    def __truediv__(self, other: FixedPoint) -> FixedPoint:
        """
        Divide two fixed-point numbers.

        Scales dividend before division to maintain precision.
        """
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        if other._raw == 0:
            raise ZeroDivisionError("Division by zero in fixed-point")
        # Scale dividend up before division
        scaled_dividend = self._raw << self.fmt.fractional_bits
        result_raw = scaled_dividend // other._raw
        return FixedPoint(result_raw, self.fmt, from_raw=True)

    def __neg__(self) -> FixedPoint:
        """Negate the fixed-point number."""
        return FixedPoint(-self._raw, self.fmt, from_raw=True)

    def __abs__(self) -> FixedPoint:
        """Absolute value."""
        return FixedPoint(abs(self._raw), self.fmt, from_raw=True)

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, FixedPoint):
            return NotImplemented
        return self.fmt == other.fmt and self._raw == other._raw

    def __lt__(self, other: FixedPoint) -> bool:
        """Less than comparison."""
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        return self._raw < other._raw

    def __le__(self, other: FixedPoint) -> bool:
        """Less than or equal comparison."""
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        return self._raw <= other._raw

    def __gt__(self, other: FixedPoint) -> bool:
        """Greater than comparison."""
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        return self._raw > other._raw

    def __ge__(self, other: FixedPoint) -> bool:
        """Greater than or equal comparison."""
        if self.fmt != other.fmt:
            raise ValueError(f"Q-format mismatch: {self.fmt} vs {other.fmt}")
        return self._raw >= other._raw

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.fmt.integer_bits, self.fmt.fractional_bits, self._raw))

    def __repr__(self) -> str:
        return f"FixedPoint({self.value:.6f}, {self.fmt})"

    def __str__(self) -> str:
        return f"{self.value:.6f}"


def float_to_fixed(
    arr: np.ndarray,
    fmt: QFormat = Q8_8,
    dtype: np.dtype = np.int32
) -> np.ndarray:
    """
    Convert float array to fixed-point integers.

    Args:
        arr: Input float array
        fmt: Q-format specification
        dtype: Output integer dtype (default int32)

    Returns:
        Array of fixed-point integer representations

    Example:
        >>> arr = np.array([1.5, 0.25, -0.5])
        >>> fixed = float_to_fixed(arr, Q8_8)
        >>> fixed
        array([384, 64, -128], dtype=int32)
    """
    scaled = np.round(arr * fmt.scale)
    clipped = np.clip(scaled, fmt.min_raw, fmt.max_raw)
    return clipped.astype(dtype)


def fixed_to_float(
    arr: np.ndarray,
    fmt: QFormat = Q8_8,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Convert fixed-point integers to float array.

    Args:
        arr: Input fixed-point integer array
        fmt: Q-format specification
        dtype: Output float dtype (default float32)

    Returns:
        Array of float values

    Example:
        >>> fixed = np.array([384, 64, -128], dtype=np.int32)
        >>> floats = fixed_to_float(fixed, Q8_8)
        >>> floats
        array([1.5, 0.25, -0.5], dtype=float32)
    """
    return arr.astype(dtype) / fmt.scale


def fixed_multiply(
    a: np.ndarray,
    b: np.ndarray,
    fmt: QFormat = Q8_8,
    dtype: np.dtype = np.int32
) -> np.ndarray:
    """
    Multiply two fixed-point arrays element-wise.

    Uses 64-bit intermediate for overflow safety, then scales back.

    Args:
        a: First fixed-point array
        b: Second fixed-point array
        fmt: Q-format specification
        dtype: Output dtype

    Returns:
        Product array in fixed-point
    """
    # Use int64 for intermediate to avoid overflow
    product = a.astype(np.int64) * b.astype(np.int64)
    # Round and scale back
    half = 1 << (fmt.fractional_bits - 1) if fmt.fractional_bits > 0 else 0
    result = (product + half) >> fmt.fractional_bits
    # Clip and convert
    clipped = np.clip(result, fmt.min_raw, fmt.max_raw)
    return clipped.astype(dtype)


def fixed_mac(
    acc: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fmt: QFormat = Q8_8
) -> np.ndarray:
    """
    Multiply-accumulate operation: acc += a * b

    Common operation in neural networks. Uses higher precision accumulator.

    Args:
        acc: Accumulator array (modified in place)
        a: First operand array
        b: Second operand array
        fmt: Q-format specification

    Returns:
        Updated accumulator
    """
    product = fixed_multiply(a, b, fmt)
    acc += product
    np.clip(acc, fmt.min_raw, fmt.max_raw, out=acc)
    return acc


def quantize_to_fixed(
    arr: np.ndarray,
    fmt: QFormat = Q8_8
) -> np.ndarray:
    """
    Quantize float array to fixed-point precision and back to float.

    Useful for simulating fixed-point effects in floating-point code.

    Args:
        arr: Input float array
        fmt: Q-format specification

    Returns:
        Float array with fixed-point quantization applied

    Example:
        >>> arr = np.array([1.6180339])
        >>> quantize_to_fixed(arr, Q8_8)
        array([1.6171875], dtype=float32)
    """
    fixed = float_to_fixed(arr, fmt)
    return fixed_to_float(fixed, fmt)
