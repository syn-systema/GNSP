"""
Fibonacci sequence utilities for timing and connectivity.

Fibonacci numbers appear naturally in:
- STDP time constants
- Network connectivity distances
- Hierarchical routing levels
- Quasicrystalline lattices

Properties:
- F(n+1)/F(n) -> phi as n -> infinity
- F(n) = round(phi^n / sqrt(5))
- Zeckendorf: every positive integer has unique Fibonacci representation
- F(n)^2 + F(n+1)^2 = F(2n+1)

Example:
    >>> from gnsp.core.fibonacci import fib, fibonacci_up_to, is_fibonacci
    >>> fib(10)
    55
    >>> fibonacci_up_to(100)
    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    >>> is_fibonacci(144)
    True
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Set, Tuple
import numpy as np

from gnsp.constants import PHI, FIBONACCI, STDP_TAU


@lru_cache(maxsize=1000)
def fib(n: int) -> int:
    """
    Compute nth Fibonacci number (1-indexed, F(1)=F(2)=1).

    Uses memoization for efficiency.

    Args:
        n: Index (1-indexed)

    Returns:
        F(n)

    Example:
        >>> fib(10)
        55
        >>> fib(1)
        1
        >>> fib(20)
        6765
    """
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    return fib(n - 1) + fib(n - 2)


def fib_binet(n: int) -> int:
    """
    Compute nth Fibonacci number using Binet's formula.

    F(n) = (phi^n - psi^n) / sqrt(5), where psi = (1 - sqrt(5)) / 2

    Faster for large n but may have floating-point errors.

    Args:
        n: Index (1-indexed)

    Returns:
        F(n)
    """
    if n <= 0:
        return 0
    sqrt5 = np.sqrt(5)
    psi = (1 - sqrt5) / 2
    return int(round((PHI ** n - psi ** n) / sqrt5))


def fibonacci_up_to(max_val: int) -> List[int]:
    """
    Get all Fibonacci numbers up to max_val.

    Args:
        max_val: Maximum value (inclusive)

    Returns:
        List of Fibonacci numbers <= max_val

    Example:
        >>> fibonacci_up_to(100)
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    """
    result = []
    i = 1
    while True:
        f = fib(i)
        if f > max_val:
            break
        result.append(f)
        i += 1
    return result


def is_fibonacci(n: int) -> bool:
    """
    Check if n is a Fibonacci number.

    Uses the property: n is Fibonacci iff 5n^2 + 4 or 5n^2 - 4 is perfect square.

    Args:
        n: Number to check

    Returns:
        True if n is a Fibonacci number

    Example:
        >>> is_fibonacci(144)
        True
        >>> is_fibonacci(145)
        False
    """
    if n < 0:
        return False
    if n == 0:
        return True  # F(0) = 0 in extended sequence

    def is_perfect_square(x: int) -> bool:
        if x < 0:
            return False
        s = int(x ** 0.5)
        return s * s == x

    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)


def fibonacci_index(n: int) -> int:
    """
    Find index of Fibonacci number n, or -1 if not Fibonacci.

    Args:
        n: Number to find

    Returns:
        Index i such that F(i) = n, or -1 if not Fibonacci

    Example:
        >>> fibonacci_index(144)
        12
        >>> fibonacci_index(100)
        -1
    """
    if not is_fibonacci(n):
        return -1
    if n == 0:
        return 0
    if n == 1:
        return 1  # Could be 1 or 2, return 1

    i = 1
    while fib(i) < n:
        i += 1
    return i if fib(i) == n else -1


def zeckendorf(n: int) -> List[int]:
    """
    Zeckendorf representation: express n as sum of non-consecutive Fibonacci numbers.

    Every positive integer has a unique such representation.

    Args:
        n: Positive integer

    Returns:
        List of Fibonacci numbers that sum to n

    Example:
        >>> zeckendorf(100)
        [89, 8, 3]  # 100 = 89 + 8 + 3
        >>> sum(zeckendorf(100))
        100
    """
    if n <= 0:
        return []

    fibs = fibonacci_up_to(n)
    result = []
    remaining = n

    for f in reversed(fibs):
        if f <= remaining:
            result.append(f)
            remaining -= f
            if remaining == 0:
                break

    return result


def fibonacci_encode(n: int) -> List[int]:
    """
    Encode n using Fibonacci representation (Zeckendorf indices).

    Returns list of Fibonacci indices used in representation.

    Args:
        n: Positive integer

    Returns:
        List of indices into Fibonacci sequence

    Example:
        >>> fibonacci_encode(100)
        [12, 6, 4]  # F(12)=89, F(6)=8, F(4)=3, sum=100
    """
    if n <= 0:
        return []

    rep = zeckendorf(n)
    indices = []
    for f in rep:
        idx = fibonacci_index(f)
        if idx > 0:
            indices.append(idx)
    return indices


def fibonacci_offsets(max_offset: int) -> List[int]:
    """
    Generate symmetric Fibonacci offsets for neighborhood definition.

    Returns: [-F_k, ..., -F_2, -F_1, 0, F_1, F_2, ..., F_k]

    Args:
        max_offset: Maximum offset magnitude

    Returns:
        Sorted list of symmetric Fibonacci offsets including 0

    Example:
        >>> fibonacci_offsets(10)
        [-8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8]
    """
    fibs = fibonacci_up_to(max_offset)
    offsets = set([0])
    for f in fibs:
        offsets.add(f)
        offsets.add(-f)
    return sorted(offsets)


def fibonacci_time_constants(n: int = 8) -> np.ndarray:
    """
    Generate Fibonacci-based time constants for STDP.

    Returns first n Fibonacci numbers as time constant values.

    Args:
        n: Number of time constants

    Returns:
        Array of Fibonacci time constants

    Example:
        >>> fibonacci_time_constants(8)
        array([ 1.,  1.,  2.,  3.,  5.,  8., 13., 21.], dtype=float32)
    """
    return np.array([fib(i) for i in range(1, n + 1)], dtype=np.float32)


def fibonacci_connectivity_pattern(n_neurons: int) -> Set[Tuple[int, int]]:
    """
    Generate connectivity pattern using Fibonacci distances.

    Neuron i connects to neurons at Fibonacci distances (circular).

    Args:
        n_neurons: Number of neurons

    Returns:
        Set of (source, target) connection tuples

    Example:
        >>> conns = fibonacci_connectivity_pattern(10)
        >>> (0, 1) in conns  # Distance 1 is Fibonacci
        True
        >>> (0, 4) in conns  # Distance 4 is not Fibonacci
        False
    """
    connections: Set[Tuple[int, int]] = set()
    fibs = fibonacci_up_to(n_neurons)

    for i in range(n_neurons):
        for f in fibs:
            # Forward connection
            j = (i + f) % n_neurons
            if i != j:
                connections.add((i, j))
            # Backward connection
            j = (i - f) % n_neurons
            if i != j:
                connections.add((i, j))

    return connections


def fibonacci_decay_lookup(max_dt: int = 64, n_tau: int = 8) -> np.ndarray:
    """
    Precompute decay values for Fibonacci STDP.

    Returns 2D array: decay[dt, tau_idx] = phi^(-dt/tau[tau_idx])

    Used in hardware lookup tables for efficient STDP computation.

    Args:
        max_dt: Maximum time difference
        n_tau: Number of time constants (from Fibonacci sequence)

    Returns:
        2D array of shape (max_dt + 1, n_tau)

    Example:
        >>> lut = fibonacci_decay_lookup(10, 4)
        >>> lut.shape
        (11, 4)
        >>> lut[0, 0]  # dt=0, tau=1: phi^0 = 1
        1.0
    """
    taus = fibonacci_time_constants(n_tau)
    decay = np.zeros((max_dt + 1, n_tau), dtype=np.float32)

    for dt in range(max_dt + 1):
        for tau_idx, tau in enumerate(taus):
            if tau > 0:
                decay[dt, tau_idx] = PHI ** (-dt / tau)

    return decay


def fibonacci_distance_matrix(n: int) -> np.ndarray:
    """
    Create distance matrix based on Fibonacci sequence.

    Entry (i, j) = min Fibonacci number >= |i - j|.

    Args:
        n: Matrix size

    Returns:
        n x n symmetric distance matrix

    Example:
        >>> D = fibonacci_distance_matrix(5)
        >>> D[0, 3]  # Distance 3 is Fibonacci
        3
        >>> D[0, 4]  # Distance 4 rounds up to 5
        5
    """
    fibs = [1] + fibonacci_up_to(n * 2)
    D = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            dist = abs(i - j)
            if dist == 0:
                D[i, j] = 0
            else:
                # Find smallest Fibonacci >= dist
                for f in fibs:
                    if f >= dist:
                        D[i, j] = f
                        break

    return D


def fibonacci_lattice_1d(n: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate 1D Fibonacci lattice positions.

    Points placed at scaled Fibonacci positions.

    Args:
        n: Number of points
        scale: Scaling factor

    Returns:
        1D array of positions
    """
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        positions[i] = scale * fib(i + 1)
    return positions


def tribonacci(n: int) -> int:
    """
    Compute nth Tribonacci number.

    T(1)=T(2)=1, T(3)=2, T(n) = T(n-1) + T(n-2) + T(n-3)

    Tribonacci constant converges to approximately 1.8393.

    Args:
        n: Index (1-indexed)

    Returns:
        T(n)
    """
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    if n == 3:
        return 2

    a, b, c = 1, 1, 2
    for _ in range(n - 3):
        a, b, c = b, c, a + b + c
    return c


def fibonacci_ratios(n: int) -> np.ndarray:
    """
    Compute ratios F(k+1)/F(k) for k = 1 to n.

    These ratios converge to phi.

    Args:
        n: Number of ratios

    Returns:
        Array of Fibonacci ratios

    Example:
        >>> ratios = fibonacci_ratios(10)
        >>> abs(ratios[-1] - PHI) < 0.001
        True
    """
    ratios = np.zeros(n, dtype=np.float64)
    for k in range(1, n + 1):
        ratios[k - 1] = fib(k + 1) / fib(k)
    return ratios


def generalized_fibonacci(a: int, b: int, n: int) -> List[int]:
    """
    Generate generalized Fibonacci sequence starting with a, b.

    G(1)=a, G(2)=b, G(n) = G(n-1) + G(n-2)

    Args:
        a: First term
        b: Second term
        n: Number of terms

    Returns:
        List of generalized Fibonacci numbers

    Example:
        >>> generalized_fibonacci(2, 1, 10)
        [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]
    """
    if n <= 0:
        return []
    if n == 1:
        return [a]
    if n == 2:
        return [a, b]

    seq = [a, b]
    for _ in range(n - 2):
        seq.append(seq[-1] + seq[-2])
    return seq
