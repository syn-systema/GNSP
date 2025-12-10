"""
Golden ratio utilities and mathematical functions.

Based on properties from quantum gravity research:
- phi is the dominant non-integer eigenvalue of binary matrices
- phi-based systems exhibit maximum stability (KAM theorem)
- phi appears in E8 lattice eigenvalues

The golden ratio phi = (1 + sqrt(5)) / 2 satisfies:
- phi^2 = phi + 1
- phi * (1/phi) = 1
- 1/phi = phi - 1
- phi = 1 + 1/phi (continued fraction)

Example:
    >>> from gnsp.core.golden import golden_decay, golden_spiral_points
    >>> decay = golden_decay(1.0, steps=3)
    >>> print(decay)  # ~0.236
    >>> points = golden_spiral_points(100)
    >>> print(points.shape)  # (100, 2)
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple
import numpy as np

from gnsp.constants import PHI, PHI_INV, PHI_SQ, GOLDEN_ANGLE


def golden_decay(value: float, steps: int = 1) -> float:
    """
    Apply golden ratio decay.

    v(t+n) = v(t) * phi^(-n) = v(t) * (0.618...)^n

    This is the natural decay rate for maximum stability per KAM theorem.

    Args:
        value: Starting value
        steps: Number of decay steps

    Returns:
        Decayed value

    Example:
        >>> golden_decay(1.0, 1)
        0.6180339887498949
        >>> golden_decay(1.0, 2)
        0.3819660112501051
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return value * (PHI_INV ** steps)


def golden_growth(value: float, steps: int = 1) -> float:
    """
    Apply golden ratio growth (inverse of decay).

    v(t+n) = v(t) * phi^n

    Args:
        value: Starting value
        steps: Number of growth steps

    Returns:
        Grown value
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return value * (PHI ** steps)


def golden_threshold(base: float, level: int) -> float:
    """
    Compute threshold at given golden level.

    level 0: base
    level 1: base * phi
    level -1: base * phi^(-1)
    level n: base * phi^n

    Args:
        base: Base threshold value
        level: Power of phi to apply

    Returns:
        Threshold at specified level

    Example:
        >>> golden_threshold(1.0, 1)
        1.618033988749895
        >>> golden_threshold(1.0, -1)
        0.6180339887498949
    """
    return base * (PHI ** level)


def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate first n Fibonacci numbers.

    F(1) = F(2) = 1, F(n) = F(n-1) + F(n-2)

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of first n Fibonacci numbers

    Example:
        >>> fibonacci_sequence(10)
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]

    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def fibonacci_generator() -> Iterator[int]:
    """
    Infinite Fibonacci sequence generator.

    Yields:
        Successive Fibonacci numbers

    Example:
        >>> gen = fibonacci_generator()
        >>> [next(gen) for _ in range(10)]
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


def nearest_fibonacci(n: int, max_fib: int = 20) -> int:
    """
    Find nearest Fibonacci number to n.

    Args:
        n: Target number
        max_fib: Maximum number of Fibonacci numbers to check

    Returns:
        Nearest Fibonacci number

    Example:
        >>> nearest_fibonacci(10)
        8
        >>> nearest_fibonacci(7)
        8
    """
    if n <= 0:
        return 1
    fibs = fibonacci_sequence(max_fib)
    return min(fibs, key=lambda x: abs(x - n))


def golden_spiral_points(n: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate points on a golden spiral (Fermat/Fibonacci spiral).

    Used for quasicrystalline neuron placement. Points are distributed
    with angular separation of the golden angle, creating optimal
    packing without clustering.

    Args:
        n: Number of points
        scale: Scaling factor for radius

    Returns:
        Array of shape (n, 2) with (x, y) coordinates

    Example:
        >>> points = golden_spiral_points(100)
        >>> points.shape
        (100, 2)
        >>> np.allclose(points[0], [0, 0])
        True
    """
    points = np.zeros((n, 2), dtype=np.float64)

    for i in range(n):
        theta = i * GOLDEN_ANGLE
        r = scale * np.sqrt(i)
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)

    return points


def golden_spiral_points_3d(
    n: int,
    scale: float = 1.0,
    use_spherical: bool = False
) -> np.ndarray:
    """
    Generate points on a 3D golden spiral.

    If use_spherical=True, distributes points on a sphere using
    the Fibonacci lattice (optimal spherical coverage).

    Args:
        n: Number of points
        scale: Scaling factor
        use_spherical: If True, distribute on unit sphere

    Returns:
        Array of shape (n, 3) with (x, y, z) coordinates
    """
    points = np.zeros((n, 3), dtype=np.float64)

    if use_spherical:
        # Fibonacci lattice on sphere
        for i in range(n):
            theta = 2 * np.pi * i / PHI
            phi_angle = np.arccos(1 - 2 * (i + 0.5) / n)
            points[i, 0] = scale * np.sin(phi_angle) * np.cos(theta)
            points[i, 1] = scale * np.sin(phi_angle) * np.sin(theta)
            points[i, 2] = scale * np.cos(phi_angle)
    else:
        # Helix with golden angle
        for i in range(n):
            theta = i * GOLDEN_ANGLE
            r = scale * np.sqrt(i)
            z = scale * i / n
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
            points[i, 2] = z

    return points


def golden_ratio_weights(n: int, normalize: bool = True) -> np.ndarray:
    """
    Generate weights based on golden ratio powers.

    w[i] = phi^(-i) for i = 0, 1, ..., n-1

    Used for distance-weighted averaging in topology and attention.

    Args:
        n: Number of weights
        normalize: If True, normalize to sum to 1

    Returns:
        Array of weights

    Example:
        >>> weights = golden_ratio_weights(4, normalize=False)
        >>> weights
        array([1.        , 0.61803399, 0.38196601, 0.23606798])
    """
    weights = np.array([PHI_INV ** i for i in range(n)], dtype=np.float64)
    if normalize:
        weights /= weights.sum()
    return weights


def is_near_golden(ratio: float, tolerance: float = 0.01) -> bool:
    """
    Check if a ratio is close to phi or its powers.

    Checks powers from phi^(-3) to phi^3.

    Args:
        ratio: Value to check
        tolerance: Maximum difference from golden power

    Returns:
        True if ratio is near a golden power

    Example:
        >>> is_near_golden(1.618)
        True
        >>> is_near_golden(1.5)
        False
        >>> is_near_golden(2.618)
        True
    """
    for power in range(-3, 4):
        target = PHI ** power
        if abs(ratio - target) < tolerance:
            return True
        # Also check negative
        if abs(ratio + target) < tolerance:
            return True
    return False


def golden_power_decomposition(value: float, max_power: int = 5) -> List[Tuple[int, float]]:
    """
    Decompose a value into golden ratio powers.

    Similar to Zeckendorf representation but for real numbers.

    Args:
        value: Value to decompose
        max_power: Maximum power to consider

    Returns:
        List of (power, coefficient) tuples
    """
    remaining = abs(value)
    sign = 1 if value >= 0 else -1
    components = []

    for power in range(max_power, -max_power - 1, -1):
        phi_power = PHI ** power
        if phi_power <= remaining * 1.001:  # Allow small tolerance
            coeff = int(remaining / phi_power)
            if coeff > 0:
                components.append((power, sign * coeff))
                remaining -= coeff * phi_power
        if remaining < 1e-10:
            break

    return components


def golden_matrix(n: int = 2) -> np.ndarray:
    """
    Create the fundamental golden matrix.

    For n=2:
        [[0, 1],
         [1, 1]]

    This matrix has eigenvalues phi and -1/phi.
    Larger matrices are built via Kronecker products.

    Args:
        n: Size of matrix (will be extended to next power of 2)

    Returns:
        Square matrix with golden ratio eigenvalues

    Example:
        >>> M = golden_matrix(2)
        >>> eigenvalues = np.linalg.eigvals(M)
        >>> sorted(eigenvalues, reverse=True)
        [1.618..., -0.618...]
    """
    base = np.array([[0, 1], [1, 1]], dtype=np.float64)

    if n <= 2:
        return base

    # Build larger matrices via Kronecker product
    result = base
    while result.shape[0] < n:
        result = np.kron(result, base)

    return result[:n, :n]


def lucas_sequence(n: int) -> List[int]:
    """
    Generate first n Lucas numbers.

    L(1) = 1, L(2) = 3, L(n) = L(n-1) + L(n-2)

    Lucas numbers are related to Fibonacci: L(n) = F(n-1) + F(n+1)
    And satisfy: phi^n = F(n)*phi + F(n-1)

    Args:
        n: Number of Lucas numbers to generate

    Returns:
        List of first n Lucas numbers
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 3]

    lucas = [1, 3]
    while len(lucas) < n:
        lucas.append(lucas[-1] + lucas[-2])
    return lucas


def verify_golden_eigenvalues(matrix: np.ndarray) -> Dict:
    """
    Analyze eigenvalues of a matrix for golden ratio presence.

    Returns dict with golden-related eigenvalues found.

    Args:
        matrix: Square matrix to analyze

    Returns:
        Dictionary with:
        - 'eigenvalues': All eigenvalues
        - 'golden_eigenvalues': Eigenvalues near golden powers
        - 'phi_powers_found': Which powers of phi were found

    Example:
        >>> M = golden_matrix(2)
        >>> result = verify_golden_eigenvalues(M)
        >>> 1 in result['phi_powers_found']
        True
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    eigenvalues = np.linalg.eigvals(matrix)

    results: Dict = {
        'eigenvalues': eigenvalues.tolist(),
        'golden_eigenvalues': [],
        'phi_powers_found': []
    }

    for ev in eigenvalues:
        if np.isreal(ev):
            ev_real = float(np.real(ev))
            for power in range(-3, 4):
                target = PHI ** power
                if abs(ev_real - target) < 0.001:
                    results['golden_eigenvalues'].append(ev_real)
                    results['phi_powers_found'].append(power)
                    break
                elif abs(ev_real + target) < 0.001:
                    results['golden_eigenvalues'].append(ev_real)
                    results['phi_powers_found'].append(-power)
                    break

    return results


def phi_continued_fraction(n: int) -> float:
    """
    Compute phi using n levels of continued fraction.

    phi = 1 + 1/(1 + 1/(1 + 1/(...)))

    Args:
        n: Number of levels

    Returns:
        Approximation to phi

    Example:
        >>> phi_continued_fraction(10)
        1.6181818181818182
        >>> phi_continued_fraction(50)
        1.6180339887498947
    """
    if n <= 0:
        return 1.0
    result = 1.0
    for _ in range(n):
        result = 1.0 + 1.0 / result
    return result


def golden_interpolate(a: float, b: float, t: float) -> float:
    """
    Interpolate between a and b using golden ratio.

    At t=0: returns a
    At t=1: returns b
    At t=PHI_INV (~0.618): returns golden section point

    Args:
        a: Start value
        b: End value
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated value
    """
    return a + (b - a) * t


def golden_section_search(
    f,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, float]:
    """
    Find minimum of unimodal function using golden section search.

    Uses the golden ratio to optimally narrow the search interval.

    Args:
        f: Function to minimize
        a: Left endpoint
        b: Right endpoint
        tol: Tolerance for convergence
        max_iter: Maximum iterations

    Returns:
        Tuple of (x_min, f_min)

    Example:
        >>> f = lambda x: (x - 1.5)**2
        >>> x_min, f_min = golden_section_search(f, 0, 3)
        >>> abs(x_min - 1.5) < 1e-5
        True
    """
    c = b - (b - a) * PHI_INV
    d = a + (b - a) * PHI_INV

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if f(c) < f(d):
            b = d
            d = c
            c = b - (b - a) * PHI_INV
        else:
            a = c
            c = d
            d = a + (b - a) * PHI_INV

    x_min = (a + b) / 2
    return x_min, f(x_min)
