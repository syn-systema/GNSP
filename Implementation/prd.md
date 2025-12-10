# GOLDEN NEUROMORPHIC SECURITY PLATFORM (GNSP) - SOFTWARE SIMULATION

## PROJECT CODENAME: P-001-GNSP

---

## EXECUTIVE SUMMARY

Build a software simulation platform for a neuromorphic intrusion detection system that integrates:
- Spiking Neural Networks (SNNs) with golden ratio-based architecture
- Automata-theoretic protocol analysis (DFA/NFA/Büchi/Weighted/Timed)
- Category-theoretic anomaly detection (functors, sheaves)
- Topological data analysis (persistent homology)
- Clifford/Geometric algebra state representations

The goal is to validate algorithms before hardware deployment on FPGA. All code should be structured for eventual hardware synthesis (clean interfaces, fixed-point compatible, parallel-friendly).

---

## CORE PRINCIPLES

### Mathematical Foundation
- Golden ratio (φ = 1.618033988749895) as fundamental constant throughout
- Fibonacci sequences for timing, connectivity, and learning
- Category theory for compositional security reasoning
- Automata theory for O(n) protocol verification
- Topology for robust feature extraction

### Engineering Principles
- Hardware-aware design (think about eventual FPGA implementation)
- Fixed-point arithmetic compatibility (use Q8.8 or Q16.16 representations)
- Explicit parallelism (no hidden sequential dependencies)
- Comprehensive testing at every level
- Documentation as first-class citizen

### Code Style
- Python 3.11+ with full type hints
- Dataclasses for structured data
- Abstract base classes for extensibility
- NumPy for numerical operations (GPU-ready with CuPy later)
- No emojis in code, comments, or documentation
- Docstrings in Google style

---

## PROJECT STRUCTURE

```
gnsp/
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
│
├── gnsp/
│   ├── __init__.py
│   ├── constants.py                 # Golden ratio, Fibonacci, fixed-point utils
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── fixed_point.py           # Q-format fixed-point arithmetic
│   │   ├── golden.py                # Golden ratio utilities
│   │   ├── fibonacci.py             # Fibonacci sequence generators
│   │   └── quasicrystal.py          # Quasicrystalline lattice generation
│   │
│   ├── snn/
│   │   ├── __init__.py
│   │   ├── neuron.py                # LIF neuron models (golden decay)
│   │   ├── synapse.py               # Synaptic models with golden weights
│   │   ├── stdp.py                  # Fibonacci STDP learning rule
│   │   ├── network.py               # SNN network container
│   │   ├── topology.py              # Quasicrystal network topology
│   │   ├── encoder.py               # Spike encoding (rate, temporal, delta)
│   │   ├── decoder.py               # Spike decoding and classification
│   │   └── simulator.py             # Event-driven SNN simulator
│   │
│   ├── automata/
│   │   ├── __init__.py
│   │   ├── dfa.py                   # Deterministic finite automata
│   │   ├── nfa.py                   # Non-deterministic finite automata
│   │   ├── weighted.py              # Weighted automata (semirings)
│   │   ├── buchi.py                 # Büchi automata (infinite words)
│   │   ├── timed.py                 # Timed automata (clocks, guards)
│   │   ├── cellular.py              # Cellular automata (SNN connection)
│   │   ├── krohn_rhodes.py          # Krohn-Rhodes decomposition
│   │   ├── monoid.py                # Syntactic monoid computation
│   │   ├── ltl.py                   # LTL formulas and compilation
│   │   └── learning.py              # L* algorithm for automaton inference
│   │
│   ├── category/
│   │   ├── __init__.py
│   │   ├── category.py              # Category, Functor, Natural Transform
│   │   ├── protocol.py              # Protocol categories (TCP, HTTP, DNS)
│   │   ├── traffic.py               # Traffic observation category
│   │   ├── functor.py               # Security functor F: Protocol -> Traffic
│   │   ├── sheaf.py                 # Sheaf-theoretic distributed detection
│   │   ├── cohomology.py            # Čech cohomology computation
│   │   └── topos.py                 # Topos-theoretic foundations (advanced)
│   │
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── simplicial.py            # Simplicial complex construction
│   │   ├── vietoris_rips.py         # Vietoris-Rips complex builder
│   │   ├── boundary.py              # Boundary matrix computation
│   │   ├── reduction.py             # Matrix reduction (persistence)
│   │   ├── persistence.py           # Persistent homology engine
│   │   ├── betti.py                 # Betti number computation
│   │   └── features.py              # TDA feature extraction for ML
│   │
│   ├── algebra/
│   │   ├── __init__.py
│   │   ├── clifford.py              # Clifford algebra Cl(3,0)
│   │   ├── multivector.py           # Multivector representation
│   │   ├── geometric_product.py     # Geometric product implementation
│   │   ├── rotor.py                 # Rotor operations (rotations)
│   │   └── spinor.py                # Spinor representations
│   │
│   ├── network/
│   │   ├── __init__.py
│   │   ├── packet.py                # Packet data structures
│   │   ├── flow.py                  # Flow aggregation
│   │   ├── parser.py                # Packet parsing (headers, features)
│   │   ├── pcap.py                  # PCAP file reading
│   │   └── generator.py             # Synthetic traffic generation
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py              # Main detector orchestration
│   │   ├── automata_detector.py     # Automata-based detection
│   │   ├── category_detector.py     # Category-theoretic detection
│   │   ├── tda_detector.py          # Topological detection
│   │   ├── snn_detector.py          # SNN-based detection
│   │   ├── ensemble.py              # Ensemble combination
│   │   └── explainer.py             # Anomaly explanation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py               # Dataset loading (NSL-KDD, UNSW-NB15, CICIDS)
│   │   ├── preprocessor.py          # Feature preprocessing
│   │   ├── surrogate.py             # Surrogate gradient training
│   │   ├── evolutionary.py          # Evolutionary optimization
│   │   ├── online.py                # Online learning (STDP-based)
│   │   └── evaluation.py            # Metrics, ROC, confusion matrix
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── snn_viz.py               # SNN activity visualization
│       ├── automata_viz.py          # Automata state diagrams
│       ├── persistence_viz.py       # Persistence diagrams/barcodes
│       ├── topology_viz.py          # Network topology visualization
│       └── dashboard.py             # Real-time monitoring dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_core/
│   ├── test_snn/
│   ├── test_automata/
│   ├── test_category/
│   ├── test_topology/
│   ├── test_algebra/
│   └── test_integration/
│
├── notebooks/
│   ├── 01_golden_ratio_exploration.ipynb
│   ├── 02_snn_basics.ipynb
│   ├── 03_automata_protocols.ipynb
│   ├── 04_category_theory_ids.ipynb
│   ├── 05_persistent_homology.ipynb
│   ├── 06_clifford_algebra.ipynb
│   ├── 07_full_pipeline.ipynb
│   └── 08_golden_hypothesis_testing.ipynb
│
├── configs/
│   ├── default.yaml
│   ├── snn_config.yaml
│   ├── training_config.yaml
│   └── detection_config.yaml
│
├── data/
│   ├── raw/                         # Raw datasets (gitignored)
│   ├── processed/                   # Processed features
│   └── synthetic/                   # Generated test data
│
├── models/
│   ├── checkpoints/                 # Training checkpoints
│   └── trained/                     # Final trained models
│
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   ├── train_snn.py
│   ├── evaluate_model.py
│   ├── run_detection.py
│   └── benchmark.py
│
└── docs/
    ├── architecture.md
    ├── mathematical_foundations.md
    ├── api_reference.md
    └── hardware_mapping.md
```

---

## PHASE 1: CORE FOUNDATIONS (Week 1-2)

### 1.1 Constants and Utilities (`gnsp/constants.py`, `gnsp/core/`)

```python
# gnsp/constants.py

"""
Core mathematical constants for GNSP.

The golden ratio appears throughout as the fundamental constant,
following insights from quantum gravity research showing phi
as the dominant non-integer eigenvalue of binary matrices.
"""

import math
from typing import List, Tuple

# Golden Ratio and Related Constants
PHI: float = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INV: float = 1 / PHI              # 0.618033988749895
PHI_SQ: float = PHI ** 2              # 2.618033988749895
PHI_INV_SQ: float = PHI_INV ** 2      # 0.381966011250105

# Fixed-point representations (Q8.8 format: 8 integer bits, 8 fractional bits)
PHI_Q8_8: int = 0x019E        # 1.617 in Q8.8
PHI_INV_Q8_8: int = 0x009E    # 0.617 in Q8.8
PHI_SQ_Q8_8: int = 0x029E     # 2.617 in Q8.8

# Fibonacci sequence (pre-computed for efficiency)
FIBONACCI: Tuple[int, ...] = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610)

# Golden angle (radians) - used for quasicrystal generation
GOLDEN_ANGLE: float = 2 * math.pi / (PHI ** 2)  # ~137.5 degrees

# Cabibbo angle from E8 lattice (arctan(1/phi^3))
CABIBBO_ANGLE: float = math.atan(1 / (PHI ** 3))  # ~13.28 degrees

# Detection thresholds based on golden ratio
THRESHOLD_LOW: float = PHI_INV ** 3     # ~0.236
THRESHOLD_MID: float = PHI_INV ** 2     # ~0.382
THRESHOLD_HIGH: float = PHI_INV         # ~0.618
THRESHOLD_CRITICAL: float = 1.0

# SNN parameters
DEFAULT_THRESHOLD: float = PHI           # Spike threshold
DEFAULT_RESET: float = PHI_INV           # Reset potential
DEFAULT_LEAK: float = PHI_INV ** 2       # Leak rate

# Weight quantization levels (golden ratio ladder)
WEIGHT_LEVELS: Tuple[float, ...] = (
    -PHI_SQ,      # -2.618
    -PHI,         # -1.618
    -1.0,
    -PHI_INV,     # -0.618
    0.0,
    PHI_INV,      # 0.618
    1.0,
    PHI,          # 1.618
    PHI_SQ,       # 2.618
)

# STDP time constants (Fibonacci in milliseconds)
STDP_TAU: Tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34)
```

Implement `gnsp/core/fixed_point.py`:

```python
"""
Fixed-point arithmetic for hardware compatibility.

Uses Q-format notation: Qm.n means m integer bits, n fractional bits.
Default is Q8.8 (16-bit total) but configurable.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class QFormat:
    """Q-format specification."""
    integer_bits: int
    fractional_bits: int
    
    @property
    def total_bits(self) -> int:
        return self.integer_bits + self.fractional_bits
    
    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits
    
    @property
    def max_value(self) -> float:
        return (1 << (self.integer_bits - 1)) - (1 / self.scale)
    
    @property
    def min_value(self) -> float:
        return -(1 << (self.integer_bits - 1))


Q8_8 = QFormat(8, 8)
Q16_16 = QFormat(16, 16)
Q1_15 = QFormat(1, 15)  # For normalized values [-1, 1)


class FixedPoint:
    """
    Fixed-point number representation.
    
    Designed for eventual hardware synthesis - all operations
    map directly to integer arithmetic.
    """
    
    def __init__(self, value: Union[float, int], fmt: QFormat = Q8_8):
        self.fmt = fmt
        if isinstance(value, float):
            self._raw = int(round(value * fmt.scale))
        else:
            self._raw = value
        
        # Clamp to valid range
        max_raw = (1 << (fmt.total_bits - 1)) - 1
        min_raw = -(1 << (fmt.total_bits - 1))
        self._raw = max(min_raw, min(max_raw, self._raw))
    
    @property
    def value(self) -> float:
        """Convert to floating point."""
        return self._raw / self.fmt.scale
    
    @property
    def raw(self) -> int:
        """Get raw integer representation."""
        return self._raw
    
    def __add__(self, other: 'FixedPoint') -> 'FixedPoint':
        assert self.fmt == other.fmt
        return FixedPoint(self._raw + other._raw, self.fmt)
    
    def __sub__(self, other: 'FixedPoint') -> 'FixedPoint':
        assert self.fmt == other.fmt
        return FixedPoint(self._raw - other._raw, self.fmt)
    
    def __mul__(self, other: 'FixedPoint') -> 'FixedPoint':
        assert self.fmt == other.fmt
        # Full precision multiply, then shift back
        result = (self._raw * other._raw) >> self.fmt.fractional_bits
        return FixedPoint(result, self.fmt)
    
    def __repr__(self) -> str:
        return f"FixedPoint({self.value:.4f}, Q{self.fmt.integer_bits}.{self.fmt.fractional_bits})"


def float_to_fixed(arr: np.ndarray, fmt: QFormat = Q8_8) -> np.ndarray:
    """Convert float array to fixed-point integers."""
    return np.clip(
        np.round(arr * fmt.scale).astype(np.int32),
        -(1 << (fmt.total_bits - 1)),
        (1 << (fmt.total_bits - 1)) - 1
    )


def fixed_to_float(arr: np.ndarray, fmt: QFormat = Q8_8) -> np.ndarray:
    """Convert fixed-point integers to float array."""
    return arr.astype(np.float32) / fmt.scale
```

Implement `gnsp/core/golden.py`:

```python
"""
Golden ratio utilities and mathematical functions.

Based on properties from quantum gravity research:
- phi is the dominant non-integer eigenvalue of binary matrices
- phi-based systems exhibit maximum stability (KAM theorem)
- phi appears in E8 lattice eigenvalues
"""

import math
import numpy as np
from typing import List, Tuple, Iterator
from gnsp.constants import PHI, PHI_INV, FIBONACCI


def golden_decay(value: float, steps: int = 1) -> float:
    """
    Apply golden ratio decay.
    
    v(t+1) = v(t) * phi^(-1) = v(t) * 0.618...
    
    This is the natural decay rate for maximum stability.
    """
    return value * (PHI_INV ** steps)


def golden_threshold(base: float, level: int) -> float:
    """
    Compute threshold at given golden level.
    
    level 0: base
    level 1: base * phi
    level -1: base * phi^(-1)
    """
    return base * (PHI ** level)


def fibonacci_sequence(n: int) -> List[int]:
    """Generate first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def fibonacci_generator() -> Iterator[int]:
    """Infinite Fibonacci sequence generator."""
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


def nearest_fibonacci(n: int) -> int:
    """Find nearest Fibonacci number to n."""
    fibs = fibonacci_sequence(20)  # Sufficient for most uses
    return min(fibs, key=lambda x: abs(x - n))


def golden_spiral_points(n: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate points on a golden spiral (Fibonacci spiral).
    
    Used for quasicrystalline neuron placement.
    
    Args:
        n: Number of points
        scale: Scaling factor
        
    Returns:
        Array of shape (n, 2) with (x, y) coordinates
    """
    points = np.zeros((n, 2))
    golden_angle = 2 * np.pi / (PHI ** 2)
    
    for i in range(n):
        theta = i * golden_angle
        r = scale * np.sqrt(i)
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
    
    return points


def golden_ratio_weights(n: int, normalize: bool = True) -> np.ndarray:
    """
    Generate weights based on golden ratio powers.
    
    w[i] = phi^(-i) for i = 0, 1, ..., n-1
    
    Used for distance-weighted averaging in topology.
    """
    weights = np.array([PHI_INV ** i for i in range(n)])
    if normalize:
        weights /= weights.sum()
    return weights


def is_near_golden(ratio: float, tolerance: float = 0.01) -> bool:
    """Check if a ratio is close to phi or its powers."""
    for power in range(-3, 4):
        if abs(ratio - PHI ** power) < tolerance:
            return True
    return False


def golden_matrix(n: int = 2) -> np.ndarray:
    """
    Create the fundamental golden matrix.
    
    [[0, 1],
     [1, 1]]
     
    This matrix has eigenvalues phi and -1/phi.
    Extended to larger sizes via tensor products.
    """
    base = np.array([[0, 1], [1, 1]], dtype=np.float64)
    
    if n == 2:
        return base
    
    # Build larger matrices via Kronecker product
    result = base
    while result.shape[0] < n:
        result = np.kron(result, base)
    
    return result[:n, :n]


def verify_golden_eigenvalues(matrix: np.ndarray) -> dict:
    """
    Analyze eigenvalues of a matrix for golden ratio presence.
    
    Returns dict with golden-related eigenvalues found.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    
    results = {
        'eigenvalues': eigenvalues.tolist(),
        'golden_eigenvalues': [],
        'phi_powers_found': []
    }
    
    for ev in eigenvalues:
        if np.isreal(ev):
            ev_real = float(np.real(ev))
            for power in range(-3, 4):
                target = PHI ** power
                if abs(ev_real - target) < 0.001 or abs(ev_real + target) < 0.001:
                    results['golden_eigenvalues'].append(ev_real)
                    results['phi_powers_found'].append(power)
    
    return results
```

Implement `gnsp/core/fibonacci.py`:

```python
"""
Fibonacci sequence utilities for timing and connectivity.

Fibonacci numbers appear naturally in:
- STDP time constants
- Network connectivity distances
- Hierarchical routing levels
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from functools import lru_cache
from gnsp.constants import FIBONACCI, PHI


@lru_cache(maxsize=100)
def fib(n: int) -> int:
    """Compute nth Fibonacci number (1-indexed, F(1)=F(2)=1)."""
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    return fib(n - 1) + fib(n - 2)


def fibonacci_up_to(max_val: int) -> List[int]:
    """Get all Fibonacci numbers up to max_val."""
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
    """Check if n is a Fibonacci number."""
    # n is Fibonacci iff 5n^2 + 4 or 5n^2 - 4 is perfect square
    def is_perfect_square(x: int) -> bool:
        s = int(x ** 0.5)
        return s * s == x
    
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)


def fibonacci_encode(n: int) -> List[int]:
    """
    Zeckendorf representation: express n as sum of non-consecutive Fibonacci numbers.
    
    Returns list of Fibonacci indices used.
    """
    if n == 0:
        return []
    
    # Find largest Fibonacci <= n
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


def fibonacci_offsets(max_offset: int) -> List[int]:
    """
    Generate symmetric Fibonacci offsets for neighborhood definition.
    
    Returns: [-F_k, ..., -F_2, -F_1, 0, F_1, F_2, ..., F_k]
    """
    fibs = fibonacci_up_to(max_offset)
    return sorted([-f for f in fibs] + [0] + fibs)


def fibonacci_time_constants(n: int = 8) -> np.ndarray:
    """
    Generate Fibonacci-based time constants for STDP.
    
    Returns first n Fibonacci numbers as millisecond values.
    """
    return np.array([fib(i) for i in range(1, n + 1)], dtype=np.float32)


def fibonacci_connectivity_pattern(n_neurons: int) -> Set[Tuple[int, int]]:
    """
    Generate connectivity pattern using Fibonacci distances.
    
    Neuron i connects to neurons at Fibonacci distances.
    """
    connections = set()
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
    
    Used in hardware lookup tables.
    """
    taus = fibonacci_time_constants(n_tau)
    decay = np.zeros((max_dt + 1, n_tau), dtype=np.float32)
    
    for dt in range(max_dt + 1):
        for tau_idx, tau in enumerate(taus):
            decay[dt, tau_idx] = PHI ** (-dt / tau)
    
    return decay
```

### 1.2 Tests for Core Module

Create comprehensive tests:

```python
# tests/test_core/test_golden.py

"""Tests for golden ratio utilities."""

import pytest
import numpy as np
from gnsp.constants import PHI, PHI_INV, PHI_SQ
from gnsp.core.golden import (
    golden_decay,
    golden_threshold,
    fibonacci_sequence,
    golden_spiral_points,
    golden_matrix,
    verify_golden_eigenvalues,
    is_near_golden,
)


class TestGoldenDecay:
    def test_single_step(self):
        result = golden_decay(1.0, steps=1)
        assert abs(result - PHI_INV) < 1e-10
    
    def test_double_step(self):
        result = golden_decay(1.0, steps=2)
        assert abs(result - PHI_INV ** 2) < 1e-10
    
    def test_preserves_zero(self):
        assert golden_decay(0.0) == 0.0


class TestFibonacciSequence:
    def test_first_ten(self):
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        assert fibonacci_sequence(10) == expected
    
    def test_ratio_converges_to_phi(self):
        fibs = fibonacci_sequence(20)
        ratios = [fibs[i+1] / fibs[i] for i in range(len(fibs) - 1)]
        assert abs(ratios[-1] - PHI) < 1e-6


class TestGoldenMatrix:
    def test_2x2_eigenvalues(self):
        matrix = golden_matrix(2)
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues = sorted(eigenvalues, reverse=True)
        
        assert abs(eigenvalues[0] - PHI) < 1e-10
        assert abs(eigenvalues[1] + PHI_INV) < 1e-10
    
    def test_verify_golden_eigenvalues(self):
        matrix = golden_matrix(2)
        result = verify_golden_eigenvalues(matrix)
        
        assert len(result['golden_eigenvalues']) == 2
        assert 1 in result['phi_powers_found']
        assert -1 in result['phi_powers_found']


class TestGoldenSpiral:
    def test_point_count(self):
        points = golden_spiral_points(100)
        assert points.shape == (100, 2)
    
    def test_origin_is_first(self):
        points = golden_spiral_points(10)
        assert np.allclose(points[0], [0, 0])
    
    def test_spiral_grows(self):
        points = golden_spiral_points(100)
        distances = np.linalg.norm(points, axis=1)
        # Should generally increase (with some variation due to spiral)
        assert distances[-1] > distances[10]


class TestIsNearGolden:
    def test_phi_itself(self):
        assert is_near_golden(PHI)
    
    def test_phi_inv(self):
        assert is_near_golden(PHI_INV)
    
    def test_phi_squared(self):
        assert is_near_golden(PHI_SQ)
    
    def test_arbitrary_number(self):
        assert not is_near_golden(1.5)
```

---

## PHASE 2: SPIKING NEURAL NETWORK MODULE (Week 3-5)

### 2.1 Neuron Model (`gnsp/snn/neuron.py`)

```python
"""
Leaky Integrate-and-Fire (LIF) neuron with golden ratio dynamics.

The neuron membrane potential decays with rate phi^(-1) per timestep,
representing the most stable decay rate per KAM theorem.

Threshold is set at phi, reset at phi^(-1).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from gnsp.constants import PHI, PHI_INV, DEFAULT_THRESHOLD, DEFAULT_RESET, DEFAULT_LEAK
from gnsp.core.fixed_point import FixedPoint, Q8_8


@dataclass
class LIFNeuronParams:
    """Parameters for LIF neuron."""
    threshold: float = DEFAULT_THRESHOLD    # Spike threshold (phi)
    reset: float = DEFAULT_RESET            # Reset potential (1/phi)
    leak: float = DEFAULT_LEAK              # Leak factor (1/phi^2)
    refractory_period: int = 2              # Refractory timesteps
    
    def validate(self):
        """Ensure parameters are valid."""
        assert self.threshold > self.reset, "Threshold must exceed reset"
        assert 0 < self.leak < 1, "Leak must be in (0, 1)"
        assert self.refractory_period >= 0


@dataclass
class LIFNeuronState:
    """State of a single LIF neuron."""
    membrane_potential: float = 0.0
    spike_trace: float = 0.0         # For STDP, decays with phi^(-1)
    refractory_counter: int = 0
    last_spike_time: int = -1
    spike_count: int = 0


class LIFNeuron:
    """
    Single Leaky Integrate-and-Fire neuron with golden ratio dynamics.
    
    Dynamics:
        V(t+1) = V(t) * (1 - leak) + I(t)
        if V(t+1) >= threshold:
            spike = 1
            V(t+1) = reset
        else:
            spike = 0
    
    Where leak = phi^(-2) for golden ratio decay.
    """
    
    def __init__(self, params: Optional[LIFNeuronParams] = None):
        self.params = params or LIFNeuronParams()
        self.params.validate()
        self.state = LIFNeuronState()
        self.timestep = 0
    
    def reset_state(self):
        """Reset neuron to initial state."""
        self.state = LIFNeuronState()
        self.timestep = 0
    
    def step(self, input_current: float) -> bool:
        """
        Advance neuron by one timestep.
        
        Args:
            input_current: Synaptic input current
            
        Returns:
            True if neuron spiked, False otherwise
        """
        self.timestep += 1
        
        # Update spike trace (golden decay)
        self.state.spike_trace *= PHI_INV
        
        # Check refractory period
        if self.state.refractory_counter > 0:
            self.state.refractory_counter -= 1
            return False
        
        # Leak (golden ratio decay)
        self.state.membrane_potential *= (1 - self.params.leak)
        
        # Integrate input
        self.state.membrane_potential += input_current
        
        # Spike generation
        if self.state.membrane_potential >= self.params.threshold:
            self.state.membrane_potential = self.params.reset
            self.state.spike_trace = 1.0
            self.state.last_spike_time = self.timestep
            self.state.spike_count += 1
            self.state.refractory_counter = self.params.refractory_period
            return True
        
        return False
    
    @property
    def voltage(self) -> float:
        return self.state.membrane_potential
    
    @property
    def trace(self) -> float:
        return self.state.spike_trace


class LIFNeuronArray:
    """
    Vectorized array of LIF neurons for efficient simulation.
    
    Uses NumPy operations for parallelism, structured for
    eventual GPU acceleration or FPGA synthesis.
    """
    
    def __init__(self, n_neurons: int, params: Optional[LIFNeuronParams] = None):
        self.n = n_neurons
        self.params = params or LIFNeuronParams()
        self.params.validate()
        
        # State arrays
        self.membrane = np.zeros(n_neurons, dtype=np.float32)
        self.spike_trace = np.zeros(n_neurons, dtype=np.float32)
        self.refractory = np.zeros(n_neurons, dtype=np.int32)
        self.last_spike = np.full(n_neurons, -1, dtype=np.int32)
        self.spike_count = np.zeros(n_neurons, dtype=np.int32)
        
        self.timestep = 0
    
    def reset(self):
        """Reset all neurons."""
        self.membrane.fill(0)
        self.spike_trace.fill(0)
        self.refractory.fill(0)
        self.last_spike.fill(-1)
        self.spike_count.fill(0)
        self.timestep = 0
    
    def step(self, input_current: np.ndarray) -> np.ndarray:
        """
        Advance all neurons by one timestep.
        
        Args:
            input_current: Shape (n_neurons,) input currents
            
        Returns:
            Boolean array of spikes
        """
        assert input_current.shape == (self.n,)
        self.timestep += 1
        
        # Update spike traces (golden decay)
        self.spike_trace *= PHI_INV
        
        # Identify neurons not in refractory period
        active = self.refractory == 0
        
        # Decrement refractory counters
        self.refractory = np.maximum(0, self.refractory - 1)
        
        # Leak (only for active neurons)
        self.membrane[active] *= (1 - self.params.leak)
        
        # Integrate input
        self.membrane[active] += input_current[active]
        
        # Spike detection
        spikes = active & (self.membrane >= self.params.threshold)
        
        # Reset spiking neurons
        self.membrane[spikes] = self.params.reset
        self.spike_trace[spikes] = 1.0
        self.last_spike[spikes] = self.timestep
        self.spike_count[spikes] += 1
        self.refractory[spikes] = self.params.refractory_period
        
        return spikes
    
    def get_state_dict(self) -> dict:
        """Get all state as dictionary (for checkpointing)."""
        return {
            'membrane': self.membrane.copy(),
            'spike_trace': self.spike_trace.copy(),
            'refractory': self.refractory.copy(),
            'last_spike': self.last_spike.copy(),
            'spike_count': self.spike_count.copy(),
            'timestep': self.timestep,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from dictionary."""
        self.membrane = state_dict['membrane'].copy()
        self.spike_trace = state_dict['spike_trace'].copy()
        self.refractory = state_dict['refractory'].copy()
        self.last_spike = state_dict['last_spike'].copy()
        self.spike_count = state_dict['spike_count'].copy()
        self.timestep = state_dict['timestep']
```

### 2.2 Synapse Model (`gnsp/snn/synapse.py`)

```python
"""
Synaptic connections with golden ratio weight quantization.

Weights are constrained to the golden ratio ladder:
{-phi^2, -phi, -1, -1/phi, 0, 1/phi, 1, phi, phi^2}

This provides 9 discrete levels with optimal dynamic range.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from gnsp.constants import PHI, PHI_INV, PHI_SQ, WEIGHT_LEVELS


@dataclass
class SynapseParams:
    """Parameters for synaptic connections."""
    delay_min: int = 1                    # Minimum synaptic delay (timesteps)
    delay_max: int = 8                    # Maximum delay (Fibonacci!)
    use_quantized_weights: bool = True    # Use golden ratio ladder
    weight_scale: float = 1.0             # Global weight scaling


def quantize_weight(weight: float) -> float:
    """
    Quantize weight to nearest golden ratio level.
    
    Levels: {-phi^2, -phi, -1, -1/phi, 0, 1/phi, 1, phi, phi^2}
    """
    return min(WEIGHT_LEVELS, key=lambda x: abs(x - weight))


def quantize_weights(weights: np.ndarray) -> np.ndarray:
    """Vectorized weight quantization."""
    levels = np.array(WEIGHT_LEVELS)
    # For each weight, find nearest level
    distances = np.abs(weights[:, np.newaxis] - levels)
    nearest_idx = np.argmin(distances, axis=1)
    return levels[nearest_idx]


class SynapseArray:
    """
    Efficient synapse storage using compressed sparse row (CSR) format.
    
    Stores:
    - Connectivity as sparse matrix
    - Weights (quantized to golden levels)
    - Delays (Fibonacci-constrained)
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[SynapseParams] = None
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or SynapseParams()
        
        # Sparse storage
        self.connections: List[Tuple[int, int]] = []
        self.weights: np.ndarray = np.array([], dtype=np.float32)
        self.delays: np.ndarray = np.array([], dtype=np.int32)
        
        # Dense adjacency (for small networks or visualization)
        self._dense_weights: Optional[np.ndarray] = None
        
    def add_connection(self, pre: int, post: int, weight: float, delay: int = 1):
        """Add a single synaptic connection."""
        assert 0 <= pre < self.n_pre
        assert 0 <= post < self.n_post
        assert self.params.delay_min <= delay <= self.params.delay_max
        
        if self.params.use_quantized_weights:
            weight = quantize_weight(weight)
        
        self.connections.append((pre, post))
        self.weights = np.append(self.weights, weight * self.params.weight_scale)
        self.delays = np.append(self.delays, delay)
        self._dense_weights = None  # Invalidate cache
    
    def add_connections_from_matrix(
        self,
        weight_matrix: np.ndarray,
        delay_matrix: Optional[np.ndarray] = None
    ):
        """
        Add connections from dense weight matrix.
        
        Non-zero entries become connections.
        """
        assert weight_matrix.shape == (self.n_pre, self.n_post)
        
        if delay_matrix is None:
            delay_matrix = np.ones_like(weight_matrix, dtype=np.int32)
        
        pre_idx, post_idx = np.nonzero(weight_matrix)
        
        for i in range(len(pre_idx)):
            pre, post = int(pre_idx[i]), int(post_idx[i])
            weight = weight_matrix[pre, post]
            delay = int(delay_matrix[pre, post])
            self.add_connection(pre, post, weight, delay)
    
    def get_dense_weights(self) -> np.ndarray:
        """Get dense weight matrix (cached)."""
        if self._dense_weights is None:
            self._dense_weights = np.zeros((self.n_pre, self.n_post), dtype=np.float32)
            for i, (pre, post) in enumerate(self.connections):
                self._dense_weights[pre, post] = self.weights[i]
        return self._dense_weights
    
    def compute_postsynaptic_input(
        self,
        pre_spikes: np.ndarray,
        spike_history: np.ndarray
    ) -> np.ndarray:
        """
        Compute input to postsynaptic neurons.
        
        Args:
            pre_spikes: Current presynaptic spikes (shape: n_pre)
            spike_history: Recent spike history for delays (shape: max_delay x n_pre)
            
        Returns:
            Postsynaptic input currents (shape: n_post)
        """
        post_input = np.zeros(self.n_post, dtype=np.float32)
        
        for i, (pre, post) in enumerate(self.connections):
            delay = self.delays[i]
            weight = self.weights[i]
            
            # Get spike from appropriate time in history
            if delay == 1:
                spike = pre_spikes[pre]
            else:
                # spike_history[0] is most recent, [delay-1] is delay timesteps ago
                if spike_history.shape[0] >= delay:
                    spike = spike_history[delay - 1, pre]
                else:
                    spike = 0
            
            post_input[post] += weight * spike
        
        return post_input
    
    @property
    def n_synapses(self) -> int:
        return len(self.connections)
    
    def get_statistics(self) -> dict:
        """Get synapse statistics."""
        if self.n_synapses == 0:
            return {'n_synapses': 0}
        
        return {
            'n_synapses': self.n_synapses,
            'sparsity': 1 - self.n_synapses / (self.n_pre * self.n_post),
            'weight_mean': float(np.mean(self.weights)),
            'weight_std': float(np.std(self.weights)),
            'weight_min': float(np.min(self.weights)),
            'weight_max': float(np.max(self.weights)),
            'delay_mean': float(np.mean(self.delays)),
            'unique_weights': len(np.unique(self.weights)),
        }
```

### 2.3 STDP Learning (`gnsp/snn/stdp.py`)

```python
"""
Spike-Timing Dependent Plasticity with Fibonacci time constants.

Uses Fibonacci sequence for STDP time windows:
tau = [1, 2, 3, 5, 8, 13, 21, 34] ms

Weight updates follow golden ratio decay:
delta_w = A * phi^(-|dt|/tau)

Weights are bounded to golden ratio ladder.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from gnsp.constants import PHI, PHI_INV, STDP_TAU, WEIGHT_LEVELS
from gnsp.core.fibonacci import fibonacci_decay_lookup
from gnsp.snn.synapse import SynapseArray, quantize_weight


@dataclass 
class FibonacciSTDPParams:
    """Parameters for Fibonacci STDP."""
    a_plus: float = 0.1              # LTP amplitude
    a_minus: float = 0.12            # LTD amplitude (slightly stronger)
    tau_plus_idx: int = 4            # Index into STDP_TAU for LTP (tau=5ms)
    tau_minus_idx: int = 5           # Index into STDP_TAU for LTD (tau=8ms)
    w_max: float = PHI ** 2          # Maximum weight (phi^2)
    w_min: float = -(PHI ** 2)       # Minimum weight (-phi^2)
    use_golden_ladder: bool = True   # Quantize to golden levels
    

class FibonacciSTDP:
    """
    STDP learning rule with Fibonacci time constants and golden ratio decay.
    
    The learning window uses:
    - Potentiation (LTP) for pre-before-post: delta_w > 0
    - Depression (LTD) for post-before-pre: delta_w < 0
    
    Time constants from Fibonacci sequence provide naturally scaled
    windows at multiple timescales.
    """
    
    def __init__(self, params: Optional[FibonacciSTDPParams] = None):
        self.params = params or FibonacciSTDPParams()
        
        # Precompute decay lookup table
        self.decay_lut = fibonacci_decay_lookup(max_dt=64, n_tau=8)
        
        # Get actual tau values
        self.tau_plus = STDP_TAU[self.params.tau_plus_idx]
        self.tau_minus = STDP_TAU[self.params.tau_minus_idx]
    
    def compute_weight_update(
        self,
        dt: int,
        current_weight: float
    ) -> float:
        """
        Compute weight update for a single synapse.
        
        Args:
            dt: Spike time difference (post - pre). 
                Positive = pre before post (LTP)
                Negative = post before pre (LTD)
            current_weight: Current synaptic weight
            
        Returns:
            Weight change (delta_w)
        """
        if dt == 0:
            return 0.0
        
        if dt > 0:
            # LTP: pre before post
            decay = self.decay_lut[min(dt, 64), self.params.tau_plus_idx]
            delta_w = self.params.a_plus * decay
        else:
            # LTD: post before pre
            decay = self.decay_lut[min(-dt, 64), self.params.tau_minus_idx]
            delta_w = -self.params.a_minus * decay
        
        # Soft bounds using tanh-like scaling
        if delta_w > 0:
            delta_w *= (self.params.w_max - current_weight) / self.params.w_max
        else:
            delta_w *= (current_weight - self.params.w_min) / (-self.params.w_min)
        
        return delta_w
    
    def update_weights(
        self,
        synapses: SynapseArray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
        current_time: int
    ) -> np.ndarray:
        """
        Update all synaptic weights based on spike timing.
        
        Args:
            synapses: Synapse array to update
            pre_spike_times: Last spike time for each presynaptic neuron
            post_spike_times: Last spike time for each postsynaptic neuron
            current_time: Current simulation time
            
        Returns:
            Array of weight changes
        """
        delta_weights = np.zeros(synapses.n_synapses, dtype=np.float32)
        
        for i, (pre, post) in enumerate(synapses.connections):
            pre_t = pre_spike_times[pre]
            post_t = post_spike_times[post]
            
            # Skip if either neuron hasn't spiked recently
            if pre_t < 0 or post_t < 0:
                continue
            
            # Only consider recent spikes
            if current_time - pre_t > 64 and current_time - post_t > 64:
                continue
            
            dt = post_t - pre_t
            delta_w = self.compute_weight_update(dt, synapses.weights[i])
            delta_weights[i] = delta_w
        
        # Apply updates
        synapses.weights += delta_weights
        
        # Clip to bounds
        synapses.weights = np.clip(
            synapses.weights,
            self.params.w_min,
            self.params.w_max
        )
        
        # Quantize if enabled
        if self.params.use_golden_ladder:
            from gnsp.snn.synapse import quantize_weights
            synapses.weights = quantize_weights(synapses.weights)
        
        return delta_weights
    
    def get_learning_window(self, dt_range: Tuple[int, int] = (-50, 50)) -> np.ndarray:
        """
        Get the STDP learning window for visualization.
        
        Returns array of delta_w values for each dt in range.
        """
        dts = np.arange(dt_range[0], dt_range[1] + 1)
        window = np.zeros_like(dts, dtype=np.float32)
        
        for i, dt in enumerate(dts):
            window[i] = self.compute_weight_update(dt, current_weight=0.5)
        
        return dts, window


class OnlineSTDP:
    """
    Online STDP that can be applied during simulation.
    
    Maintains spike traces for efficient updates without
    storing full spike history.
    """
    
    def __init__(
        self,
        synapses: SynapseArray,
        params: Optional[FibonacciSTDPParams] = None
    ):
        self.synapses = synapses
        self.params = params or FibonacciSTDPParams()
        self.stdp = FibonacciSTDP(params)
        
        # Eligibility traces for each synapse
        self.pre_trace = np.zeros(synapses.n_pre, dtype=np.float32)
        self.post_trace = np.zeros(synapses.n_post, dtype=np.float32)
    
    def reset(self):
        """Reset traces."""
        self.pre_trace.fill(0)
        self.post_trace.fill(0)
    
    def step(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray
    ) -> np.ndarray:
        """
        Process one timestep of STDP.
        
        Args:
            pre_spikes: Binary presynaptic spike array
            post_spikes: Binary postsynaptic spike array
            
        Returns:
            Weight changes applied
        """
        # Decay traces with golden ratio
        self.pre_trace *= PHI_INV
        self.post_trace *= PHI_INV
        
        # Update traces on spikes
        self.pre_trace[pre_spikes] = 1.0
        self.post_trace[post_spikes] = 1.0
        
        # Compute weight updates
        delta_weights = np.zeros(self.synapses.n_synapses, dtype=np.float32)
        
        for i, (pre, post) in enumerate(self.synapses.connections):
            current_w = self.synapses.weights[i]
            
            # LTP: post spike with pre trace
            if post_spikes[post]:
                delta_w = self.params.a_plus * self.pre_trace[pre]
                delta_w *= (self.params.w_max - current_w) / self.params.w_max
                delta_weights[i] += delta_w
            
            # LTD: pre spike with post trace
            if pre_spikes[pre]:
                delta_w = -self.params.a_minus * self.post_trace[post]
                delta_w *= (current_w - self.params.w_min) / (-self.params.w_min)
                delta_weights[i] += delta_w
        
        # Apply updates
        self.synapses.weights += delta_weights
        self.synapses.weights = np.clip(
            self.synapses.weights,
            self.params.w_min,
            self.params.w_max
        )
        
        if self.params.use_golden_ladder:
            from gnsp.snn.synapse import quantize_weights
            self.synapses.weights = quantize_weights(self.synapses.weights)
        
        return delta_weights
```

### 2.4 Network Topology (`gnsp/snn/topology.py`)

```python
"""
Quasicrystalline network topology based on golden ratio.

Neurons are positioned on Penrose-like patterns with
connections at Fibonacci/golden distances.

This provides:
- Optimal information routing (maximally connected)
- Efficient encoding (Fibonacci chains)
- Natural hierarchical structure
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
import numpy as np
from gnsp.constants import PHI, PHI_INV, GOLDEN_ANGLE
from gnsp.core.golden import golden_spiral_points
from gnsp.core.fibonacci import fibonacci_offsets
from gnsp.snn.synapse import SynapseArray


@dataclass
class QuasicrystalTopologyParams:
    """Parameters for quasicrystal network topology."""
    n_neurons: int = 1024
    connection_distances: Tuple[float, ...] = (1.0, PHI, PHI**2)
    distance_tolerance: float = 0.15
    max_connections_per_neuron: int = 16
    use_fibonacci_offsets: bool = True
    spatial_scale: float = 1.0


class QuasicrystalTopology:
    """
    Generate quasicrystalline network topology.
    
    Neurons positioned on golden spiral, connected at
    golden ratio-related distances.
    """
    
    def __init__(self, params: Optional[QuasicrystalTopologyParams] = None):
        self.params = params or QuasicrystalTopologyParams()
        
        # Generate neuron positions
        self.positions = golden_spiral_points(
            self.params.n_neurons,
            scale=self.params.spatial_scale
        )
        
        # Compute distance matrix (cached)
        self._distance_matrix: Optional[np.ndarray] = None
        
    @property
    def n_neurons(self) -> int:
        return self.params.n_neurons
    
    @property
    def distance_matrix(self) -> np.ndarray:
        """Compute or return cached distance matrix."""
        if self._distance_matrix is None:
            n = self.n_neurons
            self._distance_matrix = np.zeros((n, n), dtype=np.float32)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    self._distance_matrix[i, j] = dist
                    self._distance_matrix[j, i] = dist
        
        return self._distance_matrix
    
    def build_connectivity(self) -> np.ndarray:
        """
        Build adjacency matrix based on golden ratio distances.
        
        Returns binary adjacency matrix.
        """
        n = self.n_neurons
        adjacency = np.zeros((n, n), dtype=np.int32)
        distances = self.distance_matrix
        
        for i in range(n):
            # Find neurons at valid distances
            candidates = []
            
            for j in range(n):
                if i == j:
                    continue
                
                dist = distances[i, j]
                
                # Check if distance matches a golden ratio level
                for valid_dist in self.params.connection_distances:
                    if abs(dist - valid_dist) < self.params.distance_tolerance * valid_dist:
                        candidates.append((j, dist))
                        break
            
            # Sort by distance and take closest up to max
            candidates.sort(key=lambda x: x[1])
            for j, _ in candidates[:self.params.max_connections_per_neuron]:
                adjacency[i, j] = 1
        
        return adjacency
    
    def build_weighted_connectivity(self) -> np.ndarray:
        """
        Build weight matrix with distance-based weights.
        
        Closer neurons get stronger connections (phi^(-distance)).
        """
        n = self.n_neurons
        adjacency = self.build_connectivity()
        weights = np.zeros((n, n), dtype=np.float32)
        distances = self.distance_matrix
        
        for i in range(n):
            for j in range(n):
                if adjacency[i, j]:
                    # Weight inversely proportional to distance
                    weights[i, j] = PHI ** (-distances[i, j])
        
        return weights
    
    def create_synapses(self) -> SynapseArray:
        """Create SynapseArray from topology."""
        weights = self.build_weighted_connectivity()
        synapses = SynapseArray(self.n_neurons, self.n_neurons)
        
        # Delays based on distance (Fibonacci quantized)
        fib_delays = [1, 2, 3, 5, 8]
        distances = self.distance_matrix
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if weights[i, j] != 0:
                    # Quantize delay to nearest Fibonacci
                    dist = distances[i, j]
                    delay = min(fib_delays, key=lambda d: abs(d - dist))
                    delay = max(1, min(8, delay))
                    synapses.add_connection(i, j, weights[i, j], delay)
        
        return synapses
    
    def get_statistics(self) -> dict:
        """Get topology statistics."""
        adjacency = self.build_connectivity()
        degrees = adjacency.sum(axis=1)
        
        return {
            'n_neurons': self.n_neurons,
            'n_connections': int(adjacency.sum()),
            'mean_degree': float(degrees.mean()),
            'std_degree': float(degrees.std()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'density': float(adjacency.sum()) / (self.n_neurons ** 2),
        }


class HierarchicalTopology:
    """
    Hierarchical network topology for layered SNN.
    
    Layers: Input -> Hidden1 -> Hidden2 -> Output
    With golden ratio connectivity between layers.
    """
    
    def __init__(
        self,
        layer_sizes: Tuple[int, ...] = (80, 64, 32, 5),
        inter_layer_sparsity: float = 0.3,
        intra_layer_sparsity: float = 0.1
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.inter_sparsity = inter_layer_sparsity
        self.intra_sparsity = intra_layer_sparsity
        
        # Compute layer offsets
        self.layer_offsets = [0]
        for size in layer_sizes[:-1]:
            self.layer_offsets.append(self.layer_offsets[-1] + size)
        
        self.total_neurons = sum(layer_sizes)
    
    def build_connectivity(self) -> np.ndarray:
        """Build hierarchical adjacency matrix."""
        n = self.total_neurons
        adjacency = np.zeros((n, n), dtype=np.float32)
        
        # Inter-layer connections (feedforward)
        for layer in range(self.n_layers - 1):
            src_start = self.layer_offsets[layer]
            src_end = src_start + self.layer_sizes[layer]
            dst_start = self.layer_offsets[layer + 1]
            dst_end = dst_start + self.layer_sizes[layer + 1]
            
            # Random sparse connections with golden ratio weights
            for i in range(src_start, src_end):
                for j in range(dst_start, dst_end):
                    if np.random.random() < self.inter_sparsity:
                        # Weight from golden ladder
                        weight = np.random.choice([PHI_INV, 1.0, PHI])
                        adjacency[i, j] = weight
        
        # Intra-layer connections (recurrent within layer)
        for layer in range(1, self.n_layers - 1):  # Hidden layers only
            start = self.layer_offsets[layer]
            end = start + self.layer_sizes[layer]
            
            for i in range(start, end):
                for j in range(start, end):
                    if i != j and np.random.random() < self.intra_sparsity:
                        weight = np.random.choice([PHI_INV, 1.0]) * 0.5
                        adjacency[i, j] = weight
        
        return adjacency
    
    def create_synapses(self) -> SynapseArray:
        """Create SynapseArray from topology."""
        weights = self.build_connectivity()
        synapses = SynapseArray(self.total_neurons, self.total_neurons)
        synapses.add_connections_from_matrix(weights)
        return synapses
    
    def get_layer_indices(self, layer: int) -> Tuple[int, int]:
        """Get start and end indices for a layer."""
        start = self.layer_offsets[layer]
        end = start + self.layer_sizes[layer]
        return start, end
```

### 2.5 Main Network (`gnsp/snn/network.py`)

```python
"""
Complete SNN network integrating neurons, synapses, topology, and learning.

This is the main simulation container that orchestrates all components.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
import numpy as np
from gnsp.constants import PHI
from gnsp.snn.neuron import LIFNeuronArray, LIFNeuronParams
from gnsp.snn.synapse import SynapseArray, SynapseParams
from gnsp.snn.stdp import OnlineSTDP, FibonacciSTDPParams
from gnsp.snn.topology import QuasicrystalTopology, HierarchicalTopology


@dataclass
class SNNConfig:
    """Configuration for SNN network."""
    n_input: int = 80
    n_hidden: Tuple[int, ...] = (64, 32)
    n_output: int = 5
    
    neuron_params: LIFNeuronParams = field(default_factory=LIFNeuronParams)
    synapse_params: SynapseParams = field(default_factory=SynapseParams)
    stdp_params: FibonacciSTDPParams = field(default_factory=FibonacciSTDPParams)
    
    use_quasicrystal_topology: bool = True
    enable_stdp: bool = True
    max_spike_history: int = 16


class SpikingNeuralNetwork:
    """
    Complete Spiking Neural Network with golden ratio architecture.
    
    Features:
    - LIF neurons with golden ratio dynamics
    - Quasicrystalline or hierarchical topology
    - Fibonacci STDP learning
    - Golden ratio weight quantization
    """
    
    def __init__(self, config: Optional[SNNConfig] = None):
        self.config = config or SNNConfig()
        
        # Compute layer structure
        self.layer_sizes = (
            self.config.n_input,
            *self.config.n_hidden,
            self.config.n_output
        )
        self.n_neurons = sum(self.layer_sizes)
        self.n_layers = len(self.layer_sizes)
        
        # Initialize neurons
        self.neurons = LIFNeuronArray(self.n_neurons, self.config.neuron_params)
        
        # Initialize topology and synapses
        if self.config.use_quasicrystal_topology:
            topology = QuasicrystalTopology(
                QuasicrystalTopologyParams(n_neurons=self.n_neurons)
            )
            self.synapses = topology.create_synapses()
        else:
            topology = HierarchicalTopology(self.layer_sizes)
            self.synapses = topology.create_synapses()
        
        # Initialize STDP
        if self.config.enable_stdp:
            self.stdp = OnlineSTDP(self.synapses, self.config.stdp_params)
        else:
            self.stdp = None
        
        # Spike history for delays
        self.spike_history = np.zeros(
            (self.config.max_spike_history, self.n_neurons),
            dtype=np.float32
        )
        
        # Recording
        self.spike_record: List[np.ndarray] = []
        self.voltage_record: List[np.ndarray] = []
        self.weight_record: List[np.ndarray] = []
        
        self.timestep = 0
    
    def reset(self):
        """Reset network state."""
        self.neurons.reset()
        self.spike_history.fill(0)
        if self.stdp:
            self.stdp.reset()
        self.spike_record.clear()
        self.voltage_record.clear()
        self.weight_record.clear()
        self.timestep = 0
    
    def step(
        self,
        input_current: np.ndarray,
        record: bool = False,
        learn: bool = True
    ) -> np.ndarray:
        """
        Advance network by one timestep.
        
        Args:
            input_current: Input to first layer (shape: n_input)
            record: Whether to record state
            learn: Whether to apply STDP
            
        Returns:
            Output spikes (shape: n_output)
        """
        self.timestep += 1
        
        # Build full input current
        full_input = np.zeros(self.n_neurons, dtype=np.float32)
        full_input[:self.config.n_input] = input_current
        
        # Add synaptic input from previous spikes
        synaptic_input = self.synapses.compute_postsynaptic_input(
            self.spike_history[0],  # Most recent spikes
            self.spike_history
        )
        full_input += synaptic_input
        
        # Update neurons
        spikes = self.neurons.step(full_input)
        
        # Update spike history (shift and insert new)
        self.spike_history = np.roll(self.spike_history, 1, axis=0)
        self.spike_history[0] = spikes.astype(np.float32)
        
        # Apply STDP if enabled
        if learn and self.stdp:
            self.stdp.step(
                spikes[:self.n_neurons - self.config.n_output],
                spikes
            )
        
        # Recording
        if record:
            self.spike_record.append(spikes.copy())
            self.voltage_record.append(self.neurons.membrane.copy())
            self.weight_record.append(self.synapses.weights.copy())
        
        # Return output layer spikes
        output_start = self.n_neurons - self.config.n_output
        return spikes[output_start:]
    
    def run(
        self,
        inputs: np.ndarray,
        record: bool = False,
        learn: bool = True
    ) -> np.ndarray:
        """
        Run network on sequence of inputs.
        
        Args:
            inputs: Input sequence (shape: timesteps x n_input)
            record: Whether to record state
            learn: Whether to apply STDP
            
        Returns:
            Output spikes (shape: timesteps x n_output)
        """
        n_steps = inputs.shape[0]
        outputs = np.zeros((n_steps, self.config.n_output), dtype=np.float32)
        
        for t in range(n_steps):
            outputs[t] = self.step(inputs[t], record=record, learn=learn)
        
        return outputs
    
    def get_output_rates(
        self,
        inputs: np.ndarray,
        window: int = 50
    ) -> np.ndarray:
        """
        Get output neuron firing rates.
        
        Runs network and averages output spikes over window.
        """
        outputs = self.run(inputs, record=False, learn=False)
        
        # Compute rates over sliding window
        n_steps = outputs.shape[0]
        if n_steps < window:
            return outputs.mean(axis=0)
        
        # Rolling mean
        rates = np.zeros_like(outputs)
        for t in range(n_steps):
            start = max(0, t - window + 1)
            rates[t] = outputs[start:t+1].mean(axis=0)
        
        return rates
    
    def classify(
        self,
        inputs: np.ndarray,
        integration_time: int = 100
    ) -> int:
        """
        Classify input by running network and taking argmax of output rates.
        """
        # Extend input to integration_time if needed
        if inputs.shape[0] < integration_time:
            inputs = np.tile(inputs, (integration_time // inputs.shape[0] + 1, 1))
            inputs = inputs[:integration_time]
        
        outputs = self.run(inputs[:integration_time], record=False, learn=False)
        rates = outputs.sum(axis=0)
        return int(np.argmax(rates))
    
    def get_state_dict(self) -> dict:
        """Get network state for checkpointing."""
        return {
            'neurons': self.neurons.get_state_dict(),
            'weights': self.synapses.weights.copy(),
            'spike_history': self.spike_history.copy(),
            'timestep': self.timestep,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load network state from checkpoint."""
        self.neurons.load_state_dict(state_dict['neurons'])
        self.synapses.weights = state_dict['weights'].copy()
        self.spike_history = state_dict['spike_history'].copy()
        self.timestep = state_dict['timestep']
    
    def get_statistics(self) -> dict:
        """Get network statistics."""
        return {
            'n_neurons': self.n_neurons,
            'n_layers': self.n_layers,
            'layer_sizes': self.layer_sizes,
            'synapse_stats': self.synapses.get_statistics(),
            'total_spikes': int(self.neurons.spike_count.sum()),
            'mean_rate': float(self.neurons.spike_count.mean() / max(1, self.timestep)),
        }
```

---

## PHASE 3: AUTOMATA THEORY MODULE (Week 6-8)

### 3.1 Core Automata (`gnsp/automata/dfa.py`, `nfa.py`)

Implement DFA and NFA with full operations:
- Construction from transitions
- Complement, intersection, union
- Determinization (NFA to DFA)
- Minimization
- Language membership
- Word generation

### 3.2 Weighted Automata (`gnsp/automata/weighted.py`)

Implement weighted automata over arbitrary semirings:
- Probability semiring (Markov chains)
- Tropical semiring (shortest path)
- Viterbi semiring (most probable path)
- Quantum semiring (complex amplitudes)

### 3.3 Büchi Automata (`gnsp/automata/buchi.py`)

Implement ω-automata for infinite words:
- Büchi acceptance condition
- Online monitoring
- LTL compilation

### 3.4 Timed Automata (`gnsp/automata/timed.py`)

Implement timed automata with:
- Clock variables
- Guards and invariants
- Zone-based semantics
- Timing attack detection

### 3.5 Krohn-Rhodes (`gnsp/automata/krohn_rhodes.py`)

Implement algebraic decomposition:
- Syntactic monoid computation
- Aperiodicity testing
- Group component extraction
- Complexity measure

### 3.6 L* Learning (`gnsp/automata/learning.py`)

Implement Angluin's L* algorithm:
- Membership and equivalence queries
- Observation table
- Counterexample processing
- Online protocol learning

---

## PHASE 4: CATEGORY THEORY MODULE (Week 9-11)

### 4.1 Core Categories (`gnsp/category/category.py`)

```python
"""
Category theory foundations for security analysis.

A category consists of:
- Objects (e.g., protocol states)
- Morphisms (e.g., valid transitions)
- Composition (sequential execution)
- Identity morphisms

Functors map between categories preserving structure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Dict, Set, List, Tuple, Optional, Callable

Obj = TypeVar('Obj')
Mor = TypeVar('Mor')


@dataclass(frozen=True)
class Morphism(Generic[Obj]):
    """A morphism between objects."""
    source: Obj
    target: Obj
    label: str = ""
    
    def __repr__(self) -> str:
        return f"{self.source} --{self.label}--> {self.target}"


class Category(ABC, Generic[Obj, Mor]):
    """
    Abstract base class for categories.
    
    Subclasses must implement:
    - objects(): Set of objects
    - morphisms(): Set of morphisms
    - compose(): Morphism composition
    - identity(): Identity morphism for each object
    """
    
    @abstractmethod
    def objects(self) -> Set[Obj]:
        """Return all objects in the category."""
        pass
    
    @abstractmethod
    def morphisms(self, source: Optional[Obj] = None, target: Optional[Obj] = None) -> Set[Mor]:
        """Return morphisms, optionally filtered by source/target."""
        pass
    
    @abstractmethod
    def compose(self, f: Mor, g: Mor) -> Optional[Mor]:
        """
        Compose morphisms: f ; g (f then g).
        
        Returns None if composition is undefined (source/target mismatch).
        """
        pass
    
    @abstractmethod
    def identity(self, obj: Obj) -> Mor:
        """Return identity morphism for object."""
        pass
    
    def hom(self, source: Obj, target: Obj) -> Set[Mor]:
        """Return Hom(source, target) - all morphisms from source to target."""
        return self.morphisms(source=source, target=target)


class FiniteCategory(Category[str, Morphism[str]]):
    """
    Finite category with explicit objects and morphisms.
    
    Used for protocol state machines.
    """
    
    def __init__(self):
        self._objects: Set[str] = set()
        self._morphisms: Set[Morphism[str]] = set()
        self._composition: Dict[Tuple[Morphism[str], Morphism[str]], Morphism[str]] = {}
    
    def add_object(self, obj: str):
        """Add an object to the category."""
        self._objects.add(obj)
    
    def add_morphism(self, source: str, target: str, label: str = ""):
        """Add a morphism to the category."""
        self._objects.add(source)
        self._objects.add(target)
        mor = Morphism(source, target, label)
        self._morphisms.add(mor)
        return mor
    
    def set_composition(self, f: Morphism[str], g: Morphism[str], result: Morphism[str]):
        """Define composition f ; g = result."""
        self._composition[(f, g)] = result
    
    def objects(self) -> Set[str]:
        return self._objects.copy()
    
    def morphisms(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None
    ) -> Set[Morphism[str]]:
        result = self._morphisms.copy()
        if source is not None:
            result = {m for m in result if m.source == source}
        if target is not None:
            result = {m for m in result if m.target == target}
        return result
    
    def compose(self, f: Morphism[str], g: Morphism[str]) -> Optional[Morphism[str]]:
        # Check composability
        if f.target != g.source:
            return None
        
        # Check if composition is defined
        if (f, g) in self._composition:
            return self._composition[(f, g)]
        
        # Try to find or create composition
        # For finite categories, we need explicit definition
        return None
    
    def identity(self, obj: str) -> Morphism[str]:
        return Morphism(obj, obj, f"id_{obj}")


class Functor(Generic[Obj, Mor]):
    """
    Functor between categories.
    
    Maps objects to objects and morphisms to morphisms,
    preserving composition and identities.
    """
    
    def __init__(
        self,
        source_cat: Category,
        target_cat: Category,
        object_map: Dict[Obj, Obj],
        morphism_map: Dict[Mor, Mor]
    ):
        self.source = source_cat
        self.target = target_cat
        self.object_map = object_map
        self.morphism_map = morphism_map
    
    def apply_object(self, obj: Obj) -> Obj:
        """Apply functor to object."""
        return self.object_map.get(obj)
    
    def apply_morphism(self, mor: Mor) -> Mor:
        """Apply functor to morphism."""
        return self.morphism_map.get(mor)
    
    def verify_functoriality(self) -> List[str]:
        """
        Verify functor laws.
        
        Returns list of violations (empty if valid).
        """
        violations = []
        
        # Check identity preservation
        for obj in self.source.objects():
            id_src = self.source.identity(obj)
            id_tgt = self.target.identity(self.apply_object(obj))
            
            if self.apply_morphism(id_src) != id_tgt:
                violations.append(f"Identity not preserved for {obj}")
        
        # Check composition preservation
        for f in self.source.morphisms():
            for g in self.source.morphisms():
                fg = self.source.compose(f, g)
                if fg is not None:
                    Ff = self.apply_morphism(f)
                    Fg = self.apply_morphism(g)
                    Ffg = self.apply_morphism(fg)
                    
                    composed = self.target.compose(Ff, Fg)
                    
                    if composed != Ffg:
                        violations.append(
                            f"Composition not preserved: F({f};{g}) != F({f});F({g})"
                        )
        
        return violations
```

### 4.2 Protocol Categories (`gnsp/category/protocol.py`)

Build categories for TCP, HTTP, DNS protocols with:
- State objects
- Transition morphisms
- Composition rules
- Security properties

### 4.3 Security Functors (`gnsp/category/functor.py`)

Implement the security functor F: Protocol -> Traffic:
- Traffic observation
- Functoriality checking
- Violation detection
- Severity scoring

### 4.4 Sheaves (`gnsp/category/sheaf.py`)

Implement sheaf-theoretic distributed detection:
- Network topology as site
- Security states as sheaf
- Restriction maps
- Čech cohomology

---

## PHASE 5: TOPOLOGICAL DATA ANALYSIS (Week 12-14)

### 5.1 Simplicial Complexes (`gnsp/topology/simplicial.py`)

Build simplicial complex structures:
- 0-simplices (vertices)
- 1-simplices (edges)
- 2-simplices (triangles)
- Higher simplices
- Boundary operators

### 5.2 Vietoris-Rips Complex (`gnsp/topology/vietoris_rips.py`)

Efficient VR complex construction:
- Golden ratio scale sequence: ε, εφ, εφ², ...
- Incremental construction
- Sparse representation

### 5.3 Persistent Homology (`gnsp/topology/persistence.py`)

Full persistent homology pipeline:
- Boundary matrix construction
- Column reduction over Z₂
- Persistence pairs
- Birth-death pairs
- Persistence diagrams

### 5.4 TDA Features (`gnsp/topology/features.py`)

Extract ML features from persistence:
- Betti curves
- Persistence entropy
- Persistence landscapes
- Persistence images
- Statistical summaries

---

## PHASE 6: CLIFFORD ALGEBRA (Week 15-16)

### 6.1 Multivectors (`gnsp/algebra/multivector.py`)

Implement Cl(3,0) multivector algebra:
- Scalar, vector, bivector, trivector components
- Basis: {1, e1, e2, e3, e12, e23, e31, e123}
- Addition, scalar multiplication
- Grade extraction

### 6.2 Geometric Product (`gnsp/algebra/geometric_product.py`)

Implement geometric product:
- Sign table for basis products
- Full multivector multiplication
- Inner and outer products

### 6.3 Rotors (`gnsp/algebra/rotor.py`)

Implement rotor operations:
- Rotor construction from angle and plane
- Sandwich product: R X R†
- Rotor composition
- Application to SNN state

---

## PHASE 7: INTEGRATION AND DETECTION (Week 17-20)

### 7.1 Detector Integration (`gnsp/detection/detector.py`)

Main detector orchestrating all methods:
- Automata-based protocol checking
- Category-theoretic functor analysis
- TDA topological features
- SNN classification
- Ensemble combination

### 7.2 Training Pipeline (`gnsp/training/`)

Complete training system:
- Dataset loading (NSL-KDD, UNSW-NB15, CICIDS)
- Preprocessing (normalization, encoding)
- Surrogate gradient training for SNN
- Cross-validation
- Hyperparameter search

### 7.3 Evaluation (`gnsp/training/evaluation.py`)

Comprehensive evaluation:
- Accuracy, precision, recall, F1
- ROC curves, AUC
- Confusion matrices
- Per-class analysis
- Golden ratio hypothesis testing

---

## DEPENDENCIES

```
# requirements.txt

# Core
numpy>=1.24.0
scipy>=1.10.0

# SNN Simulation
snnTorch>=0.7.0        # Optional: for comparison
brian2>=2.5.0          # Optional: for validation

# Machine Learning
torch>=2.0.0
scikit-learn>=1.2.0

# Data Processing
pandas>=2.0.0
h5py>=3.8.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Topology
giotto-tda>=0.6.0      # For TDA validation
gudhi>=3.8.0           # Optional: alternative TDA

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0
hypothesis>=6.75.0     # Property-based testing

# Development
black>=23.0.0
mypy>=1.0.0
ruff>=0.0.260

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0
```

---

## TESTING STRATEGY

### Unit Tests
- Every function has corresponding test
- Property-based testing for mathematical properties
- Golden ratio invariants verified

### Integration Tests
- Full pipeline tests
- Dataset processing tests
- Detection accuracy tests

### Golden Ratio Hypothesis Tests
Create specific tests for:

```python
# tests/test_golden_hypothesis.py

"""
Tests for golden ratio hypothesis in SNN performance.

We test whether phi-based parameters improve:
1. Convergence speed
2. Stability
3. Adversarial robustness
4. Classification accuracy
"""

import pytest
import numpy as np
from gnsp.constants import PHI, PHI_INV
from gnsp.snn.network import SpikingNeuralNetwork, SNNConfig
from gnsp.snn.neuron import LIFNeuronParams


class TestGoldenConvergence:
    """Test if golden ratio decay improves convergence."""
    
    @pytest.fixture
    def standard_config(self):
        params = LIFNeuronParams(leak=0.5)  # Standard leak
        return SNNConfig(neuron_params=params)
    
    @pytest.fixture
    def golden_config(self):
        params = LIFNeuronParams(leak=PHI_INV ** 2)  # Golden leak
        return SNNConfig(neuron_params=params)
    
    def test_convergence_comparison(self, standard_config, golden_config):
        """Golden leak should converge faster or equally."""
        # ... implementation
        pass


class TestGoldenStability:
    """Test if golden ratio provides stability under perturbation."""
    
    def test_weight_perturbation_robustness(self):
        """Golden weights should be more robust to noise."""
        pass
    
    def test_input_noise_robustness(self):
        """Golden architecture should handle noisy input better."""
        pass


class TestGoldenAccuracy:
    """Test if golden ratio improves classification."""
    
    def test_nsl_kdd_accuracy(self):
        """Compare accuracy on NSL-KDD benchmark."""
        pass
```

---

## NOTEBOOKS

Create interactive notebooks for exploration:

1. **01_golden_ratio_exploration.ipynb**: Explore phi properties, eigenvalues, Fibonacci
2. **02_snn_basics.ipynb**: Basic SNN simulation, visualization
3. **03_automata_protocols.ipynb**: Protocol modeling with automata
4. **04_category_theory_ids.ipynb**: Category-theoretic detection
5. **05_persistent_homology.ipynb**: TDA on network traffic
6. **06_clifford_algebra.ipynb**: Geometric algebra for state representation
7. **07_full_pipeline.ipynb**: End-to-end detection demonstration
8. **08_golden_hypothesis_testing.ipynb**: Empirical tests of phi benefits

---

## CONFIGURATION

```yaml
# configs/default.yaml

project:
  name: "GNSP"
  version: "0.1.0"
  seed: 42

golden:
  phi: 1.618033988749895
  use_golden_weights: true
  use_golden_decay: true
  use_fibonacci_stdp: true

snn:
  n_input: 80
  n_hidden: [64, 32]
  n_output: 5
  
  neuron:
    threshold: 1.618
    reset: 0.618
    leak: 0.382
    refractory_period: 2
  
  synapse:
    delay_min: 1
    delay_max: 8
    quantize_weights: true
  
  stdp:
    a_plus: 0.1
    a_minus: 0.12
    tau_plus_idx: 4
    tau_minus_idx: 5
    use_golden_ladder: true
  
  topology:
    type: "quasicrystal"  # or "hierarchical"
    connection_distances: [1.0, 1.618, 2.618]

automata:
  protocol_checking: true
  ltl_monitoring: true
  timed_analysis: true

category:
  functor_checking: true
  sheaf_cohomology: true

topology:
  enable_tda: true
  scales: 8
  scale_factor: 1.618
  max_dimension: 2

training:
  dataset: "nsl-kdd"
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2

detection:
  ensemble_weights:
    snn: 0.4
    automata: 0.2
    category: 0.2
    tda: 0.2
  threshold: 0.618
```

---

## EXECUTION ORDER

When building this project, follow this order:

1. **Core module** - constants, fixed-point, golden utilities
2. **Tests for core** - verify mathematical foundations
3. **SNN neurons** - LIF with golden dynamics
4. **SNN synapses** - golden weight quantization
5. **SNN STDP** - Fibonacci learning
6. **SNN topology** - quasicrystal layout
7. **SNN network** - integration
8. **Automata DFA/NFA** - basic automata
9. **Automata weighted** - semiring operations
10. **Automata Büchi** - infinite monitoring
11. **Category core** - objects, morphisms, functors
12. **Category protocols** - TCP/HTTP/DNS categories
13. **Category detection** - functor checking
14. **Topology simplicial** - complex construction
15. **Topology persistence** - homology computation
16. **Topology features** - ML feature extraction
17. **Algebra Clifford** - geometric algebra
18. **Detection integration** - ensemble detector
19. **Training pipeline** - dataset, training, evaluation
20. **Notebooks** - interactive exploration

---

## SUCCESS CRITERIA

The software is complete when:

1. **SNN Simulation**
   - [ ] LIF neurons match analytical solution
   - [ ] STDP produces expected learning curves
   - [ ] Quasicrystal topology generates valid graphs
   - [ ] Full network runs at >1000 timesteps/second

2. **Automata Module**
   - [ ] DFA/NFA operations verified against known automata
   - [ ] L* learns simple protocols correctly
   - [ ] LTL monitoring detects known violation patterns

3. **Category Module**
   - [ ] Protocol categories model TCP correctly
   - [ ] Functor checking identifies protocol violations
   - [ ] Sheaf cohomology computes expected values

4. **Topology Module**
   - [ ] VR complex matches reference implementation
   - [ ] Persistent homology verified against GUDHI
   - [ ] TDA features distinguish attack classes

5. **Detection Performance**
   - [ ] >85% accuracy on NSL-KDD
   - [ ] >90% accuracy on UNSW-NB15
   - [ ] <1% false positive rate
   - [ ] Real-time capable (>1000 packets/second)

6. **Golden Ratio Hypothesis**
   - [ ] Controlled experiments comparing golden vs standard
   - [ ] Statistical significance testing
   - [ ] Documentation of findings (positive or negative)

---

## NOTES FOR CLAUDE CODE

When implementing this project:

1. **Start simple**: Build minimal working versions first, then add complexity
2. **Test continuously**: Write tests alongside implementation
3. **Document mathematical basis**: Include references to papers and formulas
4. **Think hardware**: Design for eventual FPGA synthesis
5. **Profile early**: Identify bottlenecks before optimization
6. **Use type hints**: Full typing for all functions
7. **No premature optimization**: Correctness first, speed second
8. **Validate against references**: Compare with established libraries
9. **Keep golden ratio optional**: Allow toggling phi-based features for comparison
10. **Log extensively**: Track experiments for reproducibility

---

## COMMANDS TO START

```bash
# Create project structure
mkdir -p gnsp/{core,snn,automata,category,topology,algebra,network,detection,training,visualization}
mkdir -p tests/{test_core,test_snn,test_automata,test_category,test_topology,test_algebra,test_integration}
mkdir -p notebooks configs data/{raw,processed,synthetic} models/{checkpoints,trained} scripts docs

# Initialize Python package
touch gnsp/__init__.py
touch gnsp/{core,snn,automata,category,topology,algebra,network,detection,training,visualization}/__init__.py

# Create requirements.txt and pyproject.toml
# ... (content as specified above)

# Install in development mode
pip install -e ".[dev]"

# Run initial tests
pytest tests/ -v

# Start with core module
# Implement gnsp/constants.py first
# Then gnsp/core/fixed_point.py, golden.py, fibonacci.py
# Write tests as you go
```

---

## FINAL CHECKLIST

Before considering Phase N complete:

- [ ] All functions have docstrings
- [ ] All functions have type hints
- [ ] All functions have unit tests
- [ ] Tests pass with >90% coverage
- [ ] Code formatted with black
- [ ] Type checked with mypy
- [ ] No linting errors (ruff)
- [ ] Example notebook demonstrating usage
- [ ] Performance benchmarked

---

This prompt provides comprehensive specifications for building the GNSP software simulation platform. The implementation should produce a research-grade codebase suitable for:

1. Algorithm validation before hardware deployment
2. Academic publication
3. DARPA/SBIR proposal support
4. Patent documentation
5. Eventually: FPGA synthesis targeting

Good luck, and remember: the golden ratio is not just a number, it is the signature of optimal organization in complex systems.
