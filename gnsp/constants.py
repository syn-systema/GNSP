"""
Core mathematical constants for GNSP.

The golden ratio appears throughout as the fundamental constant,
following insights from quantum gravity research showing phi
as the dominant non-integer eigenvalue of binary matrices.

Mathematical Properties:
    - phi^2 = phi + 1
    - phi * phi_inv = 1
    - phi_inv = phi - 1
    - phi = 1 + 1/phi (continued fraction)
    - lim(F_n+1 / F_n) = phi as n -> infinity
"""

import math
from typing import Tuple

# =============================================================================
# Golden Ratio and Related Constants
# =============================================================================

PHI: float = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INV: float = 1 / PHI              # 0.618033988749895 = phi - 1
PHI_SQ: float = PHI ** 2              # 2.618033988749895 = phi + 1
PHI_INV_SQ: float = PHI_INV ** 2      # 0.381966011250105
PHI_CUBE: float = PHI ** 3            # 4.236067977499790
PHI_INV_CUBE: float = PHI_INV ** 3    # 0.236067977499790

# Golden ratio in degrees (for angular operations)
GOLDEN_ANGLE_DEG: float = 360 / (PHI ** 2)  # ~137.5077... degrees

# =============================================================================
# Fixed-point Representations
# =============================================================================

# Q8.8 format: 8 integer bits, 8 fractional bits (16-bit total)
# Conversion: fixed = int(float_val * 256)
PHI_Q8_8: int = 0x019E        # 1.617 in Q8.8 (actual: 414/256 = 1.6171875)
PHI_INV_Q8_8: int = 0x009E    # 0.617 in Q8.8 (actual: 158/256 = 0.6171875)
PHI_SQ_Q8_8: int = 0x029E     # 2.617 in Q8.8 (actual: 670/256 = 2.6171875)

# Q16.16 format: 16 integer bits, 16 fractional bits (32-bit total)
# Conversion: fixed = int(float_val * 65536)
PHI_Q16_16: int = 0x00019E37      # 1.6180... in Q16.16
PHI_INV_Q16_16: int = 0x00009E37  # 0.6180... in Q16.16
PHI_SQ_Q16_16: int = 0x00029E37   # 2.6180... in Q16.16

# Q1.15 format: 1 sign bit, 15 fractional bits (for normalized values in [-1, 1))
PHI_INV_Q1_15: int = 0x4F1B       # 0.618... in Q1.15

# =============================================================================
# Fibonacci Sequence
# =============================================================================

# Pre-computed Fibonacci numbers (sufficient for most STDP and connectivity uses)
FIBONACCI: Tuple[int, ...] = (
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
    89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765
)

# Extended Fibonacci for large network indices
FIBONACCI_EXTENDED: Tuple[int, ...] = FIBONACCI + (
    10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040
)

# =============================================================================
# Angular Constants
# =============================================================================

# Golden angle (radians) - used for quasicrystal generation and spiral layouts
GOLDEN_ANGLE: float = 2 * math.pi / (PHI ** 2)  # ~2.399963... rad (~137.5 deg)

# Cabibbo angle from E8 lattice theory: arctan(1/phi^3)
# This appears in particle physics and connects to E8 lattice eigenvalues
CABIBBO_ANGLE: float = math.atan(1 / (PHI ** 3))  # ~0.2318... rad (~13.28 deg)

# =============================================================================
# Detection Thresholds
# =============================================================================

# Thresholds based on golden ratio powers for hierarchical anomaly detection
THRESHOLD_LOW: float = PHI_INV ** 3       # ~0.236 (subtle anomaly)
THRESHOLD_MID: float = PHI_INV ** 2       # ~0.382 (moderate anomaly)
THRESHOLD_HIGH: float = PHI_INV           # ~0.618 (significant anomaly)
THRESHOLD_CRITICAL: float = 1.0           # (critical anomaly)

# =============================================================================
# SNN Neuron Parameters
# =============================================================================

# LIF neuron defaults using golden ratio dynamics
DEFAULT_THRESHOLD: float = PHI            # Spike threshold (1.618)
DEFAULT_RESET: float = PHI_INV            # Reset potential (0.618)
DEFAULT_LEAK: float = PHI_INV_SQ          # Leak rate per timestep (0.382)
DEFAULT_REFRACTORY: int = 2               # Refractory period in timesteps

# Membrane time constant (derived from leak)
# tau = -1 / ln(1 - leak) for exponential decay model
DEFAULT_TAU_MEMBRANE: float = -1 / math.log(1 - DEFAULT_LEAK)  # ~2.236 timesteps

# =============================================================================
# Weight Quantization (Golden Ratio Ladder)
# =============================================================================

# 9-level weight quantization based on golden ratio powers
# Provides optimal dynamic range with hardware-friendly discrete levels
WEIGHT_LEVELS: Tuple[float, ...] = (
    -PHI_SQ,      # -2.618 (strong inhibitory)
    -PHI,         # -1.618 (moderate inhibitory)
    -1.0,         # -1.000 (unit inhibitory)
    -PHI_INV,     # -0.618 (weak inhibitory)
    0.0,          #  0.000 (no connection)
    PHI_INV,      #  0.618 (weak excitatory)
    1.0,          #  1.000 (unit excitatory)
    PHI,          #  1.618 (moderate excitatory)
    PHI_SQ,       #  2.618 (strong excitatory)
)

# Number of discrete weight levels
N_WEIGHT_LEVELS: int = len(WEIGHT_LEVELS)

# =============================================================================
# STDP Time Constants (Fibonacci-based)
# =============================================================================

# STDP time windows using Fibonacci sequence (in milliseconds/timesteps)
# These provide naturally scaled learning windows at multiple timescales
STDP_TAU: Tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34)

# Default indices into STDP_TAU for LTP and LTD
STDP_TAU_PLUS_IDX: int = 4   # tau_plus = 5 (LTP window)
STDP_TAU_MINUS_IDX: int = 5  # tau_minus = 8 (LTD window)

# STDP learning amplitudes
STDP_A_PLUS: float = 0.1     # LTP amplitude
STDP_A_MINUS: float = 0.12   # LTD amplitude (slightly stronger for stability)

# =============================================================================
# Network Topology Constants
# =============================================================================

# Default connection distances for quasicrystal topology (golden ratio multiples)
CONNECTION_DISTANCES: Tuple[float, ...] = (1.0, PHI, PHI_SQ)

# Maximum synaptic delay (Fibonacci: 8)
MAX_SYNAPTIC_DELAY: int = 8

# =============================================================================
# Ensemble Detection Weights
# =============================================================================

# Default weights for ensemble detector combination
ENSEMBLE_WEIGHT_SNN: float = 0.4       # SNN detector (40%)
ENSEMBLE_WEIGHT_AUTOMATA: float = 0.2  # Automata detector (20%)
ENSEMBLE_WEIGHT_CATEGORY: float = 0.2  # Category theory detector (20%)
ENSEMBLE_WEIGHT_TDA: float = 0.2       # TDA detector (20%)
