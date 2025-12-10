"""
Core mathematical utilities for GNSP.

This module provides:
- Fixed-point arithmetic for hardware compatibility
- Golden ratio utilities and functions
- Fibonacci sequence operations
- Quasicrystalline lattice generation
"""

from gnsp.core.fixed_point import (
    QFormat,
    FixedPoint,
    Q8_8,
    Q16_16,
    Q1_15,
    Q4_12,
    Q8_24,
    float_to_fixed,
    fixed_to_float,
    fixed_multiply,
    fixed_mac,
    quantize_to_fixed,
)

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

from gnsp.core.quasicrystal import (
    Vertex,
    fibonacci_chain,
    fibonacci_lattice_2d,
    penrose_vertices,
    ammann_beenker_vertices,
    cut_and_project_1d,
    voronoi_neighbors,
    delaunay_neighbors,
    golden_spiral_lattice,
    spherical_fibonacci_lattice,
    compute_packing_density,
)

__all__ = [
    # Fixed-point
    "QFormat",
    "FixedPoint",
    "Q8_8",
    "Q16_16",
    "Q1_15",
    "Q4_12",
    "Q8_24",
    "float_to_fixed",
    "fixed_to_float",
    "fixed_multiply",
    "fixed_mac",
    "quantize_to_fixed",
    # Golden
    "golden_decay",
    "golden_growth",
    "golden_threshold",
    "fibonacci_sequence",
    "fibonacci_generator",
    "nearest_fibonacci",
    "golden_spiral_points",
    "golden_spiral_points_3d",
    "golden_ratio_weights",
    "is_near_golden",
    "golden_power_decomposition",
    "golden_matrix",
    "lucas_sequence",
    "verify_golden_eigenvalues",
    "phi_continued_fraction",
    "golden_interpolate",
    "golden_section_search",
    # Fibonacci
    "fib",
    "fib_binet",
    "fibonacci_up_to",
    "is_fibonacci",
    "fibonacci_index",
    "zeckendorf",
    "fibonacci_encode",
    "fibonacci_offsets",
    "fibonacci_time_constants",
    "fibonacci_connectivity_pattern",
    "fibonacci_decay_lookup",
    "fibonacci_distance_matrix",
    "fibonacci_lattice_1d",
    "tribonacci",
    "fibonacci_ratios",
    "generalized_fibonacci",
    # Quasicrystal
    "Vertex",
    "fibonacci_chain",
    "fibonacci_lattice_2d",
    "penrose_vertices",
    "ammann_beenker_vertices",
    "cut_and_project_1d",
    "voronoi_neighbors",
    "delaunay_neighbors",
    "golden_spiral_lattice",
    "spherical_fibonacci_lattice",
    "compute_packing_density",
]
