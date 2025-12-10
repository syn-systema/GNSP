"""Network topology generators using quasicrystalline and golden spiral patterns.

This module provides topology generation for SNNs with:
- Quasicrystalline lattice-based connectivity
- Golden spiral neuron positioning
- Fibonacci-spaced hierarchical layers
- Distance-dependent connection probabilities
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from gnsp.constants import (
    PHI,
    PHI_INV,
    GOLDEN_ANGLE,
    FIBONACCI,
)
from gnsp.core.golden import golden_spiral_points, golden_spiral_points_3d
from gnsp.core.fibonacci import fibonacci_connectivity_pattern
from gnsp.core.quasicrystal import (
    fibonacci_lattice_2d,
    voronoi_neighbors,
    delaunay_neighbors,
    spherical_fibonacci_lattice,
)


@dataclass
class TopologyParams:
    """Parameters for topology generation.

    Attributes:
        connection_radius: Maximum connection distance
        connection_probability: Base connection probability
        use_fibonacci_distances: Only connect at Fibonacci distances
        bidirectional: Create bidirectional connections
        self_connections: Allow self-connections
    """

    connection_radius: float = 10.0
    connection_probability: float = 0.3
    use_fibonacci_distances: bool = True
    bidirectional: bool = False
    self_connections: bool = False


class TopologyBase(ABC):
    """Abstract base class for topology generators."""

    @abstractmethod
    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate connectivity pattern.

        Args:
            n_neurons: Number of neurons

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        pass

    @abstractmethod
    def get_positions(self) -> NDArray:
        """Get neuron positions.

        Returns:
            Position array (n_neurons, dim)
        """
        pass


class QuasicrystalTopology(TopologyBase):
    """Topology based on quasicrystalline lattice.

    Positions neurons on a 2D Fibonacci lattice and connects
    them based on Voronoi/Delaunay neighbors or Fibonacci distances.
    """

    def __init__(
        self,
        params: Optional[TopologyParams] = None,
        use_voronoi: bool = True,
    ) -> None:
        """Initialize quasicrystal topology.

        Args:
            params: Topology parameters
            use_voronoi: Use Voronoi neighbors (vs Delaunay)
        """
        self.params = params or TopologyParams()
        self.use_voronoi = use_voronoi
        self._positions: Optional[NDArray] = None
        self._n_neurons = 0

    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate connections using quasicrystalline neighbors.

        Args:
            n_neurons: Number of neurons

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        self._n_neurons = n_neurons

        # Generate Fibonacci lattice positions
        self._positions = fibonacci_lattice_2d(n_neurons)

        # Get neighbors
        if self.use_voronoi:
            neighbors = voronoi_neighbors(self._positions)
        else:
            neighbors = delaunay_neighbors(self._positions)

        # Build connection lists
        pre_list = []
        post_list = []

        for i, neighbor_set in enumerate(neighbors):
            for j in neighbor_set:
                if not self.params.self_connections and i == j:
                    continue

                # Apply connection probability
                if np.random.random() < self.params.connection_probability:
                    pre_list.append(i)
                    post_list.append(j)

                    if self.params.bidirectional:
                        pre_list.append(j)
                        post_list.append(i)

        return (
            np.array(pre_list, dtype=np.int32),
            np.array(post_list, dtype=np.int32),
        )

    def get_positions(self) -> NDArray:
        """Get neuron positions on Fibonacci lattice."""
        if self._positions is None:
            raise ValueError("Must call generate_connections first")
        return self._positions.copy()


class GoldenSpiralTopology(TopologyBase):
    """Topology using golden spiral neuron placement.

    Neurons are positioned along a golden spiral, with connections
    based on angular and radial distance.
    """

    def __init__(
        self,
        params: Optional[TopologyParams] = None,
        scale: float = 1.0,
    ) -> None:
        """Initialize golden spiral topology.

        Args:
            params: Topology parameters
            scale: Spiral scale factor
        """
        self.params = params or TopologyParams()
        self.scale = scale
        self._positions: Optional[NDArray] = None
        self._n_neurons = 0

    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate connections along golden spiral.

        Args:
            n_neurons: Number of neurons

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        self._n_neurons = n_neurons

        # Generate spiral positions
        self._positions = golden_spiral_points(n_neurons, scale=self.scale)

        # Build KD-tree for efficient neighbor search
        tree = KDTree(self._positions)

        pre_list = []
        post_list = []

        # Find neighbors within radius
        for i in range(n_neurons):
            neighbors = tree.query_ball_point(
                self._positions[i],
                self.params.connection_radius,
            )

            for j in neighbors:
                if not self.params.self_connections and i == j:
                    continue

                # Check Fibonacci distance constraint if enabled
                if self.params.use_fibonacci_distances:
                    dist = abs(i - j)
                    if dist not in FIBONACCI[:10]:
                        continue

                # Apply connection probability
                if np.random.random() < self.params.connection_probability:
                    pre_list.append(i)
                    post_list.append(j)

        return (
            np.array(pre_list, dtype=np.int32),
            np.array(post_list, dtype=np.int32),
        )

    def get_positions(self) -> NDArray:
        """Get neuron positions on golden spiral."""
        if self._positions is None:
            raise ValueError("Must call generate_connections first")
        return self._positions.copy()


class SphericalFibonacciTopology(TopologyBase):
    """Topology using spherical Fibonacci lattice.

    Neurons are distributed uniformly on a sphere using
    the Fibonacci lattice, optimal for 3D networks.
    """

    def __init__(
        self,
        params: Optional[TopologyParams] = None,
        radius: float = 1.0,
    ) -> None:
        """Initialize spherical topology.

        Args:
            params: Topology parameters
            radius: Sphere radius
        """
        self.params = params or TopologyParams()
        self.radius = radius
        self._positions: Optional[NDArray] = None
        self._n_neurons = 0

    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate connections on spherical lattice.

        Args:
            n_neurons: Number of neurons

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        self._n_neurons = n_neurons

        # Generate spherical Fibonacci points
        self._positions = spherical_fibonacci_lattice(n_neurons, self.radius)

        # Build KD-tree
        tree = KDTree(self._positions)

        pre_list = []
        post_list = []

        # Connect neighbors on sphere
        for i in range(n_neurons):
            # Find k nearest neighbors
            k = min(int(n_neurons * self.params.connection_probability * 2), n_neurons)
            _, indices = tree.query(self._positions[i], k=k)

            for j in indices:
                if not self.params.self_connections and i == j:
                    continue

                # Distance on sphere
                dist = np.linalg.norm(self._positions[i] - self._positions[j])
                if dist > self.params.connection_radius:
                    continue

                if np.random.random() < self.params.connection_probability:
                    pre_list.append(i)
                    post_list.append(int(j))

        return (
            np.array(pre_list, dtype=np.int32),
            np.array(post_list, dtype=np.int32),
        )

    def get_positions(self) -> NDArray:
        """Get neuron positions on sphere."""
        if self._positions is None:
            raise ValueError("Must call generate_connections first")
        return self._positions.copy()


class HierarchicalTopology(TopologyBase):
    """Hierarchical topology with Fibonacci-sized layers.

    Creates a multi-layer network where layer sizes follow
    Fibonacci numbers and connections use golden ratio scaling.
    """

    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        inter_layer_density: float = 0.3,
        intra_layer_density: float = 0.1,
        forward_only: bool = False,
    ) -> None:
        """Initialize hierarchical topology.

        Args:
            layer_sizes: Neurons per layer (default: Fibonacci)
            inter_layer_density: Connection density between layers
            intra_layer_density: Connection density within layers
            forward_only: Only forward inter-layer connections
        """
        if layer_sizes is None:
            # Use Fibonacci layer sizes: 8, 13, 21, 13, 8
            layer_sizes = [8, 13, 21, 13, 8]

        self.layer_sizes = layer_sizes
        self.inter_layer_density = inter_layer_density
        self.intra_layer_density = intra_layer_density
        self.forward_only = forward_only
        self._positions: Optional[NDArray] = None
        self._n_neurons = sum(layer_sizes)
        self._layer_indices: List[Tuple[int, int]] = []

    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate hierarchical connections.

        Note: n_neurons is ignored; use layer_sizes instead.

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        # Compute layer index ranges
        self._layer_indices = []
        start = 0
        for size in self.layer_sizes:
            self._layer_indices.append((start, start + size))
            start += size

        self._n_neurons = start

        # Generate 2D positions for visualization
        self._positions = np.zeros((self._n_neurons, 2), dtype=np.float32)
        for layer_idx, (start, end) in enumerate(self._layer_indices):
            layer_size = end - start
            # Arrange layer neurons in a row
            x = np.linspace(0, 1, layer_size)
            y = np.full(layer_size, layer_idx * PHI)
            self._positions[start:end, 0] = x
            self._positions[start:end, 1] = y

        pre_list = []
        post_list = []

        # Intra-layer connections
        for start, end in self._layer_indices:
            layer_size = end - start
            for i in range(start, end):
                for j in range(start, end):
                    if i == j:
                        continue
                    if np.random.random() < self.intra_layer_density:
                        pre_list.append(i)
                        post_list.append(j)

        # Inter-layer connections (between adjacent layers)
        for layer_idx in range(len(self.layer_sizes) - 1):
            pre_start, pre_end = self._layer_indices[layer_idx]
            post_start, post_end = self._layer_indices[layer_idx + 1]

            # Forward connections
            for i in range(pre_start, pre_end):
                for j in range(post_start, post_end):
                    if np.random.random() < self.inter_layer_density:
                        pre_list.append(i)
                        post_list.append(j)

            # Backward connections (if not forward only)
            if not self.forward_only:
                # Use weaker backward connectivity (scaled by 1/phi)
                backward_density = self.inter_layer_density * PHI_INV
                for i in range(post_start, post_end):
                    for j in range(pre_start, pre_end):
                        if np.random.random() < backward_density:
                            pre_list.append(i)
                            post_list.append(j)

        return (
            np.array(pre_list, dtype=np.int32),
            np.array(post_list, dtype=np.int32),
        )

    def get_positions(self) -> NDArray:
        """Get neuron positions."""
        if self._positions is None:
            raise ValueError("Must call generate_connections first")
        return self._positions.copy()

    def get_layer_indices(self) -> List[Tuple[int, int]]:
        """Get index ranges for each layer.

        Returns:
            List of (start, end) tuples for each layer
        """
        return self._layer_indices.copy()


class SmallWorldTopology(TopologyBase):
    """Small-world topology with golden ratio rewiring.

    Creates a small-world network using Watts-Strogatz model
    with rewiring probability scaled by golden ratio.
    """

    def __init__(
        self,
        k: int = 4,
        rewire_prob: float = PHI_INV,  # ~0.618
    ) -> None:
        """Initialize small-world topology.

        Args:
            k: Number of nearest neighbors in ring lattice
            rewire_prob: Probability of rewiring each edge
        """
        self.k = k
        self.rewire_prob = rewire_prob
        self._positions: Optional[NDArray] = None
        self._n_neurons = 0

    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate small-world connections.

        Args:
            n_neurons: Number of neurons

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        self._n_neurons = n_neurons

        # Position on a ring
        angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
        self._positions = np.column_stack([np.cos(angles), np.sin(angles)])

        # Build regular ring lattice
        edges = set()
        for i in range(n_neurons):
            for j in range(1, self.k // 2 + 1):
                # Connect to k/2 neighbors on each side
                target = (i + j) % n_neurons
                edges.add((i, target))
                edges.add((target, i))

        # Rewire with probability p
        rewired_edges = set()
        for i, j in edges:
            if np.random.random() < self.rewire_prob:
                # Rewire to random target
                new_target = np.random.randint(0, n_neurons)
                while new_target == i or (i, new_target) in rewired_edges:
                    new_target = np.random.randint(0, n_neurons)
                rewired_edges.add((i, new_target))
            else:
                rewired_edges.add((i, j))

        pre_list = [e[0] for e in rewired_edges]
        post_list = [e[1] for e in rewired_edges]

        return (
            np.array(pre_list, dtype=np.int32),
            np.array(post_list, dtype=np.int32),
        )

    def get_positions(self) -> NDArray:
        """Get neuron positions on ring."""
        if self._positions is None:
            raise ValueError("Must call generate_connections first")
        return self._positions.copy()


class FibonacciConnectivityTopology(TopologyBase):
    """Topology using Fibonacci connectivity pattern.

    Neurons connect to others at Fibonacci distances (1, 1, 2, 3, 5, 8, ...),
    creating long-range shortcuts without full connectivity.
    """

    def __init__(
        self,
        max_fib_idx: int = 8,
        bidirectional: bool = True,
    ) -> None:
        """Initialize Fibonacci connectivity topology.

        Args:
            max_fib_idx: Maximum Fibonacci index to use
            bidirectional: Create bidirectional connections
        """
        self.max_fib_idx = max_fib_idx
        self.bidirectional = bidirectional
        self._positions: Optional[NDArray] = None
        self._n_neurons = 0

    def generate_connections(
        self,
        n_neurons: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Generate Fibonacci-distance connections.

        Args:
            n_neurons: Number of neurons

        Returns:
            Tuple of (pre_indices, post_indices)
        """
        self._n_neurons = n_neurons

        # Linear positions
        self._positions = np.column_stack([
            np.arange(n_neurons, dtype=np.float32),
            np.zeros(n_neurons, dtype=np.float32)
        ])

        # Use core function
        connections = fibonacci_connectivity_pattern(n_neurons)

        pre_list = [c[0] for c in connections]
        post_list = [c[1] for c in connections]

        if self.bidirectional:
            # Add reverse connections
            pre_list.extend([c[1] for c in connections])
            post_list.extend([c[0] for c in connections])

        return (
            np.array(pre_list, dtype=np.int32),
            np.array(post_list, dtype=np.int32),
        )

    def get_positions(self) -> NDArray:
        """Get neuron positions (linear)."""
        if self._positions is None:
            raise ValueError("Must call generate_connections first")
        return self._positions.copy()


def compute_connection_statistics(
    pre_indices: NDArray[np.int32],
    post_indices: NDArray[np.int32],
    n_neurons: int,
) -> Dict[str, float]:
    """Compute statistics about connectivity.

    Args:
        pre_indices: Presynaptic indices
        post_indices: Postsynaptic indices
        n_neurons: Total number of neurons

    Returns:
        Dictionary with connectivity statistics
    """
    n_connections = len(pre_indices)
    max_connections = n_neurons * n_neurons

    # In-degree and out-degree
    in_degree = np.zeros(n_neurons, dtype=np.int32)
    out_degree = np.zeros(n_neurons, dtype=np.int32)

    for pre, post in zip(pre_indices, post_indices):
        out_degree[pre] += 1
        in_degree[post] += 1

    return {
        "n_connections": float(n_connections),
        "density": n_connections / max_connections,
        "mean_in_degree": float(np.mean(in_degree)),
        "mean_out_degree": float(np.mean(out_degree)),
        "std_in_degree": float(np.std(in_degree)),
        "std_out_degree": float(np.std(out_degree)),
        "max_in_degree": float(np.max(in_degree)),
        "max_out_degree": float(np.max(out_degree)),
    }
