"""Synaptic connections with golden ratio weight quantization.

This module implements synaptic connections for the SNN with:
- 9-level golden ratio weight quantization
- Sparse connectivity using CSR matrices
- Efficient synaptic current computation
- Delay support for realistic propagation
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from gnsp.constants import (
    WEIGHT_LEVELS,
    N_WEIGHT_LEVELS,
    PHI,
    PHI_INV,
)


class WeightLevel(IntEnum):
    """Enumeration of 9 golden ratio weight levels.

    Weight values are symmetric around zero:
    -phi^2, -phi, -1, -1/phi, 0, 1/phi, 1, phi, phi^2
    """

    NEG_PHI_SQ = 0    # -2.618
    NEG_PHI = 1       # -1.618
    NEG_ONE = 2       # -1.0
    NEG_PHI_INV = 3   # -0.618
    ZERO = 4          # 0.0
    PHI_INV = 5       # 0.618
    ONE = 6           # 1.0
    PHI = 7           # 1.618
    PHI_SQ = 8        # 2.618


@dataclass
class SynapseParams:
    """Parameters for synaptic connections.

    Attributes:
        delay_min: Minimum synaptic delay (timesteps)
        delay_max: Maximum synaptic delay (timesteps)
        use_quantized: Whether to use quantized weights
        sparse_threshold: Density below which to use sparse storage
    """

    delay_min: int = 1
    delay_max: int = 8  # Fibonacci: 1, 2, 3, 5, 8
    use_quantized: bool = True
    sparse_threshold: float = 0.3

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.delay_min < 1:
            raise ValueError("delay_min must be >= 1")
        if self.delay_max < self.delay_min:
            raise ValueError("delay_max must be >= delay_min")
        if not 0 < self.sparse_threshold <= 1:
            raise ValueError("sparse_threshold must be in (0, 1]")


def quantize_weight(weight: float) -> int:
    """Quantize a continuous weight to nearest golden level.

    Args:
        weight: Continuous weight value

    Returns:
        Index into WEIGHT_LEVELS (0-8)
    """
    # Find nearest weight level
    min_dist = float("inf")
    best_level = WeightLevel.ZERO

    for level in WeightLevel:
        dist = abs(weight - WEIGHT_LEVELS[level])
        if dist < min_dist:
            min_dist = dist
            best_level = level

    return int(best_level)


def quantize_weight_array(weights: NDArray) -> NDArray[np.int8]:
    """Quantize array of weights to golden levels.

    Args:
        weights: Array of continuous weights

    Returns:
        Array of weight level indices (int8)
    """
    levels = np.array(WEIGHT_LEVELS)
    # Compute distances to all levels
    distances = np.abs(weights[..., np.newaxis] - levels)
    # Find index of minimum distance
    return np.argmin(distances, axis=-1).astype(np.int8)


def dequantize_weight(level: int) -> float:
    """Convert weight level index back to float value.

    Args:
        level: Weight level index (0-8)

    Returns:
        Float weight value
    """
    if not 0 <= level < N_WEIGHT_LEVELS:
        raise ValueError(f"level must be in [0, {N_WEIGHT_LEVELS})")
    return WEIGHT_LEVELS[level]


def dequantize_weight_array(levels: NDArray[np.int8]) -> NDArray[np.float32]:
    """Convert weight level array to float values.

    Args:
        levels: Array of weight level indices

    Returns:
        Array of float weight values
    """
    weight_lut = np.array(WEIGHT_LEVELS, dtype=np.float32)
    return weight_lut[levels]


class SynapseArray:
    """Synaptic connections between neuron populations.

    Implements efficient synaptic current computation with:
    - Dense or sparse weight storage (automatic selection)
    - 9-level golden ratio weight quantization
    - Synaptic delays with circular buffer

    Attributes:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        params: Synapse parameters
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[SynapseParams] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize synapse array.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: Synapse parameters
            dtype: Data type for computations
        """
        if n_pre <= 0 or n_post <= 0:
            raise ValueError("n_pre and n_post must be positive")

        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or SynapseParams()
        self.dtype = dtype

        # Weight storage (initially empty)
        self._weights: Optional[NDArray] = None
        self._weight_levels: Optional[NDArray[np.int8]] = None
        self._is_sparse = False
        self._sparse_weights: Optional[sparse.csr_matrix] = None

        # Delay buffer for spike propagation
        self._delay_buffer: Optional[NDArray] = None
        self._delays: Optional[NDArray[np.int8]] = None
        self._buffer_idx = 0

    def initialize_dense(
        self,
        weights: Optional[NDArray] = None,
        delays: Optional[NDArray] = None,
    ) -> None:
        """Initialize with dense connectivity.

        Args:
            weights: Weight matrix (n_pre, n_post), random if None
            delays: Delay matrix (n_pre, n_post), default if None
        """
        if weights is None:
            # Initialize with small random weights centered at 0
            weights = np.random.randn(self.n_pre, self.n_post).astype(self.dtype)
            weights *= PHI_INV  # Scale by 1/phi

        if weights.shape != (self.n_pre, self.n_post):
            raise ValueError(
                f"weights shape {weights.shape} doesn't match ({self.n_pre}, {self.n_post})"
            )

        # Store weights
        self._weights = weights.astype(self.dtype)
        self._is_sparse = False

        if self.params.use_quantized:
            self._weight_levels = quantize_weight_array(self._weights)
            self._weights = dequantize_weight_array(self._weight_levels)

        # Initialize delays
        if delays is None:
            delays = np.ones((self.n_pre, self.n_post), dtype=np.int8)
        self._delays = delays.astype(np.int8)

        # Initialize delay buffer
        max_delay = self.params.delay_max + 1
        self._delay_buffer = np.zeros(
            (max_delay, self.n_post), dtype=self.dtype
        )
        self._buffer_idx = 0

    def initialize_sparse(
        self,
        pre_indices: NDArray[np.int32],
        post_indices: NDArray[np.int32],
        weights: NDArray,
        delays: Optional[NDArray] = None,
    ) -> None:
        """Initialize with sparse connectivity.

        Args:
            pre_indices: Presynaptic neuron indices for each synapse
            post_indices: Postsynaptic neuron indices for each synapse
            weights: Weight for each synapse
            delays: Delay for each synapse (default: 1)
        """
        n_synapses = len(pre_indices)
        if len(post_indices) != n_synapses or len(weights) != n_synapses:
            raise ValueError("pre_indices, post_indices, and weights must have same length")

        # Quantize if needed
        if self.params.use_quantized:
            levels = quantize_weight_array(weights)
            weights = dequantize_weight_array(levels)

        # Create sparse matrix
        self._sparse_weights = sparse.csr_matrix(
            (weights, (pre_indices, post_indices)),
            shape=(self.n_pre, self.n_post),
            dtype=self.dtype,
        )
        self._is_sparse = True
        self._weights = None

        # Store delays as sparse array
        if delays is None:
            delays = np.ones(n_synapses, dtype=np.int8)
        self._sparse_delays = sparse.csr_matrix(
            (delays, (pre_indices, post_indices)),
            shape=(self.n_pre, self.n_post),
            dtype=np.int8,
        )

        # Initialize delay buffer
        max_delay = self.params.delay_max + 1
        self._delay_buffer = np.zeros(
            (max_delay, self.n_post), dtype=self.dtype
        )
        self._buffer_idx = 0

    def initialize_random(
        self,
        density: float = 0.1,
        weight_std: float = 1.0,
    ) -> None:
        """Initialize with random sparse connectivity.

        Args:
            density: Connection probability (0 to 1)
            weight_std: Standard deviation of initial weights
        """
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")

        # Determine sparse vs dense storage
        use_sparse = density < self.params.sparse_threshold

        if use_sparse:
            # Generate random connections
            n_synapses = int(density * self.n_pre * self.n_post)
            pre_indices = np.random.randint(0, self.n_pre, n_synapses, dtype=np.int32)
            post_indices = np.random.randint(0, self.n_post, n_synapses, dtype=np.int32)

            # Random weights
            weights = np.random.randn(n_synapses).astype(self.dtype)
            weights *= weight_std * PHI_INV

            # Random delays (Fibonacci-based)
            delays = np.random.choice(
                [1, 2, 3, 5, 8],
                size=n_synapses,
                p=[0.4, 0.25, 0.2, 0.1, 0.05],  # Favor shorter delays
            ).astype(np.int8)

            self.initialize_sparse(pre_indices, post_indices, weights, delays)
        else:
            # Dense random weights
            mask = np.random.random((self.n_pre, self.n_post)) < density
            weights = np.random.randn(self.n_pre, self.n_post).astype(self.dtype)
            weights *= weight_std * PHI_INV
            weights *= mask  # Zero out unconnected

            self.initialize_dense(weights)

    def propagate(self, pre_spikes: NDArray[np.bool_]) -> NDArray:
        """Compute postsynaptic currents from presynaptic spikes.

        This method handles synaptic delays by using a circular buffer.

        Args:
            pre_spikes: Boolean array of presynaptic spikes (n_pre,)

        Returns:
            Postsynaptic currents (n_post,)
        """
        if pre_spikes.shape[0] != self.n_pre:
            raise ValueError(
                f"pre_spikes shape {pre_spikes.shape} doesn't match n_pre {self.n_pre}"
            )

        # Get current output from delay buffer
        output = self._delay_buffer[self._buffer_idx].copy()

        # Clear current buffer slot
        self._delay_buffer[self._buffer_idx] = 0

        # Compute new contributions and add to buffer
        if self._is_sparse:
            # Sparse computation
            spike_indices = np.where(pre_spikes)[0]
            for idx in spike_indices:
                # Get row of sparse matrix
                row_start = self._sparse_weights.indptr[idx]
                row_end = self._sparse_weights.indptr[idx + 1]

                for j in range(row_start, row_end):
                    post_idx = self._sparse_weights.indices[j]
                    weight = self._sparse_weights.data[j]
                    delay = self._sparse_delays.data[j]

                    # Add to appropriate buffer slot
                    buffer_slot = (self._buffer_idx + delay) % len(self._delay_buffer)
                    self._delay_buffer[buffer_slot, post_idx] += weight
        else:
            # Dense computation
            # For unit delay, direct matrix multiplication
            if self._delays is None or np.all(self._delays == 1):
                contribution = pre_spikes.astype(self.dtype) @ self._weights
                next_slot = (self._buffer_idx + 1) % len(self._delay_buffer)
                self._delay_buffer[next_slot] += contribution
            else:
                # Handle variable delays
                spike_indices = np.where(pre_spikes)[0]
                for idx in spike_indices:
                    for post_idx in range(self.n_post):
                        weight = self._weights[idx, post_idx]
                        if weight != 0:
                            delay = self._delays[idx, post_idx]
                            buffer_slot = (self._buffer_idx + delay) % len(self._delay_buffer)
                            self._delay_buffer[buffer_slot, post_idx] += weight

        # Advance buffer index
        self._buffer_idx = (self._buffer_idx + 1) % len(self._delay_buffer)

        return output

    def propagate_instant(self, pre_spikes: NDArray[np.bool_]) -> NDArray:
        """Compute postsynaptic currents without delays.

        Faster version for networks without synaptic delays.

        Args:
            pre_spikes: Boolean array of presynaptic spikes (n_pre,)

        Returns:
            Postsynaptic currents (n_post,)
        """
        spike_vec = pre_spikes.astype(self.dtype)

        if self._is_sparse:
            return np.asarray(self._sparse_weights.T @ spike_vec).flatten()
        else:
            return spike_vec @ self._weights

    def get_weights(self) -> NDArray:
        """Get weight matrix as dense array.

        Returns:
            Weight matrix (n_pre, n_post)
        """
        if self._is_sparse:
            return self._sparse_weights.toarray()
        return self._weights.copy() if self._weights is not None else np.zeros(
            (self.n_pre, self.n_post), dtype=self.dtype
        )

    def set_weights(self, weights: NDArray) -> None:
        """Set weight matrix.

        Args:
            weights: New weight matrix (n_pre, n_post)
        """
        if weights.shape != (self.n_pre, self.n_post):
            raise ValueError("weights shape doesn't match")

        if self.params.use_quantized:
            levels = quantize_weight_array(weights)
            weights = dequantize_weight_array(levels)
            self._weight_levels = levels

        if self._is_sparse:
            self._sparse_weights = sparse.csr_matrix(weights, dtype=self.dtype)
        else:
            self._weights = weights.astype(self.dtype)

    def get_weight_levels(self) -> Optional[NDArray[np.int8]]:
        """Get quantized weight levels.

        Returns:
            Weight levels (n_pre, n_post) or None if not quantized
        """
        if not self.params.use_quantized:
            return None

        if self._is_sparse:
            # Convert sparse to dense levels
            weights = self._sparse_weights.toarray()
            return quantize_weight_array(weights)
        return self._weight_levels.copy() if self._weight_levels is not None else None

    def update_weight(
        self,
        pre_idx: int,
        post_idx: int,
        delta: float,
    ) -> None:
        """Update a single synaptic weight.

        Args:
            pre_idx: Presynaptic neuron index
            post_idx: Postsynaptic neuron index
            delta: Weight change
        """
        if self._is_sparse:
            # Find synapse in sparse matrix
            row_start = self._sparse_weights.indptr[pre_idx]
            row_end = self._sparse_weights.indptr[pre_idx + 1]

            for j in range(row_start, row_end):
                if self._sparse_weights.indices[j] == post_idx:
                    new_weight = self._sparse_weights.data[j] + delta
                    if self.params.use_quantized:
                        level = quantize_weight(new_weight)
                        new_weight = WEIGHT_LEVELS[level]
                    self._sparse_weights.data[j] = new_weight
                    return
        else:
            if self._weights is None:
                return
            new_weight = self._weights[pre_idx, post_idx] + delta
            if self.params.use_quantized:
                level = quantize_weight(new_weight)
                new_weight = WEIGHT_LEVELS[level]
                if self._weight_levels is not None:
                    self._weight_levels[pre_idx, post_idx] = level
            self._weights[pre_idx, post_idx] = new_weight

    def get_connection_count(self) -> int:
        """Get number of non-zero connections.

        Returns:
            Number of active synapses
        """
        if self._is_sparse:
            return self._sparse_weights.nnz
        elif self._weights is not None:
            return int(np.count_nonzero(self._weights))
        return 0

    def get_density(self) -> float:
        """Get connection density.

        Returns:
            Fraction of possible connections that exist
        """
        max_connections = self.n_pre * self.n_post
        return self.get_connection_count() / max_connections

    def reset_delay_buffer(self) -> None:
        """Clear the delay buffer."""
        if self._delay_buffer is not None:
            self._delay_buffer.fill(0)
            self._buffer_idx = 0


def create_fibonacci_delays(
    n_pre: int,
    n_post: int,
    max_fib_idx: int = 5,
) -> NDArray[np.int8]:
    """Create delay matrix using Fibonacci numbers.

    Delays are assigned based on neuron distance, using Fibonacci
    numbers (1, 1, 2, 3, 5, 8, 13, ...) as possible delay values.

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        max_fib_idx: Maximum Fibonacci index to use

    Returns:
        Delay matrix (n_pre, n_post)
    """
    # Fibonacci delays
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34][:max_fib_idx + 1]

    delays = np.ones((n_pre, n_post), dtype=np.int8)

    # Assign delays based on distance (for ordered neurons)
    for i in range(n_pre):
        for j in range(n_post):
            # Use normalized distance to select Fibonacci delay
            if n_pre > 1 and n_post > 1:
                dist = abs(i / (n_pre - 1) - j / (n_post - 1))
                fib_idx = min(int(dist * len(fibs)), len(fibs) - 1)
                delays[i, j] = fibs[fib_idx]

    return delays


def create_golden_weight_matrix(
    n_pre: int,
    n_post: int,
    pattern: str = "random",
    excitatory_ratio: float = 0.8,
) -> NDArray[np.float32]:
    """Create weight matrix using golden ratio values.

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        pattern: Weight pattern ("random", "structured", "balanced")
        excitatory_ratio: Fraction of excitatory (positive) weights

    Returns:
        Weight matrix with golden ratio values
    """
    weights = np.zeros((n_pre, n_post), dtype=np.float32)

    if pattern == "random":
        # Random assignment from weight levels
        for i in range(n_pre):
            for j in range(n_post):
                if np.random.random() < excitatory_ratio:
                    # Excitatory: positive levels (5, 6, 7, 8)
                    level = np.random.choice([5, 6, 7, 8])
                else:
                    # Inhibitory: negative levels (0, 1, 2, 3)
                    level = np.random.choice([0, 1, 2, 3])
                weights[i, j] = WEIGHT_LEVELS[level]

    elif pattern == "structured":
        # Distance-based weights using golden spiral
        for i in range(n_pre):
            for j in range(n_post):
                # Normalized positions
                pre_pos = i / max(n_pre - 1, 1)
                post_pos = j / max(n_post - 1, 1)
                dist = abs(pre_pos - post_pos)

                # Closer neurons have stronger connections
                if dist < PHI_INV ** 2:
                    level = 8  # phi^2
                elif dist < PHI_INV:
                    level = 7  # phi
                elif dist < 0.5:
                    level = 6  # 1.0
                else:
                    level = 5  # 1/phi

                # Add some inhibitory connections
                if np.random.random() > excitatory_ratio:
                    level = 8 - level  # Mirror to negative

                weights[i, j] = WEIGHT_LEVELS[level]

    elif pattern == "balanced":
        # Equal excitatory and inhibitory influence per post neuron
        for j in range(n_post):
            n_exc = int(n_pre * excitatory_ratio)
            exc_indices = np.random.choice(n_pre, n_exc, replace=False)

            for i in range(n_pre):
                if i in exc_indices:
                    level = np.random.choice([5, 6, 7, 8])
                else:
                    level = np.random.choice([0, 1, 2, 3])
                weights[i, j] = WEIGHT_LEVELS[level]

    return weights
