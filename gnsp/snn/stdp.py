"""Spike-Timing Dependent Plasticity with Fibonacci time constants.

This module implements STDP learning rules using Fibonacci-based
time windows for biologically-inspired synaptic plasticity.

Key features:
- Fibonacci time constants (1, 2, 3, 5, 8, 13, 21, 34)
- Multi-timescale learning with golden ratio decay
- Online and batch learning modes
- Weight quantization to golden ratio levels
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from gnsp.constants import (
    STDP_TAU,
    PHI,
    PHI_INV,
    PHI_INV_SQ,
    WEIGHT_LEVELS,
    N_WEIGHT_LEVELS,
)
from gnsp.core.fibonacci import fibonacci_decay_lookup
from gnsp.snn.synapse import SynapseArray, quantize_weight


@dataclass
class STDPParams:
    """Parameters for STDP learning rule.

    Attributes:
        a_plus: Amplitude for potentiation (pre before post)
        a_minus: Amplitude for depression (post before pre)
        tau_plus: Time constant for potentiation window
        tau_minus: Time constant for depression window
        w_max: Maximum weight value
        w_min: Minimum weight value
        learning_rate: Global learning rate multiplier
        use_quantized: Whether to quantize weights after update
    """

    a_plus: float = PHI_INV        # ~0.618
    a_minus: float = PHI_INV_SQ    # ~0.382
    tau_plus: int = 8              # Fibonacci time constant
    tau_minus: int = 13            # Fibonacci time constant
    w_max: float = PHI * PHI       # phi^2 ~ 2.618
    w_min: float = -PHI * PHI      # -phi^2 ~ -2.618
    learning_rate: float = 0.01
    use_quantized: bool = True

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.a_plus <= 0 or self.a_minus <= 0:
            raise ValueError("a_plus and a_minus must be positive")
        if self.tau_plus <= 0 or self.tau_minus <= 0:
            raise ValueError("tau values must be positive")
        if self.w_max <= self.w_min:
            raise ValueError("w_max must be greater than w_min")


class STDPBase(ABC):
    """Abstract base class for STDP implementations."""

    @abstractmethod
    def update(
        self,
        synapses: SynapseArray,
        pre_spikes: NDArray[np.bool_],
        post_spikes: NDArray[np.bool_],
    ) -> None:
        """Update synaptic weights based on spike timing.

        Args:
            synapses: Synapse array to update
            pre_spikes: Presynaptic spikes this timestep
            post_spikes: Postsynaptic spikes this timestep
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset STDP state."""
        pass


class FibonacciSTDP(STDPBase):
    """STDP with multiple Fibonacci time constants.

    This implementation uses multiple exponential traces with
    Fibonacci time constants for both potentiation and depression,
    providing multi-scale temporal integration.

    The learning rule is:
        dw = sum_i(A_plus * trace_plus_i * post_spike) - sum_i(A_minus * trace_minus_i * pre_spike)

    where traces decay exponentially with Fibonacci time constants.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[STDPParams] = None,
        n_timescales: int = 4,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize Fibonacci STDP.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: STDP parameters
            n_timescales: Number of Fibonacci timescales to use
            dtype: Data type for computations
        """
        if n_pre <= 0 or n_post <= 0:
            raise ValueError("n_pre and n_post must be positive")
        if n_timescales <= 0 or n_timescales > len(STDP_TAU):
            raise ValueError(f"n_timescales must be in [1, {len(STDP_TAU)}]")

        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        self.n_timescales = n_timescales
        self.dtype = dtype

        # Fibonacci time constants
        self.tau_values = np.array(STDP_TAU[:n_timescales], dtype=dtype)

        # Compute decay factors: exp(-1/tau)
        self.decay_plus = np.exp(-1.0 / self.tau_values).astype(dtype)
        self.decay_minus = np.exp(-1.0 / self.tau_values).astype(dtype)

        # Golden ratio weighting for different timescales
        # Earlier timescales (smaller tau) have higher weights
        self.timescale_weights = np.array(
            [PHI_INV ** i for i in range(n_timescales)], dtype=dtype
        )
        self.timescale_weights /= self.timescale_weights.sum()

        # Eligibility traces: (timescales, neurons)
        self.trace_pre = np.zeros((n_timescales, n_pre), dtype=dtype)
        self.trace_post = np.zeros((n_timescales, n_post), dtype=dtype)

    def reset(self) -> None:
        """Reset eligibility traces."""
        self.trace_pre.fill(0)
        self.trace_post.fill(0)

    def update(
        self,
        synapses: SynapseArray,
        pre_spikes: NDArray[np.bool_],
        post_spikes: NDArray[np.bool_],
    ) -> None:
        """Update synaptic weights based on spike timing.

        Args:
            synapses: Synapse array to update
            pre_spikes: Presynaptic spikes this timestep (n_pre,)
            post_spikes: Postsynaptic spikes this timestep (n_post,)
        """
        if pre_spikes.shape[0] != self.n_pre:
            raise ValueError("pre_spikes shape doesn't match n_pre")
        if post_spikes.shape[0] != self.n_post:
            raise ValueError("post_spikes shape doesn't match n_post")

        # Update traces with decay
        for i in range(self.n_timescales):
            self.trace_pre[i] *= self.decay_plus[i]
            self.trace_post[i] *= self.decay_minus[i]

        # Add spike contributions to traces
        pre_spike_float = pre_spikes.astype(self.dtype)
        post_spike_float = post_spikes.astype(self.dtype)

        for i in range(self.n_timescales):
            self.trace_pre[i] += pre_spike_float
            self.trace_post[i] += post_spike_float

        # Compute weight updates
        weights = synapses.get_weights()

        # LTP: pre trace * post spike (pre before post -> strengthen)
        # LTD: post trace * pre spike (post before pre -> weaken)
        dw = np.zeros_like(weights)

        for i in range(self.n_timescales):
            w = self.timescale_weights[i]

            # LTP: for each post spike, use pre trace
            if np.any(post_spikes):
                ltp = w * self.params.a_plus * np.outer(
                    self.trace_pre[i], post_spike_float
                )
                dw += ltp

            # LTD: for each pre spike, use post trace
            if np.any(pre_spikes):
                ltd = w * self.params.a_minus * np.outer(
                    pre_spike_float, self.trace_post[i]
                )
                dw -= ltd

        # Apply learning rate and update
        dw *= self.params.learning_rate

        # Clip to weight bounds
        new_weights = np.clip(
            weights + dw,
            self.params.w_min,
            self.params.w_max,
        )

        # Quantize if needed
        if self.params.use_quantized:
            from gnsp.snn.synapse import quantize_weight_array, dequantize_weight_array
            levels = quantize_weight_array(new_weights)
            new_weights = dequantize_weight_array(levels)

        synapses.set_weights(new_weights)


class OnlineSTDP(STDPBase):
    """Efficient online STDP with single exponential traces.

    Simpler implementation using single time constants for
    potentiation and depression, suitable for real-time learning.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[STDPParams] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize online STDP.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: STDP parameters
            dtype: Data type for computations
        """
        if n_pre <= 0 or n_post <= 0:
            raise ValueError("n_pre and n_post must be positive")

        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        self.dtype = dtype

        # Decay factors
        self.decay_plus = np.exp(-1.0 / self.params.tau_plus).astype(dtype)
        self.decay_minus = np.exp(-1.0 / self.params.tau_minus).astype(dtype)

        # Single eligibility traces
        self.trace_pre = np.zeros(n_pre, dtype=dtype)
        self.trace_post = np.zeros(n_post, dtype=dtype)

    def reset(self) -> None:
        """Reset eligibility traces."""
        self.trace_pre.fill(0)
        self.trace_post.fill(0)

    def update(
        self,
        synapses: SynapseArray,
        pre_spikes: NDArray[np.bool_],
        post_spikes: NDArray[np.bool_],
    ) -> None:
        """Update synaptic weights based on spike timing.

        Args:
            synapses: Synapse array to update
            pre_spikes: Presynaptic spikes this timestep (n_pre,)
            post_spikes: Postsynaptic spikes this timestep (n_post,)
        """
        # Update traces with decay
        self.trace_pre *= self.decay_plus
        self.trace_post *= self.decay_minus

        # Add spike contributions
        self.trace_pre += pre_spikes.astype(self.dtype)
        self.trace_post += post_spikes.astype(self.dtype)

        # Compute weight updates
        weights = synapses.get_weights()

        # LTP: pre trace * post spike
        ltp = self.params.a_plus * np.outer(self.trace_pre, post_spikes.astype(self.dtype))

        # LTD: post trace * pre spike
        ltd = self.params.a_minus * np.outer(pre_spikes.astype(self.dtype), self.trace_post)

        dw = self.params.learning_rate * (ltp - ltd)

        # Update with bounds
        new_weights = np.clip(
            weights + dw,
            self.params.w_min,
            self.params.w_max,
        )

        # Quantize if needed
        if self.params.use_quantized:
            from gnsp.snn.synapse import quantize_weight_array, dequantize_weight_array
            levels = quantize_weight_array(new_weights)
            new_weights = dequantize_weight_array(levels)

        synapses.set_weights(new_weights)


class TripleSTDP(STDPBase):
    """Triplet STDP rule with golden ratio parameters.

    Extends classical pair-based STDP to consider triplets of spikes,
    capturing more complex temporal correlations.

    Based on: Pfister & Gerstner (2006)
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[STDPParams] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize triplet STDP.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: STDP parameters
            dtype: Data type for computations
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        self.dtype = dtype

        # Fast and slow traces with Fibonacci time constants
        # Fast traces (tau = 3, 5)
        self.tau_fast_pre = 3
        self.tau_fast_post = 5
        # Slow traces (tau = 13, 21)
        self.tau_slow_pre = 13
        self.tau_slow_post = 21

        # Decay factors
        self.decay_fast_pre = np.exp(-1.0 / self.tau_fast_pre).astype(dtype)
        self.decay_fast_post = np.exp(-1.0 / self.tau_fast_post).astype(dtype)
        self.decay_slow_pre = np.exp(-1.0 / self.tau_slow_pre).astype(dtype)
        self.decay_slow_post = np.exp(-1.0 / self.tau_slow_post).astype(dtype)

        # Traces
        self.trace_fast_pre = np.zeros(n_pre, dtype=dtype)
        self.trace_fast_post = np.zeros(n_post, dtype=dtype)
        self.trace_slow_pre = np.zeros(n_pre, dtype=dtype)
        self.trace_slow_post = np.zeros(n_post, dtype=dtype)

        # Triplet-specific amplitudes (golden ratio scaled)
        self.a2_plus = self.params.a_plus
        self.a2_minus = self.params.a_minus
        self.a3_plus = self.params.a_plus * PHI_INV  # Weaker triplet contribution
        self.a3_minus = self.params.a_minus * PHI_INV

    def reset(self) -> None:
        """Reset eligibility traces."""
        self.trace_fast_pre.fill(0)
        self.trace_fast_post.fill(0)
        self.trace_slow_pre.fill(0)
        self.trace_slow_post.fill(0)

    def update(
        self,
        synapses: SynapseArray,
        pre_spikes: NDArray[np.bool_],
        post_spikes: NDArray[np.bool_],
    ) -> None:
        """Update synaptic weights using triplet rule.

        Args:
            synapses: Synapse array to update
            pre_spikes: Presynaptic spikes this timestep
            post_spikes: Postsynaptic spikes this timestep
        """
        pre_float = pre_spikes.astype(self.dtype)
        post_float = post_spikes.astype(self.dtype)

        # Update traces with decay before adding new spikes
        self.trace_fast_pre *= self.decay_fast_pre
        self.trace_fast_post *= self.decay_fast_post
        self.trace_slow_pre *= self.decay_slow_pre
        self.trace_slow_post *= self.decay_slow_post

        weights = synapses.get_weights()

        # Pair-based LTP: A2+ * fast_pre * post
        ltp_pair = self.a2_plus * np.outer(self.trace_fast_pre, post_float)

        # Triplet LTP: A3+ * fast_pre * slow_post * post
        # Modulated by slow post trace (recent post activity)
        ltp_triplet = self.a3_plus * np.outer(
            self.trace_fast_pre,
            post_float * self.trace_slow_post
        )

        # Pair-based LTD: A2- * pre * fast_post
        ltd_pair = self.a2_minus * np.outer(pre_float, self.trace_fast_post)

        # Triplet LTD: A3- * slow_pre * pre * fast_post
        ltd_triplet = self.a3_minus * np.outer(
            pre_float * self.trace_slow_pre,
            self.trace_fast_post
        )

        # Total weight change
        dw = self.params.learning_rate * (
            ltp_pair + ltp_triplet - ltd_pair - ltd_triplet
        )

        # Update traces after computing weight changes
        self.trace_fast_pre += pre_float
        self.trace_fast_post += post_float
        self.trace_slow_pre += pre_float
        self.trace_slow_post += post_float

        # Apply weight update
        new_weights = np.clip(
            weights + dw,
            self.params.w_min,
            self.params.w_max,
        )

        if self.params.use_quantized:
            from gnsp.snn.synapse import quantize_weight_array, dequantize_weight_array
            levels = quantize_weight_array(new_weights)
            new_weights = dequantize_weight_array(levels)

        synapses.set_weights(new_weights)


class RewardModulatedSTDP(STDPBase):
    """STDP modulated by reward signal for reinforcement learning.

    Implements three-factor learning where synaptic updates are
    gated by a reward signal, enabling goal-directed learning.

    The learning rule is:
        dw = R(t) * STDP_trace(t)

    where R is the reward signal and STDP_trace accumulates
    pre-post correlations with Fibonacci time constants.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[STDPParams] = None,
        reward_tau: int = 21,  # Fibonacci time constant for reward integration
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize reward-modulated STDP.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: STDP parameters
            reward_tau: Time constant for reward eligibility trace
            dtype: Data type for computations
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or STDPParams()
        self.dtype = dtype
        self.reward_tau = reward_tau

        # Decay factors
        self.decay_pre = np.exp(-1.0 / self.params.tau_plus).astype(dtype)
        self.decay_post = np.exp(-1.0 / self.params.tau_minus).astype(dtype)
        self.decay_reward = np.exp(-1.0 / reward_tau).astype(dtype)

        # Eligibility traces
        self.trace_pre = np.zeros(n_pre, dtype=dtype)
        self.trace_post = np.zeros(n_post, dtype=dtype)

        # STDP eligibility trace (accumulates pre-post correlations)
        self.eligibility = np.zeros((n_pre, n_post), dtype=dtype)

        # Current reward signal
        self.reward = dtype(0.0)

    def reset(self) -> None:
        """Reset all traces."""
        self.trace_pre.fill(0)
        self.trace_post.fill(0)
        self.eligibility.fill(0)
        self.reward = self.dtype(0.0)

    def set_reward(self, reward: float) -> None:
        """Set the current reward signal.

        Args:
            reward: Reward value (positive for reward, negative for punishment)
        """
        self.reward = self.dtype(reward)

    def update(
        self,
        synapses: SynapseArray,
        pre_spikes: NDArray[np.bool_],
        post_spikes: NDArray[np.bool_],
    ) -> None:
        """Update weights with reward modulation.

        Args:
            synapses: Synapse array to update
            pre_spikes: Presynaptic spikes this timestep
            post_spikes: Postsynaptic spikes this timestep
        """
        pre_float = pre_spikes.astype(self.dtype)
        post_float = post_spikes.astype(self.dtype)

        # Decay traces
        self.trace_pre *= self.decay_pre
        self.trace_post *= self.decay_post
        self.eligibility *= self.decay_reward

        # Update traces
        self.trace_pre += pre_float
        self.trace_post += post_float

        # Accumulate STDP eligibility
        # LTP contribution
        ltp = self.params.a_plus * np.outer(self.trace_pre, post_float)
        # LTD contribution
        ltd = self.params.a_minus * np.outer(pre_float, self.trace_post)
        self.eligibility += ltp - ltd

        # Apply reward-modulated update
        if abs(self.reward) > 1e-8:
            weights = synapses.get_weights()
            dw = self.params.learning_rate * self.reward * self.eligibility

            new_weights = np.clip(
                weights + dw,
                self.params.w_min,
                self.params.w_max,
            )

            if self.params.use_quantized:
                from gnsp.snn.synapse import quantize_weight_array, dequantize_weight_array
                levels = quantize_weight_array(new_weights)
                new_weights = dequantize_weight_array(levels)

            synapses.set_weights(new_weights)


def compute_stdp_window(
    dt_range: Tuple[int, int],
    tau_plus: int = 8,
    tau_minus: int = 13,
    a_plus: float = PHI_INV,
    a_minus: float = PHI_INV_SQ,
) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Compute STDP learning window.

    Args:
        dt_range: Range of time differences (min, max)
        tau_plus: Potentiation time constant
        tau_minus: Depression time constant
        a_plus: Potentiation amplitude
        a_minus: Depression amplitude

    Returns:
        Tuple of (dt values, weight change values)
    """
    dt = np.arange(dt_range[0], dt_range[1] + 1, dtype=np.int32)
    dw = np.zeros(len(dt), dtype=np.float32)

    for i, t in enumerate(dt):
        if t > 0:
            # Post after pre -> LTP
            dw[i] = a_plus * np.exp(-t / tau_plus)
        elif t < 0:
            # Pre after post -> LTD
            dw[i] = -a_minus * np.exp(t / tau_minus)
        # dt == 0 -> no change

    return dt, dw
