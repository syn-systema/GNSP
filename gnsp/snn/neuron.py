"""Leaky Integrate-and-Fire (LIF) neuron models with golden ratio dynamics.

This module provides LIF neuron implementations optimized for neuromorphic
security applications. Key features:
- Golden ratio-based parameters (threshold=phi, reset=1/phi, leak=1/phi^2)
- Both single-neuron and vectorized array implementations
- Support for fixed-point arithmetic for hardware compatibility
- Refractory period support
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from gnsp.constants import (
    DEFAULT_THRESHOLD,
    DEFAULT_RESET,
    DEFAULT_LEAK,
    PHI,
    PHI_INV,
    PHI_INV_SQ,
)


@dataclass
class LIFNeuronParams:
    """Parameters for LIF neuron model.

    Attributes:
        threshold: Firing threshold voltage (default: phi)
        reset: Reset voltage after spike (default: 1/phi)
        leak: Leak factor per timestep (default: 1/phi^2)
        refractory_period: Timesteps neuron is inactive after spike
        v_rest: Resting membrane potential
    """

    threshold: float = DEFAULT_THRESHOLD  # phi
    reset: float = DEFAULT_RESET          # 1/phi
    leak: float = DEFAULT_LEAK            # 1/phi^2
    refractory_period: int = 1
    v_rest: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.threshold <= self.reset:
            raise ValueError("threshold must be greater than reset")
        if not 0 < self.leak < 1:
            raise ValueError("leak must be in (0, 1)")
        if self.refractory_period < 0:
            raise ValueError("refractory_period must be non-negative")


@dataclass
class LIFNeuron:
    """Single Leaky Integrate-and-Fire neuron.

    Implements the LIF dynamics:
        v(t+1) = leak * v(t) + I(t)  if not in refractory
        spike if v(t) >= threshold, then v(t) = reset

    Attributes:
        params: Neuron parameters
        v: Current membrane potential
        refractory_counter: Remaining refractory timesteps
        spike_count: Total number of spikes fired
    """

    params: LIFNeuronParams = field(default_factory=LIFNeuronParams)
    v: float = field(default=0.0, init=False)
    refractory_counter: int = field(default=0, init=False)
    spike_count: int = field(default=0, init=False)

    def reset_state(self) -> None:
        """Reset neuron to initial state."""
        self.v = self.params.v_rest
        self.refractory_counter = 0
        self.spike_count = 0

    def step(self, current: float) -> bool:
        """Advance neuron by one timestep.

        Args:
            current: Input current for this timestep

        Returns:
            True if neuron spiked, False otherwise
        """
        # Check refractory period
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return False

        # Leak and integrate
        self.v = self.params.leak * self.v + current

        # Check for spike
        if self.v >= self.params.threshold:
            self.v = self.params.reset
            self.refractory_counter = self.params.refractory_period
            self.spike_count += 1
            return True

        return False

    def get_state(self) -> Tuple[float, int, int]:
        """Get current neuron state.

        Returns:
            Tuple of (membrane potential, refractory counter, spike count)
        """
        return (self.v, self.refractory_counter, self.spike_count)


class LIFNeuronArray:
    """Vectorized array of LIF neurons for efficient batch processing.

    This class provides an optimized implementation for simulating many
    neurons in parallel using numpy vectorized operations.

    Attributes:
        n_neurons: Number of neurons in the array
        params: Shared parameters for all neurons
        v: Membrane potentials (n_neurons,)
        refractory: Refractory counters (n_neurons,)
        spike_counts: Cumulative spike counts (n_neurons,)
    """

    def __init__(
        self,
        n_neurons: int,
        params: Optional[LIFNeuronParams] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize neuron array.

        Args:
            n_neurons: Number of neurons
            params: Neuron parameters (uses defaults if None)
            dtype: Data type for computations
        """
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")

        self.n_neurons = n_neurons
        self.params = params or LIFNeuronParams()
        self.dtype = dtype

        # State arrays
        self.v: NDArray = np.full(n_neurons, self.params.v_rest, dtype=dtype)
        self.refractory: NDArray[np.int32] = np.zeros(n_neurons, dtype=np.int32)
        self.spike_counts: NDArray[np.int64] = np.zeros(n_neurons, dtype=np.int64)

        # Pre-compute parameters as arrays for vectorized ops
        self._threshold = self.dtype(self.params.threshold)
        self._reset = self.dtype(self.params.reset)
        self._leak = self.dtype(self.params.leak)

    def reset_state(self) -> None:
        """Reset all neurons to initial state."""
        self.v.fill(self.params.v_rest)
        self.refractory.fill(0)
        self.spike_counts.fill(0)

    def step(self, currents: NDArray) -> NDArray[np.bool_]:
        """Advance all neurons by one timestep.

        Args:
            currents: Input currents for each neuron (n_neurons,)

        Returns:
            Boolean array indicating which neurons spiked
        """
        if currents.shape[0] != self.n_neurons:
            raise ValueError(
                f"currents shape {currents.shape} doesn't match n_neurons {self.n_neurons}"
            )

        # Create masks
        not_refractory = self.refractory == 0

        # Decrement refractory counters
        self.refractory = np.maximum(self.refractory - 1, 0)

        # Leak and integrate (only for non-refractory neurons)
        self.v = np.where(
            not_refractory,
            self._leak * self.v + currents,
            self.v
        )

        # Check for spikes
        spikes = self.v >= self._threshold

        # Reset spiking neurons
        self.v = np.where(spikes, self._reset, self.v)

        # Set refractory period for spiking neurons
        self.refractory = np.where(
            spikes,
            self.params.refractory_period,
            self.refractory
        )

        # Update spike counts
        self.spike_counts += spikes.astype(np.int64)

        return spikes

    def step_batch(
        self,
        currents: NDArray,
    ) -> NDArray[np.bool_]:
        """Process multiple timesteps at once.

        Args:
            currents: Input currents (timesteps, n_neurons)

        Returns:
            Boolean array of spikes (timesteps, n_neurons)
        """
        timesteps = currents.shape[0]
        spikes = np.zeros((timesteps, self.n_neurons), dtype=np.bool_)

        for t in range(timesteps):
            spikes[t] = self.step(currents[t])

        return spikes

    def get_membrane_potentials(self) -> NDArray:
        """Get current membrane potentials.

        Returns:
            Copy of membrane potential array
        """
        return self.v.copy()

    def get_spike_counts(self) -> NDArray[np.int64]:
        """Get cumulative spike counts.

        Returns:
            Copy of spike count array
        """
        return self.spike_counts.copy()

    def get_firing_rates(self, timesteps: int) -> NDArray:
        """Calculate firing rates as spikes per timestep.

        Args:
            timesteps: Number of timesteps elapsed

        Returns:
            Firing rate for each neuron
        """
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        return self.spike_counts / timesteps

    def set_membrane_potentials(self, potentials: NDArray) -> None:
        """Set membrane potentials directly.

        Args:
            potentials: New membrane potentials (n_neurons,)
        """
        if potentials.shape[0] != self.n_neurons:
            raise ValueError("potentials shape doesn't match n_neurons")
        self.v = potentials.astype(self.dtype)

    def get_active_fraction(self) -> float:
        """Get fraction of neurons not in refractory period.

        Returns:
            Fraction of active neurons (0 to 1)
        """
        return float(np.mean(self.refractory == 0))


class AdaptiveLIFNeuronArray:
    """LIF neurons with spike-frequency adaptation.

    Extends basic LIF with an adaptation current that increases
    with each spike and decays exponentially, implementing a
    form of negative feedback that regulates firing rates.

    The adaptation follows golden ratio dynamics:
        a(t+1) = a(t) * phi_inv + delta_a * spike(t)

    where delta_a = phi_inv^2 by default.
    """

    def __init__(
        self,
        n_neurons: int,
        params: Optional[LIFNeuronParams] = None,
        adaptation_increment: float = PHI_INV_SQ,
        adaptation_decay: float = PHI_INV,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize adaptive neuron array.

        Args:
            n_neurons: Number of neurons
            params: Base LIF parameters
            adaptation_increment: Adaptation increase per spike
            adaptation_decay: Adaptation decay factor per timestep
            dtype: Data type for computations
        """
        self.base = LIFNeuronArray(n_neurons, params, dtype)
        self.adaptation_increment = dtype(adaptation_increment)
        self.adaptation_decay = dtype(adaptation_decay)

        # Adaptation current for each neuron
        self.adaptation: NDArray = np.zeros(n_neurons, dtype=dtype)

    @property
    def n_neurons(self) -> int:
        """Number of neurons."""
        return self.base.n_neurons

    @property
    def params(self) -> LIFNeuronParams:
        """Neuron parameters."""
        return self.base.params

    def reset_state(self) -> None:
        """Reset all neurons to initial state."""
        self.base.reset_state()
        self.adaptation.fill(0)

    def step(self, currents: NDArray) -> NDArray[np.bool_]:
        """Advance all neurons by one timestep with adaptation.

        Args:
            currents: Input currents for each neuron (n_neurons,)

        Returns:
            Boolean array indicating which neurons spiked
        """
        # Subtract adaptation current from input
        effective_current = currents - self.adaptation

        # Run base LIF step
        spikes = self.base.step(effective_current)

        # Update adaptation: decay + increment on spike
        self.adaptation = (
            self.adaptation * self.adaptation_decay +
            spikes.astype(self.base.dtype) * self.adaptation_increment
        )

        return spikes

    def get_adaptation(self) -> NDArray:
        """Get current adaptation values.

        Returns:
            Copy of adaptation array
        """
        return self.adaptation.copy()


class IzhikevichNeuronArray:
    """Izhikevich neuron model with golden ratio-tuned parameters.

    The Izhikevich model provides more biologically realistic dynamics
    while remaining computationally efficient:
        dv/dt = 0.04v^2 + 5v + 140 - u + I
        du/dt = a(bv - u)
        if v >= 30: v = c, u = u + d

    This implementation uses golden ratio-derived default parameters.
    """

    def __init__(
        self,
        n_neurons: int,
        a: float = PHI_INV_SQ * 0.1,  # Recovery time constant
        b: float = PHI_INV * 0.5,      # Sensitivity of recovery
        c: float = -65.0,              # Reset potential
        d: float = PHI * 4.0,          # Recovery jump after spike
        dt: float = 1.0,               # Integration timestep (ms)
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize Izhikevich neuron array.

        Args:
            n_neurons: Number of neurons
            a, b, c, d: Izhikevich model parameters
            dt: Integration timestep
            dtype: Data type for computations
        """
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")

        self.n_neurons = n_neurons
        self.dtype = dtype
        self.dt = dtype(dt)

        # Parameters
        self.a = dtype(a)
        self.b = dtype(b)
        self.c = dtype(c)
        self.d = dtype(d)

        # State variables
        self.v: NDArray = np.full(n_neurons, c, dtype=dtype)
        self.u: NDArray = np.full(n_neurons, b * c, dtype=dtype)
        self.spike_counts: NDArray[np.int64] = np.zeros(n_neurons, dtype=np.int64)

    def reset_state(self) -> None:
        """Reset all neurons to initial state."""
        self.v.fill(self.c)
        self.u.fill(self.b * self.c)
        self.spike_counts.fill(0)

    def step(self, currents: NDArray) -> NDArray[np.bool_]:
        """Advance all neurons by one timestep.

        Args:
            currents: Input currents for each neuron (n_neurons,)

        Returns:
            Boolean array indicating which neurons spiked
        """
        if currents.shape[0] != self.n_neurons:
            raise ValueError(
                f"currents shape {currents.shape} doesn't match n_neurons {self.n_neurons}"
            )

        # Euler integration for membrane potential
        dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + currents)
        self.v = self.v + dv * self.dt

        # Update recovery variable
        du = self.a * (self.b * self.v - self.u)
        self.u = self.u + du * self.dt

        # Check for spikes (threshold at 30mV)
        spikes = self.v >= 30.0

        # Reset spiking neurons
        self.v = np.where(spikes, self.c, self.v)
        self.u = np.where(spikes, self.u + self.d, self.u)

        # Update spike counts
        self.spike_counts += spikes.astype(np.int64)

        return spikes

    def get_state(self) -> Tuple[NDArray, NDArray]:
        """Get current neuron state.

        Returns:
            Tuple of (membrane potentials, recovery variables)
        """
        return (self.v.copy(), self.u.copy())


def create_golden_neuron_params(
    threshold_level: int = 0,
    reset_level: int = -1,
    leak_level: int = -2,
) -> LIFNeuronParams:
    """Create LIF parameters using golden ratio powers.

    Each parameter is set to phi^level, allowing systematic
    variation of neuron dynamics on the golden ratio scale.

    Args:
        threshold_level: Power of phi for threshold (default: 0 -> phi^0 * phi = phi)
        reset_level: Power of phi for reset (default: -1 -> phi^-1)
        leak_level: Power of phi for leak (default: -2 -> phi^-2)

    Returns:
        LIFNeuronParams with golden ratio-based values
    """
    # Threshold is phi * phi^threshold_level
    threshold = PHI * (PHI ** threshold_level)
    # Reset is phi^reset_level
    reset = PHI ** reset_level if reset_level >= 0 else PHI_INV ** (-reset_level)
    # Leak is phi^leak_level (must be in (0,1))
    leak = PHI ** leak_level if leak_level >= 0 else PHI_INV ** (-leak_level)

    # Ensure leak is in valid range
    if leak >= 1:
        leak = PHI_INV

    return LIFNeuronParams(
        threshold=threshold,
        reset=reset,
        leak=leak,
    )
