"""Input encoders for converting data to spike trains.

This module provides various encoding schemes for converting
continuous or discrete data into spike patterns suitable for SNN input.

Encoders include:
- Rate encoding: spike probability proportional to input
- Temporal encoding: spike timing encodes value
- Delta encoding: spikes on value changes
- Population encoding: distributed representation
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from gnsp.constants import PHI, PHI_INV, FIBONACCI


class EncoderBase(ABC):
    """Abstract base class for spike encoders."""

    @abstractmethod
    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data into spike trains.

        Args:
            data: Input data to encode
            timesteps: Number of simulation timesteps

        Returns:
            Spike array (timesteps, n_neurons)
        """
        pass

    @abstractmethod
    def get_n_neurons(self) -> int:
        """Get number of output neurons."""
        pass


class RateEncoder(EncoderBase):
    """Rate-based spike encoding.

    Input values are mapped to spike probabilities. Higher values
    produce more frequent spikes over the encoding window.
    """

    def __init__(
        self,
        n_neurons: int,
        max_rate: float = 100.0,  # Hz
        dt: float = 1.0,  # ms
        normalize: bool = True,
    ) -> None:
        """Initialize rate encoder.

        Args:
            n_neurons: Number of input neurons
            max_rate: Maximum spike rate (Hz)
            dt: Simulation timestep (ms)
            normalize: Normalize input to [0, 1]
        """
        self.n_neurons = n_neurons
        self.max_rate = max_rate
        self.dt = dt
        self.normalize = normalize

        # Compute max probability per timestep
        self.max_prob = max_rate * dt / 1000.0  # Convert to probability

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data using rate coding.

        Args:
            data: Input values (n_neurons,) or (batch, n_neurons)
            timesteps: Number of timesteps to generate

        Returns:
            Spike array (timesteps, n_neurons)
        """
        # Handle batch dimension
        if data.ndim == 1:
            data = data[np.newaxis, :]

        if data.shape[-1] != self.n_neurons:
            raise ValueError(f"data shape {data.shape} doesn't match n_neurons {self.n_neurons}")

        # Normalize if needed
        if self.normalize:
            data_range = data.max() - data.min()
            if data_range > 1e-8:
                data = (data - data.min()) / data_range
            # else: data is constant, keep as-is (values should be in [0,1] already)

        # Compute spike probabilities
        probs = np.clip(data, 0, 1) * self.max_prob

        # Generate spikes
        spikes = np.random.random((timesteps, self.n_neurons)) < probs

        return spikes.astype(np.bool_)

    def get_n_neurons(self) -> int:
        return self.n_neurons


class TemporalEncoder(EncoderBase):
    """Temporal (latency) encoding.

    Input values are encoded as spike times - higher values spike
    earlier. Uses inverse relationship: time = max_time * (1 - value).
    """

    def __init__(
        self,
        n_neurons: int,
        max_latency: int = 100,
        min_latency: int = 1,
    ) -> None:
        """Initialize temporal encoder.

        Args:
            n_neurons: Number of input neurons
            max_latency: Maximum spike latency (timesteps)
            min_latency: Minimum spike latency (timesteps)
        """
        self.n_neurons = n_neurons
        self.max_latency = max_latency
        self.min_latency = min_latency

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data using temporal coding.

        Args:
            data: Input values (n_neurons,), should be in [0, 1]
            timesteps: Number of timesteps

        Returns:
            Spike array (timesteps, n_neurons)
        """
        if data.shape[-1] != self.n_neurons:
            raise ValueError(f"data shape doesn't match n_neurons")

        # Clip to valid range
        data = np.clip(data, 0, 1)

        # Compute spike times (higher value = earlier spike)
        latency_range = self.max_latency - self.min_latency
        spike_times = self.min_latency + latency_range * (1 - data)
        spike_times = spike_times.astype(np.int32)

        # Generate spike trains
        spikes = np.zeros((timesteps, self.n_neurons), dtype=np.bool_)
        for i in range(self.n_neurons):
            t = spike_times[i] if data.ndim == 1 else spike_times[0, i]
            if 0 <= t < timesteps:
                spikes[t, i] = True

        return spikes

    def get_n_neurons(self) -> int:
        return self.n_neurons


class DeltaEncoder(EncoderBase):
    """Delta (change-based) encoding.

    Generates spikes when input values change significantly.
    Uses two neuron populations: ON cells for increases,
    OFF cells for decreases.
    """

    def __init__(
        self,
        n_channels: int,
        threshold: float = 0.1,
        use_off_cells: bool = True,
    ) -> None:
        """Initialize delta encoder.

        Args:
            n_channels: Number of input channels
            threshold: Change threshold for spike generation
            use_off_cells: Include OFF cells for decreases
        """
        self.n_channels = n_channels
        self.threshold = threshold
        self.use_off_cells = use_off_cells

        self._prev_values: Optional[NDArray] = None

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data changes as spikes.

        Args:
            data: Time series input (timesteps, n_channels)
            timesteps: Ignored (uses data length)

        Returns:
            Spike array (timesteps, n_neurons)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        actual_timesteps = data.shape[0]
        n_neurons = self.n_channels * (2 if self.use_off_cells else 1)

        spikes = np.zeros((actual_timesteps, n_neurons), dtype=np.bool_)

        # Initialize previous values
        prev = data[0].copy()

        for t in range(1, actual_timesteps):
            delta = data[t] - prev

            # ON spikes for positive changes
            on_spikes = delta > self.threshold
            spikes[t, :self.n_channels] = on_spikes

            # OFF spikes for negative changes
            if self.use_off_cells:
                off_spikes = delta < -self.threshold
                spikes[t, self.n_channels:] = off_spikes

            prev = data[t].copy()

        return spikes

    def get_n_neurons(self) -> int:
        return self.n_channels * (2 if self.use_off_cells else 1)

    def reset(self) -> None:
        """Reset encoder state."""
        self._prev_values = None


class PopulationEncoder(EncoderBase):
    """Population (place cell) encoding.

    Each input value activates a population of neurons with
    Gaussian tuning curves, providing distributed representation.
    Uses golden ratio spacing for tuning curve centers.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons_per_input: int = 10,
        value_range: Tuple[float, float] = (0.0, 1.0),
        sigma: Optional[float] = None,
    ) -> None:
        """Initialize population encoder.

        Args:
            n_inputs: Number of input channels
            n_neurons_per_input: Neurons per input channel
            value_range: Expected input value range
            sigma: Tuning curve width (default: golden ratio spacing)
        """
        self.n_inputs = n_inputs
        self.n_neurons_per_input = n_neurons_per_input
        self.value_range = value_range

        # Compute tuning curve centers using golden ratio
        self.centers = np.zeros((n_inputs, n_neurons_per_input))
        for i in range(n_inputs):
            # Golden ratio-based spacing
            self.centers[i] = np.linspace(
                value_range[0],
                value_range[1],
                n_neurons_per_input,
            )

        # Default sigma based on spacing
        if sigma is None:
            spacing = (value_range[1] - value_range[0]) / n_neurons_per_input
            sigma = spacing * PHI_INV  # Overlap controlled by golden ratio

        self.sigma = sigma

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data using population coding.

        Args:
            data: Input values (n_inputs,)
            timesteps: Number of timesteps

        Returns:
            Spike array (timesteps, n_neurons)
        """
        if data.shape[-1] != self.n_inputs:
            raise ValueError(f"data shape doesn't match n_inputs")

        # Compute activation for each neuron (Gaussian tuning)
        n_neurons = self.n_inputs * self.n_neurons_per_input
        activations = np.zeros(n_neurons)

        for i in range(self.n_inputs):
            value = data[i] if data.ndim == 1 else data[0, i]
            start_idx = i * self.n_neurons_per_input

            for j in range(self.n_neurons_per_input):
                center = self.centers[i, j]
                # Gaussian activation
                activations[start_idx + j] = np.exp(
                    -0.5 * ((value - center) / self.sigma) ** 2
                )

        # Convert activations to spike probabilities and generate spikes
        spikes = np.zeros((timesteps, n_neurons), dtype=np.bool_)
        for t in range(timesteps):
            spikes[t] = np.random.random(n_neurons) < activations

        return spikes

    def get_n_neurons(self) -> int:
        return self.n_inputs * self.n_neurons_per_input


class FibonacciPhaseEncoder(EncoderBase):
    """Phase encoding with Fibonacci-based periods.

    Encodes values as spike phase relative to multiple oscillations
    with Fibonacci period lengths, providing multi-scale temporal coding.
    """

    def __init__(
        self,
        n_inputs: int,
        n_phases: int = 5,
    ) -> None:
        """Initialize Fibonacci phase encoder.

        Args:
            n_inputs: Number of input channels
            n_phases: Number of Fibonacci periods to use
        """
        self.n_inputs = n_inputs
        self.n_phases = n_phases

        # Fibonacci periods
        self.periods = np.array(FIBONACCI[:n_phases], dtype=np.int32)

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data using phase coding.

        Args:
            data: Input values (n_inputs,) in [0, 1]
            timesteps: Number of timesteps

        Returns:
            Spike array (timesteps, n_neurons)
        """
        n_neurons = self.n_inputs * self.n_phases
        spikes = np.zeros((timesteps, n_neurons), dtype=np.bool_)

        data = np.clip(data, 0, 1)

        for i in range(self.n_inputs):
            value = data[i] if data.ndim == 1 else data[0, i]

            for j, period in enumerate(self.periods):
                neuron_idx = i * self.n_phases + j

                # Spike phase within period (value determines phase)
                phase_offset = int(value * period)

                # Generate spikes at this phase in each period
                for t in range(timesteps):
                    if t % period == phase_offset:
                        spikes[t, neuron_idx] = True

        return spikes

    def get_n_neurons(self) -> int:
        return self.n_inputs * self.n_phases


class BurstEncoder(EncoderBase):
    """Burst encoding with golden ratio burst structure.

    Values are encoded as burst patterns where stronger inputs
    produce longer bursts with golden ratio inter-spike intervals.
    """

    def __init__(
        self,
        n_neurons: int,
        max_burst_length: int = 8,
        base_interval: int = 2,
    ) -> None:
        """Initialize burst encoder.

        Args:
            n_neurons: Number of input neurons
            max_burst_length: Maximum spikes per burst
            base_interval: Base inter-spike interval
        """
        self.n_neurons = n_neurons
        self.max_burst_length = max_burst_length
        self.base_interval = base_interval

        # Pre-compute burst patterns
        self._burst_patterns = self._generate_burst_patterns()

    def _generate_burst_patterns(self) -> List[NDArray]:
        """Generate burst patterns with golden ratio intervals."""
        patterns = []

        for length in range(self.max_burst_length + 1):
            if length == 0:
                patterns.append(np.array([], dtype=np.int32))
            else:
                # Golden ratio-scaled intervals
                intervals = [
                    int(self.base_interval * (PHI_INV ** (i % 3)))
                    for i in range(length - 1)
                ]
                times = [0]
                for interval in intervals:
                    times.append(times[-1] + max(1, interval))
                patterns.append(np.array(times, dtype=np.int32))

        return patterns

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode data as burst patterns.

        Args:
            data: Input values (n_neurons,) in [0, 1]
            timesteps: Number of timesteps

        Returns:
            Spike array (timesteps, n_neurons)
        """
        data = np.clip(data, 0, 1)
        spikes = np.zeros((timesteps, self.n_neurons), dtype=np.bool_)

        for i in range(self.n_neurons):
            value = data[i] if data.ndim == 1 else data[0, i]

            # Burst length proportional to value
            burst_length = int(value * self.max_burst_length)
            pattern = self._burst_patterns[burst_length]

            # Apply pattern at start of window
            for t in pattern:
                if t < timesteps:
                    spikes[t, i] = True

        return spikes

    def get_n_neurons(self) -> int:
        return self.n_neurons


class NetworkPacketEncoder(EncoderBase):
    """Specialized encoder for network packet features.

    Encodes network packet features (port numbers, protocol, flags, etc.)
    into spike patterns suitable for intrusion detection.
    """

    def __init__(
        self,
        feature_dims: List[int],
        encoding_type: str = "population",
    ) -> None:
        """Initialize network packet encoder.

        Args:
            feature_dims: Number of values per feature
            encoding_type: Type of encoding ("population", "rate", "temporal")
        """
        self.feature_dims = feature_dims
        self.encoding_type = encoding_type
        self.n_features = len(feature_dims)

        # Create sub-encoders for each feature
        self._encoders: List[EncoderBase] = []
        for dim in feature_dims:
            if encoding_type == "population":
                self._encoders.append(PopulationEncoder(1, n_neurons_per_input=dim))
            elif encoding_type == "rate":
                self._encoders.append(RateEncoder(dim))
            else:  # temporal
                self._encoders.append(TemporalEncoder(dim))

        self._n_neurons = sum(e.get_n_neurons() for e in self._encoders)

    def encode(
        self,
        data: NDArray,
        timesteps: int,
    ) -> NDArray[np.bool_]:
        """Encode packet features.

        Args:
            data: Feature values (n_features,) normalized to [0, 1]
            timesteps: Number of timesteps

        Returns:
            Spike array (timesteps, n_neurons)
        """
        all_spikes = []

        for i, encoder in enumerate(self._encoders):
            feature_data = data[i:i+1] if data.ndim == 1 else data[:, i:i+1]
            spikes = encoder.encode(feature_data, timesteps)
            all_spikes.append(spikes)

        return np.concatenate(all_spikes, axis=1)

    def get_n_neurons(self) -> int:
        return self._n_neurons
