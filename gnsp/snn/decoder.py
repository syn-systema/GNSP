"""Output decoders for converting spike trains to values.

This module provides various decoding schemes for interpreting
SNN output spike patterns as continuous values or class labels.

Decoders include:
- Rate decoding: spike count to value
- Temporal decoding: first spike time to value
- Population decoding: winner-take-all classification
- Weighted decoding: golden ratio weighted spike integration
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from gnsp.constants import PHI, PHI_INV, THRESHOLD_HIGH


class DecoderBase(ABC):
    """Abstract base class for spike decoders."""

    @abstractmethod
    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode spike trains to output values.

        Args:
            spikes: Spike array (timesteps, n_neurons)

        Returns:
            Decoded output values
        """
        pass


class RateDecoder(DecoderBase):
    """Rate-based spike decoding.

    Counts spikes over a time window and converts to a rate
    or normalized value.
    """

    def __init__(
        self,
        n_outputs: int,
        normalize: bool = True,
        max_rate: float = 100.0,  # Hz
        dt: float = 1.0,  # ms
    ) -> None:
        """Initialize rate decoder.

        Args:
            n_outputs: Number of output channels
            normalize: Normalize output to [0, 1]
            max_rate: Maximum expected rate (Hz)
            dt: Simulation timestep (ms)
        """
        self.n_outputs = n_outputs
        self.normalize = normalize
        self.max_rate = max_rate
        self.dt = dt

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode spike rates.

        Args:
            spikes: Spike array (timesteps, n_outputs)

        Returns:
            Rate values (n_outputs,)
        """
        timesteps = spikes.shape[0]

        # Count spikes
        spike_counts = np.sum(spikes, axis=0)

        # Convert to rate (Hz)
        rates = spike_counts / (timesteps * self.dt / 1000.0)

        if self.normalize:
            rates = np.clip(rates / self.max_rate, 0, 1)

        return rates.astype(np.float32)

    def decode_windowed(
        self,
        spikes: NDArray[np.bool_],
        window_size: int,
    ) -> NDArray:
        """Decode rates using sliding window.

        Args:
            spikes: Spike array (timesteps, n_outputs)
            window_size: Window size in timesteps

        Returns:
            Rate values over time (n_windows, n_outputs)
        """
        timesteps = spikes.shape[0]
        n_windows = timesteps - window_size + 1

        rates = np.zeros((n_windows, self.n_outputs), dtype=np.float32)

        for i in range(n_windows):
            window_spikes = spikes[i:i + window_size]
            rates[i] = np.sum(window_spikes, axis=0) / window_size

        if self.normalize:
            rates = np.clip(rates / (self.max_rate * self.dt / 1000.0), 0, 1)

        return rates


class TemporalDecoder(DecoderBase):
    """Temporal (first spike) decoding.

    Decodes values based on the timing of the first spike,
    with earlier spikes indicating higher values.
    """

    def __init__(
        self,
        n_outputs: int,
        max_latency: int = 100,
    ) -> None:
        """Initialize temporal decoder.

        Args:
            n_outputs: Number of output channels
            max_latency: Maximum expected latency
        """
        self.n_outputs = n_outputs
        self.max_latency = max_latency

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode first spike times.

        Args:
            spikes: Spike array (timesteps, n_outputs)

        Returns:
            Decoded values (n_outputs,) in [0, 1]
        """
        timesteps = spikes.shape[0]
        values = np.zeros(self.n_outputs, dtype=np.float32)

        for i in range(self.n_outputs):
            spike_times = np.where(spikes[:, i])[0]
            if len(spike_times) > 0:
                first_spike = spike_times[0]
                # Earlier spike = higher value
                values[i] = 1.0 - (first_spike / min(self.max_latency, timesteps))
            else:
                values[i] = 0.0  # No spike = minimum value

        return np.clip(values, 0, 1)


class PopulationDecoder(DecoderBase):
    """Population (winner-take-all) decoding.

    For classification tasks, returns the class with the
    highest spike count or rate.
    """

    def __init__(
        self,
        n_classes: int,
        method: str = "count",  # "count", "rate", "first"
    ) -> None:
        """Initialize population decoder.

        Args:
            n_classes: Number of output classes
            method: Decoding method
        """
        self.n_classes = n_classes
        self.method = method

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode class labels.

        Args:
            spikes: Spike array (timesteps, n_classes)

        Returns:
            Softmax-like class probabilities (n_classes,)
        """
        if self.method == "count":
            # Sum spike counts
            counts = np.sum(spikes, axis=0).astype(np.float32)
            # Softmax normalization
            if np.sum(counts) > 0:
                probs = counts / (np.sum(counts) + 1e-8)
            else:
                probs = np.ones(self.n_classes) / self.n_classes

        elif self.method == "rate":
            # Same as count but normalized by time
            timesteps = spikes.shape[0]
            rates = np.sum(spikes, axis=0).astype(np.float32) / timesteps
            probs = rates / (np.sum(rates) + 1e-8)

        elif self.method == "first":
            # Based on first spike time (earlier = stronger)
            scores = np.zeros(self.n_classes, dtype=np.float32)
            timesteps = spikes.shape[0]

            for i in range(self.n_classes):
                spike_times = np.where(spikes[:, i])[0]
                if len(spike_times) > 0:
                    scores[i] = 1.0 - (spike_times[0] / timesteps)

            probs = scores / (np.sum(scores) + 1e-8)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return probs

    def predict(
        self,
        spikes: NDArray[np.bool_],
    ) -> int:
        """Get predicted class label.

        Args:
            spikes: Spike array (timesteps, n_classes)

        Returns:
            Predicted class index
        """
        probs = self.decode(spikes)
        return int(np.argmax(probs))


class GoldenWeightedDecoder(DecoderBase):
    """Decoder with golden ratio temporal weighting.

    Recent spikes are weighted more heavily using golden ratio
    decay for temporal integration.
    """

    def __init__(
        self,
        n_outputs: int,
        decay_factor: float = PHI_INV,
    ) -> None:
        """Initialize golden weighted decoder.

        Args:
            n_outputs: Number of output channels
            decay_factor: Temporal decay factor (default: 1/phi)
        """
        self.n_outputs = n_outputs
        self.decay_factor = decay_factor

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode with golden ratio weighting.

        Args:
            spikes: Spike array (timesteps, n_outputs)

        Returns:
            Weighted spike values (n_outputs,)
        """
        timesteps = spikes.shape[0]

        # Create decay weights (more recent = higher weight)
        weights = np.array([
            self.decay_factor ** (timesteps - t - 1)
            for t in range(timesteps)
        ], dtype=np.float32)

        # Normalize weights
        weights /= np.sum(weights)

        # Weighted sum of spikes
        spike_float = spikes.astype(np.float32)
        values = np.sum(spike_float * weights[:, np.newaxis], axis=0)

        return values


class ThresholdDecoder(DecoderBase):
    """Binary threshold decoder for anomaly detection.

    Uses golden ratio-based thresholds to classify spike
    activity as normal or anomalous.
    """

    def __init__(
        self,
        n_outputs: int,
        threshold: float = THRESHOLD_HIGH,  # 1/phi
        window_size: int = 100,
    ) -> None:
        """Initialize threshold decoder.

        Args:
            n_outputs: Number of output channels
            threshold: Detection threshold
            window_size: Integration window
        """
        self.n_outputs = n_outputs
        self.threshold = threshold
        self.window_size = window_size

        # Internal state
        self._buffer: Optional[NDArray] = None
        self._buffer_idx = 0

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode binary detection signals.

        Args:
            spikes: Spike array (timesteps, n_outputs)

        Returns:
            Detection signals (n_outputs,) as 0 or 1
        """
        # Compute spike rates
        rates = np.mean(spikes, axis=0)

        # Apply threshold
        detections = (rates > self.threshold).astype(np.float32)

        return detections

    def decode_streaming(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode with streaming buffer.

        Args:
            spikes: Single timestep spikes (n_outputs,)

        Returns:
            Detection signals (n_outputs,)
        """
        if self._buffer is None:
            self._buffer = np.zeros(
                (self.window_size, self.n_outputs),
                dtype=np.bool_,
            )

        # Update buffer
        self._buffer[self._buffer_idx] = spikes
        self._buffer_idx = (self._buffer_idx + 1) % self.window_size

        # Compute rate over buffer
        rates = np.mean(self._buffer, axis=0)

        return (rates > self.threshold).astype(np.float32)

    def reset(self) -> None:
        """Reset streaming buffer."""
        self._buffer = None
        self._buffer_idx = 0


class EnsembleDecoder(DecoderBase):
    """Ensemble decoder combining multiple decoding strategies.

    Combines outputs from multiple decoders using golden ratio
    weighted averaging.
    """

    def __init__(
        self,
        decoders: List[DecoderBase],
        weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize ensemble decoder.

        Args:
            decoders: List of decoder instances
            weights: Decoder weights (default: golden ratio powers)
        """
        self.decoders = decoders

        if weights is None:
            # Golden ratio weighting
            weights = [PHI_INV ** i for i in range(len(decoders))]

        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Decode using ensemble.

        Args:
            spikes: Spike array (timesteps, n_outputs)

        Returns:
            Ensemble decoded values
        """
        outputs = []
        for decoder, weight in zip(self.decoders, self.weights):
            output = decoder.decode(spikes)
            outputs.append(weight * output)

        return np.sum(outputs, axis=0).astype(np.float32)


class AnomalyScoreDecoder(DecoderBase):
    """Specialized decoder for anomaly scoring.

    Computes an anomaly score based on spike patterns using
    multiple features with golden ratio weighting.
    """

    def __init__(
        self,
        n_neurons: int,
        baseline_rate: Optional[float] = None,
    ) -> None:
        """Initialize anomaly score decoder.

        Args:
            n_neurons: Number of neurons to decode
            baseline_rate: Expected normal spike rate
        """
        self.n_neurons = n_neurons
        self.baseline_rate = baseline_rate

        # Running statistics for adaptive thresholds
        self._rate_history: List[float] = []
        self._max_history = 1000

    def decode(
        self,
        spikes: NDArray[np.bool_],
    ) -> NDArray:
        """Compute anomaly score.

        Args:
            spikes: Spike array (timesteps, n_neurons)

        Returns:
            Anomaly score (scalar array)
        """
        timesteps = spikes.shape[0]

        # Feature 1: Overall spike rate deviation
        current_rate = np.mean(spikes)
        self._rate_history.append(current_rate)
        if len(self._rate_history) > self._max_history:
            self._rate_history.pop(0)

        if self.baseline_rate is not None:
            baseline = self.baseline_rate
        elif len(self._rate_history) > 10:
            baseline = np.mean(self._rate_history[:-1])
        else:
            baseline = 0.5

        rate_deviation = abs(current_rate - baseline) / (baseline + 1e-8)

        # Feature 2: Spike synchrony (unusual if too synchronized)
        spike_counts = np.sum(spikes, axis=1)
        synchrony = np.std(spike_counts) / (np.mean(spike_counts) + 1e-8)

        # Feature 3: Burst detection
        burst_threshold = 3
        bursts = np.sum(spike_counts > burst_threshold) / timesteps

        # Combine features with golden ratio weights
        score = (
            PHI_INV ** 0 * np.clip(rate_deviation, 0, 1) +
            PHI_INV ** 1 * np.clip(synchrony, 0, 1) +
            PHI_INV ** 2 * np.clip(bursts, 0, 1)
        )

        # Normalize
        score = score / (1 + PHI_INV + PHI_INV ** 2)

        return np.array([score], dtype=np.float32)

    def reset(self) -> None:
        """Reset decoder state."""
        self._rate_history = []


def compute_spike_statistics(
    spikes: NDArray[np.bool_],
) -> dict:
    """Compute statistics about spike patterns.

    Args:
        spikes: Spike array (timesteps, n_neurons)

    Returns:
        Dictionary of spike statistics
    """
    timesteps, n_neurons = spikes.shape

    # Per-neuron statistics
    spike_counts = np.sum(spikes, axis=0)
    rates = spike_counts / timesteps

    # Temporal statistics
    per_timestep = np.sum(spikes, axis=1)

    # ISI statistics (inter-spike intervals)
    all_isis = []
    for i in range(n_neurons):
        spike_times = np.where(spikes[:, i])[0]
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            all_isis.extend(isis.tolist())

    return {
        "mean_rate": float(np.mean(rates)),
        "std_rate": float(np.std(rates)),
        "max_rate": float(np.max(rates)),
        "min_rate": float(np.min(rates)),
        "total_spikes": int(np.sum(spikes)),
        "active_neurons": int(np.sum(spike_counts > 0)),
        "mean_per_timestep": float(np.mean(per_timestep)),
        "std_per_timestep": float(np.std(per_timestep)),
        "mean_isi": float(np.mean(all_isis)) if all_isis else 0.0,
        "std_isi": float(np.std(all_isis)) if all_isis else 0.0,
    }
