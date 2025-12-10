"""Main SNN network container and configuration.

This module provides the SpikingNeuralNetwork class that integrates
all SNN components into a cohesive network for neuromorphic processing.

Features:
- Multi-layer network architecture
- Golden ratio-based default parameters
- Configurable topology and learning
- Integrated encoding/decoding
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from gnsp.constants import (
    PHI,
    PHI_INV,
    PHI_INV_SQ,
    DEFAULT_THRESHOLD,
    DEFAULT_RESET,
    DEFAULT_LEAK,
    FIBONACCI,
)
from gnsp.snn.neuron import LIFNeuronArray, LIFNeuronParams
from gnsp.snn.synapse import SynapseArray, SynapseParams
from gnsp.snn.stdp import FibonacciSTDP, OnlineSTDP, STDPParams
from gnsp.snn.topology import (
    TopologyBase,
    TopologyParams,
    HierarchicalTopology,
    GoldenSpiralTopology,
    QuasicrystalTopology,
)
from gnsp.snn.encoder import EncoderBase, RateEncoder
from gnsp.snn.decoder import DecoderBase, PopulationDecoder
from gnsp.snn.simulator import TimeSteppedSimulator, SimulationConfig, SimulationResult


class TopologyType(Enum):
    """Available network topology types."""

    HIERARCHICAL = "hierarchical"
    GOLDEN_SPIRAL = "golden_spiral"
    QUASICRYSTAL = "quasicrystal"
    CUSTOM = "custom"


class LearningRule(Enum):
    """Available learning rules."""

    NONE = "none"
    FIBONACCI_STDP = "fibonacci_stdp"
    ONLINE_STDP = "online_stdp"


@dataclass
class SNNConfig:
    """Configuration for spiking neural network.

    Attributes:
        n_inputs: Number of input neurons
        n_outputs: Number of output neurons
        hidden_sizes: Sizes of hidden layers (Fibonacci by default)
        topology: Network topology type
        learning_rule: Learning rule type
        use_quantized_weights: Use 9-level golden weight quantization
        neuron_params: Neuron parameters
        synapse_params: Synapse parameters
        stdp_params: STDP parameters
        dtype: Data type for computations
    """

    n_inputs: int = 64
    n_outputs: int = 2
    hidden_sizes: Optional[List[int]] = None
    topology: TopologyType = TopologyType.HIERARCHICAL
    learning_rule: LearningRule = LearningRule.FIBONACCI_STDP
    use_quantized_weights: bool = True
    neuron_params: Optional[LIFNeuronParams] = None
    synapse_params: Optional[SynapseParams] = None
    stdp_params: Optional[STDPParams] = None
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        """Set defaults using Fibonacci/golden ratio values."""
        if self.hidden_sizes is None:
            # Default: Fibonacci layer sizes
            self.hidden_sizes = [21, 34, 21]

        if self.neuron_params is None:
            self.neuron_params = LIFNeuronParams(
                threshold=DEFAULT_THRESHOLD,
                reset=DEFAULT_RESET,
                leak=DEFAULT_LEAK,
            )

        if self.synapse_params is None:
            self.synapse_params = SynapseParams(
                use_quantized=self.use_quantized_weights,
            )

        if self.stdp_params is None:
            self.stdp_params = STDPParams()


@dataclass
class Layer:
    """A single layer in the network.

    Attributes:
        neurons: Neuron array for this layer
        input_synapses: Synapses from previous layer
        name: Layer name for identification
    """

    neurons: LIFNeuronArray
    input_synapses: Optional[SynapseArray] = None
    name: str = ""


class SpikingNeuralNetwork:
    """Main spiking neural network container.

    Integrates neurons, synapses, topology, and learning into
    a complete network for neuromorphic processing.

    Example usage:
        >>> config = SNNConfig(n_inputs=64, n_outputs=2)
        >>> network = SpikingNeuralNetwork(config)
        >>> network.build()
        >>> output = network.forward(input_spikes, timesteps=100)
    """

    def __init__(
        self,
        config: SNNConfig,
    ) -> None:
        """Initialize spiking neural network.

        Args:
            config: Network configuration
        """
        self.config = config
        self.layers: List[Layer] = []
        self.stdp_rules: List[Any] = []
        self.is_built = False

        # Encoder/decoder
        self.encoder: Optional[EncoderBase] = None
        self.decoder: Optional[DecoderBase] = None

        # Total neuron count
        self._total_neurons = 0

    def build(self) -> None:
        """Build the network architecture."""
        if self.is_built:
            return

        layer_sizes = [self.config.n_inputs]
        if self.config.hidden_sizes:
            layer_sizes.extend(self.config.hidden_sizes)
        layer_sizes.append(self.config.n_outputs)

        # Create layers
        prev_size = 0
        for i, size in enumerate(layer_sizes):
            # Create neurons
            neurons = LIFNeuronArray(
                n_neurons=size,
                params=self.config.neuron_params,
                dtype=self.config.dtype,
            )

            # Create input synapses (except for input layer)
            input_synapses = None
            if i > 0 and prev_size > 0:
                input_synapses = SynapseArray(
                    n_pre=prev_size,
                    n_post=size,
                    params=self.config.synapse_params,
                    dtype=self.config.dtype,
                )
                # Initialize with random connectivity
                input_synapses.initialize_random(
                    density=PHI_INV,  # Golden ratio density
                    weight_std=1.0,
                )

                # Create STDP rule if enabled
                if self.config.learning_rule == LearningRule.FIBONACCI_STDP:
                    stdp = FibonacciSTDP(
                        n_pre=prev_size,
                        n_post=size,
                        params=self.config.stdp_params,
                        dtype=self.config.dtype,
                    )
                    self.stdp_rules.append((i - 1, i, stdp))
                elif self.config.learning_rule == LearningRule.ONLINE_STDP:
                    stdp = OnlineSTDP(
                        n_pre=prev_size,
                        n_post=size,
                        params=self.config.stdp_params,
                        dtype=self.config.dtype,
                    )
                    self.stdp_rules.append((i - 1, i, stdp))

            layer = Layer(
                neurons=neurons,
                input_synapses=input_synapses,
                name=f"layer_{i}",
            )
            self.layers.append(layer)
            prev_size = size

        # Compute total neurons
        self._total_neurons = sum(layer.neurons.n_neurons for layer in self.layers)

        # Setup default encoder/decoder
        self.encoder = RateEncoder(self.config.n_inputs)
        self.decoder = PopulationDecoder(self.config.n_outputs)

        self.is_built = True

    def forward(
        self,
        inputs: NDArray,
        timesteps: int,
        return_all_spikes: bool = False,
        apply_learning: bool = True,
    ) -> NDArray:
        """Forward pass through the network.

        Args:
            inputs: Input data (features) or spikes
            timesteps: Number of timesteps to simulate
            return_all_spikes: Return spikes from all layers
            apply_learning: Apply STDP learning

        Returns:
            Output layer spikes or decoded output
        """
        if not self.is_built:
            self.build()

        # Encode inputs if needed
        if inputs.ndim == 1:
            # Single sample - encode to spikes
            input_spikes = self.encoder.encode(inputs, timesteps)
        elif inputs.shape[0] == timesteps:
            # Already spike format
            input_spikes = inputs
        else:
            # Encode each timestep
            input_spikes = self.encoder.encode(inputs[0], timesteps)

        # Storage for layer spikes
        layer_spikes = [np.zeros((timesteps, layer.neurons.n_neurons), dtype=np.bool_)
                        for layer in self.layers]
        layer_spikes[0] = input_spikes.astype(np.bool_)

        # Simulate timesteps
        for t in range(timesteps):
            # Input layer gets external input
            current_spikes = layer_spikes[0][t]

            # Propagate through layers
            for i, layer in enumerate(self.layers[1:], 1):
                prev_spikes = current_spikes

                # Compute synaptic current
                if layer.input_synapses is not None:
                    currents = layer.input_synapses.propagate_instant(prev_spikes)
                else:
                    currents = np.zeros(layer.neurons.n_neurons, dtype=self.config.dtype)

                # Update neurons
                current_spikes = layer.neurons.step(currents)
                layer_spikes[i][t] = current_spikes

                # Apply STDP
                if apply_learning:
                    for pre_idx, post_idx, stdp in self.stdp_rules:
                        if post_idx == i:
                            stdp.update(
                                layer.input_synapses,
                                prev_spikes,
                                current_spikes,
                            )

        if return_all_spikes:
            return layer_spikes

        # Return output layer spikes
        return layer_spikes[-1]

    def predict(
        self,
        inputs: NDArray,
        timesteps: int = 100,
    ) -> int:
        """Get classification prediction.

        Args:
            inputs: Input data
            timesteps: Simulation timesteps

        Returns:
            Predicted class index
        """
        output_spikes = self.forward(inputs, timesteps, apply_learning=False)
        return self.decoder.predict(output_spikes)

    def predict_proba(
        self,
        inputs: NDArray,
        timesteps: int = 100,
    ) -> NDArray:
        """Get class probabilities.

        Args:
            inputs: Input data
            timesteps: Simulation timesteps

        Returns:
            Class probabilities (n_classes,)
        """
        output_spikes = self.forward(inputs, timesteps, apply_learning=False)
        return self.decoder.decode(output_spikes)

    def reset(self) -> None:
        """Reset network state."""
        for layer in self.layers:
            layer.neurons.reset_state()
            if layer.input_synapses is not None:
                layer.input_synapses.reset_delay_buffer()

        for _, _, stdp in self.stdp_rules:
            stdp.reset()

    def get_weights(self, layer_idx: int) -> Optional[NDArray]:
        """Get weight matrix for a layer.

        Args:
            layer_idx: Layer index (1 to n_layers-1)

        Returns:
            Weight matrix or None if no synapses
        """
        if 0 < layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if layer.input_synapses is not None:
                return layer.input_synapses.get_weights()
        return None

    def set_weights(self, layer_idx: int, weights: NDArray) -> None:
        """Set weight matrix for a layer.

        Args:
            layer_idx: Layer index (1 to n_layers-1)
            weights: New weight matrix
        """
        if 0 < layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if layer.input_synapses is not None:
                layer.input_synapses.set_weights(weights)

    def get_layer_sizes(self) -> List[int]:
        """Get sizes of all layers.

        Returns:
            List of layer sizes
        """
        return [layer.neurons.n_neurons for layer in self.layers]

    def get_total_synapses(self) -> int:
        """Get total number of synapses.

        Returns:
            Total synapse count
        """
        total = 0
        for layer in self.layers:
            if layer.input_synapses is not None:
                total += layer.input_synapses.get_connection_count()
        return total

    def get_state_dict(self) -> Dict[str, Any]:
        """Get network state for serialization.

        Returns:
            Dictionary of network state
        """
        state = {
            "config": {
                "n_inputs": self.config.n_inputs,
                "n_outputs": self.config.n_outputs,
                "hidden_sizes": self.config.hidden_sizes,
                "topology": self.config.topology.value,
                "learning_rule": self.config.learning_rule.value,
            },
            "weights": [],
        }

        for i, layer in enumerate(self.layers):
            if layer.input_synapses is not None:
                state["weights"].append(layer.input_synapses.get_weights().tolist())
            else:
                state["weights"].append(None)

        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load network state from dictionary.

        Args:
            state: State dictionary
        """
        if not self.is_built:
            self.build()

        for i, weights in enumerate(state["weights"]):
            if weights is not None and i < len(self.layers):
                self.set_weights(i, np.array(weights))

    @property
    def n_neurons(self) -> int:
        """Total number of neurons."""
        return self._total_neurons

    @property
    def neurons(self) -> LIFNeuronArray:
        """Get output layer neurons (for compatibility)."""
        return self.layers[-1].neurons

    @property
    def synapses(self) -> Optional[SynapseArray]:
        """Get output layer synapses (for compatibility)."""
        return self.layers[-1].input_synapses


def create_intrusion_detection_network(
    n_features: int = 41,  # NSL-KDD feature count
    n_classes: int = 2,    # Normal vs Attack
    hidden_multiplier: float = PHI,
) -> SpikingNeuralNetwork:
    """Create SNN configured for intrusion detection.

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        hidden_multiplier: Multiplier for hidden layer sizes

    Returns:
        Configured SpikingNeuralNetwork
    """
    # Fibonacci-inspired hidden layer sizes
    h1 = int(n_features * hidden_multiplier)  # ~66
    h2 = int(h1 * hidden_multiplier)          # ~107
    h3 = int(h1 * PHI_INV)                    # ~41

    config = SNNConfig(
        n_inputs=n_features,
        n_outputs=n_classes,
        hidden_sizes=[h1, h2, h3],
        topology=TopologyType.HIERARCHICAL,
        learning_rule=LearningRule.FIBONACCI_STDP,
        use_quantized_weights=True,
    )

    network = SpikingNeuralNetwork(config)
    network.build()

    return network


def create_golden_autoencoder(
    input_size: int,
    bottleneck_ratio: float = PHI_INV_SQ,  # ~0.382
) -> SpikingNeuralNetwork:
    """Create autoencoder with golden ratio compression.

    The encoder compresses by phi at each layer, bottleneck
    is phi^-2 of input, decoder expands by phi.

    Args:
        input_size: Input dimension
        bottleneck_ratio: Compression ratio for bottleneck

    Returns:
        Autoencoder network
    """
    # Encoder: input -> phi^-1 -> phi^-2
    # Decoder: phi^-2 -> phi^-1 -> input
    bottleneck = max(2, int(input_size * bottleneck_ratio))
    mid_size = int(input_size * PHI_INV)

    config = SNNConfig(
        n_inputs=input_size,
        n_outputs=input_size,  # Reconstruction
        hidden_sizes=[mid_size, bottleneck, mid_size],
        learning_rule=LearningRule.ONLINE_STDP,
    )

    network = SpikingNeuralNetwork(config)
    network.build()

    return network


def create_recurrent_reservoir(
    n_inputs: int,
    n_reservoir: int = 100,
    n_outputs: int = 2,
    spectral_radius: float = PHI_INV,  # Edge of chaos
) -> SpikingNeuralNetwork:
    """Create reservoir computing network.

    Uses quasicrystal topology for reservoir with
    golden ratio spectral radius for optimal dynamics.

    Args:
        n_inputs: Number of inputs
        n_reservoir: Reservoir size
        n_outputs: Number of outputs
        spectral_radius: Reservoir spectral radius

    Returns:
        Reservoir network
    """
    config = SNNConfig(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_sizes=[n_reservoir],
        topology=TopologyType.QUASICRYSTAL,
        learning_rule=LearningRule.NONE,  # Only train readout
    )

    network = SpikingNeuralNetwork(config)
    network.build()

    # Scale reservoir weights for desired spectral radius
    # Note: For true reservoir, we'd need recurrent connections.
    # Here we scale input-to-reservoir weights by spectral_radius factor
    if len(network.layers) > 1:
        layer = network.layers[1]
        if layer.input_synapses is not None:
            weights = layer.input_synapses.get_weights()
            # Only compute eigenvalues for square matrices
            if weights.shape[0] == weights.shape[1]:
                eigenvalues = np.linalg.eigvals(weights)
                current_radius = np.max(np.abs(eigenvalues))
                if current_radius > 0:
                    weights *= spectral_radius / current_radius
            else:
                # For non-square matrices, just scale by factor
                weights *= spectral_radius
            layer.input_synapses.set_weights(weights)

    return network
