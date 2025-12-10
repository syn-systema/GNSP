"""Spiking Neural Network module.

This module provides components for building and simulating
spiking neural networks with golden ratio-based architecture.

Components:
- Neurons: LIF neuron models with golden ratio dynamics
- Synapses: Golden ratio weight quantization
- STDP: Fibonacci-based spike-timing dependent plasticity
- Topology: Quasicrystalline network structures
- Encoders: Convert data to spike trains
- Decoders: Convert spikes to outputs
- Simulator: Time-stepped and event-driven simulation
- Network: Main SNN container
"""

from gnsp.snn.neuron import (
    LIFNeuron,
    LIFNeuronArray,
    LIFNeuronParams,
    AdaptiveLIFNeuronArray,
    IzhikevichNeuronArray,
    create_golden_neuron_params,
)

from gnsp.snn.synapse import (
    SynapseArray,
    SynapseParams,
    WeightLevel,
    quantize_weight,
    quantize_weight_array,
    dequantize_weight,
    dequantize_weight_array,
    create_fibonacci_delays,
    create_golden_weight_matrix,
)

from gnsp.snn.stdp import (
    STDPBase,
    STDPParams,
    FibonacciSTDP,
    OnlineSTDP,
    TripleSTDP,
    RewardModulatedSTDP,
    compute_stdp_window,
)

from gnsp.snn.topology import (
    TopologyBase,
    TopologyParams,
    QuasicrystalTopology,
    GoldenSpiralTopology,
    SphericalFibonacciTopology,
    HierarchicalTopology,
    SmallWorldTopology,
    FibonacciConnectivityTopology,
    compute_connection_statistics,
)

from gnsp.snn.encoder import (
    EncoderBase,
    RateEncoder,
    TemporalEncoder,
    DeltaEncoder,
    PopulationEncoder,
    FibonacciPhaseEncoder,
    BurstEncoder,
    NetworkPacketEncoder,
)

from gnsp.snn.decoder import (
    DecoderBase,
    RateDecoder,
    TemporalDecoder,
    PopulationDecoder,
    GoldenWeightedDecoder,
    ThresholdDecoder,
    EnsembleDecoder,
    AnomalyScoreDecoder,
    compute_spike_statistics,
)

from gnsp.snn.simulator import (
    SimulationConfig,
    SimulationResult,
    SpikeRecorder,
    PotentialRecorder,
    SimulatorBase,
    TimeSteppedSimulator,
    EventDrivenSimulator,
    BatchSimulator,
    create_simulation_input,
)

from gnsp.snn.network import (
    SNNConfig,
    TopologyType,
    LearningRule,
    Layer,
    SpikingNeuralNetwork,
    create_intrusion_detection_network,
    create_golden_autoencoder,
    create_recurrent_reservoir,
)

__all__ = [
    # Neuron
    "LIFNeuron",
    "LIFNeuronArray",
    "LIFNeuronParams",
    "AdaptiveLIFNeuronArray",
    "IzhikevichNeuronArray",
    "create_golden_neuron_params",
    # Synapse
    "SynapseArray",
    "SynapseParams",
    "WeightLevel",
    "quantize_weight",
    "quantize_weight_array",
    "dequantize_weight",
    "dequantize_weight_array",
    "create_fibonacci_delays",
    "create_golden_weight_matrix",
    # STDP
    "STDPBase",
    "STDPParams",
    "FibonacciSTDP",
    "OnlineSTDP",
    "TripleSTDP",
    "RewardModulatedSTDP",
    "compute_stdp_window",
    # Topology
    "TopologyBase",
    "TopologyParams",
    "QuasicrystalTopology",
    "GoldenSpiralTopology",
    "SphericalFibonacciTopology",
    "HierarchicalTopology",
    "SmallWorldTopology",
    "FibonacciConnectivityTopology",
    "compute_connection_statistics",
    # Encoder
    "EncoderBase",
    "RateEncoder",
    "TemporalEncoder",
    "DeltaEncoder",
    "PopulationEncoder",
    "FibonacciPhaseEncoder",
    "BurstEncoder",
    "NetworkPacketEncoder",
    # Decoder
    "DecoderBase",
    "RateDecoder",
    "TemporalDecoder",
    "PopulationDecoder",
    "GoldenWeightedDecoder",
    "ThresholdDecoder",
    "EnsembleDecoder",
    "AnomalyScoreDecoder",
    "compute_spike_statistics",
    # Simulator
    "SimulationConfig",
    "SimulationResult",
    "SpikeRecorder",
    "PotentialRecorder",
    "SimulatorBase",
    "TimeSteppedSimulator",
    "EventDrivenSimulator",
    "BatchSimulator",
    "create_simulation_input",
    # Network
    "SNNConfig",
    "TopologyType",
    "LearningRule",
    "Layer",
    "SpikingNeuralNetwork",
    "create_intrusion_detection_network",
    "create_golden_autoencoder",
    "create_recurrent_reservoir",
]
