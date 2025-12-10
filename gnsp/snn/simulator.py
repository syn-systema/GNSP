"""SNN simulation engine with event-driven and time-stepped modes.

This module provides simulation infrastructure for running SNNs:
- Time-stepped simulation for synchronous updates
- Event-driven simulation for sparse activity
- Recording and monitoring capabilities
- Performance optimization utilities
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable, Any
from abc import ABC, abstractmethod
import time

import numpy as np
from numpy.typing import NDArray

from gnsp.constants import PHI_INV


@dataclass
class SimulationConfig:
    """Configuration for SNN simulation.

    Attributes:
        dt: Simulation timestep (ms)
        duration: Total simulation duration (ms)
        record_spikes: Record all spikes
        record_potentials: Record membrane potentials
        record_weights: Record weight snapshots
        weight_record_interval: Timesteps between weight records
    """

    dt: float = 1.0
    duration: float = 1000.0
    record_spikes: bool = True
    record_potentials: bool = False
    record_weights: bool = False
    weight_record_interval: int = 100


@dataclass
class SimulationResult:
    """Results from simulation run.

    Attributes:
        spikes: Spike records (timesteps, n_neurons) or sparse list
        potentials: Membrane potential records (timesteps, n_neurons)
        weights: Weight snapshots at intervals
        metrics: Performance metrics
        timesteps: Number of timesteps simulated
    """

    spikes: Optional[NDArray[np.bool_]] = None
    potentials: Optional[NDArray[np.float32]] = None
    weights: Optional[List[NDArray[np.float32]]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    timesteps: int = 0


class SpikeRecorder:
    """Records spike events during simulation."""

    def __init__(
        self,
        n_neurons: int,
        max_timesteps: int,
        sparse: bool = True,
    ) -> None:
        """Initialize spike recorder.

        Args:
            n_neurons: Number of neurons to record
            max_timesteps: Maximum timesteps to record
            sparse: Use sparse storage (list of spike times)
        """
        self.n_neurons = n_neurons
        self.max_timesteps = max_timesteps
        self.sparse = sparse

        if sparse:
            # List of (timestep, neuron_id) tuples
            self._spike_list: List[Tuple[int, int]] = []
        else:
            # Dense array
            self._spike_array = np.zeros(
                (max_timesteps, n_neurons),
                dtype=np.bool_,
            )

        self._current_timestep = 0

    def record(
        self,
        timestep: int,
        spikes: NDArray[np.bool_],
    ) -> None:
        """Record spikes for a timestep.

        Args:
            timestep: Current timestep
            spikes: Boolean spike array (n_neurons,)
        """
        if self.sparse:
            spike_indices = np.where(spikes)[0]
            for idx in spike_indices:
                self._spike_list.append((timestep, int(idx)))
        else:
            if timestep < self.max_timesteps:
                self._spike_array[timestep] = spikes

        self._current_timestep = timestep

    def get_spikes(self) -> NDArray[np.bool_]:
        """Get recorded spikes as dense array.

        Returns:
            Spike array (recorded_timesteps, n_neurons)
        """
        if self.sparse:
            # Convert sparse to dense
            n_timesteps = self._current_timestep + 1
            array = np.zeros((n_timesteps, self.n_neurons), dtype=np.bool_)
            for t, idx in self._spike_list:
                if t < n_timesteps:
                    array[t, idx] = True
            return array
        else:
            return self._spike_array[:self._current_timestep + 1]

    def get_spike_times(self, neuron_id: int) -> NDArray[np.int32]:
        """Get spike times for a specific neuron.

        Args:
            neuron_id: Neuron index

        Returns:
            Array of spike times
        """
        if self.sparse:
            times = [t for t, idx in self._spike_list if idx == neuron_id]
            return np.array(times, dtype=np.int32)
        else:
            return np.where(self._spike_array[:, neuron_id])[0].astype(np.int32)

    def get_spike_count(self) -> int:
        """Get total number of spikes recorded."""
        if self.sparse:
            return len(self._spike_list)
        else:
            return int(np.sum(self._spike_array[:self._current_timestep + 1]))

    def reset(self) -> None:
        """Reset recorder."""
        if self.sparse:
            self._spike_list.clear()
        else:
            self._spike_array.fill(False)
        self._current_timestep = 0


class PotentialRecorder:
    """Records membrane potentials during simulation."""

    def __init__(
        self,
        n_neurons: int,
        max_timesteps: int,
        record_interval: int = 1,
        neuron_subset: Optional[NDArray[np.int32]] = None,
    ) -> None:
        """Initialize potential recorder.

        Args:
            n_neurons: Total neurons in network
            max_timesteps: Maximum timesteps to record
            record_interval: Record every N timesteps
            neuron_subset: Subset of neurons to record (all if None)
        """
        self.n_neurons = n_neurons
        self.max_timesteps = max_timesteps
        self.record_interval = record_interval

        if neuron_subset is not None:
            self.neuron_subset = neuron_subset
            n_record = len(neuron_subset)
        else:
            self.neuron_subset = np.arange(n_neurons)
            n_record = n_neurons

        n_records = max_timesteps // record_interval + 1
        self._potentials = np.zeros((n_records, n_record), dtype=np.float32)
        self._record_idx = 0

    def record(
        self,
        timestep: int,
        potentials: NDArray[np.float32],
    ) -> None:
        """Record potentials for a timestep.

        Args:
            timestep: Current timestep
            potentials: Membrane potentials (n_neurons,)
        """
        if timestep % self.record_interval == 0:
            if self._record_idx < len(self._potentials):
                self._potentials[self._record_idx] = potentials[self.neuron_subset]
                self._record_idx += 1

    def get_potentials(self) -> NDArray[np.float32]:
        """Get recorded potentials.

        Returns:
            Potential array (recorded_timesteps, n_recorded_neurons)
        """
        return self._potentials[:self._record_idx]

    def reset(self) -> None:
        """Reset recorder."""
        self._potentials.fill(0)
        self._record_idx = 0


class SimulatorBase(ABC):
    """Abstract base class for SNN simulators."""

    @abstractmethod
    def run(
        self,
        inputs: NDArray,
        config: SimulationConfig,
    ) -> SimulationResult:
        """Run simulation.

        Args:
            inputs: Input spike trains or currents
            config: Simulation configuration

        Returns:
            Simulation results
        """
        pass

    @abstractmethod
    def step(
        self,
        inputs: NDArray,
    ) -> NDArray[np.bool_]:
        """Execute single timestep.

        Args:
            inputs: Input for this timestep

        Returns:
            Output spikes
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset simulator state."""
        pass


class TimeSteppedSimulator(SimulatorBase):
    """Time-stepped (synchronous) SNN simulator.

    Updates all neurons in parallel at each timestep.
    Suitable for most network sizes and topologies.
    """

    def __init__(
        self,
        neurons: Any,  # LIFNeuronArray
        synapses: Optional[Any] = None,  # SynapseArray
        stdp: Optional[Any] = None,  # STDPBase
    ) -> None:
        """Initialize time-stepped simulator.

        Args:
            neurons: Neuron array
            synapses: Optional synapse array
            stdp: Optional STDP learning rule
        """
        self.neurons = neurons
        self.synapses = synapses
        self.stdp = stdp

        self._timestep = 0

    def step(
        self,
        inputs: NDArray,
    ) -> NDArray[np.bool_]:
        """Execute single timestep.

        Args:
            inputs: External input currents (n_neurons,)

        Returns:
            Output spikes (n_neurons,)
        """
        # Compute recurrent input if synapses exist
        if self.synapses is not None:
            # Get previous spikes for recurrent computation
            recurrent = self.synapses.propagate_instant(self._prev_spikes)
            total_input = inputs + recurrent
        else:
            total_input = inputs

        # Update neurons
        spikes = self.neurons.step(total_input)

        # Apply STDP if enabled
        if self.stdp is not None and self.synapses is not None:
            self.stdp.update(self.synapses, self._prev_spikes, spikes)

        self._prev_spikes = spikes
        self._timestep += 1

        return spikes

    def run(
        self,
        inputs: NDArray,
        config: SimulationConfig,
    ) -> SimulationResult:
        """Run time-stepped simulation.

        Args:
            inputs: Input currents/spikes (timesteps, n_inputs)
            config: Simulation configuration

        Returns:
            Simulation results
        """
        self.reset()

        timesteps = inputs.shape[0]
        n_neurons = self.neurons.n_neurons

        # Setup recorders
        spike_recorder = None
        potential_recorder = None

        if config.record_spikes:
            spike_recorder = SpikeRecorder(n_neurons, timesteps)

        if config.record_potentials:
            potential_recorder = PotentialRecorder(n_neurons, timesteps)

        weight_snapshots = []

        # Run simulation
        start_time = time.time()

        for t in range(timesteps):
            # Step simulation
            spikes = self.step(inputs[t])

            # Record
            if spike_recorder is not None:
                spike_recorder.record(t, spikes)

            if potential_recorder is not None:
                potential_recorder.record(t, self.neurons.get_membrane_potentials())

            if config.record_weights and self.synapses is not None:
                if t % config.weight_record_interval == 0:
                    weight_snapshots.append(self.synapses.get_weights().copy())

        elapsed = time.time() - start_time

        # Compile results
        result = SimulationResult(
            timesteps=timesteps,
            metrics={
                "elapsed_time": elapsed,
                "timesteps_per_second": timesteps / elapsed if elapsed > 0 else 0,
                "total_spikes": spike_recorder.get_spike_count() if spike_recorder else 0,
            }
        )

        if spike_recorder is not None:
            result.spikes = spike_recorder.get_spikes()

        if potential_recorder is not None:
            result.potentials = potential_recorder.get_potentials()

        if weight_snapshots:
            result.weights = weight_snapshots

        return result

    def reset(self) -> None:
        """Reset simulator state."""
        self.neurons.reset_state()
        if self.synapses is not None:
            self.synapses.reset_delay_buffer()
        if self.stdp is not None:
            self.stdp.reset()

        n_neurons = self.neurons.n_neurons
        self._prev_spikes = np.zeros(n_neurons, dtype=np.bool_)
        self._timestep = 0


@dataclass
class SpikeEvent:
    """Event for event-driven simulation."""

    time: float
    neuron_id: int
    event_type: str = "spike"  # "spike", "input", "threshold"


class EventDrivenSimulator(SimulatorBase):
    """Event-driven SNN simulator.

    Processes events (spikes) as they occur rather than
    stepping through fixed timesteps. More efficient for
    sparse activity patterns.
    """

    def __init__(
        self,
        n_neurons: int,
        threshold: float,
        reset_potential: float,
        leak_factor: float,
    ) -> None:
        """Initialize event-driven simulator.

        Args:
            n_neurons: Number of neurons
            threshold: Spike threshold
            reset_potential: Reset potential after spike
            leak_factor: Leak time constant
        """
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.leak_factor = leak_factor

        # State
        self.potentials = np.zeros(n_neurons, dtype=np.float32)
        self.last_update = np.zeros(n_neurons, dtype=np.float32)

        # Event queue (sorted by time)
        self._event_queue: List[SpikeEvent] = []

    def _insert_event(self, event: SpikeEvent) -> None:
        """Insert event maintaining sorted order."""
        # Binary search for insertion point
        lo, hi = 0, len(self._event_queue)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._event_queue[mid].time < event.time:
                lo = mid + 1
            else:
                hi = mid
        self._event_queue.insert(lo, event)

    def _update_potential(
        self,
        neuron_id: int,
        current_time: float,
    ) -> None:
        """Update potential with leak since last update."""
        dt = current_time - self.last_update[neuron_id]
        if dt > 0:
            # Exponential decay
            self.potentials[neuron_id] *= np.exp(-dt / self.leak_factor)
            self.last_update[neuron_id] = current_time

    def add_input(
        self,
        neuron_id: int,
        time: float,
        current: float,
    ) -> None:
        """Add external input event.

        Args:
            neuron_id: Target neuron
            time: Event time
            current: Input current
        """
        event = SpikeEvent(time=time, neuron_id=neuron_id, event_type="input")
        event.current = current  # type: ignore
        self._insert_event(event)

    def step(
        self,
        inputs: NDArray,
    ) -> NDArray[np.bool_]:
        """Not used in event-driven mode."""
        raise NotImplementedError("Use run() for event-driven simulation")

    def run(
        self,
        inputs: NDArray,
        config: SimulationConfig,
    ) -> SimulationResult:
        """Run event-driven simulation.

        Args:
            inputs: Input events as (time, neuron_id, current) tuples
            config: Simulation configuration

        Returns:
            Simulation results
        """
        self.reset()

        # Add input events
        for t, neuron_id, current in inputs:
            self.add_input(int(neuron_id), float(t), float(current))

        spike_times: List[Tuple[float, int]] = []
        max_time = config.duration

        start_time = time.time()

        # Process events
        while self._event_queue:
            event = self._event_queue.pop(0)

            if event.time > max_time:
                break

            neuron_id = event.neuron_id

            if event.event_type == "input":
                # Update potential with leak
                self._update_potential(neuron_id, event.time)

                # Add input current
                self.potentials[neuron_id] += event.current  # type: ignore

                # Check for spike
                if self.potentials[neuron_id] >= self.threshold:
                    spike_times.append((event.time, neuron_id))
                    self.potentials[neuron_id] = self.reset_potential

            elif event.event_type == "spike":
                spike_times.append((event.time, neuron_id))

        elapsed = time.time() - start_time

        # Convert to dense spikes if needed
        if config.record_spikes:
            timesteps = int(config.duration / config.dt)
            spikes = np.zeros((timesteps, self.n_neurons), dtype=np.bool_)
            for t, neuron_id in spike_times:
                timestep = int(t / config.dt)
                if 0 <= timestep < timesteps:
                    spikes[timestep, neuron_id] = True
        else:
            spikes = None

        return SimulationResult(
            spikes=spikes,
            timesteps=int(config.duration / config.dt),
            metrics={
                "elapsed_time": elapsed,
                "total_spikes": len(spike_times),
                "events_processed": len(spike_times),
            }
        )

    def reset(self) -> None:
        """Reset simulator state."""
        self.potentials.fill(0)
        self.last_update.fill(0)
        self._event_queue.clear()


class BatchSimulator:
    """Batched simulation for processing multiple inputs.

    Efficiently simulates the same network on multiple input
    patterns in parallel.
    """

    def __init__(
        self,
        create_network_fn: Callable,
        n_parallel: int = 1,
    ) -> None:
        """Initialize batch simulator.

        Args:
            create_network_fn: Function to create a network instance
            n_parallel: Number of parallel networks
        """
        self.create_network_fn = create_network_fn
        self.n_parallel = n_parallel

        # Create network instances
        self.networks = [create_network_fn() for _ in range(n_parallel)]

    def run_batch(
        self,
        batch_inputs: List[NDArray],
        config: SimulationConfig,
    ) -> List[SimulationResult]:
        """Run simulation on batch of inputs.

        Args:
            batch_inputs: List of input arrays
            config: Simulation configuration

        Returns:
            List of simulation results
        """
        results = []

        # Process in chunks
        for i in range(0, len(batch_inputs), self.n_parallel):
            chunk = batch_inputs[i:i + self.n_parallel]

            # Reset networks
            for network in self.networks:
                network.reset()

            # Run simulations
            chunk_results = []
            for j, inputs in enumerate(chunk):
                if j < len(self.networks):
                    # Create simulator for this network
                    sim = TimeSteppedSimulator(
                        self.networks[j].neurons,
                        self.networks[j].synapses,
                    )
                    result = sim.run(inputs, config)
                    chunk_results.append(result)

            results.extend(chunk_results)

        return results


def create_simulation_input(
    n_neurons: int,
    timesteps: int,
    input_type: str = "poisson",
    rate: float = 10.0,
    dt: float = 1.0,
) -> NDArray:
    """Create simulation input patterns.

    Args:
        n_neurons: Number of input neurons
        timesteps: Number of timesteps
        input_type: Type of input ("poisson", "constant", "burst")
        rate: Spike rate (Hz) for Poisson or current for constant
        dt: Timestep (ms)

    Returns:
        Input array (timesteps, n_neurons)
    """
    if input_type == "poisson":
        # Poisson spike trains
        prob = rate * dt / 1000.0
        return (np.random.random((timesteps, n_neurons)) < prob).astype(np.float32)

    elif input_type == "constant":
        # Constant current
        return np.full((timesteps, n_neurons), rate, dtype=np.float32)

    elif input_type == "burst":
        # Burst pattern with golden ratio timing
        inputs = np.zeros((timesteps, n_neurons), dtype=np.float32)
        burst_interval = int(100 * PHI_INV)  # ~62ms between bursts
        burst_duration = 5

        for t in range(0, timesteps, burst_interval):
            for dt in range(min(burst_duration, timesteps - t)):
                inputs[t + dt] = 1.0

        return inputs

    else:
        raise ValueError(f"Unknown input type: {input_type}")
