"""Tests for gnsp.snn.neuron module."""

import numpy as np
import pytest

from gnsp.snn.neuron import (
    LIFNeuron,
    LIFNeuronArray,
    LIFNeuronParams,
    AdaptiveLIFNeuronArray,
    IzhikevichNeuronArray,
    create_golden_neuron_params,
)
from gnsp.constants import PHI, PHI_INV, PHI_INV_SQ


class TestLIFNeuronParams:
    """Test LIF neuron parameters."""

    def test_default_params(self):
        """Test default parameters use golden ratio values."""
        params = LIFNeuronParams()
        assert params.threshold == pytest.approx(PHI, rel=1e-10)
        assert params.reset == pytest.approx(PHI_INV, rel=1e-10)
        assert params.leak == pytest.approx(PHI_INV_SQ, rel=1e-10)

    def test_invalid_threshold_reset(self):
        """Test threshold must be greater than reset."""
        with pytest.raises(ValueError):
            LIFNeuronParams(threshold=0.5, reset=0.6)

    def test_invalid_leak(self):
        """Test leak must be in (0, 1)."""
        with pytest.raises(ValueError):
            LIFNeuronParams(leak=1.5)
        with pytest.raises(ValueError):
            LIFNeuronParams(leak=-0.1)

    def test_invalid_refractory(self):
        """Test refractory period must be non-negative."""
        with pytest.raises(ValueError):
            LIFNeuronParams(refractory_period=-1)


class TestLIFNeuron:
    """Test single LIF neuron."""

    def test_reset_state(self):
        """Test state reset."""
        neuron = LIFNeuron()
        neuron.v = 1.0
        neuron.spike_count = 5
        neuron.reset_state()
        assert neuron.v == 0.0
        assert neuron.spike_count == 0

    def test_spike_on_threshold(self):
        """Test neuron spikes at threshold."""
        params = LIFNeuronParams(threshold=1.0, reset=0.0, leak=0.9)
        neuron = LIFNeuron(params)

        # Below threshold - no spike
        spiked = neuron.step(0.5)
        assert not spiked

        # Above threshold - spike
        spiked = neuron.step(1.0)
        assert spiked
        assert neuron.v == 0.0  # Reset

    def test_refractory_period(self):
        """Test neuron doesn't spike during refractory."""
        params = LIFNeuronParams(threshold=1.0, reset=0.0, leak=0.9, refractory_period=3)
        neuron = LIFNeuron(params)

        # Force spike
        neuron.step(2.0)
        assert neuron.spike_count == 1

        # During refractory - no spike even with high input
        for _ in range(3):
            spiked = neuron.step(10.0)
            assert not spiked

        # After refractory - can spike again
        spiked = neuron.step(2.0)
        assert spiked

    def test_leak(self):
        """Test membrane potential leak."""
        params = LIFNeuronParams(threshold=10.0, reset=0.0, leak=0.5)
        neuron = LIFNeuron(params)

        neuron.step(2.0)  # v = 2.0
        neuron.step(0.0)  # v = 2.0 * 0.5 = 1.0

        assert neuron.v == pytest.approx(1.0)


class TestLIFNeuronArray:
    """Test vectorized LIF neuron array."""

    def test_create_array(self):
        """Test creating neuron array."""
        neurons = LIFNeuronArray(100)
        assert neurons.n_neurons == 100
        assert neurons.v.shape == (100,)

    def test_invalid_size(self):
        """Test invalid array size."""
        with pytest.raises(ValueError):
            LIFNeuronArray(0)
        with pytest.raises(ValueError):
            LIFNeuronArray(-5)

    def test_step(self):
        """Test single step update."""
        neurons = LIFNeuronArray(10)
        currents = np.ones(10) * 2.0
        spikes = neurons.step(currents)

        assert spikes.shape == (10,)
        assert spikes.dtype == np.bool_

    def test_step_batch(self):
        """Test batch step update."""
        neurons = LIFNeuronArray(10)
        currents = np.ones((50, 10)) * 0.5
        spikes = neurons.step_batch(currents)

        assert spikes.shape == (50, 10)
        assert spikes.dtype == np.bool_

    def test_spike_counts(self):
        """Test spike counting."""
        params = LIFNeuronParams(threshold=1.0, reset=0.0, leak=0.9, refractory_period=0)
        neurons = LIFNeuronArray(10, params)

        # Strong input should cause spikes
        for _ in range(100):
            neurons.step(np.ones(10) * 2.0)

        counts = neurons.get_spike_counts()
        assert np.all(counts > 0)

    def test_firing_rates(self):
        """Test firing rate calculation."""
        params = LIFNeuronParams(threshold=1.0, reset=0.0, leak=0.9, refractory_period=0)
        neurons = LIFNeuronArray(10, params)

        for _ in range(100):
            neurons.step(np.ones(10) * 2.0)

        rates = neurons.get_firing_rates(100)
        assert np.all(rates >= 0)
        assert np.all(rates <= 1)

    def test_reset_state(self):
        """Test state reset."""
        neurons = LIFNeuronArray(10)
        neurons.step(np.ones(10) * 2.0)
        neurons.reset_state()

        assert np.all(neurons.v == 0.0)
        assert np.all(neurons.spike_counts == 0)


class TestAdaptiveLIFNeuronArray:
    """Test adaptive LIF neurons."""

    def test_adaptation_increases_on_spike(self):
        """Test adaptation increases when neuron spikes."""
        params = LIFNeuronParams(threshold=1.0, reset=0.0, leak=0.9, refractory_period=0)
        neurons = AdaptiveLIFNeuronArray(10, params)

        initial_adaptation = neurons.get_adaptation().copy()

        # Strong input causes spikes
        neurons.step(np.ones(10) * 2.0)

        new_adaptation = neurons.get_adaptation()
        assert np.all(new_adaptation >= initial_adaptation)

    def test_adaptation_decays(self):
        """Test adaptation decays over time."""
        neurons = AdaptiveLIFNeuronArray(10)

        # Build up adaptation
        for _ in range(10):
            neurons.step(np.ones(10) * 2.0)

        high_adaptation = neurons.get_adaptation().copy()

        # Let it decay with no input
        for _ in range(50):
            neurons.step(np.zeros(10))

        low_adaptation = neurons.get_adaptation()
        assert np.all(low_adaptation < high_adaptation)


class TestIzhikevichNeuronArray:
    """Test Izhikevich neuron model."""

    def test_create(self):
        """Test creating Izhikevich neurons."""
        neurons = IzhikevichNeuronArray(10)
        assert neurons.n_neurons == 10

    def test_spike_at_threshold(self):
        """Test neurons spike at 30mV threshold."""
        neurons = IzhikevichNeuronArray(10)

        # Strong sustained input should cause spikes
        total_spikes = 0
        for _ in range(100):
            spikes = neurons.step(np.ones(10) * 20.0)
            total_spikes += np.sum(spikes)

        assert total_spikes > 0

    def test_get_state(self):
        """Test getting state variables."""
        neurons = IzhikevichNeuronArray(10)
        v, u = neurons.get_state()

        assert v.shape == (10,)
        assert u.shape == (10,)


class TestCreateGoldenNeuronParams:
    """Test golden neuron parameter creation."""

    def test_default_levels(self):
        """Test default level parameters."""
        params = create_golden_neuron_params()

        assert params.threshold == pytest.approx(PHI, rel=1e-10)
        assert params.reset == pytest.approx(PHI_INV, rel=1e-10)
        assert params.leak == pytest.approx(PHI_INV_SQ, rel=1e-10)

    def test_positive_threshold_level(self):
        """Test positive threshold level."""
        params = create_golden_neuron_params(threshold_level=1)
        assert params.threshold == pytest.approx(PHI * PHI, rel=1e-10)

    def test_leak_clipped(self):
        """Test leak is clipped to valid range."""
        params = create_golden_neuron_params(leak_level=2)
        assert 0 < params.leak < 1
