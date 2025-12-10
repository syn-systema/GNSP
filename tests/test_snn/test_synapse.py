"""Tests for gnsp.snn.synapse module."""

import numpy as np
import pytest

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
from gnsp.constants import PHI, PHI_INV, WEIGHT_LEVELS


class TestWeightQuantization:
    """Test weight quantization functions."""

    def test_quantize_zero(self):
        """Test zero quantizes to zero level."""
        level = quantize_weight(0.0)
        assert level == WeightLevel.ZERO

    def test_quantize_phi(self):
        """Test phi quantizes correctly."""
        level = quantize_weight(PHI)
        assert level == WeightLevel.PHI

    def test_quantize_phi_inv(self):
        """Test 1/phi quantizes correctly."""
        level = quantize_weight(PHI_INV)
        assert level == WeightLevel.PHI_INV

    def test_quantize_negative_phi(self):
        """Test negative phi quantizes correctly."""
        level = quantize_weight(-PHI)
        assert level == WeightLevel.NEG_PHI

    def test_quantize_array(self):
        """Test array quantization."""
        weights = np.array([0.0, PHI, -PHI, PHI_INV])
        levels = quantize_weight_array(weights)

        assert levels[0] == WeightLevel.ZERO
        assert levels[1] == WeightLevel.PHI
        assert levels[2] == WeightLevel.NEG_PHI
        assert levels[3] == WeightLevel.PHI_INV

    def test_dequantize_roundtrip(self):
        """Test quantize then dequantize gives weight level value."""
        for level in WeightLevel:
            weight = WEIGHT_LEVELS[level]
            quantized = quantize_weight(weight)
            dequantized = dequantize_weight(quantized)
            assert dequantized == pytest.approx(weight, rel=1e-10)

    def test_dequantize_array(self):
        """Test array dequantization."""
        levels = np.array([0, 4, 8], dtype=np.int8)
        weights = dequantize_weight_array(levels)

        assert weights[0] == pytest.approx(WEIGHT_LEVELS[0])
        assert weights[1] == pytest.approx(WEIGHT_LEVELS[4])
        assert weights[2] == pytest.approx(WEIGHT_LEVELS[8])


class TestSynapseParams:
    """Test synapse parameters."""

    def test_defaults(self):
        """Test default parameters."""
        params = SynapseParams()
        assert params.delay_min == 1
        assert params.delay_max >= params.delay_min
        assert params.use_quantized is True

    def test_invalid_delay(self):
        """Test invalid delay raises error."""
        with pytest.raises(ValueError):
            SynapseParams(delay_min=0)
        with pytest.raises(ValueError):
            SynapseParams(delay_min=5, delay_max=3)


class TestSynapseArray:
    """Test synapse array."""

    def test_create(self):
        """Test creating synapse array."""
        synapses = SynapseArray(10, 20)
        assert synapses.n_pre == 10
        assert synapses.n_post == 20

    def test_invalid_size(self):
        """Test invalid sizes raise error."""
        with pytest.raises(ValueError):
            SynapseArray(0, 10)
        with pytest.raises(ValueError):
            SynapseArray(10, 0)

    def test_initialize_dense(self):
        """Test dense initialization."""
        synapses = SynapseArray(10, 20)
        weights = np.random.randn(10, 20).astype(np.float32)
        synapses.initialize_dense(weights)

        stored = synapses.get_weights()
        # Weights should be quantized
        assert stored.shape == (10, 20)

    def test_initialize_sparse(self):
        """Test sparse initialization."""
        synapses = SynapseArray(10, 20)
        pre = np.array([0, 1, 2], dtype=np.int32)
        post = np.array([5, 10, 15], dtype=np.int32)
        weights = np.array([PHI, PHI_INV, -PHI], dtype=np.float32)

        synapses.initialize_sparse(pre, post, weights)
        assert synapses.get_connection_count() == 3

    def test_initialize_random(self):
        """Test random initialization."""
        synapses = SynapseArray(100, 100)
        synapses.initialize_random(density=0.1)

        # Should have approximately 10% connections
        density = synapses.get_density()
        assert 0.05 < density < 0.2

    def test_propagate_instant(self):
        """Test instant propagation without delays."""
        synapses = SynapseArray(10, 20)
        weights = np.ones((10, 20), dtype=np.float32) * PHI_INV
        synapses.initialize_dense(weights)

        pre_spikes = np.zeros(10, dtype=np.bool_)
        pre_spikes[0] = True
        pre_spikes[5] = True

        currents = synapses.propagate_instant(pre_spikes)
        assert currents.shape == (20,)
        assert np.all(currents > 0)  # Two presynaptic spikes

    def test_get_set_weights(self):
        """Test getting and setting weights."""
        synapses = SynapseArray(10, 20)
        synapses.initialize_random(density=1.0)

        weights = synapses.get_weights()
        weights *= 2  # Modify
        synapses.set_weights(weights)

        new_weights = synapses.get_weights()
        # Should be quantized version of 2x original
        assert new_weights.shape == weights.shape

    def test_update_single_weight(self):
        """Test updating a single weight."""
        synapses = SynapseArray(10, 20, SynapseParams(use_quantized=False))
        synapses.initialize_dense(np.zeros((10, 20), dtype=np.float32))

        synapses.update_weight(0, 0, 1.0)
        weights = synapses.get_weights()
        assert weights[0, 0] == pytest.approx(1.0)


class TestFibonacciDelays:
    """Test Fibonacci delay creation."""

    def test_delay_shape(self):
        """Test delay matrix shape."""
        delays = create_fibonacci_delays(10, 20)
        assert delays.shape == (10, 20)

    def test_delays_positive(self):
        """Test all delays are positive."""
        delays = create_fibonacci_delays(10, 20)
        assert np.all(delays >= 1)


class TestGoldenWeightMatrix:
    """Test golden weight matrix creation."""

    def test_random_pattern(self):
        """Test random weight pattern."""
        weights = create_golden_weight_matrix(10, 20, pattern="random")
        assert weights.shape == (10, 20)

        # All values should be from WEIGHT_LEVELS
        unique = np.unique(weights)
        for val in unique:
            assert any(abs(val - w) < 1e-6 for w in WEIGHT_LEVELS)

    def test_structured_pattern(self):
        """Test structured weight pattern."""
        weights = create_golden_weight_matrix(10, 20, pattern="structured")
        assert weights.shape == (10, 20)

    def test_balanced_pattern(self):
        """Test balanced weight pattern."""
        weights = create_golden_weight_matrix(10, 20, pattern="balanced")
        assert weights.shape == (10, 20)

    def test_excitatory_ratio(self):
        """Test excitatory ratio is approximately correct."""
        weights = create_golden_weight_matrix(100, 100, excitatory_ratio=0.8)

        positive = np.sum(weights > 0)
        total = np.sum(weights != 0)
        ratio = positive / total

        # Should be approximately 0.8 (with some variance)
        assert 0.6 < ratio < 1.0
