# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNSP (Golden Neuromorphic Security Platform) is a research platform for neuromorphic intrusion detection using spiking neural networks with golden ratio-based architecture. The project is a software simulation platform designed to validate algorithms before hardware deployment on FPGA.

## Build and Development Commands

```bash
# Bootstrap project structure (run from project root)
python gnsp-bootstrap.py

# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=gnsp --cov-report=html

# Run a single test file
pytest tests/test_core/test_golden.py -v

# Run a specific test
pytest tests/test_core/test_golden.py::TestGoldenDecay::test_single_step -v

# Type checking
mypy gnsp/

# Format code
black gnsp/ tests/

# Lint code
ruff gnsp/ tests/
```

## Architecture

### Core Mathematical Foundation
- **Golden ratio (phi = 1.618...)** is the fundamental constant throughout the system
- **Fibonacci sequences** used for timing, connectivity, and learning rules
- **Fixed-point arithmetic** (Q8.8 or Q16.16) for hardware compatibility

### Module Structure

**gnsp/core/** - Mathematical utilities
- `fixed_point.py` - Q-format fixed-point arithmetic for FPGA compatibility
- `golden.py` - Golden ratio utilities (decay, spiral points, matrices)
- `fibonacci.py` - Fibonacci sequence generators and utilities

**gnsp/snn/** - Spiking Neural Networks
- `neuron.py` - LIF neurons with golden ratio dynamics (threshold=phi, reset=1/phi, leak=1/phi^2)
- `synapse.py` - Synaptic connections with golden ratio weight quantization (9 levels)
- `stdp.py` - Spike-Timing Dependent Plasticity with Fibonacci time constants
- `topology.py` - Quasicrystalline network topology using golden spiral positioning
- `network.py` - Main SNN container integrating all components

**gnsp/automata/** - Automata-theoretic protocol analysis
- DFA/NFA, weighted automata, Buchi automata, timed automata
- Krohn-Rhodes decomposition, L* learning algorithm

**gnsp/category/** - Category-theoretic anomaly detection
- Protocol categories (TCP, HTTP, DNS)
- Security functors mapping Protocol -> Traffic
- Sheaf-theoretic distributed detection

**gnsp/topology/** - Topological Data Analysis
- Vietoris-Rips complex construction at golden ratio scales
- Persistent homology computation
- TDA feature extraction for ML

**gnsp/algebra/** - Clifford/Geometric algebra Cl(3,0)
- Multivector representation and geometric product
- Rotor operations for state transformations

**gnsp/detection/** - Integrated detection system
- Ensemble combining SNN, automata, category, and TDA methods

**gnsp/training/** - Training pipeline
- Dataset loading (NSL-KDD, UNSW-NB15, CICIDS)
- Surrogate gradient training for SNN

## Code Style Requirements

- Python 3.11+ with full type hints
- Dataclasses for structured data
- Abstract base classes for extensibility
- NumPy for numerical operations
- **No emojis in code, comments, or documentation**
- Docstrings in Google style
- Hardware-aware design (think FPGA synthesis)
- Fixed-point arithmetic compatibility

## Key Constants

```python
PHI = 1.618033988749895           # Golden ratio
PHI_INV = 0.618033988749895       # 1/phi
WEIGHT_LEVELS = (-2.618, -1.618, -1.0, -0.618, 0.0, 0.618, 1.0, 1.618, 2.618)
STDP_TAU = (1, 2, 3, 5, 8, 13, 21, 34)  # Fibonacci time constants
```

## Testing Philosophy

- Every function needs unit tests
- Property-based testing for mathematical properties (use hypothesis library)
- Golden ratio invariants must be verified
- Integration tests for full pipelines
- Specific tests for "golden ratio hypothesis" comparing phi-based vs standard parameters

## Implementation Order

1. Core module (constants, fixed-point, golden, fibonacci)
2. SNN neurons, synapses, STDP, topology, network
3. Automata (DFA/NFA, weighted, Buchi)
4. Category theory (categories, functors, sheaves)
5. Topology (simplicial complexes, persistence)
6. Algebra (Clifford/geometric)
7. Detection integration and training pipeline

## Configuration

Configuration files are in `configs/` directory using YAML format. Default config at `configs/default.yaml`.
