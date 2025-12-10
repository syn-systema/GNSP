# Golden Neuromorphic Security Platform (GNSP)

A research platform for neuromorphic intrusion detection using spiking neural networks
with golden ratio-based architecture.

## Features

- Spiking Neural Networks with golden ratio dynamics
- Automata-theoretic protocol analysis
- Category-theoretic anomaly detection
- Topological data analysis (persistent homology)
- Clifford algebra state representations

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from gnsp.snn.network import SpikingNeuralNetwork, SNNConfig

# Create network with golden ratio architecture
config = SNNConfig(n_input=80, n_hidden=(64, 32), n_output=5)
network = SpikingNeuralNetwork(config)

# Run simulation
outputs = network.run(inputs)
```

## Project Structure

```
gnsp/
  core/       - Mathematical foundations (golden ratio, Fibonacci)
  snn/        - Spiking neural network implementation
  automata/   - Automata theory (DFA, NFA, Buchi, weighted)
  category/   - Category theory (functors, sheaves)
  topology/   - Topological data analysis
  algebra/    - Clifford/geometric algebra
  detection/  - Integrated detection system
  training/   - Training and evaluation
```

## License

MIT
