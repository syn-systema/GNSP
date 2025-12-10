#!/usr/bin/env python3
"""
GNSP Project Bootstrap Script

Run this to create the full project structure and initial files.
"""

import os
from pathlib import Path

PROJECT_NAME = "gnsp"

# Directory structure
DIRECTORIES = [
    f"{PROJECT_NAME}",
    f"{PROJECT_NAME}/core",
    f"{PROJECT_NAME}/snn",
    f"{PROJECT_NAME}/automata",
    f"{PROJECT_NAME}/category",
    f"{PROJECT_NAME}/topology",
    f"{PROJECT_NAME}/algebra",
    f"{PROJECT_NAME}/network",
    f"{PROJECT_NAME}/detection",
    f"{PROJECT_NAME}/training",
    f"{PROJECT_NAME}/visualization",
    "tests",
    "tests/test_core",
    "tests/test_snn",
    "tests/test_automata",
    "tests/test_category",
    "tests/test_topology",
    "tests/test_algebra",
    "tests/test_integration",
    "notebooks",
    "configs",
    "data/raw",
    "data/processed",
    "data/synthetic",
    "models/checkpoints",
    "models/trained",
    "scripts",
    "docs",
]

# Files to create with initial content
FILES = {
    "pyproject.toml": '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gnsp"
version = "0.1.0"
description = "Golden Neuromorphic Security Platform - SNN-based IDS with golden ratio architecture"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "GNSP Team"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Security",
]
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "torch>=2.0.0",
    "scikit-learn>=1.2.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "ruff>=0.0.260",
    "hypothesis>=6.75.0",
]
notebooks = [
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
]
tda = [
    "giotto-tda>=0.6.0",
    "gudhi>=3.8.0",
]
snn = [
    "snnTorch>=0.7.0",
    "brian2>=2.5.0",
]
all = ["gnsp[dev,notebooks,tda,snn]"]

[tool.setuptools.packages.find]
where = ["."]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
''',

    "requirements.txt": '''# Core
numpy>=1.24.0
scipy>=1.10.0

# Machine Learning
torch>=2.0.0
scikit-learn>=1.2.0

# Data Processing
pandas>=2.0.0
h5py>=3.8.0
pyyaml>=6.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0

# Development
black>=23.0.0
mypy>=1.0.0
ruff>=0.0.260
''',

    "README.md": '''# Golden Neuromorphic Security Platform (GNSP)

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
''',

    f"{PROJECT_NAME}/__init__.py": '''"""
Golden Neuromorphic Security Platform (GNSP)

A neuromorphic intrusion detection system using spiking neural networks
with golden ratio-based architecture.
"""

__version__ = "0.1.0"
__author__ = "GNSP Team"

from gnsp.constants import PHI, PHI_INV, FIBONACCI
''',

    f"{PROJECT_NAME}/constants.py": '''"""
Core mathematical constants for GNSP.

The golden ratio appears throughout as the fundamental constant,
following insights from quantum gravity research showing phi
as the dominant non-integer eigenvalue of binary matrices.
"""

import math
from typing import Tuple

# Golden Ratio and Related Constants
PHI: float = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INV: float = 1 / PHI              # 0.618033988749895
PHI_SQ: float = PHI ** 2              # 2.618033988749895
PHI_INV_SQ: float = PHI_INV ** 2      # 0.381966011250105

# Fixed-point representations (Q8.8 format)
PHI_Q8_8: int = 0x019E        # 1.617 in Q8.8
PHI_INV_Q8_8: int = 0x009E    # 0.617 in Q8.8
PHI_SQ_Q8_8: int = 0x029E     # 2.617 in Q8.8

# Fibonacci sequence (pre-computed)
FIBONACCI: Tuple[int, ...] = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610)

# Golden angle (radians)
GOLDEN_ANGLE: float = 2 * math.pi / (PHI ** 2)  # ~137.5 degrees

# Cabibbo angle from E8 lattice
CABIBBO_ANGLE: float = math.atan(1 / (PHI ** 3))  # ~13.28 degrees

# Detection thresholds based on golden ratio
THRESHOLD_LOW: float = PHI_INV ** 3     # ~0.236
THRESHOLD_MID: float = PHI_INV ** 2     # ~0.382
THRESHOLD_HIGH: float = PHI_INV         # ~0.618
THRESHOLD_CRITICAL: float = 1.0

# SNN parameters
DEFAULT_THRESHOLD: float = PHI           # Spike threshold
DEFAULT_RESET: float = PHI_INV           # Reset potential
DEFAULT_LEAK: float = PHI_INV ** 2       # Leak rate

# Weight quantization levels (golden ratio ladder)
WEIGHT_LEVELS: Tuple[float, ...] = (
    -PHI_SQ,      # -2.618
    -PHI,         # -1.618
    -1.0,
    -PHI_INV,     # -0.618
    0.0,
    PHI_INV,      # 0.618
    1.0,
    PHI,          # 1.618
    PHI_SQ,       # 2.618
)

# STDP time constants (Fibonacci in milliseconds)
STDP_TAU: Tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34)
''',

    f"{PROJECT_NAME}/core/__init__.py": '''"""Core mathematical utilities."""

from gnsp.core.golden import (
    golden_decay,
    golden_threshold,
    fibonacci_sequence,
    golden_spiral_points,
    golden_matrix,
)
from gnsp.core.fibonacci import (
    fib,
    fibonacci_up_to,
    is_fibonacci,
    fibonacci_offsets,
)
''',

    f"{PROJECT_NAME}/snn/__init__.py": '''"""Spiking Neural Network module."""

from gnsp.snn.neuron import LIFNeuron, LIFNeuronArray, LIFNeuronParams
from gnsp.snn.synapse import SynapseArray, SynapseParams
from gnsp.snn.stdp import FibonacciSTDP, OnlineSTDP
from gnsp.snn.network import SpikingNeuralNetwork, SNNConfig
''',

    f"{PROJECT_NAME}/automata/__init__.py": '''"""Automata theory module."""
''',

    f"{PROJECT_NAME}/category/__init__.py": '''"""Category theory module."""
''',

    f"{PROJECT_NAME}/topology/__init__.py": '''"""Topological data analysis module."""
''',

    f"{PROJECT_NAME}/algebra/__init__.py": '''"""Clifford/geometric algebra module."""
''',

    f"{PROJECT_NAME}/network/__init__.py": '''"""Network traffic processing module."""
''',

    f"{PROJECT_NAME}/detection/__init__.py": '''"""Detection and classification module."""
''',

    f"{PROJECT_NAME}/training/__init__.py": '''"""Training and evaluation module."""
''',

    f"{PROJECT_NAME}/visualization/__init__.py": '''"""Visualization utilities."""
''',

    "tests/__init__.py": '''"""GNSP test suite."""
''',

    "tests/test_core/__init__.py": '''"""Core module tests."""
''',

    "tests/test_snn/__init__.py": '''"""SNN module tests."""
''',

    "configs/default.yaml": '''# GNSP Default Configuration

project:
  name: "GNSP"
  version: "0.1.0"
  seed: 42

golden:
  phi: 1.618033988749895
  use_golden_weights: true
  use_golden_decay: true
  use_fibonacci_stdp: true

snn:
  n_input: 80
  n_hidden: [64, 32]
  n_output: 5
  
  neuron:
    threshold: 1.618
    reset: 0.618
    leak: 0.382
    refractory_period: 2
  
  topology:
    type: "quasicrystal"
    connection_distances: [1.0, 1.618, 2.618]

training:
  dataset: "nsl-kdd"
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2

detection:
  ensemble_weights:
    snn: 0.4
    automata: 0.2
    category: 0.2
    tda: 0.2
  threshold: 0.618
''',

    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
.venv/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/*
!data/raw/.gitkeep

# Models
models/checkpoints/*
models/trained/*
!models/checkpoints/.gitkeep
!models/trained/.gitkeep

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# OS
.DS_Store
Thumbs.db
''',

    "data/raw/.gitkeep": "",
    "models/checkpoints/.gitkeep": "",
    "models/trained/.gitkeep": "",
}


def create_project():
    """Create the full project structure."""
    base = Path(".")
    
    print("Creating GNSP project structure...")
    print("=" * 50)
    
    # Create directories
    for dir_path in DIRECTORIES:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR]  {dir_path}/")
    
    print()
    
    # Create files
    for file_path, content in FILES.items():
        full_path = base / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        print(f"[FILE] {file_path}")
    
    print()
    print("=" * 50)
    print("Project structure created successfully!")
    print()
    print("Next steps:")
    print("  1. cd gnsp")
    print("  2. pip install -e '.[dev]'")
    print("  3. pytest tests/ -v")
    print("  4. Start implementing gnsp/core/golden.py")
    print()
    print("Refer to gnsp-claude-code-prompt.md for detailed specifications.")


if __name__ == "__main__":
    create_project()
