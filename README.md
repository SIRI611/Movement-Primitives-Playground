# Movement Primitives Playground

A comprehensive Python framework for implementing various Movement Primitives algorithms including DMP, ProMP, ProDMP, CNMP, and other classic movement primitive approaches.

## Overview

Movement Primitives (MPs) are a powerful framework for learning and reproducing complex motor behaviors. This repository provides implementations of several state-of-the-art movement primitive algorithms:

- **Dynamic Movement Primitives (DMP)**: Learn and reproduce complex movements using dynamical systems
- **Probabilistic Movement Primitives (ProMP)**: Extend DMPs with probabilistic modeling for uncertainty quantification
- **Probabilistic Dynamic Movement Primitives (ProDMP)**: Combine benefits of DMPs and ProMPs
- **Conditional Neural Movement Primitives (CNMP)**: Use neural networks for context-conditioned movement generation

## Features

- 🚀 **Multiple Algorithms**: Implementations of DMP, ProMP, ProDMP, and CNMP
- 📊 **Uncertainty Quantification**: Probabilistic methods provide uncertainty estimates
- 🎯 **Goal Modification**: Easy goal and start position modification
- ⏱️ **Temporal Scaling**: Adjustable execution times
- 🧠 **Context Conditioning**: CNMP supports context-dependent movement generation
- 📈 **Visualization**: Built-in plotting and analysis tools
- 🔧 **Extensible**: Clean, modular design for easy extension
- 📚 **Examples**: Comprehensive examples and demonstrations

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- PyTorch (for CNMP)
- Optional: TensorFlow (for additional neural network support)

### Install from Source

```bash
git clone https://github.com/SIRI611/Movement-Primitives-Playground.git
cd Movement-Primitives-Playground
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic DMP Usage

```python
import numpy as np
from movement_primitives import DMP, DMPConfig

# Create demonstration trajectory
t = np.linspace(0, 1, 100)
trajectory = np.column_stack([t, 0.5 * np.sin(2 * np.pi * t)])

# Configure and train DMP
config = DMPConfig(n_dims=2, dt=0.01, execution_time=1.0)
dmp = DMP(config)
dmp.fit([trajectory])

# Generate new trajectory
new_trajectory = dmp.generate(
    start=np.array([0.0, 0.0]),
    goal=np.array([1.0, 0.0])
)
```

### Probabilistic Movement Primitives

```python
from movement_primitives import ProMP, ProMPConfig

# Create multiple demonstrations
trajectories = [create_trajectory() for _ in range(5)]

# Train ProMP
config = ProMPConfig(n_dims=2, dt=0.01, execution_time=1.0)
promp = ProMP(config)
promp.fit(trajectories)

# Generate mean trajectory
mean_traj = promp.generate(start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))

# Sample multiple trajectories
samples = promp.sample_trajectories(10, start=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))

# Get uncertainty
variance = promp.get_trajectory_variance()
```

### Conditional Neural Movement Primitives

```python
from movement_primitives import CNMP, CNMPConfig

# Create contextual demonstrations
trajectories = [create_trajectory(ctx) for ctx in contexts]
contexts = [np.array([amp, freq]) for amp, freq in context_params]

# Train CNMP
config = CNMPConfig(n_dims=2, context_dim=2, n_epochs=500)
cnmp = CNMP(config)
cnmp.fit(trajectories, contexts)

# Generate with new context
new_context = np.array([0.5, 1.5])
trajectory = cnmp.generate(
    start=np.array([0.0, 0.0]),
    goal=np.array([1.0, 0.0]),
    context=new_context
)
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

- `dmp_demo.py`: Basic DMP usage and visualization
- `promp_demo.py`: Probabilistic movement primitives with uncertainty
- `prodmp_demo.py`: Probabilistic dynamic movement primitives
- `cnmp_demo.py`: Conditional neural movement primitives
- `comparison_demo.py`: Comprehensive comparison of all methods

Run examples:

```bash
python examples/dmp_demo.py
python examples/promp_demo.py
python examples/prodmp_demo.py
python examples/cnmp_demo.py
python examples/comparison_demo.py
```

## Algorithm Details

### Dynamic Movement Primitives (DMP)

DMPs encode movements as a set of nonlinear differential equations that can be learned from demonstrations and reproduced with different goals and temporal scales.

**Key Features:**
- Fast training and execution
- Goal and temporal scaling
- Smooth trajectory generation
- Robust to perturbations

### Probabilistic Movement Primitives (ProMP)

ProMPs extend DMPs by incorporating probabilistic modeling to handle variations in demonstrations and enable probabilistic inference.

**Key Features:**
- Uncertainty quantification
- Waypoint conditioning
- Probabilistic sampling
- Bayesian learning

### Probabilistic Dynamic Movement Primitives (ProDMP)

ProDMPs combine the benefits of DMPs and ProMPs by incorporating probabilistic modeling into the DMP framework.

**Key Features:**
- All DMP features
- Probabilistic modeling
- Full trajectory analysis (position, velocity, acceleration)
- Uncertainty quantification

### Conditional Neural Movement Primitives (CNMP)

CNMPs use neural networks to learn complex movement patterns conditioned on contextual information.

**Key Features:**
- Context conditioning
- Neural network flexibility
- Complex pattern learning
- High-dimensional contexts

## API Reference

### Base Classes

- `BaseMovementPrimitive`: Abstract base class for all MPs
- `MovementPrimitiveConfig`: Base configuration class

### Implementations

- `DMP`: Dynamic Movement Primitives
- `ProMP`: Probabilistic Movement Primitives  
- `ProDMP`: Probabilistic Dynamic Movement Primitives
- `CNMP`: Conditional Neural Movement Primitives

### Utilities

- `PhaseGenerator`: Phase variable generation
- `GaussianBasis`: Gaussian basis functions
- `RadialBasisFunction`: Radial basis functions
- `TrajectoryGenerator`: Trajectory manipulation utilities
- `plot_trajectory`: Visualization functions

## Configuration

Each MP type has its own configuration class with specific parameters:

```python
# DMP Configuration
config = DMPConfig(
    n_dims=2,                    # Number of dimensions
    dt=0.01,                     # Time step
    execution_time=1.0,          # Execution time
    n_basis=50,                  # Number of basis functions
    basis_width=0.1,             # Basis function width
    alpha=25.0,                  # Phase convergence rate
    beta=6.25,                   # Phase convergence rate
    regularization=1e-6          # Regularization parameter
)

# ProMP Configuration (extends DMPConfig)
config = ProMPConfig(
    # ... all DMP parameters ...
    noise_variance=1e-6,         # Observation noise variance
    prior_variance=1.0           # Prior variance for weights
)

# CNMP Configuration (extends DMPConfig)
config = CNMPConfig(
    # ... all DMP parameters ...
    context_dim=2,               # Context dimension
    hidden_dim=128,              # Hidden layer dimension
    n_hidden_layers=2,           # Number of hidden layers
    activation='relu',           # Activation function
    learning_rate=1e-3,          # Learning rate
    n_epochs=1000                # Number of training epochs
)
```

## Performance Comparison

| Method | Training Time | Generation Time | Probabilistic | Context | Uncertainty |
|--------|---------------|-----------------|---------------|---------|-------------|
| DMP    | Fast          | Very Fast       | No            | No      | No          |
| ProMP  | Fast          | Fast            | Yes           | No      | Yes         |
| ProDMP | Fast          | Fast            | Yes           | No      | Yes         |
| CNMP   | Slow          | Medium          | No            | Yes     | No          |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/yourusername/Movement-Primitives-Playground.git
cd Movement-Primitives-Playground
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{movement_primitives_playground,
  title={Movement Primitives Playground: A Comprehensive Framework for Movement Primitive Algorithms},
  author={Xirui Shi},
  year={2024},
  url={https://github.com/SIRI611/Movement-Primitives-Playground.git}
}
```

## References

1. Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., & Schaal, S. (2013). Dynamical movement primitives: learning attractor models for motor behaviors. Neural computation, 25(2), 328-373.

2. Paraschos, A., Daniel, C., Peters, J., & Neumann, G. (2013). Probabilistic movement primitives. Advances in neural information processing systems, 26.

3. Maeda, G. J., Neumann, G., Ewerton, M., Lioutikov, R., Kroemer, O., & Peters, J. (2017). Probabilistic movement primitives for coordination of multiple human–robot collaborative tasks. Autonomous Robots, 41(3), 593-612.

4. Ewerton, M., Maeda, G., Koert, D., Kolev, Z., Takahashi, M., Peters, J., & Neumann, G. (2019). Reinforcement learning of trajectory distributions for continuous motion planning. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

## Acknowledgments

- Original DMP implementation by Auke Ijspeert
- ProMP implementation by Alexandros Paraschos
- CNMP implementation inspired by recent neural MP research
