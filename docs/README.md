# Liquid Neural Network

A PyTorch implementation of Liquid Neural Networks with advanced continuous-time modeling capabilities.

## Features

- **LiquidS4 Model**: State-of-the-art sequence modeling with liquid time-constant networks
- **CfC Model**: Closed-form Continuous-time model for enhanced temporal processing
- **Visualization Tools**: Built-in tools for analyzing model behavior and performance
- **Comprehensive Testing**: Full test coverage with advanced test cases
- **Example Notebooks**: Interactive tutorials and usage examples

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from liquid_neural_network import LiquidS4Model

# Create model
model = LiquidS4Model(
    input_size=10,
    hidden_size=64,
    output_size=10
)

# Generate sample data
x = torch.randn(32, 100, 10)  # (batch_size, seq_length, input_size)

# Get predictions
output = model(x)
```

## Models

### LiquidS4Model

The LiquidS4Model combines liquid networks with the S4 (Structured State Space Sequence) model:

```python
from liquid_neural_network import LiquidS4Model

model = LiquidS4Model(
    input_size=10,    # Input feature dimension
    hidden_size=64,   # Hidden state dimension
    output_size=10    # Output feature dimension
)
```

### CfCModel

The CfC (Closed-form Continuous-time) model provides enhanced temporal processing:

```python
from liquid_neural_network import CfCModel

model = CfCModel(
    input_size=10,    # Input feature dimension
    hidden_size=64,   # Hidden state dimension
    output_size=10    # Output feature dimension
)

# Process sequence with optional hidden state
output, hidden = model(x, hidden=None)
```

## Examples

See the `examples/` directory for interactive notebooks demonstrating:
- Basic usage and model training
- Sequence prediction tasks
- Time series forecasting
- Model visualization and analysis

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=./
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{liquid_neural_network,
  title={Liquid Neural Network},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/liquid-neural-network}
}
```

## Table of Contents

1. [Getting Started](getting_started.md)
   - Installation
   - Basic Usage
   - Quick Examples

2. [Architecture](architecture/README.md)
   - [Core Components](architecture/core_components.md)
   - [Liquid-S4 Model](architecture/liquid_s4.md)
   - [CfC Model](architecture/cfc_model.md)
   - [Visualization Tools](architecture/visualization.md)

3. [Tutorials](tutorials/README.md)
   - [Basic Tutorial](tutorials/basic_tutorial.md)
   - [Advanced Usage](tutorials/advanced_usage.md)
   - [Real-world Examples](tutorials/real_world_examples.md)

4. [API Reference](api/README.md)
   - [LiquidS4Model](api/liquid_s4.md)
   - [CfCModel](api/cfc_model.md)
   - [NetworkVisualizer](api/visualization.md)

5. [Benchmarks](benchmarks/README.md)
   - Performance Metrics
   - Comparison with Other Models
   - Hardware Requirements
