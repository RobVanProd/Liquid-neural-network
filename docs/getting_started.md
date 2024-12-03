# Getting Started with Liquid Neural Network

This guide will help you get started with the Liquid Neural Network implementation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RobVanProd/Liquid-neural-network.git
cd Liquid-neural-network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

### Using Liquid-S4 Model

```python
from liquid_s4 import LiquidS4Model

# Initialize model
model = LiquidS4Model(
    input_size=10,
    hidden_size=64,
    output_size=10,
    num_layers=4
)

# Process sequence data
batch_size = 32
seq_length = 100
input_data = torch.randn(batch_size, seq_length, input_size)
output = model(input_data)
```

### Using CfC Model

```python
from cfc_model import CfCModel

# Initialize model
model = CfCModel(
    input_size=10,
    hidden_size=64,
    output_size=10
)

# Process sequence data
output = model(input_data)
```

### Visualization Tools

```python
from visualization import NetworkVisualizer

# Initialize visualizer
visualizer = NetworkVisualizer(log_dir='runs/experiment')

# Log metrics
metrics = {'loss': 0.5, 'accuracy': 0.85}
visualizer.log_metrics(metrics, step=0)

# Visualize activations
visualizer.visualize_activations(
    activations=model_output,
    layer_name='output_layer',
    step=0
)

# Plot training progress
visualizer.plot_training_progress('training_progress.png')
```

## Quick Examples

### Time Series Prediction

```python
import torch
import numpy as np
from liquid_s4 import LiquidS4Model

# Generate synthetic time series data
def generate_sine_wave(seq_length, frequency=1.0):
    t = np.linspace(0, 10, seq_length)
    return np.sin(2 * np.pi * frequency * t)

# Create dataset
seq_length = 100
x = generate_sine_wave(seq_length)
x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(-1)

# Initialize model
model = LiquidS4Model(
    input_size=1,
    hidden_size=32,
    output_size=1
)

# Make predictions
with torch.no_grad():
    predictions = model(x)
```

### Visualization Example

```python
import matplotlib.pyplot as plt

# Plot predictions vs actual
plt.figure(figsize=(12, 4))
plt.plot(x[0, :, 0].numpy(), label='Actual')
plt.plot(predictions[0, :, 0].numpy(), label='Predicted')
plt.legend()
plt.title('Time Series Prediction Example')
plt.show()
```

## Next Steps

- Check out the [tutorials](tutorials/README.md) for more detailed examples
- Read about the [architecture](architecture/README.md) to understand the implementation
- Learn how to [contribute](contributing.md) to the project

## Support

If you encounter any issues or have questions:
1. Check the [documentation](docs/README.md)
2. Search existing [GitHub Issues](https://github.com/RobVanProd/Liquid-neural-network/issues)
3. Create a new issue if needed
