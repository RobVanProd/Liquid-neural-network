# Advanced Liquid Neural Network

A sophisticated self-improving neural network with dynamic temporal learning capabilities, designed for complex time series prediction and pattern recognition.

## 🔬 Features

- **Dynamic Network Structure**: Self-adapting architecture that evolves based on data complexity
- **Multi-scale Temporal Processing**: Advanced time series handling at multiple scales
- **Transformer-based Attention**: Sophisticated pattern recognition using attention mechanisms
- **Neuro-evolution Module**: Self-improvement capabilities through evolutionary algorithms
- **Advanced Visualization**: Comprehensive visualization tools for network analysis
- **Liquid-S4 Architecture**: Enhanced sequence modeling with state-space formulation
- **CfC Models**: Efficient continuous-time processing
- **Advanced Visualization Tools**: Comprehensive analysis and debugging capabilities

## 🛠️ Components

1. **super_liquid_network.py**
   - Core neural network architecture
   - Multi-scale liquid layers
   - Self-improvement mechanisms
   - Visualization utilities
2. **liquid_s4.py**
   - Implementation of Liquid-S4 architecture
3. **cfc_model.py**
   - Implementation of CfC models
4. **visualization.py**
   - Comprehensive visualization tools
5. **super_test.py**
   - Complex data generation
   - Multi-dataset testing framework
   - Performance evaluation
6. **test_improvements.py**
   - Test suite for new features

## 📊 Supported Datasets

- Chaotic Systems (Lorenz, Rössler attractors)
- Financial Time Series
- Complex Synthetic Patterns
- Custom data support

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11+
PyTorch 2.0.1+
```

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from super_liquid_network import SuperLiquidNetwork
from liquid_s4 import LiquidS4Model
from cfc_model import CfCModel
from visualization import NetworkVisualizer
from super_test import generate_complex_datasets

# Create model
model = SuperLiquidNetwork(
    input_size=10,
    hidden_size=64,
    output_size=1
)

# Initialize Liquid-S4 model
s4_model = LiquidS4Model(input_size=10, hidden_size=64, output_size=10)

# Initialize CfC model
cfc_model = CfCModel(input_size=10, hidden_size=64, output_size=10)

# Initialize visualizer
visualizer = NetworkVisualizer(log_dir='runs/experiment')

# Generate and prepare data
datasets = generate_complex_datasets()

# Train model
model.train(datasets['chaotic'])
```

## 🔧 Configuration

Key hyperparameters:
- `hidden_size`: Size of hidden layers (default: 64)
- `num_layers`: Number of liquid layers (default: 4)
- `num_heads`: Number of attention heads (default: 4)
- `dropout`: Dropout rate (default: 0.1)

## 📈 Performance

The network shows strong performance on:
- Chaotic time series prediction
- Market data forecasting
- Complex pattern recognition

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

- torch
- numpy
- matplotlib
- pandas
- scikit-learn
- yfinance
- seaborn
- plotly
- tensorboard


## 🧪 Proof-of-Concept Modules (TorusAI Exploration)

These modules are initial explorations into components inspired by the TorusAI Cognitive Stack. They are standalone proofs-of-concept at this stage.

1.  **`torusai_affect_encoding.py`**
    *   Implements a 6-dimensional `AffectVector` (valence, arousal, dominance, novelty, certainty, effort) based on Layer 9 (Affective Modulator) of the TorusAI specification.
    *   Includes an update rule: `A_t[d] = α*A_t-1[d] + β*reward + γ_d*Δ_intero(d)`.
    *   To run the demonstration: `python torusai_affect_encoding.py`

2.  **`torusai_dcg_hebbian.py`**
    *   Provides a basic implementation of an affective Hebbian update rule for edge weights in a conceptual Dynamic Concept Graph (DCG), related to Layer 4 and Layer 9 interaction.
    *   The rule is: `Δw = η * f_valence * f_dominance * pre_act * post_act`.
    *   Includes weight decay and pruning of edges below a threshold.
    *   Requires `torusai_affect_encoding.py` to be present.
    *   To run the demonstration: `python torusai_dcg_hebbian.py`

