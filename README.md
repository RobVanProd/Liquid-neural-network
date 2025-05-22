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


## TorusAI 15-Layer Cognitive Stack (v0.3+) — Design Specification
===============================================================

TABLE OF CONTENTS / LAYER INDEX
#  ID   LAYER NAME                         STATUS
0  L0   Geometry Runtime Substrate         Implemented
1  L1   Symbolic Tagging & Typing          Implemented
2  L2   Activation Dynamics (CA)           Implemented
3  L3   Feedback Modulation                Implemented
4  L4   Hebbian Edge Memory                Implemented
5  L5   Identity Tracking                  Implemented
6  L6   Symbolic Reflection                Implemented
7  L7   Metrics Observation                Implemented
8  L8   Language Module (Morph-Syntax)     Implemented
9  L9   Affective Body                     Implemented
10 L10  Social Mind                        Implemented
11 L11  Offline Dreaming                   Implemented
12 L12  Learning Orchestrator              Implemented
13 L13  Monitoring / Telemetry             Implemented
14 L14  API Embedder / Interface           Implemented

GLOBAL CONVENTIONS & CORE DATA OBJECTS
• ConceptGraph (cg):
    nodes -> Dict[ConceptID, NodeData]
    edges -> Dict[(source,target), EdgeData]
• EngineState (state) shared keys:
    cg, query, last_response, valence (-1..+1),
    reflection, metrics, clusters

LAYER SPECIFICATIONS (ABBREVIATED)
L0 Geometry ............ decay_awake=0.02 | decay_dream=0.05
L1 SymbolicTagging ..... guarantee each node has “type”
L2 ActivationDynamics .. Δ[t] += act(s) * weight * 0.3
L3 FeedbackModulation .. apply ±0.03 to all activations from valence
L4 HebbianEdgeMemory ... weight *=1.1 for co-active nodes; prune<0.2; optional top-K cap
L5 IdentityTracking .... maintain clusters[state]
L6 SymbolicReflection .. reflection = top active path ≥0.7
L7 MetricsObservation .. avg_activation, node_count, edge_count → state['metrics']
L8 LanguageModule ...... morph-syntax templates; output → last_response
L9 AffectiveBody ....... sets state['valence'] via heuristics
L10 SocialMind ......... if valence<0 dampen activations ×0.95
L11 OfflineDreaming .... dream_n_candidates=5 | keep_top_k=2 | log→state['dream_log']
L12 LearningOrchestrator patternMemory.record(reflection, last_response)
L13 Monitoring ......... emit state['metrics'], clusters, dream_log
L14 APIEmbedder ........ final hand-off to CLI/UI (print for now)

DEFAULT PARAMETERS
decay_awake:0.02  •  decay_dream:0.05  •  low_ticks:3
hebbian_multiplier:1.1  •  prune_threshold:0.2
dream_n_candidates:5  •  dream_keep_top_k:2

PERFORMANCE TARGETS (CPU only)
5 000 nodes → 0.8 ms per user-cycle
50 000 nodes → 2.3 ms per user-cycle

## 🧪 Implemented Modules Overview (aligned with 15-Layer Spec)

This section provides a quick reference to the Python modules implemented within the `torusai_cognitive_engine` package, corresponding to the layers defined in the "TorusAI 15-Layer Cognitive Stack (v0.3+) — Design Specification".

1.  **L0 Geometry Runtime Substrate (`layer_04_dcg/graph_dynamics.py` - `GeometryLayer` class):**
    *   Handles activation decay in the concept graph. Parameters: `decay_awake`, `decay_dream`.
2.  **L1 Symbolic Tagging & Typing (`layer_04_dcg/symbolic_tagging.py` - `SymbolicTaggingLayer` class):**
    *   Ensures every node in the concept graph has a "type" attribute.
3.  **L2 Activation Dynamics (CA) (`layer_04_dcg/graph_dynamics.py` - `DynamicsLayer` class):**
    *   Propagates activation energy through the concept graph based on node activations and edge weights.
4.  **L3 Feedback Modulation (`layer_03_toroidal_lattice/feedback_modulation.py` - `FeedbackModulationLayer` class):**
    *   Adjusts all node activations uniformly based on a global `last_sentiment` value.
5.  **L4 Hebbian Edge Memory (`layer_04_dcg/graph_dynamics.py` - `HebbianLayer` class):**
    *   Modifies edge weights between co-active nodes and prunes weak edges. Parameters: `hebbian_multiplier`, `prune_threshold`.
6.  **Core Concept Graph System (`layer_04_dcg/concept_graph.py` - `EnhancedConceptGraph` class):**
    *   The central data structure for representing concepts, their activations, properties, linguistic forms, and weighted relationships. This is used by many layers.
7.  **L5 Identity Tracking (`layer_05_smm/identity_tracking.py` - `IdentityTrackingLayer` class):**
    *   Groups concept nodes by their "type" and stores these groupings in `state['clusters']`.
8.  **L6 Symbolic Reflection (`layer_06_errb/symbolic_reflection.py` - `SymbolicReflectionLayer` class):**
    *   Identifies and records the most active path of concepts (activation >= 0.7) into `state['reflection']`.
9.  **L7 Metrics Observation (`layer_07_mrmce/metrics_observation.py` - `MetricsObservationLayer` class):**
    *   Calculates overall metrics about the concept graph (e.g., average activation, node/edge counts) and stores them in `state['metrics']`.
10. **L8 Language Module (Morph-Syntax) (`layer_13_imr/language_generator.py` - `EnhancedLanguageGenerator` class):**
    *   Generates natural language sentences based on the current state of active concepts in the graph. Output stored in `state['last_response']`.
11. **L9 Affective Body (`layer_09_abm/affective_body.py` - `AffectiveBodyLayer` class):**
    *   Determines a global `state['valence']` based on sentiment heuristics from the `state['last_response']`.
12. **L10 Social Mind (`layer_10_ire/social_mind_influence.py` - `SocialMindLayer` class):**
    *   Applies a damping effect on all node activations if the global `state['valence']` is negative.
13. **L11 Offline Dreaming (`layer_08_egdpe/dream_logic.py` - `DreamLayer` class):**
    *   Simulates a "dream" cycle by randomly activating concept nodes and generating a linguistic description of this state. Logs dreams to `state['dream_log']`. Parameters: `dream_n_candidates`, `dream_keep_top_k`.
14. **L12 Learning Orchestrator (`layer_12_learning_orchestrator/orchestrator.py` - `LearningOrchestrator` & `PatternMemory` classes):**
    *   Uses `PatternMemory` to record trigger-response pairs (e.g., from reflection and last_response) along with associated metadata like valence.
    *   Can retrieve nearest matching patterns to a given query. Stores retrieved patterns in `state['retrieved_patterns']`.
15. **L13 Monitoring / Telemetry (`layer_13_imr/telemetry.py` - `MonitoringLayer` class):**
    *   Prints out metrics, clusters, or dream logs for observation.
16. **L14 API Embedder / Interface (`layer_14_api_embedders/output_embedding.py` - `APIEmbedderLayer` class):**
    *   Handles the final output of the system's response (e.g., printing `state['last_response']`).
17. **Engine & Utilities:**
    *   `engine/torus_engine.py`: The main engine that can load and run sequences of these layers.
    *   `tests/stress_test.py`: A script for testing the performance and behavior of the engine with a large concept graph.
    *   `main.py`: An example script to initialize and run the engine and its layers.
    *   The original POCs (`torusai_affect_encoding.py`, `torusai_dcg_hebbian.py`) also exist at the root but are not directly part of this integrated engine.

