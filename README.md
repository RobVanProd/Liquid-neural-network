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


## TorusAI Cognitive Stack Architecture (v0.3)

Technical Architecture Specification — Revision May 14, 2025

This outlines the 14-layer cognitive architecture governing the TorusAI runtime. Each layer plays a distinct role in enabling symbolic cognition, affect modulation, and adaptive control over meta-rule behavior.

---

**Layer 1: Runtime Substrate**

*   Implementation: Python (main), Rust (subsystems), emerging WASM runtime for deployment.
*   Purpose: Provides memory management, timing loops, tick scheduler, and interfacing for peripheral and telemetry layers.

---

**Layer 2: Message Bus & Synchronization Primitives**

*   Purpose:
    *   Orchestrates communication between layers via event-based messaging and channelized topic routing.
    *   Manages tick-based global synchronization using barriers and futures.
    *   Handles ordered execution pipelines and abortable cascades.
*   Protocols: Internally defined tick, update, and emit schemas via protobuf-lite objects.

---

**Layer 3: 64-Node Toroidal Lattice**

*   Purpose: Symbolic inference and spatiotemporal pattern propagation.
*   Details:
    *   Each node maintains rule_set = {pattern, action, salience, lastFired, complexity}.
    *   Firing order: descending salience, ascending lastFired, ascending complexity.
    *   Rules decay with factor 0.95 per tick if not activated.
    *   Ring-based neighbor influence: ±2 toroidal hops.
    *   Active node threshold: act > 0.3.
    *   Output propagates to DCG (L4) and ABM (L9).

---

**Layer 4: Dynamic Concept Graph (DCG)**

*   Purpose: Represents evolving symbolic concepts and their contextual relationships.
*   Interaction:
    *   Receives reinforcement via Hebbian updates from ABM-derived affect state. *(POC for Hebbian update implemented)*
    *   Nodes represent symbolic tokens; edges carry temporal weights.
    *   Supports attention focus shifting and concept salience decay.
*   Prototype: Implemented and functional within reference JS core.

---

**Layer 5: Symbolic Memory Module (SMM)**

*   Purpose:
    *   Stores symbolic sequences (motifs, scripts, rule firings).
    *   Performs recall and context-sensitive priming based on partial matches.
*   Mechanism:
    *   Sequence buffer uses compressed trie + temporal weighting.
    *   Retrieval strategy uses fuzzy prefix scoring and ABM modulation.
*   Integration: Feeds Layer 3 lattice with historically successful rule chains.

---

**Layer 6: Episodic Recorder & Replay Buffer (ERRB)**

*   Purpose:
    *   Logs runtime state snapshots, affective valence, reward signals, and symbolic outputs for retrospective analysis.
    *   Enables replay of meaningful sequences for training or introspective refinement.
*   Format:
    *   Indexed ring buffer (2k depth) with timestamped entries {Tick, AffectiveState, SymbolSequence, Reward, GoalActive}.

---

**Layer 7: Meta-Rule Monitor & Confidence Engine (MRMCE)**

*   Purpose:
    *   Tracks confidence, activation frequency, and utility of meta-rules over time.
    *   Performs periodic decay, reinforcement, and arbitration.
*   Mechanism:
    *   Composite Score = (1 − λ) × CreditScore + λ × GoalAlignmentScore.
    *   Confidence updated via adaptive reinforcement (η learning rate).
    *   Rule pruning threshold: confidence < 0.1 over 5 epochs.

---

**Layer 8: Emergent Goal Discovery & Proposal Engine (EGDPE)**

*   Purpose:
    *   Detects internal state anomalies and long-term trends.
    *   Proposes new symbolic goals using trigger → goal_template mappings.
*   Structure:
    *   Includes Memory-Driven Refinement (heuristic layers), Trigger Quality Tracking, and Repetition Penalty.
    *   Output includes provisional goals with confidence_init, trigger_type, and expected_utility.
*   Evaluation:
    *   Provisional goals enter Layer 10 (validation loop).

---

**Layer 9: Affective Modulator + Body Schema (ABM)**

*   Purpose: Encodes interoception, affective state, and symbol-affect linking. *(POC for Affect Encoding implemented)*
*   Details:
    *   Affect vector A ∈ [-1,1]^6 = [valence, arousal, dominance, novelty, certainty, effort].
    *   Updated: A_t[d] = α·A_t−1[d] + β·reward + γ_d·Δ_intero(d).
    *   21x2 body schema (θ, ω) modulates symbol propagation via Hebbian weights.
    *   Fuzzy affect symbols retrieved via 0xFddB signature; thresholding with θ_weight = 0.2, θ_fuzzy = 0.2.

---

**Layer 10: Internal Reward Engine (IRE)**

*   Purpose:
    *   Computes internal reinforcement signal R_int = Σ r_i, where each r_i is a weighted function of prediction error, goal proximity, novelty, coherence, and affect match.
    *   Feeds ABM (L9) and Actor-Critic (L11).
*   Reward Channels:
    *   r_novelty, r_prediction_error, r_affect_alignment, r_goal_congruence.
    *   Tunable weights via YAML config or hot-reload endpoint.

---

**Layer 11: Actor-Critic Meta-Policy Engine (ACMPE)**

*   Purpose:
    *   Learns macro-behaviors over time via reinforcement on symbolic strategy effectiveness.
*   Structure:
    *   Actor proposes meta-rule policies.
    *   Critic evaluates long-term utility (based on R_int trends, symbolic entropy stability).
*   Policy Scope:
    *   Can enable/disable rules, adjust salience schedules, or invoke goal reformulation via Layer 8.

---

**Layer 12: Symbolic Attention Shaping Module (SASM)**

*   Purpose:
    *   Directs lattice rule firing and concept graph weighting based on recent trends, anomalies, or meta-cognitive prioritization.
*   Functionality:
    *   Implements “spotlight” over regions of DCG or rule lattice.
    *   Uses entropy gradient and ABM modulator feedback.
*   Result:
    *   Enables task- or anomaly-focused symbolic shifts (e.g., “explain surprise” behavior).

---

**Layer 13: Introspective Meta-Reporter (IMR)**

*   Purpose:
    *   Provides narrative and interpretable output summarizing current symbolic state, rule confidence changes, active goals, and affective status.
*   Output Forms:
    *   Human-readable symbolic trace summaries.
    *   Diagnostic reports for debugging or research visibility.
*   Endpoint: /report/trace.

---

**Layer 14: High-Level APIs / Embedders**

*   Purpose:
    *   Provides interface to embed the system in external hosts, query internal state, inject symbolic feedback or reward.
*   Endpoints:
    *   /affect GET, /reward/internal POST, /goal/active, /meta_rules/status.
*   Integration:
    *   WebSocket or REST-based telemetry with optional Grafana binding.
    *   Intended for runtime ablation experiments, visualization dashboards, or supervisory AI agents.

---

**Integration Cycle (Per Cognitive Tick)**

1.  ABM-derived activation modifiers computed.
2.  Top-k active symbols collected from lattice (L3).
3.  Hebbian updates applied between L3, L4, L9.
4.  Internal rewards computed via L10.
5.  Affect symbols projected into L3.
6.  Meta-rules evaluated by L7.
7.  Actor-Critic policies (L11) optionally adjust strategy.
8.  Symbolic goals adjusted/refined in L8.
9.  Introspective trace emitted by L13.
10. Symbolic attention shape adjusted (L12).
11. Rules fired (L3), salience decayed.
12. API data exported (L14).

---

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

