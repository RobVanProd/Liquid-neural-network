# Main entry point for the TorusAI Cognitive Engine simulation

# Standard library imports
import time
import random
import json # For pretty printing dicts

# --- Relative imports for engine components ---
# Layer 01: Runtime Substrate (Placeholder, no specific classes to import directly for main)
# from .layer_01_runtime_substrate.runtime_utils import SomeRuntimeUtil # Example

# Layer 02: Message Bus (Placeholder, no specific classes to import directly for main)
# from .layer_02_message_bus.bus_primitives import MessageBus # Example

# Layer 03: Toroidal Lattice (FeedbackModulationLayer)
from .layer_03_toroidal_lattice.feedback_modulation import FeedbackModulationLayer

# Layer 04: Dynamic Concept Graph (DCG)
from .layer_04_dcg.concept_graph import EnhancedConceptGraph
from .layer_04_dcg.graph_dynamics import GeometryLayer, DynamicsLayer, HebbianLayer
from .layer_04_dcg.symbolic_tagging import SymbolicTaggingLayer

# Layer 05: Symbolic Memory Module (SMM)
from .layer_05_smm.identity_tracking import IdentityTrackingLayer

# Layer 06: Episodic Recorder & Replay Buffer (ERRB)
from .layer_06_errb.symbolic_reflection import SymbolicReflectionLayer

# Layer 07: Meta-Rule Monitor & Confidence Engine (MRMCE)
from .layer_07_mrmce.metrics_observation import MetricsObservationLayer

# Layer 08: Emergent Goal Discovery & Proposal Engine (EGDPE)
from .layer_08_egdpe.dream_logic import DreamLayer

# Layer 09: Affective Modulator + Body Schema (ABM)
from .layer_09_abm.affective_body import AffectiveBodyLayer

# Layer 10: Internal Reward Engine (IRE)
from .layer_10_ire.social_mind_influence import SocialMindLayer

# Layer 11: Actor-Critic Meta-Policy Engine (ACMPE) (Placeholder)
# from .layer_11_acmpe.policy_engine_stubs import ACMPE_Stub # Example

# Layer 12: Symbolic Attention Shaping Module (SASM) (Placeholder)
# from .layer_12_sasm.attention_shaping_stubs import SASM_Stub # Example

# Layer 13: Introspective Meta-Reporter (IMR)
from .layer_13_imr.language_generator import EnhancedLanguageGenerator
from .layer_13_imr.telemetry import MonitoringLayer

# Layer 14: High-Level APIs / Embedders
from .layer_14_api_embedders.output_embedding import APIEmbedderLayer

# Core Engine
from .engine.torus_engine import TorusEngine

# Test utility (if choosing to run stress test from main)
from .tests.stress_test import run_stress_test

# --- Simulation Setup ---
def initialize_simulation_components():
    """Initializes and returns all core components for the simulation."""
    
    # Core data structures
    cg = EnhancedConceptGraph()
    narrative_log = [] # Shared narrative log
    shared_state = {
        "cg": cg,
        "narrative": narrative_log,
        "dream_log": [],
        "metrics": {},
        "valence": 0.1, # Initial affective state
        "last_response": "System boot complete."
    }

    # Instantiate layers/modules
    # Layer 03
    feedback_modulation_layer = FeedbackModulationLayer()
    # Layer 04
    geometry_layer = GeometryLayer(decay=0.01) # Slightly lower decay for longer persistence
    dynamics_layer = DynamicsLayer()
    hebbian_layer = HebbianLayer(mult=1.05, prune=0.15) # Slightly adjusted params
    symbolic_tagging_layer = SymbolicTaggingLayer()
    # Layer 05
    identity_tracking_layer = IdentityTrackingLayer()
    # Layer 06
    symbolic_reflection_layer = SymbolicReflectionLayer()
    # Layer 07
    metrics_observation_layer = MetricsObservationLayer()
    # Layer 08
    # LanguageGenerator is needed by DreamLayer, so instantiate it first
    language_generator = EnhancedLanguageGenerator(cg_instance=cg, narrative_log_list=narrative_log)
    dream_layer = DreamLayer(lg_instance=language_generator)
    # Layer 09
    affective_body_layer = AffectiveBodyLayer()
    # Layer 10
    social_mind_layer = SocialMindLayer()
    # Layer 13
    # language_generator already instantiated
    monitoring_layer = MonitoringLayer()
    # Layer 14
    api_embedder_layer = APIEmbedderLayer()

    # Instantiate the main engine (if using the TorusEngine orchestrator from Part 2)
    # This TorusEngine orchestrates a subset of layers as per its original design.
    # For a full 14-layer orchestration, a more comprehensive engine loop would be needed here.
    torus_engine_orchestrator = TorusEngine(
        cg_instance=cg,
        lg_instance=language_generator,
        geom_layer_instance=geometry_layer,
        dyn_layer_instance=dynamics_layer,
        hebb_layer_instance=hebbian_layer,
        dream_layer_instance=dream_layer,
        ticks=1 # Reduced ticks for faster individual main loop cycles
    )
    
    # Store layers and engine in a dictionary for easier access in the main loop
    components = {
        "cg": cg,
        "lg": language_generator,
        "engine": torus_engine_orchestrator, # The specific engine from Part 2
        "shared_state": shared_state,
        "layers": {
            "feedback_modulation": feedback_modulation_layer,
            "geometry": geometry_layer, # Also part of torus_engine_orchestrator
            "dynamics": dynamics_layer,   # Also part of torus_engine_orchestrator
            "hebbian": hebbian_layer,     # Also part of torus_engine_orchestrator
            "symbolic_tagging": symbolic_tagging_layer,
            "identity_tracking": identity_tracking_layer,
            "symbolic_reflection": symbolic_reflection_layer,
            "metrics_observation": metrics_observation_layer,
            "dream": dream_layer,         # Also part of torus_engine_orchestrator
            "affective_body": affective_body_layer,
            "social_mind": social_mind_layer,
            "monitoring": monitoring_layer,
            "api_embedder": api_embedder_layer
        }
    }
    return components

def run_full_cycle_simulation(components, num_cycles=5):
    """
    Runs a more comprehensive simulation loop, invoking layers in a plausible order.
    This is a conceptual main loop; the exact order and interaction would be refined
    based on the full TorusAI specification.
    """
    print("\n--- Starting Full Cycle Simulation ---")
    
    cg = components["cg"]
    state = components["shared_state"] # Use the shared state dictionary
    layers = components["layers"]

    # Initial data for the concept graph
    initial_concepts = {
        "cat": {"type": "animal", "wordForms": {"base": "cat"}},
        "dog": {"type": "animal", "wordForms": {"base": "dog"}},
        "chase": {"type": "action", "wordForms": {"base": "chase"}},
        "play": {"type": "action", "wordForms": {"base": "play"}},
        "fast": {"type": "property", "wordForms": {"base": "fast"}},
        "happy": {"type": "emotion", "wordForms": {"base": "happy"}}
    }
    for cid, data in initial_concepts.items():
        cg.addEnhancedNode(cid, properties={"type": data["type"]}, linguistics={"wordForms": data["wordForms"]})
        cg.setActivation(cid, random.uniform(0.1, 0.4))

    cg.addEdge("cat", "chase", weight=0.8)
    cg.addEdge("dog", "chase", weight=0.7)
    cg.addEdge("cat", "play", weight=0.6)
    cg.addEdge("dog", "play", weight=0.9)
    cg.addEdge("chase", "fast", weight=0.5)
    cg.addEdge("play", "happy", weight=0.7)
    
    print(f"Initial CG: {len(cg.nodes)} nodes, {len(cg.edges)} edges.")
    print("-" * 20)

    for i in range(num_cycles):
        print(f"\n--- Cycle {i+1}/{num_cycles} ---")
        
        # 1. Affective Body Update (Layer 9)
        layers["affective_body"].run(state)
        
        # 2. Symbolic Tagging (Layer 4 Utility)
        layers["symbolic_tagging"].run(state) # Modifies cg in state

        # 3. DCG Dynamics (Layer 4)
        layers["geometry"].run(cg) # Decay
        layers["dynamics"].run(cg) # Propagation
        layers["hebbian"].run(cg)  # Learning
        
        # 4. Feedback Modulation (Layer 3 based on affect)
        # This layer needs access to the sentiment/valence from Layer 9
        layers["feedback_modulation"].last_sentiment = state.get("valence", 0.1)
        layers["feedback_modulation"].run(state) # Modifies cg in state

        # 5. Identity Tracking (Layer 5)
        layers["identity_tracking"].run(state) # Updates state['clusters']
        
        # 6. Symbolic Reflection (Layer 6)
        layers["symbolic_reflection"].run(state) # Updates state['reflection']
        
        # 7. Social Mind Influence (Layer 10, using affect)
        layers["social_mind"].run(state) # Modifies cg based on valence in state

        # 8. Dream Logic (Layer 8 - occasionally)
        if i % 3 == 0: # Let's say dreaming happens every 3 cycles
            print("[Dreaming...]")
            # DreamLayer uses the main language generator instance
            layers["dream"].run(cg, state) # Modifies cg and state['dream_log']
            if state["dream_log"]:
                 print(f"[Dream Output]: {state['dream_log'][-1]}")


        # 9. Language Generation (Layer 13 - IMR)
        # For this demo, let's generate a statement based on most active concepts
        # Or use a predefined query for simplicity
        query = "What is happening?" if i % 2 == 0 else "Tell me about active concepts."
        generated_output = components["lg"].generate(query) # Access lg from components
        state["last_response"] = generated_output # Update shared state

        # 10. API/Output Embedding (Layer 14)
        layers["api_embedder"].run(state) # Prints last_response

        # 11. Metrics and Monitoring (Layer 7 & 13)
        layers["metrics_observation"].run(state) # Updates state['metrics']
        layers["monitoring"].run(state) # Prints metrics

        time.sleep(0.1) # Small delay to make output readable

    print("\n--- Simulation Complete ---")
    print("Final Narrative Log:", json.dumps(state["narrative"], indent=2))
    print("Final Dream Log:", json.dumps(state["dream_log"], indent=2))


if __name__ == "__main__":
    print("--- TorusAI Cognitive Engine: Main Simulation Runner ---")
    
    # Option 1: Run the comprehensive simulation loop
    sim_components = initialize_simulation_components()
    run_full_cycle_simulation(sim_components, num_cycles=3) # Run for a few cycles

    # --- Stress Test (Optional) ---
    # Note: The stress test uses the class definitions, not the initialized instances above.
    # It creates its own instances internally.
    # print("\n\n--- Preparing for Stress Test (will re-initialize components) ---")
    # print("This might take a moment...")
    # stress_test_results = run_stress_test(
    #     cg_class=EnhancedConceptGraph,
    #     lg_class=EnhancedLanguageGenerator,
    #     engine_class=TorusEngine, # The simpler orchestrator from Part 2
    #     geom_class=GeometryLayer,
    #     dyn_class=DynamicsLayer,
    #     hebb_class=HebbianLayer,
    #     dream_class=DreamLayer
    # )
    # print("\n--- Stress Test Results ---")
    # print(json.dumps(stress_test_results, indent=2))

    print("\n--- End of Main ---")
