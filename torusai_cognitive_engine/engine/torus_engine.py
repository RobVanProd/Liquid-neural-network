# Assuming these are the necessary imports for the classes used by TorusEngine
from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
from ..layer_13_imr.language_generator import EnhancedLanguageGenerator
from ..layer_04_dcg.graph_dynamics import GeometryLayer, DynamicsLayer, HebbianLayer
from ..layer_04_dcg.symbolic_tagging import SymbolicTaggingLayer
from ..layer_03_toroidal_lattice.feedback_modulation import FeedbackModulationLayer
from ..layer_05_smm.identity_tracking import IdentityTrackingLayer
from ..layer_06_errb.symbolic_reflection import SymbolicReflectionLayer
from ..layer_07_mrmce.metrics_observation import MetricsObservationLayer
from ..layer_09_abm.affective_body import AffectiveBodyLayer
from ..layer_10_ire.social_mind_influence import SocialMindLayer
from ..layer_08_egdpe.dream_logic import DreamLayer # This is L11 (Offline Dreaming) in the new spec
from ..layer_12_learning_orchestrator.orchestrator import LearningOrchestrator # L12
from ..layer_13_imr.telemetry import MonitoringLayer # L13
from ..layer_14_api_embedders.output_embedding import APIEmbedderLayer # L14

class TorusEngine:
    def __init__(self, cg_instance: EnhancedConceptGraph, lg_instance: EnhancedLanguageGenerator, params: dict = None):
        self.cg = cg_instance  # Concept Graph instance
        self.lg = lg_instance  # Language Generator instance (L8)
        self.params = params if params is not None else {}

        # Retrieve default parameters from the spec, allowing overrides from self.params
        self.decay_awake = self.params.get('decay_awake', 0.02)
        self.decay_dream = self.params.get('decay_dream', 0.05)
        self.hebbian_multiplier = self.params.get('hebbian_multiplier', 1.1)
        self.prune_threshold = self.params.get('prune_threshold', 0.2)
        self.dream_n_candidates = self.params.get('dream_n_candidates', 5)
        self.dream_keep_top_k = self.params.get('dream_keep_top_k', 2)
        self.low_ticks = self.params.get('low_ticks', 3) # General ticks for non-query cycles if needed

        # Initialize shared engine state. This state will be passed to layers that need it.
        self.state = {
            "cg": self.cg, # Direct reference to the concept graph instance
            "narrative": self.lg.narrative if self.lg else [], # Reference to lg's narrative log
            "query": None,
            "last_response": "",
            "valence": 0.0, 
            "reflection": [],
            "metrics": {},
            "clusters": {},
            "dream_log": [],
            "retrieved_patterns": []
        }

        # Instantiate layers (L0-L14 based on 15-Layer spec)
        # L0 Geometry Runtime Substrate
        self.l0_geometry = GeometryLayer(decay_rate=self.decay_awake)
        # L1 Symbolic Tagging & Typing
        self.l1_symbolic_tagging = SymbolicTaggingLayer()
        # L2 Activation Dynamics (CA)
        self.l2_activation_dynamics = DynamicsLayer()
        # L3 Feedback Modulation
        self.l3_feedback_modulation = FeedbackModulationLayer() 
        # L4 Hebbian Edge Memory
        self.l4_hebbian_memory = HebbianLayer(multiplier=self.hebbian_multiplier, prune_threshold=self.prune_threshold)
        # L5 Identity Tracking
        self.l5_identity_tracking = IdentityTrackingLayer()
        # L6 Symbolic Reflection
        self.l6_symbolic_reflection = SymbolicReflectionLayer()
        # L7 Metrics Observation
        self.l7_metrics_observation = MetricsObservationLayer()
        # L8 Language Module (is self.lg, passed in)
        # L9 Affective Body
        self.l9_affective_body = AffectiveBodyLayer()
        # L10 Social Mind
        self.l10_social_mind = SocialMindLayer()
        # L11 Offline Dreaming
        self.l11_offline_dreaming = DreamLayer(lg_instance=self.lg, n_candidates=self.dream_n_candidates, keep_top_k=self.dream_keep_top_k)
        # L12 Learning Orchestrator
        self.l12_learning_orchestrator = LearningOrchestrator(shared_engine_state=self.state)
        # L13 Monitoring / Telemetry
        self.l13_monitoring = MonitoringLayer()
        # L14 API Embedder / Interface
        self.l14_api_embedder = APIEmbedderLayer()
            
    def run_cycle(self, query: str = None, is_dream_cycle: bool = False):
        self.state["query"] = query
        if query and not is_dream_cycle: # Clear previous response only for new, non-dream queries
            self.state["last_response"] = ""

        # Layer Execution Order (based on a general cognitive flow)
        
        # L0 Geometry (Perception/Decay)
        current_decay = self.decay_dream if is_dream_cycle else self.decay_awake
        # Temporarily set decay for L0 if it's a dream, then restore
        original_l0_decay = self.l0_geometry.decay_rate
        self.l0_geometry.decay_rate = current_decay
        self.l0_geometry.run(self.cg)
        self.l0_geometry.decay_rate = original_l0_decay # Restore

        # L1 Symbolic Tagging (Basic graph maintenance)
        self.l1_symbolic_tagging.run(self.state)
        
        # L2 Activation Dynamics (Graph propagation)
        self.l2_activation_dynamics.run(self.cg)
        
        # L9 Affective Body (Update valence based on previous cycle's output if any)
        # This might run earlier or later depending on when last_response is finalized.
        # For now, run it before feedback modulation that might use the new valence.
        if not is_dream_cycle: # Affective body might not run or run differently during dreams
             self.l9_affective_body.run(self.state)

        # L3 Feedback Modulation (Valence influences activations)
        # Ensure L9 (AffectiveBody) runs before L3 if L3 depends on fresh valence
        self.l3_feedback_modulation.last_sentiment = self.state.get("valence", 0.0) # Update layer's sentiment
        self.l3_feedback_modulation.run(self.state)
        
        # L4 Hebbian Edge Memory (Learning based on new activations)
        self.l4_hebbian_memory.run(self.cg)
        
        # L5 Identity Tracking (Clustering based on types)
        self.l5_identity_tracking.run(self.state)
        
        # L6 Symbolic Reflection (Capture highly active path)
        self.l6_symbolic_reflection.run(self.state)

        # L7 Metrics Observation (Observe current graph state)
        self.l7_metrics_observation.run(self.state)
        
        if is_dream_cycle:
            # L11 Offline Dreaming
            self.l11_offline_dreaming.run(self.cg, self.state) # DreamLayer logs to state['dream_log']
        elif query and self.lg:
            # L8 Language Module (Generate response if query)
            self.state["last_response"] = self.lg.generate(query)
        
        # L10 Social Mind (Dampen activations if negative valence)
        # Run after L9 has potentially updated valence
        self.l10_social_mind.run(self.state)
        
        # L12 Learning Orchestrator (Record patterns, retrieve)
        self.l12_learning_orchestrator.step()
        
        # L13 Monitoring / Telemetry (Output internal metrics/logs)
        self.l13_monitoring.run(self.state) # Prints metrics, dream_log etc.
        
        # L14 API Embedder / Interface (Output final response)
        self.l14_api_embedder.run(self.state) # Prints last_response

        return self.state.get("last_response")

    # Convenience methods for standard and dream cycles
    def standard_cycle(self, query: str):
        # print("\n--- Running Standard Cycle ---")
        return self.run_cycle(query=query, is_dream_cycle=False)

    def dream_cycle_execution(self): # Renamed from user's dream_cycle to avoid clash if they define one
        # print("\n--- Running Dream Cycle ---")
        # No query is typically passed for a dream cycle
        return self.run_cycle(query=None, is_dream_cycle=True)
