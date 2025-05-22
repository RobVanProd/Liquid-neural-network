# From User Part 2 - TorusEngine
# Orchestrates several of the provided layer components
# Needs imports from the structured layer modules.

from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
from ..layer_04_dcg.graph_dynamics import GeometryLayer, DynamicsLayer, HebbianLayer
from ..layer_08_egdpe.dream_logic import DreamLayer
from ..layer_13_imr.language_generator import EnhancedLanguageGenerator


class TorusEngine:
    def __init__(self, 
                 cg_instance, # Should be an EnhancedConceptGraph instance
                 lg_instance, # Should be an EnhancedLanguageGenerator instance
                 geom_layer_instance, 
                 dyn_layer_instance, 
                 hebb_layer_instance, 
                 dream_layer_instance,
                 ticks=3):
        
        self.cg = cg_instance
        self.lg = lg_instance
        self.ticks = ticks
        
        self.geom_layer = geom_layer_instance
        self.dyn_layer = dyn_layer_instance
        self.hebb_layer = hebb_layer_instance
        self.dream_layer = dream_layer_instance

        # Initialize state; narrative comes from lg instance's attribute
        self.state = {"narrative": self.lg.narrative if self.lg else []}


    def cycle(self, query_string=None):
        for _ in range(self.ticks):
            if self.geom_layer: self.geom_layer.run(self.cg)
            if self.dyn_layer: self.dyn_layer.run(self.cg)
            if self.hebb_layer: self.hebb_layer.run(self.cg)
        
        if query_string and self.lg:
            # The generate method of EnhancedLanguageGenerator updates its own narrative list
            # and returns the generated sentence.
            generated_sentence = self.lg.generate(query_string)
            self.state["last_response"] = generated_sentence

    def dream_cycle(self):
        if not self.dream_layer:
            # print("Warning: DreamLayer not provided to TorusEngine for dream_cycle.")
            return

        if self.geom_layer: self.geom_layer.run(self.cg)
        if self.dyn_layer: self.dyn_layer.run(self.cg)
        if self.hebb_layer: self.hebb_layer.run(self.cg)
        # DreamLayer's run method expects cg_instance and state_dict
        self.dream_layer.run(self.cg, self.state)
