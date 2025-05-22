# From User Part 2 - DreamLayer
# Mapped to README Layer 8: Emergent Goal Discovery & Proposal Engine (EGDPE) - as dreams can be exploratory
import random # DreamLayer uses random.sample

class DreamLayer:
    def __init__(self, lg_instance, n_candidates: int = 5, keep_top_k: int = 2): # lg is an EnhancedLanguageGenerator instance
        self.lg = lg_instance
        self.n_candidates = n_candidates
        self.keep_top_k = keep_top_k # Stored, though not fully used by current run logic
    def run(self, cg_instance, state_dict): # cg is EnhancedConceptGraph, state is a dict
        if not self.lg:
            # print("Warning: Language generator not available for DreamLayer.")
            return
        if not cg_instance:
            # print("Warning: Concept graph not available for DreamLayer.")
            return

        node_ids = list(cg_instance.nodes.keys())
        if not node_ids:
            # print("Warning: No nodes in concept graph for DreamLayer.")
            return

        sample_size = min(self.n_candidates, len(node_ids))
        
        if sample_size > 0:
            picked_node_ids = random.sample(node_ids, sample_size)
            for cid in picked_node_ids: 
                cg_instance.setActivation(cid, 1.0)
        
        # Generate dream line using the language generator
        dream_line = self.lg.generate("dream?") 
        state_dict.setdefault("dream_log", []).append(dream_line)
