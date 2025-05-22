# From User Part 2 - DreamLayer
# Mapped to README Layer 8: Emergent Goal Discovery & Proposal Engine (EGDPE) - as dreams can be exploratory
import random # DreamLayer uses random.sample

class DreamLayer:
    def __init__(self, lg_instance): # lg is an EnhancedLanguageGenerator instance
        self.lg = lg_instance
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

        sample_size = min(3, len(node_ids)) # Sample up to 3 nodes, or fewer if not enough
        
        if sample_size > 0:
            picked_node_ids = random.sample(node_ids, sample_size)
            for cid in picked_node_ids: 
                cg_instance.setActivation(cid, 1.0)
        
        # Generate dream line using the language generator
        dream_line = self.lg.generate("dream?") 
        state_dict.setdefault("dream_log", []).append(dream_line)
