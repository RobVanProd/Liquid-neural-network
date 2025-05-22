# From User Part 3 - Labelled "Layer 10: Social Mind Layer"
# Remapped to Layer 10 (IRE) as an influence on behavior/reward, or could be L9.

class SocialMindLayer:
    def run(self, state_dict): # state is a dict
        cg_instance = state_dict.get('cg')
        current_valence = state_dict.get("valence", 0.0) # Get valence from shared state

        if not cg_instance:
            # print("Warning: Concept graph 'cg' not found in state for SocialMindLayer.")
            return

        if current_valence < 0:
            for cid in list(cg_instance.nodes.keys()): # Iterate over keys
                current_activation = cg_instance.getActivation(cid)
                cg_instance.setActivation(cid, current_activation * 0.95)
