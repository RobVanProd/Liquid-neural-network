# From User Part 3 - Labelled "Layer 5: Identity Tracking"
# Maps to README Layer 5: Symbolic Memory Module (SMM)

class IdentityTrackingLayer:
    def run(self, state): # Assumes state contains 'cg'
        cg = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for IdentityTrackingLayer.")
            return

        clusters = {}
        for cid, node_data in cg.nodes.items():
            node_type = node_data.get("type", "unknown") # Use .get for safety
            clusters.setdefault(node_type, []).append(cid)
        state['clusters'] = clusters
