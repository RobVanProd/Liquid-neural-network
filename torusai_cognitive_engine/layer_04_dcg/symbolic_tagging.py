# From User Part 3 - Labelled "Layer 1: Symbolic Tagging"
# Remapped to Layer 4 (DCG) as a utility/maintenance function for concept nodes

class SymbolicTaggingLayer:
    def run(self, state): # Assumes state contains 'cg' which is EnhancedConceptGraph
        cg = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for SymbolicTaggingLayer.")
            return
            
        for cid in cg.nodes: # Iterates over keys of the nodes dictionary
            # setdefault on the node's dictionary itself
            cg.nodes[cid].setdefault("type", "concept")
