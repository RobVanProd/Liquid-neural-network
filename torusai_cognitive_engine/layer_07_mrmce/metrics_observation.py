# From User Part 3 - Labelled "Layer 7: Metrics Observation"
# Maps to README Layer 7: Meta-Rule Monitor & Confidence Engine (MRMCE) - observation part

class MetricsObservationLayer:
    def run(self, state): # Assumes state contains 'cg'
        cg = state.get('cg')
        if not cg:
            # print("Warning: Concept graph 'cg' not found in state for MetricsObservationLayer.")
            state['metrics'] = {} # Ensure metrics key exists
            return

        activations = [data.get("activation", 0.0) for data in cg.nodes.values()]
        avg_activation = sum(activations) / len(activations) if activations else 0.0
        
        state['metrics'] = {
            "avg_activation": round(avg_activation, 4),
            "node_count": len(cg.nodes),
            "edge_count": len(cg.edges)
        }
