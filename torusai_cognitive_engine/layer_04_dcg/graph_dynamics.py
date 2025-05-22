# From User Part 2 - GeometryLayer, DynamicsLayer, HebbianLayer
# Maps to README Layer 4: Dynamic Concept Graph (DCG) - as operational dynamics
from collections import defaultdict
import random # HebbianLayer might implicitly need random if new edges are added with random weights, though current code uses 0.5

class GeometryLayer:
    def __init__(self, decay_rate: float = 0.02): 
        self.decay_rate = decay_rate
    def run(self, cg): # cg is an EnhancedConceptGraph instance
        for cid in list(cg.nodes.keys()): # Iterate over keys for safe modification if needed elsewhere
            current_activation = cg.getActivation(cid)
            cg.setActivation(cid, current_activation - self.decay_rate)

class DynamicsLayer:
    def run(self, cg): # cg is an EnhancedConceptGraph instance
        delta = defaultdict(float)
        for (s, t), edge_data in cg.edges.items():
            source_activation = cg.getActivation(s)
            delta[t] += source_activation * edge_data.get("weight", 0.0) * 0.3 # Use .get for safety
        
        for t_cid, val_delta in delta.items():
            current_activation = cg.getActivation(t_cid)
            cg.setActivation(t_cid, min(1.0, current_activation + val_delta))

class HebbianLayer: # General Hebbian, not the Affective Hebbian POC
    def __init__(self, multiplier: float = 1.1, prune_threshold: float = 0.2): 
        self.multiplier = multiplier
        self.prune_threshold = prune_threshold
    def run(self, cg): # cg is an EnhancedConceptGraph instance
        active_nodes = [cid for cid, data in cg.nodes.items() if data.get("activation", 0.0) > 0.6]
        
        # Strengthen existing edges or add new ones between co-active nodes
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                s_cid, t_cid = active_nodes[i], active_nodes[j]
                
                # Ensure order for consistent edge key, or handle both (s,t) and (t,s) if graph is undirected
                # For now, assume directed or that order in active_nodes is sufficient for pairing
                
                current_edge_data = cg.edges.get((s_cid, t_cid), {"weight": 0.5, "relation": "assoc_hebbian"}) # Default for new edge
                new_weight = current_edge_data["weight"] * self.multiplier
                cg.addEdge(s_cid, t_cid, relation=current_edge_data.get("relation","assoc_hebbian"), weight=min(new_weight, 5.0))

        # Prune weak edges
        for (s, t), edge_data in list(cg.edges.items()): # Use list() for safe deletion during iteration
            if edge_data.get("weight", 0.0) < self.prune_threshold:
                del cg.edges[(s, t)]
