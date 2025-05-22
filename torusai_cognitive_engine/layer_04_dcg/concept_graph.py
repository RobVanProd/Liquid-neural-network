# From User Part 1 - EnhancedConceptGraph
# Maps to README Layer 4: Dynamic Concept Graph (DCG)
# import numpy as np # Not strictly needed by this class as provided

class EnhancedConceptGraph:
    def __init__(self):
        self.nodes = {} # cid: {properties}
        self.edges = {} # (source_cid, target_cid): {properties}

    def addEnhancedNode(self, cid, properties=None, linguistics=None):
        properties = properties if properties is not None else {}
        linguistics = linguistics if linguistics is not None else {}
        
        # Ensure base activation is present
        current_properties = {"activation": 0.0}
        current_properties.update(properties) # Merge provided properties
        
        self.nodes[cid] = current_properties
        self.nodes[cid]["wordForms"] = linguistics.get("wordForms", {"base": str(cid)}) # Ensure cid is string for base
        self.nodes[cid]["syntacticRoles"] = linguistics.get("syntacticRoles", [])

    def setActivation(self, cid, value):
        if cid in self.nodes:
            self.nodes[cid]["activation"] = max(0.0, min(1.0, float(value))) # Ensure float
        # else:
            # Optionally handle case where node doesn't exist, e.g., print warning or auto-add
            # print(f"Warning: Node {cid} not found for setActivation.")

    def getActivation(self, cid):
        return self.nodes.get(cid, {}).get("activation", 0.0) # Default to 0.0 if node or activation missing

    def addEdge(self, s_cid, t_cid, relation="assoc", weight=1.0):
        if s_cid in self.nodes and t_cid in self.nodes:
            self.edges[(s_cid, t_cid)] = {"relation": relation, "weight": float(weight)} # Ensure float
        # else:
            # Optionally handle case where one or both nodes don't exist
            # print(f"Warning: Node(s) {s_cid} or {t_cid} not found for addEdge.")


    def getActiveConceptPath(self, threshold=0.3):
        # Ensure 'type' exists or provide a default
        return sorted(
            [{"id": cid, "activation": data["activation"], "type": data.get("type", "concept")}
             for cid, data in self.nodes.items() if data.get("activation", 0.0) >= threshold],
            key=lambda x: -x["activation"]
        )
