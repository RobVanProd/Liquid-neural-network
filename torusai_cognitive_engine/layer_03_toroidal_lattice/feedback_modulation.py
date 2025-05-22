# From User Part 3 - Labelled "Layer 3: Feedback Modulation"
# Maps to README Layer 3: Toroidal Lattice (dynamics, activation modulation)

class FeedbackModulationLayer:
    def __init__(self): 
        self.last_sentiment = 0.1  # default

    def run(self, state):
        cg = state['cg']
        delta = 0.03 if self.last_sentiment > 0 else -0.03
        for cid in cg.nodes: # cg.nodes should be iterable, e.g. a dict
            if cid in cg.nodes: # Ensure cid is a valid key
                 a = cg.getActivation(cid) # Assumes cg has getActivation
                 cg.setActivation(cid, max(0.0, min(1.0, a + delta))) # Assumes cg has setActivation
