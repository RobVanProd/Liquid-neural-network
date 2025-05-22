# From User Part 3 - Labelled "Layer 9: Affective Body Layer"
# Maps to README Layer 9: Affective Modulator + Body Schema (ABM)

class AffectiveBodyLayer:
    def run(self, state_dict): # state is a dict
        last_response = state_dict.get("last_response", "").lower()
        valence_value = 0.1 if "good" in last_response else -0.05
        state_dict['valence'] = valence_value # Store valence in the shared state
