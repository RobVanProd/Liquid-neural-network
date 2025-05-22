# From User Part 3 - Labelled "Layer 14: API / Output Embedder"
# Maps to README Layer 14: High-Level APIs / Embedders

class APIEmbedderLayer:
    def run(self, state_dict): # state is a dict
        if "last_response" in state_dict: # last_response is from TorusEngine via LanguageGenerator
            # Basic print output. A real API might format as JSON, XML, etc.
            print("[TorusAI Output]:", state_dict["last_response"])
