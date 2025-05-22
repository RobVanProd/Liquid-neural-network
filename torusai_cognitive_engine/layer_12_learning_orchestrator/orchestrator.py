import time
import hashlib
import json # Added for potential serialization in LearningOrchestrator
from collections import deque
from typing import Any, Dict, List, Tuple, Optional

class PatternMemory:
    MAX_ITEMS, RETAIN_PCT, RECENT_N, TOP_K, THRESH = 50_000, 0.8, 1000, 5, 0.35
    
    def __init__(self, max_items: int = MAX_ITEMS):
        self.max_items = max_items
        self.buf = deque() # type: deque[Tuple[str, str, Dict[str, Any]]]
        self.sha_idx = {} # type: Dict[str, int]
        self.written = 0
        self.last_prune = 0.0

    def record(self, trigger: str, response: str,
               meta: Optional[Dict[str, Any]] = None) -> None:
        if len(self.buf) >= self.max_items:
            self._prune()
        
        current_meta = meta if meta is not None else {}
        current_meta["ts"] = time.time()
        
        self.buf.append((trigger, response, current_meta))
        self.sha_idx[hashlib.sha1(trigger.encode('utf-8')).hexdigest()] = len(self.buf) - 1
        self.written += 1

    def nearest(self, query: str, k: int = TOP_K, t: float = THRESH) -> List[Tuple[str, str, Dict[str, Any], float]]:
        if not query: 
            return []
            
        qv = self._vec(query)
        candidate_items = list(self._recent()) if len(self.buf) >= self.RECENT_N else list(self.buf)
        if not candidate_items:
            return []

        out = []
        for trig, resp, meta_info in candidate_items:
            # Corrected variable name from self_vec_trig to self._vec(trig)
            sim = self._cos(qv, self._vec(trig)) 
            if sim >= t:
                out.append((trig, resp, meta_info, sim))
        
        return sorted(out, key=lambda x: -x[3])[:k]

    def stats(self) -> Dict[str, Any]:
        return {"items": len(self.buf), "written": self.written,
                "last_prune_timestamp": self.last_prune}

    def _vec(self, s: str) -> List[int]:
        v = [0]*128
        for char_val in s[:1024].encode('utf-8', 'ignore'): 
            v[char_val % 128] += 1 
        return v

    def _cos(self, a: List[int], b: List[int]) -> float:
        dot_product = sum(x_val * y_val for x_val, y_val in zip(a, b))
        norm_a = sum(x_val * x_val for x_val in a)**0.5
        norm_b = sum(y_val * y_val for y_val in b)**0.5
        
        if norm_a == 0 or norm_b == 0: 
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _recent(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        num_recent = min(len(self.buf), self.RECENT_N)
        # Correctly create a list of recent items from the right end of the deque
        return [self.buf[len(self.buf) - 1 - i] for i in range(num_recent)]


    def _prune(self) -> None:
        items_to_keep = int(self.max_items * self.RETAIN_PCT)
        while len(self.buf) > items_to_keep:
            old_trigger, _, _ = self.buf.popleft() 
            self.sha_idx.pop(hashlib.sha1(old_trigger.encode('utf-8')).hexdigest(), None)
        self.last_prune = time.time()

class LearningOrchestrator:
    def __init__(self, shared_engine_state: Dict[str, Any]):
        self.state = shared_engine_state
        self.pattern_memory = PatternMemory()

    def step(self) -> None:
        reflection_data = self.state.get("reflection", "") 
        last_response_str = str(self.state.get("last_response", ""))
        current_valence = self.state.get("valence", 0.0)

        # Determine reflection_str based on type of reflection_data
        if isinstance(reflection_data, list) and reflection_data: # Assuming it's list of dicts from getActiveConceptPath
            # Serialize the list of active concepts (e.g., join their IDs or full JSON)
            # A simpler approach for now: just use the ID of the first concept if available
            reflection_str = reflection_data[0].get("id", "") if reflection_data[0] else ""
            # For a more robust key, one might join all IDs:
            # reflection_str = "_".join(sorted([item.get("id","") for item in reflection_data]))
        elif isinstance(reflection_data, str):
            reflection_str = reflection_data
        else: # Handle other types or empty reflection_data
            reflection_str = ""


        if reflection_str and last_response_str:
            meta_data = {"valence_at_recording": current_valence, "timestamp": time.time()}
            self.pattern_memory.record(reflection_str, last_response_str, meta=meta_data)
        
        if reflection_str:
            retrieved = self.pattern_memory.nearest(reflection_str)
            self.state["retrieved_patterns"] = retrieved
        else:
            self.state["retrieved_patterns"] = []

# Optional: Comment out or remove the self-test block for final integration
# if __name__ == "__main__":
#     example_state = {
#         "reflection": [{"id":"conceptA", "activation":0.8}, {"id":"conceptB", "activation":0.7}], 
#         "last_response": "a generated response to conceptA",
#         "valence": 0.5 
#     }
# 
#     orchestrator = LearningOrchestrator(example_state)
#     orchestrator.step()
#     print("State after LearningOrchestrator step:", json.dumps(example_state, indent=2))
# 
#     pm = orchestrator.pattern_memory # Test the pm instance from the orchestrator
#     print("\nPatternMemory Stats:", pm.stats())
#     
#     # Example of how reflection might be stored if it's a list of dicts
#     reflection_as_key = example_state["reflection"][0].get("id", "") # Simplified key
#     # reflection_as_key = "_".join(sorted([item.get("id","") for item in example_state["reflection"]]))

#     print(f"Nearest to '{reflection_as_key}':", pm.nearest(reflection_as_key))
