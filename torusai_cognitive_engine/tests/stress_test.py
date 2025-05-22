import random
import time
# Relative imports for classes used by the stress test
from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
from ..layer_13_imr.language_generator import EnhancedLanguageGenerator
from ..engine.torus_engine import TorusEngine
# Note: Specific layer classes like GeometryLayer etc. are no longer directly needed here
# as TorusEngine instantiates them internally.

def run_stress_test(cg_class, lg_class, engine_class):
    
    cg_instance = cg_class() 
    
    node_ids = [f"node{i}" for i in range(500)] 
    for i, node_id in enumerate(node_ids):
        role = 'subject' if i % 5 == 0 else ('verb' if i % 5 in (1, 2) else 'object')
        cg_instance.addEnhancedNode(node_id, 
                                  properties={"type": role}, 
                                  linguistics={"wordForms": {"base": node_id}, "syntacticRoles": [role]})
        cg_instance.setActivation(node_id, random.random() * 0.2)
    
    for _ in range(1000): 
        if len(node_ids) < 2: break
        s_node_id, t_node_id = random.sample(node_ids, 2)
        cg_instance.addEdge(s_node_id, t_node_id, weight=random.random())

    narrative_log = []
    lg_instance = lg_class(cg_instance, narrative_log) 
    
    engine_params = {
        'decay_awake': 0.02, 
        'decay_dream': 0.05,
        'hebbian_multiplier': 1.1,
        'prune_threshold': 0.2,
        'dream_n_candidates': 5,
        'dream_keep_top_k': 2
        # 'low_ticks' is a parameter defined in the 15-layer spec for TorusEngine,
        # but the current TorusEngine.run_cycle doesn't use self.low_ticks internally for its loop.
        # The old engine took 'ticks' for its own loop. This is a minor discrepancy to note.
    }
    # Instantiate TorusEngine with the graph, language generator, and parameters
    engine_instance = engine_class(cg_instance=cg_instance, lg_instance=lg_instance, params=engine_params)
    
    start_run_time = time.perf_counter()

    # Use the new standard_cycle method
    for i in range(min(10, len(node_ids))):
        engine_instance.standard_cycle(query=f"What does {node_ids[i]} do?")
    
    # Use the new dream_cycle_execution method
    for _ in range(min(2, len(node_ids))): 
        if not node_ids: break
        engine_instance.dream_cycle_execution() 
    
    end_run_time = time.perf_counter()

    results = {
        "run_time_sec_total": round(end_run_time - start_run_time, 4),
        "node_count_final": len(cg_instance.nodes),
        "edge_count_final": len(cg_instance.edges),
        "dreams_logged": engine_instance.state.get("dream_log", []), # Access dream_log from engine's state
        "narrative_samples": narrative_log[-5:] # narrative_log is directly modified by lg_instance
    }
    return results
