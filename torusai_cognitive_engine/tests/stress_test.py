# From User Part 2 - run_stress_test
# Test utility for the provided components
import random
import time # Ensure time is imported

from ..layer_04_dcg.concept_graph import EnhancedConceptGraph
from ..layer_13_imr.language_generator import EnhancedLanguageGenerator
from ..engine.torus_engine import TorusEngine
from ..layer_04_dcg.graph_dynamics import GeometryLayer, DynamicsLayer, HebbianLayer
from ..layer_08_egdpe.dream_logic import DreamLayer


def run_stress_test(cg_class, lg_class, engine_class, 
                    geom_class, dyn_class, hebb_class, dream_class):
    
    cg_instance = cg_class() 
    
    # Create nodes
    node_ids = [f"node{i}" for i in range(500)] # Reduced for faster testing
    for i, node_id in enumerate(node_ids):
        role = 'subject' if i % 5 == 0 else ('verb' if i % 5 in (1, 2) else 'object')
        cg_instance.addEnhancedNode(node_id, 
                                  properties={"type": role}, 
                                  linguistics={"wordForms": {"base": node_id}, "syntacticRoles": [role]})
        cg_instance.setActivation(node_id, random.random() * 0.2)
    
    # Create edges
    # Reduced edge creation for potentially faster testing
    for _ in range(1000): # Reduced from 100000
        if len(node_ids) < 2: break # Need at least 2 nodes to form an edge
        s_node_id, t_node_id = random.sample(node_ids, 2) # Ensure s_node_id != t_node_id
        cg_instance.addEdge(s_node_id, t_node_id, weight=random.random())

    narrative_log = [] # Initialize an empty list for the narrative
    lg_instance = lg_class(cg_instance, narrative_log) 
    
    # Instantiate layers for the engine
    geom_instance = geom_class()
    dyn_instance = dyn_class()
    hebb_instance = hebb_class()
    dream_instance = dream_class(lg_instance) # DreamLayer needs lg instance

    engine_instance = engine_class(
        cg_instance=cg_instance, 
        lg_instance=lg_instance, 
        geom_layer_instance=geom_instance,
        dyn_layer_instance=dyn_instance,
        hebb_layer_instance=hebb_instance,
        dream_layer_instance=dream_instance,
        ticks=2
    )
    
    start_run_time = time.perf_counter()

    # Cycle the engine
    for i in range(min(10, len(node_ids))): # Cycle based on available nodes
        engine_instance.cycle(f"What does {node_ids[i]} do?")
    
    # Dream cycle
    for _ in range(min(2, len(node_ids))): # Dream cycle if nodes exist
        if not node_ids: break
        engine_instance.dream_cycle()
    
    end_run_time = time.perf_counter()

    results = {
        "run_time_sec_total": round(end_run_time - start_run_time, 4),
        "node_count_final": len(cg_instance.nodes),
        "edge_count_final": len(cg_instance.edges),
        "dreams_logged": engine_instance.state.get("dream_log", []),
        "narrative_samples": narrative_log[-5:] # narrative_log is directly modified by lg_instance
    }
    return results
