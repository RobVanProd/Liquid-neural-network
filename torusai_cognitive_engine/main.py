# To run this script as part of the package, execute from the directory
# containing 'torusai_cognitive_engine':
# python -m torusai_cognitive_engine.main

if __name__ == "__main__":
    try:
        from .layer_04_dcg.concept_graph import EnhancedConceptGraph
        from .layer_13_imr.language_generator import EnhancedLanguageGenerator
        from .engine.torus_engine import TorusEngine
        from .tests.stress_test import run_stress_test 
    except ImportError as e:
        print(f"ImportError: {e}. Ensure you are running this as part of the torusai_cognitive_engine package.")
        print("For example, from the directory containing 'torusai_cognitive_engine', run: python -m torusai_cognitive_engine.main")
        exit(1)

    print("--- Running TorusAI Cognitive Engine Stress Test (via main.py) ---")
    
    stress_test_results = run_stress_test(
        cg_class=EnhancedConceptGraph,
        lg_class=EnhancedLanguageGenerator,
        engine_class=TorusEngine
    )
    
    print("\n--- Stress Test Results ---")
    for key, value in stress_test_results.items():
        if isinstance(value, list) and key in ["dreams_logged", "narrative_samples"]:
            print(f"{key}:")
            for item_idx, item_val in enumerate(value): # Changed variable names to avoid conflict
                print(f"  - {item_val}")
        else:
            print(f"{key}: {value}")

    print("\n\n--- Example of Direct TorusEngine Usage (Few Cycles) ---")
    # Setup for direct engine usage
    cg_direct = EnhancedConceptGraph()
    cg_direct.addEnhancedNode("user", linguistics={"wordForms":{"base":"user"}})
    cg_direct.addEnhancedNode("query", linguistics={"wordForms":{"base":"query"}}) # Changed from "greet"
    cg_direct.addEnhancedNode("info", linguistics={"wordForms":{"base":"information"}}) # Changed from "system"
    cg_direct.addEdge("user", "query", weight=1.0, relation="asks")
    cg_direct.addEdge("query", "info", weight=1.0, relation="requests")
    cg_direct.setActivation("user", 0.9)
    cg_direct.setActivation("query", 0.8)


    narrative_for_direct_lg = []
    lg_direct = EnhancedLanguageGenerator(cg_direct, narrative_for_direct_lg)
    
    # Engine parameters (using defaults from the 15-layer spec)
    engine_params = {
        'decay_awake': 0.02, 
        'decay_dream': 0.05,
        'hebbian_multiplier': 1.1,
        'prune_threshold': 0.2,
        'dream_n_candidates': 5,
        'dream_keep_top_k': 2
    }
    engine_direct = TorusEngine(cg_instance=cg_direct, lg_instance=lg_direct, params=engine_params)

    print("\nInitial state of direct CG nodes:", cg_direct.nodes)
    print("Initial state of direct CG edges:", cg_direct.edges)
    print(f"Initial direct engine state sample (valence): {engine_direct.state.get('valence')}")
    
    print("\nRunning a few standard cycles with direct engine:")
    queries = ["User asks for current status", "User requests system info"]
    for i, q_str in enumerate(queries):
        print(f"\nCycle {i+1} with query: '{q_str}'")
        response = engine_direct.standard_cycle(query=q_str)
        print(f"  Response: {response}")
        print(f"  Narrative Log: {narrative_for_direct_lg[-1 if narrative_for_direct_lg else 'N/A']}")
        print(f"  Engine State Valence: {engine_direct.state.get('valence')}")
        print(f"  Engine State Reflection: {engine_direct.state.get('reflection')}")
        print(f"  Engine State Metrics: {engine_direct.state.get('metrics')}")
        print(f"  Engine State Clusters: {engine_direct.state.get('clusters')}")


    print("\nRunning a dream cycle with direct engine:")
    engine_direct.dream_cycle_execution()
    print(f"  Dream Log after dream cycle: {engine_direct.state.get('dream_log')}")
    print(f"  Engine State Metrics after dream: {engine_direct.state.get('metrics')}") # Metrics might change after dream cycle processing
    
    print("\n--- End of Direct Engine Usage Example ---")
