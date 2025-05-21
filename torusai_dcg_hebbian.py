from torusai_affect_encoding import AffectVector

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.activation = 0.0

    def __repr__(self):
        return f"Node(id='{self.node_id}', activation={self.activation})"

class Edge:
    def __init__(self, source_node: Node, target_node: Node, weight: float):
        self.source_node = source_node
        self.target_node = target_node
        self.weight = weight

    def __repr__(self):
        return f"Edge({self.source_node.node_id} -> {self.target_node.node_id}, weight={self.weight})"

# Constants for the Affective Hebbian update rule
ETA = 0.05  # Learning rate η
LAMBDA_DECAY = 0.01  # Decay rate λ
PRUNING_THRESHOLD = 0.01 # Pruning threshold

def update_edge_weight_affective_hebbian(
    edge_to_update: Edge,
    pre_synaptic_activation: float,
    post_synaptic_activation: float,
    valence: float,
    dominance: float
):
    """
    Updates the weight of an edge based on the Affective Hebbian learning rule.

    Args:
        edge_to_update: The edge whose weight is to be updated.
        pre_synaptic_activation: Activation of the source node.
        post_synaptic_activation: Activation of the target node.
        valence: Current valence from the affect system.
        dominance: Current dominance from the affect system.
    """
    # Calculate delta_w based on Affective Hebbian rule
    delta_w = ETA * valence * dominance * pre_synaptic_activation * post_synaptic_activation

    # Update the edge's weight
    current_weight = edge_to_update.weight
    new_weight_before_decay = current_weight + delta_w

    # Apply decay to the updated weight
    final_new_weight = new_weight_before_decay * (1 - LAMBDA_DECAY)

    # Assign the final new weight back to the edge
    edge_to_update.weight = final_new_weight

    # Pruning check
    if abs(edge_to_update.weight) < PRUNING_THRESHOLD:
        edge_to_update.weight = 0.0
        print(f"INFO: Edge connecting {edge_to_update.source_node.node_id} to {edge_to_update.target_node.node_id} was pruned (weight set to 0.0).")

if __name__ == "__main__":
    # 1. Initialization
    print("--- Initializing Nodes, Edges, and AffectVector ---")
    node_A = Node(node_id='A')
    node_B = Node(node_id='B')
    node_C = Node(node_id='C')

    edge_AB = Edge(source_node=node_A, target_node=node_B, weight=0.5)
    edge_BC = Edge(source_node=node_B, target_node=node_C, weight=0.01005) # Extremely close to pruning threshold

    affect_sim = AffectVector()

    print(node_A)
    print(node_B)
    print(node_C)
    print(edge_AB)
    print(edge_BC)
    print(affect_sim)
    print("-" * 50)

    # 2. First Update Scenario (for edge_AB)
    print("\n--- First Update Scenario (edge_AB) ---")
    node_A.activation = 0.8
    node_B.activation = 0.7
    print(f"Set Node Activations: {node_A}, {node_B}")

    affect_sim.update(reward=0.5, delta_intero=[0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
    valence = affect_sim.get_vector()[0] # Index for valence
    dominance = affect_sim.get_vector()[2] # Index for dominance

    print("Current Affect State:", affect_sim)
    print("Edge before update:", edge_AB)
    print(f"Applying update with: pre_activation={node_A.activation}, post_activation={node_B.activation}, valence={valence:.4f}, dominance={dominance:.4f}")
    
    update_edge_weight_affective_hebbian(edge_AB, node_A.activation, node_B.activation, valence, dominance)
    print("Edge after update:", edge_AB)
    print("-" * 50)

    # 3. Second Update Scenario (for edge_AB, different affect)
    print("\n--- Second Update Scenario (edge_AB, different affect) ---")
    node_A.activation = 0.6
    node_B.activation = 0.9
    print(f"Set Node Activations: {node_A}, {node_B}")

    affect_sim.update(reward=-0.2, delta_intero=[0.0, 0.0, -0.3, 0.0, 0.0, 0.0])
    valence = affect_sim.get_vector()[0] # Index for valence
    dominance = affect_sim.get_vector()[2] # Index for dominance

    print("Current Affect State:", affect_sim)
    print("Edge before update:", edge_AB)
    print(f"Applying update with: pre_activation={node_A.activation}, post_activation={node_B.activation}, valence={valence:.4f}, dominance={dominance:.4f}")

    update_edge_weight_affective_hebbian(edge_AB, node_A.activation, node_B.activation, valence, dominance)
    print("Edge after update:", edge_AB)
    print("-" * 50)

    # 4. Pruning Scenario (for edge_BC)
    print("\n--- Pruning Scenario (edge_BC) ---")
    node_B.activation = 0.1 # Low activation
    node_C.activation = 0.1 # Low activation
    print(f"Set Node Activations: {node_B}, {node_C}")

    # Update affect to produce low valence/dominance or conditions that lead to small/negative delta_w
    # Previous state of affect_sim might already be suitable, or we can force one.
    # Let's use a specific update to ensure a small delta_w or a negative one.
    # The current affect_sim valence is likely positive, dominance likely negative from previous step.
    # A small positive valence and small positive dominance (or one negative) with small activations should result in a small delta_w.
    # The existing affect_sim state after step 3 will be used.
    # If valence or dominance is negative, delta_w will be negative, reducing the weight.
    # If both are small positive, delta_w will be small positive.
    # The key is that the existing weight (0.01005) is extremely close to PRUNING_THRESHOLD (0.01)
    # The goal is for the decay factor to reduce the weight below the threshold.
    # A small positive delta_w or slightly negative delta_w will achieve this.
    # W_new = (W_old + delta_w) * (1 - LAMBDA_DECAY)
    # If W_old = 0.01005, (0.01005 + delta_w) * 0.99 < 0.01
    # 0.01005 + delta_w < 0.01 / 0.99 = 0.010101...
    # delta_w < 0.010101 - 0.01005 = 0.00005101
    # We need delta_w = 0.0005 * valence * dominance < 0.00005101
    # So, valence * dominance < 0.00005101 / 0.0005 = 0.102

    print("Forcing affect state for pruning test (aiming for small val*dom product)...")
    reward_prune = 0.0
    # Aim for valence approx 0.0935, dominance approx 0.1045 from previous calculations.
    # This gives val*dom approx 0.0097, which is < 0.102.
    intero_prune = [-0.1, 0.0, 0.1, 0.0, 0.0, 0.0]
    affect_sim.update(reward=reward_prune, delta_intero=intero_prune)
    
    valence = affect_sim.get_vector()[0] # Index for valence
    dominance = affect_sim.get_vector()[2] # Index for dominance
    
    print("Current Affect State (after forcing for pruning):", affect_sim)
    print("Edge before update:", edge_BC)
    print(f"Applying update with: pre_activation={node_B.activation}, post_activation={node_C.activation}, valence={valence:.4f}, dominance={dominance:.4f}")

    update_edge_weight_affective_hebbian(edge_BC, node_B.activation, node_C.activation, valence, dominance)
    print("Edge after update:", edge_BC) # Should show pruning message and weight 0.0
    print("-" * 50)
