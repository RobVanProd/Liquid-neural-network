class InternalRewardEngine:
    # Define weights as class attributes
    W_PREDICTION_ERROR = -0.5
    W_GOAL_PROXIMITY = 0.6
    W_NOVELTY = 0.2
    W_COHERENCE = 0.3
    W_AFFECT_MATCH = 0.2

    def __init__(self):
        # Constructor can be pass for now if weights are class attributes.
        # Could be used later to load weights from a config.
        pass

    def calculate_R_int(self, 
                        prediction_error_signal: float, 
                        goal_proximity_signal: float, 
                        novelty_signal: float, 
                        coherence_signal: float, 
                        affect_match_signal: float) -> float:
        """
        Calculates the internal reward signal (R_int).

        Args:
            prediction_error_signal (float): Assumed range [0, 1] (0=no error, 1=max error).
            goal_proximity_signal (float): Assumed range [0, 1] (0=far, 1=at goal).
            novelty_signal (float): Assumed range [0, 1] (higher is more novel).
            coherence_signal (float): Assumed range [0, 1] (higher is more coherent).
            affect_match_signal (float): Assumed range [0, 1] (higher means better affect alignment).

        Returns:
            float: The calculated R_int, clipped to [0, 1].
        """

        r_prediction_error = self.W_PREDICTION_ERROR * prediction_error_signal
        r_goal_proximity = self.W_GOAL_PROXIMITY * goal_proximity_signal
        r_novelty = self.W_NOVELTY * novelty_signal
        r_coherence = self.W_COHERENCE * coherence_signal
        r_affect_match = self.W_AFFECT_MATCH * affect_match_signal

        # Sum of all weighted components
        R_int_raw = (r_prediction_error + 
                     r_goal_proximity + 
                     r_novelty + 
                     r_coherence + 
                     r_affect_match)

        # Clip the final reward to the range [0, 1]
        R_int_clipped = max(0.0, min(1.0, R_int_raw))

        return R_int_clipped

if __name__ == "__main__":
    engine = InternalRewardEngine()

    print("--- Internal Reward Engine (IRE) POC Demonstration ---")

    test_cases = [
        {"name": "Good Scenario", "inputs": {"prediction_error_signal": 0.1, "goal_proximity_signal": 0.9, "novelty_signal": 0.7, "coherence_signal": 0.8, "affect_match_signal": 0.9}},
        {"name": "Bad Scenario (low reward expected)", "inputs": {"prediction_error_signal": 0.9, "goal_proximity_signal": 0.1, "novelty_signal": 0.2, "coherence_signal": 0.3, "affect_match_signal": 0.1}},
        {"name": "Force Clipping Low (R_int = 0)", "inputs": {"prediction_error_signal": 1.0, "goal_proximity_signal": 0.0, "novelty_signal": 0.0, "coherence_signal": 0.0, "affect_match_signal": 0.0}},
        {"name": "Force Clipping High (R_int = 1)", "inputs": {"prediction_error_signal": 0.0, "goal_proximity_signal": 1.0, "novelty_signal": 1.0, "coherence_signal": 1.0, "affect_match_signal": 1.0}},
        {"name": "Mid-Range Scenario 1", "inputs": {"prediction_error_signal": 0.5, "goal_proximity_signal": 0.5, "novelty_signal": 0.5, "coherence_signal": 0.5, "affect_match_signal": 0.5}},
        {"name": "Mid-Range Scenario 2 (high novelty focus)", "inputs": {"prediction_error_signal": 0.2, "goal_proximity_signal": 0.3, "novelty_signal": 0.9, "coherence_signal": 0.4, "affect_match_signal": 0.3}}
    ]

    for case in test_cases:
        print(f"\n--- Test Case: {case['name']} ---")
        # Unpack the dictionary of inputs for the method call
        current_inputs = case['inputs']
        print(f"Input Signals: {current_inputs}")
        
        r_int = engine.calculate_R_int(
            prediction_error_signal=current_inputs['prediction_error_signal'],
            goal_proximity_signal=current_inputs['goal_proximity_signal'],
            novelty_signal=current_inputs['novelty_signal'],
            coherence_signal=current_inputs['coherence_signal'],
            affect_match_signal=current_inputs['affect_match_signal']
        )
        
        # For verification, let's also calculate R_int_raw here in the demo
        r_pe_raw = InternalRewardEngine.W_PREDICTION_ERROR * current_inputs['prediction_error_signal']
        r_gp_raw = InternalRewardEngine.W_GOAL_PROXIMITY * current_inputs['goal_proximity_signal']
        r_nov_raw = InternalRewardEngine.W_NOVELTY * current_inputs['novelty_signal']
        r_coh_raw = InternalRewardEngine.W_COHERENCE * current_inputs['coherence_signal']
        r_am_raw = InternalRewardEngine.W_AFFECT_MATCH * current_inputs['affect_match_signal']
        R_int_raw_demo = r_pe_raw + r_gp_raw + r_nov_raw + r_coh_raw + r_am_raw
        
        print(f"Calculated R_int_raw (for verification): {R_int_raw_demo:.4f}")
        print(f"Final Clipped R_int: {r_int:.4f}")
