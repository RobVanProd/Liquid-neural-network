class AffectVector:
    DIMENSIONS = ["valence", "arousal", "dominance", "novelty", "certainty", "effort"]

    def __init__(self):
        self._vector = [0.0] * len(self.DIMENSIONS)

    def get_vector(self):
        return self._vector

    def _clamp_values(self):
        self._vector = [max(-1.0, min(1.0, v)) for v in self._vector]

    def __repr__(self):
        return f"AffectVector({', '.join(f'{dim}: {val}' for dim, val in zip(self.DIMENSIONS, self._vector))})"

    def update(self, reward: float, delta_intero: list[float]):
        alpha = 0.9
        beta = 0.5
        gamma_d_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        if len(delta_intero) != len(self.DIMENSIONS):
            raise ValueError(f"delta_intero must have {len(self.DIMENSIONS)} elements.")

        for d in range(len(self.DIMENSIONS)):
            a_t_minus_1_d = self._vector[d]
            delta_intero_d = delta_intero[d]
            gamma_d = gamma_d_factors[d]

            a_t_d = alpha * a_t_minus_1_d + beta * reward + gamma_d * delta_intero_d
            self._vector[d] = a_t_d

        self._clamp_values()

if __name__ == "__main__":
    # 1. Create an instance of AffectVector
    affect_vector = AffectVector()

    # 2. Print the initial state
    print("Initial state:")
    print(affect_vector)
    print("-" * 30)

    # 3. Define a first set of sample inputs
    reward1 = 0.25
    intero1 = [0.1, -0.05, 0.03, 0.15, -0.02, -0.1]

    # 4. Call the update method with these first inputs
    print(f"Updating with reward: {reward1}, intero: {intero1}")
    affect_vector.update(reward1, intero1)

    # 5. Print the vector's state after the first update
    print("State after first update:")
    print(affect_vector)
    print("-" * 30)

    # 6. Define a second set of sample inputs
    reward2 = -0.1
    intero2 = [0.05, 0.1, -0.1, -0.05, 0.1, 0.05]

    # 7. Call the update method with these second inputs
    print(f"Updating with reward: {reward2}, intero: {intero2}")
    affect_vector.update(reward2, intero2)

    # 8. Print the vector's state after the second update
    print("State after second update:")
    print(affect_vector)
    print("-" * 30)

    # 9. Define a third set of inputs designed to test clamping
    reward3 = 0.8  # High reward
    # Strong interoceptive signals for first two dimensions, others zero to isolate effect
    intero3 = [0.7, -0.9, 0.0, 0.0, 0.0, 0.0] 

    # 10. Call the update method with these third inputs
    print(f"Updating with reward: {reward3}, intero: {intero3} (testing clamping)")
    affect_vector.update(reward3, intero3)

    # 11. Print the vector's state after the third update
    print("State after third update (clamping test):")
    print(affect_vector)
    print("-" * 30)
