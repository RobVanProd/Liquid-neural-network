import torch
import numpy as np
import matplotlib.pyplot as plt
from liquid_neural_network import LiquidNeuralNetwork

def generate_sine_data(samples=1000, sequence_length=20):
    """Generate sine wave data with noise"""
    t = np.linspace(0, 8*np.pi, samples)
    sine = np.sin(t) + np.random.normal(0, 0.1, samples)
    
    X, y = [], []
    for i in range(len(sine) - sequence_length):
        X.append(sine[i:i+sequence_length])
        y.append(sine[i+sequence_length])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

def test_temporal_dynamics(model, X, steps_range=[1, 3, 5, 10]):
    """Test network performance with different temporal steps"""
    results = []
    for steps in steps_range:
        with torch.no_grad():
            output = model(X[:1], steps=steps)
        results.append(output.numpy())
    
    plt.figure(figsize=(12, 4))
    for i, steps in enumerate(steps_range):
        plt.plot(results[i][0], label=f'Steps={steps}')
    plt.title('Output for Different Temporal Steps')
    plt.legend()
    plt.show()

def visualize_predictions(model, X, y):
    """Visualize model predictions vs actual values"""
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    plt.figure(figsize=(12, 4))
    plt.plot(y.numpy()[:100], label='Actual')
    plt.plot(predictions.numpy()[:100], label='Predicted')
    plt.title('Model Predictions vs Actual Values')
    plt.legend()
    plt.show()

def main():
    # Generate dataset
    X, y = generate_sine_data()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Create model with modified parameters
    model = LiquidNeuralNetwork(
        input_size=20,
        hidden_size=30,
        output_size=1
    )
    
    # Train with different learning rates
    learning_rates = [0.01, 0.001]
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        losses = model.train(X, y, epochs=100, lr=lr)
        
        # Plot training progress
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.title(f'Training Loss (lr={lr})')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        
        # Let the network self-improve
        model.self_improve()
    
    # Test temporal dynamics
    print("\nTesting temporal dynamics...")
    test_temporal_dynamics(model, X)
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(model, X, y)
    
    # Print network statistics
    print("\nNetwork Statistics:")
    print(f"Final hidden layer size: {model.hidden_size}")
    print(f"Time constants mean: {model.tau.mean().item():.3f}")
    print(f"Time constants std: {model.tau.std().item():.3f}")

if __name__ == "__main__":
    main()
