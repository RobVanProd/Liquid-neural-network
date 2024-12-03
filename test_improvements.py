import torch
import numpy as np
from liquid_s4 import LiquidS4Model
from cfc_model import CfCModel
from visualization import NetworkVisualizer
import matplotlib.pyplot as plt
import os

def generate_synthetic_data(batch_size: int, seq_length: int, input_size: int):
    """Generate synthetic sequential data for testing"""
    # Generate sinusoidal data with different frequencies
    t = torch.linspace(0, 8*np.pi, seq_length)
    frequencies = torch.randn(batch_size, 1) * 0.5 + 1.0  # Random frequencies
    
    # Create input sequences
    x = torch.sin(frequencies * t.unsqueeze(0))
    # Add more features
    x = torch.cat([
        x,
        torch.sin(2 * frequencies * t.unsqueeze(0)),
        torch.cos(frequencies * t.unsqueeze(0))
    ], dim=-1)
    
    # Expand to desired input size
    if input_size > x.size(-1):
        x = torch.cat([x] * (input_size // x.size(-1) + 1), dim=-1)
    x = x[:, :, :input_size]
    
    # Generate target: predict next value in sequence
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1, :] = 0  # Zero out last prediction
    
    return x, y

def test_models():
    """Test the Liquid-S4 and CfC models"""
    # Parameters
    batch_size = 32
    seq_length = 100
    input_size = 10
    hidden_size = 64
    output_size = input_size  # Predicting next value in sequence
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize visualizer
    visualizer = NetworkVisualizer(log_dir='runs/test_models')
    
    # Generate data
    x, y = generate_synthetic_data(batch_size, seq_length, input_size)
    
    # Test Liquid-S4 Model
    print("Testing Liquid-S4 Model...")
    s4_model = LiquidS4Model(input_size, hidden_size, output_size)
    s4_output = s4_model(x)
    print(f"Liquid-S4 output shape: {s4_output.shape}")
    
    # Test CfC Model
    print("\nTesting CfC Model...")
    cfc_model = CfCModel(input_size, hidden_size, output_size)
    cfc_output = cfc_model(x)
    print(f"CfC output shape: {cfc_output.shape}")
    
    # Visualize sample predictions
    sample_idx = 0
    plt.figure(figsize=(15, 5))
    
    # Plot input sequence
    plt.subplot(1, 2, 1)
    plt.plot(x[sample_idx, :, 0].detach().numpy(), label='Input')
    plt.plot(s4_output[sample_idx, :, 0].detach().numpy(), label='S4 Prediction')
    plt.title('Liquid-S4 Prediction')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x[sample_idx, :, 0].detach().numpy(), label='Input')
    plt.plot(cfc_output[sample_idx, :, 0].detach().numpy(), label='CfC Prediction')
    plt.title('CfC Prediction')
    plt.legend()
    
    plt.savefig('visualizations/model_predictions.png')
    plt.close()
    
    # Test visualization tools
    print("\nTesting visualization tools...")
    
    # Log some metrics
    for i in range(10):
        metrics = {
            'loss': np.exp(-i * 0.1) + np.random.random() * 0.1,
            'accuracy': 1 - np.exp(-i * 0.1) + np.random.random() * 0.1
        }
        visualizer.log_metrics(metrics, i)
    
    # Plot training progress
    visualizer.plot_training_progress('visualizations/training_progress.png')
    
    # Visualize activations
    visualizer.visualize_activations(
        s4_output,
        'liquid_s4_output',
        step=0
    )
    
    # Close visualizer
    visualizer.close()
    
    print("\nTesting completed! Check the 'visualizations' directory for outputs.")

if __name__ == "__main__":
    test_models()
