import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LiquidNeuralNetwork, self).__init__()
        
        # Network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Liquid time constants
        self.tau = nn.Parameter(torch.rand(hidden_size))
        
        # Network layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.liquid_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation functions
        self.tanh = nn.Tanh()
        
        # Performance history for self-improvement
        self.performance_history = []
        self.architecture_history = []
        
    def liquid_activation(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Liquid activation function with temporal dynamics"""
        dh = (-h + self.tanh(self.liquid_layer(h) + self.input_layer(x))) / self.tau.unsqueeze(0)
        return h + dh
    
    def forward(self, x: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """Forward pass with multiple liquid timesteps"""
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size)
        
        # Multiple steps through liquid layer
        for _ in range(steps):
            h = self.liquid_activation(x, h)
        
        # Output projection
        return self.output_layer(h)
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Single training step"""
        self.train_mode()
        optimizer.zero_grad()
        
        output = self(X)
        loss = nn.MSELoss()(output, y)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_mode(self):
        """Set the network to training mode"""
        self.training = True
    
    def eval(self):
        """Set the network to evaluation mode"""
        self.training = False
        return self
    
    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.01) -> List[float]:
        """Train the network"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_step(X, y, optimizer)
            losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.performance_history.append(min(losses))
        return losses
    
    def self_improve(self) -> None:
        """Self-improvement mechanism"""
        # Analyze recent performance
        if len(self.performance_history) < 2:
            return
        
        recent_performance = self.performance_history[-1]
        previous_performance = self.performance_history[-2]
        
        # If performance is degrading, adapt the network
        if recent_performance > previous_performance:
            # Adjust time constants
            with torch.no_grad():
                self.tau.data += torch.randn_like(self.tau) * 0.1
                
            # Adapt hidden layer size if needed
            if recent_performance > 1.5 * previous_performance:
                self._expand_hidden_layer()
        
        # Record current architecture
        self.architecture_history.append({
            'hidden_size': self.hidden_size,
            'tau_mean': self.tau.mean().item()
        })
    
    def _expand_hidden_layer(self) -> None:
        """Expand the hidden layer size"""
        new_hidden_size = self.hidden_size + 5
        
        # Create new layers with expanded size
        new_input_layer = nn.Linear(self.input_size, new_hidden_size)
        new_liquid_layer = nn.Linear(new_hidden_size, new_hidden_size)
        new_output_layer = nn.Linear(new_hidden_size, self.output_size)
        
        # Copy existing weights
        with torch.no_grad():
            new_input_layer.weight[:self.hidden_size] = self.input_layer.weight
            new_liquid_layer.weight[:self.hidden_size, :self.hidden_size] = self.liquid_layer.weight
            new_output_layer.weight[:, :self.hidden_size] = self.output_layer.weight
            
        # Update network
        self.hidden_size = new_hidden_size
        self.input_layer = new_input_layer
        self.liquid_layer = new_liquid_layer
        self.output_layer = new_output_layer
        
        # Expand tau parameter
        new_tau = nn.Parameter(torch.rand(new_hidden_size))
        new_tau.data[:self.hidden_size] = self.tau.data
        self.tau = new_tau
    
    def visualize_performance(self) -> None:
        """Visualize network performance over time"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.performance_history)
        plt.title('Network Performance Over Time')
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create synthetic dataset
    X = torch.randn(100, 10)
    y = torch.randn(100, 2)
    
    # Create and train network
    model = LiquidNeuralNetwork(input_size=10, hidden_size=20, output_size=2)
    losses = model.train(X, y)
    
    # Self-improve
    model.self_improve()
    
    # Visualize results
    model.visualize_performance()
