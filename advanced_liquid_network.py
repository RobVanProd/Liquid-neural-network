import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

class AttentionModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.attention(x, x, x)
        return self.norm(attn_out + x).squeeze(1)

class ResidualLiquidLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.liquid = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.liquid(h))  # Use h instead of x for the residual connection

class AdvancedLiquidNeuralNetwork(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int,
                 num_liquid_layers: int = 3,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super().__init__()
        
        # Network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Adaptive time constants with learnable initialization
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.5)
        self.tau_learning_rate = nn.Parameter(torch.ones(hidden_size) * 0.01)
        
        # Network layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.liquid_layers = nn.ModuleList([
            ResidualLiquidLayer(hidden_size) if use_residual
            else nn.Linear(hidden_size, hidden_size)
            for _ in range(num_liquid_layers)
        ])
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionModule(hidden_size)
        
        # Output layers with skip connection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
        
        # Adaptive components
        self.complexity_estimator = nn.Linear(hidden_size, 1)
        self.performance_history = []
        self.architecture_history = []
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def liquid_activation(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Enhanced liquid activation with adaptive time constants"""
        total_liquid = torch.zeros_like(h)
        
        # Process through liquid layers
        for layer in self.liquid_layers:
            if self.use_residual:
                liquid_out = layer(x, h)
            else:
                liquid_out = layer(h)
            total_liquid = total_liquid + liquid_out
        
        # Apply attention if enabled
        if self.use_attention:
            total_liquid = self.attention(total_liquid)
        
        # Adaptive temporal dynamics
        dh = (-h + F.tanh(total_liquid + self.input_layer(x))) / (self.tau.unsqueeze(0) + 1e-6)
        
        # Update time constants based on gradient information
        if self.training:
            with torch.no_grad():
                self.tau.data += self.tau_learning_rate * dh.abs().mean(0)
                self.tau.data.clamp_(min=0.1, max=10.0)
        
        return h + dh
    
    def forward(self, x: torch.Tensor, steps: int = 3) -> torch.Tensor:
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Multiple steps through liquid layers
        h_states = []
        for _ in range(steps):
            h = self.liquid_activation(x, h)
            h_states.append(h)
        
        # Combine temporal information
        h_temporal = torch.stack(h_states, dim=1)
        h_final = torch.cat([h, h_states[0]], dim=1)  # Skip connection
        
        # Estimate complexity for self-improvement
        if self.training:
            complexity = self.complexity_estimator(h).mean()
            self.current_complexity = complexity.item()
        
        return self.output_projection(h_final)
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, 
                  optimizer: torch.optim.Optimizer) -> float:
        """Enhanced training step with gradient clipping"""
        self.train()
        optimizer.zero_grad()
        
        output = self(X)
        loss = F.mse_loss(output, y)
        
        # Add complexity regularization
        if hasattr(self, 'current_complexity'):
            loss = loss + 0.01 * self.current_complexity
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        return loss.item()
    
    def train_model(self, X: torch.Tensor, y: torch.Tensor, 
                   epochs: int = 100, lr: float = 0.001,
                   patience: int = 10) -> List[float]:
        """Train with early stopping and learning rate scheduling"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            loss = self.train_step(X, y, optimizer)
            losses.append(loss)
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        self.performance_history.append(min(losses))
        return losses
    
    def self_improve(self) -> None:
        """Advanced self-improvement mechanism"""
        if len(self.performance_history) < 2:
            return
        
        recent_performance = self.performance_history[-1]
        previous_performance = self.performance_history[-2]
        
        # Calculate improvement ratio
        improvement_ratio = recent_performance / previous_performance
        
        # If performance is degrading
        if improvement_ratio > 1.2:  # 20% worse performance
            # Adapt time constants more aggressively
            with torch.no_grad():
                self.tau.data *= torch.exp(torch.randn_like(self.tau) * 0.1)
                self.tau.data.clamp_(min=0.1, max=10.0)
            
            # Increase network capacity if needed
            if improvement_ratio > 1.5:  # 50% worse performance
                self._expand_network()
        
        # Record architecture state
        self.architecture_history.append({
            'hidden_size': self.hidden_size,
            'tau_mean': self.tau.mean().item(),
            'tau_std': self.tau.std().item(),
            'performance': recent_performance
        })
    
    def _expand_network(self) -> None:
        """Expand network capacity with smart initialization"""
        expansion_size = self.hidden_size // 2
        new_hidden_size = self.hidden_size + expansion_size
        
        # Expand layers
        new_input_layer = nn.Linear(self.input_size, new_hidden_size)
        new_liquid_layers = nn.ModuleList([
            ResidualLiquidLayer(new_hidden_size) if self.use_residual
            else nn.Linear(new_hidden_size, new_hidden_size)
            for _ in range(len(self.liquid_layers))
        ])
        
        # Smart initialization: copy existing weights
        with torch.no_grad():
            new_input_layer.weight[:self.hidden_size] = self.input_layer.weight
            new_input_layer.weight[self.hidden_size:] = self.input_layer.weight[:expansion_size]
            
            for i, layer in enumerate(self.liquid_layers):
                if self.use_residual:
                    new_liquid_layers[i].liquid.weight[:self.hidden_size, :self.hidden_size] = \
                        layer.liquid.weight
                else:
                    new_liquid_layers[i].weight[:self.hidden_size, :self.hidden_size] = \
                        layer.weight
        
        # Update network
        self.hidden_size = new_hidden_size
        self.input_layer = new_input_layer
        self.liquid_layers = new_liquid_layers
        
        # Expand time constants
        new_tau = nn.Parameter(torch.ones(new_hidden_size) * self.tau.mean())
        new_tau.data[:self.hidden_size] = self.tau.data
        self.tau = new_tau
        
        print(f"Network expanded to hidden size: {self.hidden_size}")
    
    def visualize_dynamics(self) -> None:
        """Visualize network dynamics and adaptation"""
        if not self.architecture_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot network size and performance
        history = self.architecture_history
        epochs = range(len(history))
        
        ax1.plot([h['hidden_size'] for h in history], label='Hidden Size')
        ax1.set_ylabel('Network Size')
        ax1.set_xlabel('Adaptation Step')
        ax1.legend()
        
        # Plot time constant statistics
        ax2.plot([h['tau_mean'] for h in history], label='Tau Mean')
        ax2.fill_between(epochs,
                        [h['tau_mean'] - h['tau_std'] for h in history],
                        [h['tau_mean'] + h['tau_std'] for h in history],
                        alpha=0.3)
        ax2.set_ylabel('Time Constants')
        ax2.set_xlabel('Adaptation Step')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
