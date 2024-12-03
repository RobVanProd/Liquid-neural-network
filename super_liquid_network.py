import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class NeuroEvolutionModule(nn.Module):
    """Module that can evolve its architecture dynamically"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.PReLU()  # Learnable activation
        self.complexity_score = 0.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer(x))
    
    def mutate(self) -> None:
        """Mutate weights slightly"""
        with torch.no_grad():
            self.layer.weight.data += torch.randn_like(self.layer.weight) * 0.1

class MultiScaleLiquidLayer(nn.Module):
    """Liquid layer that operates at multiple time scales"""
    def __init__(self, hidden_size: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        self.dropout = nn.Dropout(dropout)
        
        # Different time scales
        self.time_constants = nn.Parameter(torch.linspace(0.1, 2.0, num_scales))
        
        # Layers for each time scale
        self.scale_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_scales)
        ])
        
        # Scale attention
        self.scale_attention = nn.Linear(hidden_size, num_scales)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Compute contributions at each time scale
        scale_outputs = []
        for i in range(self.num_scales):
            dh = (-h + torch.tanh(self.scale_layers[i](h))) / self.time_constants[i]
            scale_outputs.append(dh)
        
        # Compute attention weights with dropout
        attention_weights = F.softmax(self.dropout(self.scale_attention(h)), dim=-1)
        
        # Combine scales using attention
        combined_dh = torch.zeros_like(h)
        for i, dh in enumerate(scale_outputs):
            combined_dh += dh * attention_weights[:, i:i+1]
        
        return h + combined_dh

class SuperLiquidNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 hidden_dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01):
        super().__init__()
        
        # Store hyperparameters
        self.hparams = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'attention_dropout': attention_dropout,
            'hidden_dropout': hidden_dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Input projection with position encoding and dropout
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, hidden_size))
        self.dropout = nn.Dropout(dropout)
        
        # Multi-scale liquid layers with different dropouts
        self.liquid_layers = nn.ModuleList([
            MultiScaleLiquidLayer(hidden_size, dropout=hidden_dropout)
            for _ in range(num_layers)
        ])
        
        # Transformer with attention dropout
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            activation=F.gelu,  # Using GELU activation
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        
        # Evolution modules with dropout
        self.evolution_modules = nn.ModuleList([
            nn.Sequential(
                NeuroEvolutionModule(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(hidden_dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.complexity_history: List[float] = []
        self.attention_patterns: List[torch.Tensor] = []
        self.gradient_norms: List[float] = []
        self.layer_activations: List[Dict] = []
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_state = None
        
        # Initialize visualization tools
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1) if len(x.size()) > 2 else 1
        
        # Add positional encoding
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x) + self.pos_encoder[:, :seq_len, :]
        
        # Initial hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Store intermediate features for visualization
        features = {'input': x.detach(), 'hidden_states': []}
        
        # Process through liquid layers
        for i, (liquid, evolver) in enumerate(zip(self.liquid_layers, self.evolution_modules)):
            h = liquid(x, h)
            h = evolver(h)
            features['hidden_states'].append(h.detach())
        
        # Apply transformer for global temporal dependencies
        h = self.transformer(h.unsqueeze(1)).squeeze(1)
        features['transformer_output'] = h.detach()
        
        # Output projection
        output = self.output_proj(h)  
        features['output'] = output.detach()
        
        if return_features:
            return output, features
        return output
    
    def self_improve(self, loss: float, threshold: float = 1.2) -> None:
        """Advanced self-improvement mechanism"""
        if len(self.performance_history) < 2:
            self.performance_history.append({'loss': loss})
            return
        
        recent_loss = loss
        prev_loss = self.performance_history[-1]['loss']
        
        # Calculate improvement metrics
        relative_improvement = (prev_loss - recent_loss) / prev_loss
        
        # Update evolution modules based on performance
        if relative_improvement < -threshold:  # Performance degraded
            print("Performance degraded, evolving network...")
            for module in self.evolution_modules:
                module[0].mutate()
        
        # Track complexity
        complexity = sum(p.numel() for p in self.parameters())
        self.complexity_history.append(complexity)
        
        # Store performance metrics
        self.performance_history.append({
            'loss': loss,
            'improvement': relative_improvement,
            'complexity': complexity
        })
        
        # Early stopping
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter > 5:
                print("Early stopping: no improvement in 5 epochs")
                exit()
    
    def visualize_network_state(self, features: Dict[str, torch.Tensor]) -> None:
        """Advanced network state visualization with multiple plots"""
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Hidden State Evolution
        plt.subplot(3, 2, 1)
        hidden_states = torch.stack([h[0] for h in features['hidden_states']])
        sns.heatmap(hidden_states.cpu().numpy(), cmap='viridis')
        plt.title('Hidden State Evolution Across Layers')
        plt.xlabel('Hidden Dimensions')
        plt.ylabel('Layer')
        
        # Plot 2: Layer-wise Activation Distribution
        plt.subplot(3, 2, 2)
        activations = [h.cpu().numpy().flatten() for h in features['hidden_states']]
        plt.violinplot(activations)
        plt.title('Layer-wise Activation Distribution')
        plt.xlabel('Layer')
        plt.ylabel('Activation Value')
        
        # Plot 3: Training Metrics
        plt.subplot(3, 2, 3)
        if len(self.performance_history) > 0:
            losses = [p['loss'] for p in self.performance_history]
            improvements = [p.get('improvement', 0) for p in self.performance_history]
            plt.plot(losses, label='Loss', color='blue')
            plt.plot(improvements, label='Improvement', color='green', alpha=0.6)
            plt.yscale('log')
            plt.title('Training Metrics History')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.legend()
        
        # Plot 4: Network Complexity
        plt.subplot(3, 2, 4)
        if len(self.complexity_history) > 0:
            plt.plot(self.complexity_history)
            plt.title('Network Complexity Evolution')
            plt.xlabel('Step')
            plt.ylabel('Number of Parameters')
        
        # Plot 5: Gradient Flow
        plt.subplot(3, 2, 5)
        if len(self.gradient_norms) > 0:
            plt.plot(self.gradient_norms)
            plt.title('Gradient Flow')
            plt.xlabel('Step')
            plt.ylabel('Gradient Norm')
            plt.yscale('log')
        
        # Plot 6: Learning Rate Schedule
        plt.subplot(3, 2, 6)
        if len(self.performance_history) > 0:
            lrs = [p.get('lr', 0) for p in self.performance_history]
            plt.plot(lrs)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, true_values: torch.Tensor, 
                            predictions: torch.Tensor,
                            uncertainty: Optional[torch.Tensor] = None,
                            num_steps: int = 100) -> None:
        """Enhanced prediction visualization with uncertainty"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Predictions vs True Values
        plt.subplot(2, 1, 1)
        x = np.arange(num_steps)
        true_np = true_values[:num_steps].cpu().numpy()
        pred_np = predictions[:num_steps].cpu().numpy()
        
        plt.plot(x, true_np, label='True', color='blue', alpha=0.7)
        plt.plot(x, pred_np, label='Predicted', color='red', alpha=0.7)
        
        if uncertainty is not None:
            unc_np = uncertainty[:num_steps].cpu().numpy()
            plt.fill_between(x, pred_np - unc_np, pred_np + unc_np, 
                           color='red', alpha=0.2, label='Uncertainty')
        
        plt.title('Predictions vs True Values')
        plt.legend()
        
        # Plot 2: Error Analysis
        plt.subplot(2, 2, 3)
        error = (predictions - true_values).abs()
        plt.hist(error.cpu().numpy(), bins=50, density=True, alpha=0.7)
        plt.title('Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Density')
        
        # Plot 3: Error Over Time
        plt.subplot(2, 2, 4)
        plt.plot(error[:num_steps].cpu().numpy())
        plt.title('Error Evolution')
        plt.xlabel('Time Step')
        plt.ylabel('Absolute Error')
        
        plt.tight_layout()
        plt.show()
    
    def track_gradient_flow(self) -> None:
        """Track gradient flow through the network"""
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
