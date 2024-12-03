import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from scipy.linalg import expm

class LiquidS4Cell(nn.Module):
    """
    Liquid Structural State Space (S4) cell that combines liquid networks with S4 properties
    for enhanced sequence modeling capabilities.
    """
    def __init__(self, d_model: int, N: int = 64):
        super().__init__()
        self.N = N
        self.d_model = d_model
        
        # Initialize parameters
        self.Lambda = nn.Parameter(torch.randn(N))  # [N]
        self.log_dt = nn.Parameter(torch.zeros(1))  # [1]
        self.B = nn.Parameter(torch.randn(N, d_model))  # [N, d_model]
        self.C = nn.Parameter(torch.randn(d_model, N))  # [d_model, N]
        self.D = nn.Parameter(torch.zeros(d_model))  # [d_model]
        
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (batch, length, d_model)
        Returns:
            Output tensor of shape (batch, length, d_model)
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        
        # Initialize state
        x = torch.zeros(batch_size, self.N, device=device)
        outputs = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        dt = torch.exp(self.log_dt)
        L = -torch.exp(self.Lambda)  # ensure eigenvalues have negative real part
        A = torch.diag(torch.exp(dt * L))  # [N, N]
        
        # Process sequence with memory-efficient implementation
        for t in range(seq_len):
            # Update state: x = Ax + Bu
            x = torch.matmul(x, A.T) + torch.matmul(u[:, t], self.B.T)
            
            # Compute output: y = Cx + Du
            outputs[:, t] = torch.matmul(x, self.C.T) + self.D * u[:, t]
            
        return outputs

class LiquidS4Layer(nn.Module):
    """
    A layer that combines multiple Liquid S4 cells with skip connections and layer normalization.
    """
    def __init__(self, d_model: int, n_cells: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cells = nn.ModuleList([LiquidS4Cell(d_model) for _ in range(n_cells)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length, d_model)
        Returns:
            Output tensor of shape (batch, length, d_model)
        """
        for cell in self.cells:
            residual = x
            x = self.dropout(cell(x))
            x = self.norm(x + residual)
        return x

class LiquidS4Model(nn.Module):
    """Complete Liquid S4 model for sequence modeling tasks."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Set deterministic initialization
        torch.manual_seed(42)
        
        # Input projection with fixed initialization
        self.input_projection = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=1.0)
        nn.init.zeros_(self.input_projection.bias)
        
        # Create layers with deterministic initialization
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = LiquidS4Layer(hidden_size)
            # Initialize layer parameters deterministically
            for name, param in layer.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    nn.init.zeros_(param)
            self.layers.append(layer)
            
        # Dropout after each layer
        self.dropout = nn.Dropout(dropout, inplace=False)
        
        # Output projection with fixed initialization
        self.output_projection = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=1.0)
        nn.init.zeros_(self.output_projection.bias)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_size)
        
        # Reset RNG state
        torch.manual_seed(torch.initial_seed())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length, input_size)
        Returns:
            Output tensor of shape (batch, length, output_size)
        """
        # Project input
        x = self.input_projection(x)
        x = self.norm(x)
        
        # Process through S4 layers with residual connections
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = self.dropout(x)
            x = x + residual
            x = self.norm(x)
            
        # Project to output space
        x = self.output_projection(x)
        return x
        
    def _reset_parameters(self):
        """Reset parameters to ensure reproducibility"""
        torch.manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        torch.manual_seed(torch.initial_seed())
