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
    def __init__(self, d_model: int, N: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.N = N
        self.d_model = d_model

        # Learn continuous-time parameters
        self.Lambda = nn.Parameter(torch.randn(N, dtype=torch.float32))
        self.P = nn.Parameter(torch.randn(d_model, N, dtype=torch.float32) / np.sqrt(N))
        self.Q = nn.Parameter(torch.randn(N, d_model, dtype=torch.float32) / np.sqrt(N))
        self.B = nn.Parameter(torch.randn(N, dtype=torch.float32))

        # Time step parameters
        log_dt = torch.rand(1) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (batch, length, d_model)
        Returns:
            Output tensor of shape (batch, length, d_model)
        """
        batch_size = u.size(0)
        seq_len = u.size(1)
        
        # Initialize state
        x = torch.zeros(batch_size, self.N, device=u.device)
        outputs = []

        dt = torch.exp(self.log_dt)
        L = -torch.exp(self.Lambda)  # ensure eigenvalues have negative real part
        
        # Discretize continuous-time system
        A = torch.diag(torch.exp(dt * L))  # [N, N]
        B = dt * self.B  # [N]
        
        # Process sequence
        for t in range(seq_len):
            # Update state: x = Ax + Bu
            x = torch.matmul(x, A.T) + B.unsqueeze(0) * u[:, t]  # [batch, N]
            
            # Compute output: y = Px
            y = torch.matmul(x, self.Q)  # [batch, d_model]
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]

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
    """
    Complete Liquid S4 model for sequence modeling tasks.
    """
    def __init__(self, 
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            LiquidS4Layer(d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, d_output)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length, d_input)
        Returns:
            Output tensor of shape (batch, length, d_output)
        """
        x = self.input_projection(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.output_projection(x)
