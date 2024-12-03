import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell that enables efficient continuous-time processing
    """
    def __init__(self, input_size: int, hidden_size: int, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Continuous-time dynamics parameters
        self.W_ih = nn.Linear(input_size, hidden_size * 3)  # input weights
        self.W_hh = nn.Linear(hidden_size, hidden_size * 3)  # hidden weights
        
        # Time-scale parameters
        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using best practices for continuous-time models"""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)
            
    def forward(self, input: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Input tensor of shape (batch, input_size)
            h: Hidden state tensor of shape (batch, hidden_size)
        Returns:
            h_new: New hidden state
            output: Output tensor
        """
        batch_size = input.size(0)
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=input.device)
            
        dt = torch.exp(self.log_dt)
        
        # Compute gates
        gates_x = self.W_ih(input)
        gates_h = self.W_hh(h)
        
        # Split gates
        i_x, f_x, g_x = gates_x.chunk(3, dim=1)
        i_h, f_h, g_h = gates_h.chunk(3, dim=1)
        
        # Combine input and hidden gates
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)
        
        # Continuous-time update
        dh = (-f_t * h + i_t * g_t) * dt
        h_new = h + dh
        
        return h_new, h_new

class CfCLayer(nn.Module):
    """
    A layer that processes sequences using CfC cells
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            h: Optional initial hidden state
        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size)
            h_n: Final hidden state
        """
        outputs = []
        steps = range(x.size(1))
        
        for t in steps:
            h, output = self.cell(x[:, t], h)
            outputs.append(output)
            
        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.norm(outputs)
        
        return outputs, h

class CfCModel(nn.Module):
    """
    Complete Closed-form Continuous-time model for sequence processing
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # CfC layers
        self.layers = nn.ModuleList([
            CfCLayer(hidden_size if i > 0 else hidden_size,
                    hidden_size,
                    dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        Returns:
            Output tensor of shape (batch, seq_len, output_size)
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x, _ = layer(x)
            
        return self.output_proj(x)
