import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell implementation.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights
        self.W_ih = nn.Linear(input_size, 3 * hidden_size, bias=True)
        # Hidden-to-hidden weights
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        # Layer normalization with fixed parameters
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with uniform distribution"""
        torch.manual_seed(42)  # For reproducibility
        for name, param in self.named_parameters():
            if 'weight' in name:
                bound = 1 / math.sqrt(param.size(1) if len(param.size()) > 1 else param.size(0))
                nn.init.uniform_(param, -bound, bound)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x, h=None):
        """
        Forward pass of CfC cell.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            h: Hidden state tensor of shape (batch, hidden_size)
            
        Returns:
            tuple: (new_h, output)
                - new_h: New hidden state
                - output: Cell output
        """
        batch_size = x.size(0)
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
        # Normalize inputs for stability
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute gates
        gates_x = self.W_ih(x)
        gates_h = self.W_hh(h)
        
        # Split gates
        i_x, f_x, g_x = gates_x.chunk(3, dim=1)
        i_h, f_h, g_h = gates_h.chunk(3, dim=1)
        
        # Compute gate values with careful normalization
        i_t = torch.sigmoid((i_x + i_h) / math.sqrt(2))
        f_t = torch.sigmoid((f_x + f_h) / math.sqrt(2))
        g_t = torch.tanh((g_x + g_h) / math.sqrt(2))
        
        # Update hidden state
        new_h = f_t * h + i_t * g_t
        
        # Apply layer normalization
        new_h = self.norm(new_h)
        
        return new_h, new_h

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
    Complete Closed-form Continuous-time model.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # CfC layers
        self.layers = nn.ModuleList([
            CfCCell(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Layer normalization with fixed parameters
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with uniform distribution"""
        torch.manual_seed(42)  # For reproducibility
        for name, param in self.named_parameters():
            if 'weight' in name:
                bound = 1 / math.sqrt(param.size(1) if len(param.size()) > 1 else param.size(0))
                nn.init.uniform_(param, -bound, bound)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x, hidden=None):
        """
        Forward pass of the CfC model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            tuple: (output, hidden_state)
                - output: Output tensor of shape (batch, seq_len, output_size)
                - hidden_state: Final hidden state
        """
        # Normalize input
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Project input
        x = self.input_proj(x)
        x = self.norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = [None] * self.num_layers
        elif not isinstance(hidden, list):
            hidden = [hidden] * self.num_layers
            
        outputs = []
        current_hidden = list(hidden)  # Create a copy to avoid modifying input
        
        # Process sequence
        for t in range(seq_len):
            layer_input = x[:, t]
            for i, layer in enumerate(self.layers):
                # Update hidden state
                new_h, layer_input = layer(layer_input, current_hidden[i])
                current_hidden[i] = new_h
            
            # Project to output space
            out = self.output_proj(layer_input)
            outputs.append(out)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)
        return outputs, current_hidden
