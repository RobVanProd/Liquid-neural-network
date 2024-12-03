import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE

class NetworkVisualizer:
    """
    Comprehensive visualization tools for analyzing network behavior,
    training progress, and data patterns
    """
    def __init__(self, log_dir: str = 'runs'):
        self.writer = SummaryWriter(log_dir)
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'gradients': [],
            'activations': []
        }
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics to TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
            if name in self.training_history:
                self.training_history[name].append(value)
                
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training metrics over time"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss plot
        axes[0].plot(self.training_history['loss'], label='Training Loss')
        axes[0].set_title('Training Loss Over Time')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy plot
        if self.training_history['accuracy']:
            axes[1].plot(self.training_history['accuracy'], label='Accuracy')
            axes[1].set_title('Model Accuracy Over Time')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def visualize_activations(self, 
                            activations: torch.Tensor,
                            layer_name: str,
                            step: int):
        """Visualize layer activations"""
        if activations.dim() > 2:
            activations = activations.mean(dim=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=activations.detach().cpu().numpy(),
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f'Activation Pattern - {layer_name}',
            xaxis_title='Neuron Index',
            yaxis_title='Sample Index'
        )
        
        self.writer.add_figure(f'activations/{layer_name}', fig, step)
        
    def plot_gradient_flow(self, named_parameters: Dict[str, torch.Tensor]):
        """Visualize gradient flow through the network"""
        ave_grads = []
        layers = []
        
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                
        fig = go.Figure(data=go.Bar(
            x=layers,
            y=ave_grads,
            name='Average Gradient'
        ))
        
        fig.update_layout(
            title='Gradient Flow',
            xaxis_title='Layers',
            yaxis_title='Average Gradient',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def visualize_state_space(self,
                            hidden_states: torch.Tensor,
                            labels: Optional[torch.Tensor] = None):
        """Visualize hidden state representations using t-SNE"""
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        states_2d = tsne.fit_transform(hidden_states.detach().cpu().numpy())
        
        if labels is not None:
            fig = px.scatter(
                x=states_2d[:, 0],
                y=states_2d[:, 1],
                color=labels.cpu().numpy(),
                title='Hidden State Space Visualization'
            )
        else:
            fig = px.scatter(
                x=states_2d[:, 0],
                y=states_2d[:, 1],
                title='Hidden State Space Visualization'
            )
            
        fig.update_layout(
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2'
        )
        
        return fig
    
    def plot_attention_weights(self,
                             attention_weights: torch.Tensor,
                             save_path: Optional[str] = None):
        """Visualize attention weights"""
        weights = attention_weights.detach().cpu().numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=weights,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Attention Weights Visualization',
            xaxis_title='Key',
            yaxis_title='Query'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
