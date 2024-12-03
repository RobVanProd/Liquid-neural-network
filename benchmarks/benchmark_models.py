import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from liquid_s4 import LiquidS4Model
from cfc_model import CfCModel
import matplotlib.pyplot as plt
import seaborn as sns

class ModelBenchmark:
    def __init__(self, save_dir: str = 'benchmark_results'):
        self.save_dir = save_dir
        self.results = []
        
    def generate_synthetic_dataset(self, 
                                 num_samples: int,
                                 seq_length: int,
                                 input_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic dataset for benchmarking"""
        # Generate sinusoidal data with different frequencies
        t = torch.linspace(0, 8*np.pi, seq_length).unsqueeze(0).unsqueeze(-1)
        frequencies = torch.randn(num_samples, 1, 1) * 0.5 + 1.0
        
        # Create input sequences
        x = torch.cat([
            torch.sin(frequencies * t),
            torch.sin(2 * frequencies * t),
            torch.cos(frequencies * t)
        ], dim=-1)
        
        if input_size > x.size(-1):
            x = torch.cat([x] * (input_size // x.size(-1) + 1), dim=-1)
            x = x[:, :, :input_size]
            
        # Generate targets (next-step prediction)
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1, :] = 0
        
        return x, y
    
    def benchmark_model(self,
                       model: torch.nn.Module,
                       dataloader: DataLoader,
                       model_name: str,
                       device: torch.device) -> Dict:
        """Benchmark a single model"""
        model = model.to(device)
        model.eval()
        
        # Metrics
        total_time = 0
        total_memory = 0
        batch_times = []
        
        # Warmup
        for _ in range(5):
            x, _ = next(iter(dataloader))
            x = x.to(device)
            with torch.no_grad():
                _ = model(x)
                
        # Actual benchmark
        start_time = time.time()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                batch_start = time.time()
                output = model(x)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Memory usage
                if device.type == 'cuda':
                    memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    total_memory = max(total_memory, memory)
                    
        total_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'std_batch_time': np.std(batch_times),
            'peak_memory_mb': total_memory,
            'device': device.type
        }
    
    def run_benchmarks(self,
                      batch_sizes: List[int] = [32, 64, 128],
                      seq_lengths: List[int] = [100, 500, 1000],
                      input_sizes: List[int] = [10, 50, 100],
                      hidden_size: int = 64,
                      num_samples: int = 1000):
        """Run comprehensive benchmarks"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                for input_size in input_sizes:
                    print(f"\nBenchmarking with batch_size={batch_size}, "
                          f"seq_length={seq_length}, input_size={input_size}")
                    
                    # Generate dataset
                    x, y = self.generate_synthetic_dataset(num_samples, seq_length, input_size)
                    dataset = TensorDataset(x, y)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    
                    # Initialize models
                    models = {
                        'Liquid-S4': LiquidS4Model(input_size, hidden_size, input_size),
                        'CfC': CfCModel(input_size, hidden_size, input_size)
                    }
                    
                    # Benchmark each model
                    for model_name, model in models.items():
                        result = self.benchmark_model(model, dataloader, model_name, device)
                        result.update({
                            'batch_size': batch_size,
                            'seq_length': seq_length,
                            'input_size': input_size
                        })
                        self.results.append(result)
                        
    def plot_results(self):
        """Generate visualization of benchmark results"""
        df = pd.DataFrame(self.results)
        
        # Plot execution time comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='model_name', y='avg_batch_time')
        plt.title('Model Execution Time Comparison')
        plt.ylabel('Average Batch Time (seconds)')
        plt.savefig(f'{self.save_dir}/execution_time.png')
        plt.close()
        
        # Plot memory usage
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='model_name', y='peak_memory_mb')
        plt.title('Peak Memory Usage Comparison')
        plt.ylabel('Memory Usage (MB)')
        plt.savefig(f'{self.save_dir}/memory_usage.png')
        plt.close()
        
        # Plot scaling with sequence length
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='seq_length', y='avg_batch_time', hue='model_name')
        plt.title('Execution Time vs Sequence Length')
        plt.xlabel('Sequence Length')
        plt.ylabel('Average Batch Time (seconds)')
        plt.savefig(f'{self.save_dir}/scaling_seq_length.png')
        plt.close()
        
        # Save raw results
        df.to_csv(f'{self.save_dir}/benchmark_results.csv', index=False)
        
def main():
    benchmark = ModelBenchmark()
    benchmark.run_benchmarks()
    benchmark.plot_results()
    
if __name__ == '__main__':
    main()
