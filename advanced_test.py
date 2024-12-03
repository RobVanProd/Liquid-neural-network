import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from advanced_liquid_network import AdvancedLiquidNeuralNetwork
from typing import Tuple, Dict
import requests
from datetime import datetime, timedelta

class AdvancedDatasetGenerator:
    @staticmethod
    def get_crypto_data(symbol='BTC-USD', period='2y'):
        """Get cryptocurrency data"""
        crypto = yf.Ticker(symbol)
        df = crypto.history(period=period)
        return df[['Close', 'Volume', 'High', 'Low']].values

    @staticmethod
    def get_multi_stock_data(symbols=['AAPL', 'GOOGL', 'MSFT'], period='2y'):
        """Get multiple stock data for correlation learning"""
        data = []
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            data.append(df['Close'].values)
        return np.column_stack(data)

    @staticmethod
    def generate_chaotic_system(samples=1000):
        """Generate Lorenz attractor data"""
        def lorenz(x, y, z, s=10, r=28, b=2.667):
            x_dot = s*(y - x)
            y_dot = r*x - y - x*z
            z_dot = x*y - b*z
            return x_dot, y_dot, z_dot

        dt = 0.01
        num_steps = samples
        
        # Need one more for the initial values
        xs = np.empty(num_steps + 1)
        ys = np.empty(num_steps + 1)
        zs = np.empty(num_steps + 1)
        
        # Set initial values
        xs[0], ys[0], zs[0] = (0., 1., 1.05)
        
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
            
        return np.column_stack((xs[:-1], ys[:-1], zs[:-1]))

def prepare_sequence_data(data: np.ndarray, 
                         sequence_length: int = 20) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Prepare multivariate sequence data for training"""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to (samples, sequence_length * features)
    X = X.reshape(X.shape[0], -1)
    
    return torch.FloatTensor(X), torch.FloatTensor(y), scaler

def test_model_configurations(dataset_name: str, 
                            X: torch.Tensor, 
                            y: torch.Tensor,
                            configs: Dict) -> Dict:
    """Test different model configurations"""
    results = {}
    
    for config_name, params in configs.items():
        print(f"\nTesting {config_name} on {dataset_name}")
        
        model = AdvancedLiquidNeuralNetwork(
            input_size=X.shape[1],
            output_size=y.shape[1],
            **params
        )
        
        # Train model
        losses = model.train_model(X, y, epochs=150)
        
        # Record results
        results[config_name] = {
            'model': model,
            'final_loss': losses[-1],
            'loss_history': losses
        }
        
        # Visualize network dynamics
        model.visualize_dynamics()
        
        # Plot training progress
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.title(f'Training Loss - {config_name} on {dataset_name}')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    return results

def main():
    # Generate datasets
    data_generator = AdvancedDatasetGenerator()
    
    datasets = {
        'Crypto': data_generator.get_crypto_data(),
        'Multi-Stock': data_generator.get_multi_stock_data(),
        'Chaotic': data_generator.generate_chaotic_system()
    }
    
    # Define model configurations to test
    configs = {
        'Base': {
            'hidden_size': 64,
            'num_liquid_layers': 3,
            'use_attention': False,
            'use_residual': False
        },
        'Attention': {
            'hidden_size': 64,
            'num_liquid_layers': 3,
            'use_attention': True,
            'use_residual': False
        },
        'Residual': {
            'hidden_size': 64,
            'num_liquid_layers': 3,
            'use_attention': False,
            'use_residual': True
        },
        'Full': {
            'hidden_size': 64,
            'num_liquid_layers': 3,
            'use_attention': True,
            'use_residual': True
        }
    }
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, data in datasets.items():
        print(f"\nProcessing {dataset_name} dataset:")
        X, y, scaler = prepare_sequence_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Test different configurations
        results = test_model_configurations(dataset_name, X_train, y_train, configs)
        all_results[dataset_name] = results
        
        # Compare predictions
        plt.figure(figsize=(15, 5))
        plt.plot(scaler.inverse_transform(y_test[:100].numpy()), label='Actual')
        
        for config_name, result in results.items():
            model = result['model']
            model.eval()
            with torch.no_grad():
                predictions = model(X_test[:100])
                predictions = scaler.inverse_transform(predictions.numpy())
                plt.plot(predictions, label=f'{config_name} Predicted', alpha=0.7)
        
        plt.title(f'Predictions Comparison - {dataset_name}')
        plt.legend()
        plt.show()
    
    # Print final comparison
    print("\nFinal Results:")
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name} Dataset:")
        for config_name, result in results.items():
            print(f"{config_name}: Final Loss = {result['final_loss']:.6f}")

if __name__ == "__main__":
    main()
