import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from liquid_neural_network import LiquidNeuralNetwork
import requests
from datetime import datetime, timedelta

class DatasetGenerator:
    @staticmethod
    def get_stock_data(symbol='AAPL', period='2y'):
        """Get stock market data"""
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df['Close'].values

    @staticmethod
    def get_weather_data():
        """Get weather data from a public API"""
        api_key = "demo_key"  # Replace with your API key for production
        city = "London"
        url = f"https://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt=2023-01-01"
        
        # For demo, generate synthetic weather data
        days = 730  # 2 years
        temp = np.sin(np.linspace(0, 8*np.pi, days)) * 10 + 20  # Base temperature
        noise = np.random.normal(0, 2, days)  # Daily variations
        seasonal = np.sin(np.linspace(0, 2*np.pi, days)) * 5  # Seasonal pattern
        return temp + noise + seasonal

    @staticmethod
    def generate_complex_pattern(samples=1000):
        """Generate a complex pattern with multiple frequencies and trends"""
        t = np.linspace(0, 10*np.pi, samples)
        # Combine multiple frequencies with non-linear terms
        pattern = (np.sin(t) + 
                  0.5 * np.sin(2*t) + 
                  0.3 * np.sin(0.5*t) + 
                  0.2 * t/10 +  # Linear trend
                  0.1 * (t/10)**2)  # Quadratic term
        return pattern

def prepare_sequence_data(data, sequence_length=20):
    """Prepare sequence data for training"""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length].reshape(-1))  # Flatten the sequence
        y.append(data_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    return (torch.FloatTensor(X), torch.FloatTensor(y), scaler)

class EnhancedLiquidNeuralNetwork(LiquidNeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, 
                 num_liquid_layers=2, dropout_rate=0.1):
        super().__init__(input_size, hidden_size, output_size)
        
        # Additional liquid layers
        self.liquid_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_liquid_layers-1)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # More aggressive self-improvement thresholds
        self.improvement_threshold = 1.2  # 20% performance degradation triggers improvement
        self.expansion_factor = 1.5  # 50% size increase on expansion
        
    def forward(self, x, steps=3):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size)
        
        for _ in range(steps):
            h_new = self.input_layer(x)
            for layer in self.liquid_layers:
                h_new = h_new + layer(h)
            h = self.liquid_activation(x, h_new)
            h = self.dropout(h)
        
        return self.output_layer(h)

def test_model(model_class, dataset_name, data, sequence_length=20, 
               hidden_size=50, epochs=100, lr=0.001):
    """Test model on a specific dataset"""
    print(f"\nTesting on {dataset_name} dataset:")
    
    # Prepare data
    X, y, scaler = prepare_sequence_data(data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train model
    model = model_class(
        input_size=sequence_length,
        hidden_size=hidden_size,
        output_size=1
    )
    
    # Training
    losses = model.train(X_train, y_train, epochs=epochs, lr=lr)
    
    # Plot training progress
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title(f'Training Loss - {dataset_name}')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        
    # Inverse transform predictions
    y_test_inv = scaler.inverse_transform(y_test.numpy())
    y_pred_inv = scaler.inverse_transform(y_pred.numpy())
    
    # Plot predictions
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_inv[:100], label='Actual')
    plt.plot(y_pred_inv[:100], label='Predicted')
    plt.title(f'Predictions vs Actual - {dataset_name}')
    plt.legend()
    plt.show()
    
    return model

def main():
    # Generate datasets
    data_generator = DatasetGenerator()
    
    datasets = {
        'Stock Market': data_generator.get_stock_data(),
        'Weather': data_generator.get_weather_data(),
        'Complex Pattern': data_generator.generate_complex_pattern()
    }
    
    # Test both original and enhanced models
    models = {
        'Original LNN': LiquidNeuralNetwork,
        'Enhanced LNN': EnhancedLiquidNeuralNetwork
    }
    
    results = {}
    
    for model_name, model_class in models.items():
        print(f"\nTesting {model_name}:")
        model_results = {}
        
        for dataset_name, data in datasets.items():
            model = test_model(
                model_class=model_class,
                dataset_name=f"{model_name} - {dataset_name}",
                data=data,
                hidden_size=50,
                epochs=100
            )
            model_results[dataset_name] = model
            
            # Test temporal dynamics
            sequence = torch.FloatTensor(data[:20]).unsqueeze(0)
            steps_range = [1, 3, 5, 10]
            
            plt.figure(figsize=(12, 4))
            for steps in steps_range:
                with torch.no_grad():
                    pred = model(sequence, steps=steps)
                plt.plot(pred[0].numpy(), label=f'Steps={steps}')
            plt.title(f'Temporal Dynamics - {model_name} on {dataset_name}')
            plt.legend()
            plt.show()
        
        results[model_name] = model_results

if __name__ == "__main__":
    main()
