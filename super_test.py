import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from super_liquid_network import SuperLiquidNetwork
from typing import Tuple, Dict, List
from datetime import datetime, timedelta
import torch.nn.functional as F

class ComplexDataGenerator:
    @staticmethod
    def generate_multidimensional_chaos(samples=1000, dimensions=3):
        """Generate high-dimensional chaotic system data"""
        def lorenz(state, s=10, r=28, b=2.667):
            x, y, z = state
            dx = s*(y - x)
            dy = r*x - y - x*z
            dz = x*y - b*z
            # Clip values to prevent overflow
            return np.clip(np.array([dx, dy, dz]), -1e3, 1e3)
        
        def rossler(state, a=0.2, b=0.2, c=5.7):
            x, y, z = state
            dx = -y - z
            dy = x + a*y
            dz = b + z*(x - c)
            # Clip values to prevent overflow
            return np.clip(np.array([dx, dy, dz]), -1e3, 1e3)
        
        # Initialize states with smaller values
        dt = 0.001  # Smaller time step
        states = np.random.randn(dimensions, samples) * 0.1
        
        # Generate trajectories
        for i in range(1, samples):
            if i % 2 == 0:
                derivative = lorenz(states[:3, i-1])
            else:
                derivative = rossler(states[:3, i-1])
            
            states[:3, i] = states[:3, i-1] + derivative * dt
            
            # Add coupled oscillators for higher dimensions with controlled frequency
            if dimensions > 3:
                for d in range(3, dimensions):
                    freq = (d-2) * 0.1  # Reduced frequency
                    states[d, i] = np.sin(states[0, i] * freq)
            
            # Clip all values to prevent overflow
            states = np.clip(states, -1e3, 1e3)
        
        return states.T

    @staticmethod
    def get_market_data(symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
                       period='2y', interval='1d'):
        """Get comprehensive market data"""
        data = []
        features = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period, interval=interval)
                
                if len(hist) == 0:
                    print(f"Warning: No data found for {symbol}, skipping...")
                    continue
                
                # Calculate technical indicators
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['RSI'] = ComplexDataGenerator._calculate_rsi(hist['Close'])
                hist['MACD'] = ComplexDataGenerator._calculate_macd(hist['Close'])
                
                # Handle NaN values
                hist = hist.fillna(method='ffill').fillna(method='bfill')
                
                # Select features
                selected_features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
                data.append(hist[selected_features])
                features.extend([f"{symbol}_{feat}" for feat in selected_features])
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not data:
            raise ValueError("No valid market data found")
        
        # Combine all stock data
        combined_data = pd.concat(data, axis=1)
        combined_data.columns = features
        
        # Handle any remaining NaN or infinite values
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        return combined_data

    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2

def prepare_complex_data(data: np.ndarray, 
                        sequence_length: int = 50,
                        prediction_horizon: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare complex sequential data with multi-step prediction"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])  # Single step prediction instead of sequence
    
    # Convert lists to numpy arrays first
    X = np.array(X)
    y = np.array(y)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_and_evaluate(model: SuperLiquidNetwork,
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      test_data: Tuple[torch.Tensor, torch.Tensor],
                      epochs: int = 200,
                      batch_size: int = 32,
                      learning_rate: float = 0.001) -> Dict:
    """Train and evaluate the model with advanced monitoring"""
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,       # Increased restart period
        T_mult=2,     # Double the restart interval after each restart
        eta_min=1e-7  # Lower minimum learning rate
    )
    
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    patience = 20     # Increased patience
    patience_counter = 0
    min_epochs = 50   # Minimum number of epochs before early stopping
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # Mini-batch training with gradient accumulation
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass with dropout
            output = model(batch_X)
            
            # Calculate loss with L2 regularization
            l2_lambda = 0.001  # Reduced L2 regularization
            l2_reg = torch.tensor(0., device=output.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            
            loss = F.mse_loss(output, batch_y) + l2_lambda * l2_reg
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced gradient clipping
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Step the scheduler
        scheduler.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = F.mse_loss(test_output, y_test)
            test_losses.append(test_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # Early stopping check (only after min_epochs)
        if epoch >= min_epochs:
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        if epoch % 5 == 0:  # Print every 5 epochs
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Test Loss = {test_loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': best_loss,
        'stopped_epoch': epoch
    }

def generate_mackey_glass(n_points=1000, tau=17, beta=0.2, gamma=0.1, n=10, seed=42):
    """Generate Mackey-Glass chaotic time series"""
    np.random.seed(seed)
    x = np.zeros(n_points)
    x[0] = 1.2
    
    for i in range(1, n_points):
        if i <= tau:
            x[i] = x[i-1]
        else:
            x[i] = x[i-1] + beta * x[i-tau] / (1 + x[i-tau]**n) - gamma * x[i-1]
    
    return x

def generate_complex_datasets():
    """Generate multiple complex datasets for testing"""
    datasets = {}
    
    # 1. Chaotic Systems
    # Lorenz system
    lorenz = ComplexDataGenerator.generate_multidimensional_chaos(samples=2000, dimensions=3)
    datasets['lorenz'] = lorenz
    
    # Rössler system
    rossler = ComplexDataGenerator.generate_multidimensional_chaos(samples=2000, dimensions=3)
    datasets['rossler'] = rossler
    
    # Mackey-Glass system
    mackey = generate_mackey_glass(n_points=2000)
    datasets['mackey'] = mackey
    
    # 2. Financial Data
    try:
        # Multiple timeframes and indicators
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
        market_data = ComplexDataGenerator.get_market_data(symbols, period='2y', interval='1d')
        datasets['market'] = market_data
        
        # Crypto data
        crypto_symbols = ['BTC-USD', 'ETH-USD']
        crypto_data = ComplexDataGenerator.get_market_data(crypto_symbols, period='1y', interval='1h')
        datasets['crypto'] = crypto_data
    except Exception as e:
        print(f"Warning: Could not fetch some market data: {e}")
    
    # 3. Synthetic Complex Patterns
    # Composite sinusoidal
    t = np.linspace(0, 100, 2000)
    composite = np.sin(0.1*t) + 0.5*np.sin(0.5*t) + 0.2*np.sin(2*t)
    datasets['composite'] = composite
    
    # Nonlinear AR process
    ar = np.zeros(2000)
    ar[0:2] = np.random.randn(2)
    for i in range(2, 2000):
        ar[i] = 0.8*ar[i-1] - 0.5*ar[i-2] + 0.2*np.sin(ar[i-1]) + 0.1*np.random.randn()
    datasets['nonlinear_ar'] = ar
    
    return datasets

def get_hyperparameter_configs():
    """Generate different hyperparameter configurations for testing"""
    configs = [
        {
            'hidden_size': 128,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.15,
            'attention_dropout': 0.1,
            'hidden_dropout': 0.2,
            'learning_rate': 0.0001,  # Reduced learning rate
            'weight_decay': 0.001,    # Reduced weight decay
            'batch_size': 32,
            'name': 'large_model'
        },
        {
            'hidden_size': 64,
            'num_layers': 4,
            'num_heads': 4,
            'dropout': 0.1,
            'attention_dropout': 0.05,
            'hidden_dropout': 0.15,
            'learning_rate': 0.0002,  # Reduced learning rate
            'weight_decay': 0.0005,   # Reduced weight decay
            'batch_size': 16,
            'name': 'medium_model'
        },
        {
            'hidden_size': 32,
            'num_layers': 3,
            'num_heads': 2,
            'dropout': 0.05,
            'attention_dropout': 0.03,
            'hidden_dropout': 0.1,
            'learning_rate': 0.0003,  # Reduced learning rate
            'weight_decay': 0.0001,   # Reduced weight decay
            'batch_size': 8,
            'name': 'small_model'
        }
    ]
    return configs

def generate_sine_data(n_points: int) -> np.ndarray:
    """Generate synthetic sine wave data"""
    t = np.linspace(0, 20*np.pi, n_points)
    # Combine multiple frequencies
    y = np.sin(t) + 0.5*np.sin(2*t) + 0.3*np.sin(3*t)
    # Add some noise
    y += np.random.normal(0, 0.1, n_points)
    return y.reshape(-1, 1)

def generate_lorenz_data(n_points: int) -> np.ndarray:
    """Generate Lorenz attractor data"""
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    dt = 0.01
    num_steps = n_points
    
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
    
    # Remove the last point because it's only used for the derivative
    data = np.column_stack((xs[:-1], ys[:-1], zs[:-1]))
    return data

def prepare_complex_data(data: np.ndarray, sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for sequence prediction"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)

def fetch_market_data(symbols: List[str], period: str = '6mo') -> np.ndarray:
    """Fetch and prepare market data"""
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        import yfinance as yf
    
    data = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            # Use basic features: Open, High, Low, Close, Volume
            features = hist[['Open', 'High', 'Low', 'Close', 'Volume']].values
            data.append(features)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    
    if not data:
        # Generate synthetic data if fetching fails
        print("Falling back to synthetic data...")
        return generate_sine_data(2000)
    
    # Combine data from all symbols
    combined_data = np.concatenate(data, axis=1)
    return combined_data

def main():
    print("Starting training with multiple configurations...")
    
    # Get datasets and configs
    print("\nGenerating datasets...")
    datasets = {
        'sine': generate_sine_data(2000),
        'lorenz': generate_lorenz_data(2000),
        'market': fetch_market_data(['AAPL', 'MSFT'], period='6mo')
    }
    
    configs = [
        {
            'hidden_size': 64,
            'num_layers': 4,
            'num_heads': 4,
            'dropout': 0.1,
            'learning_rate': 0.0002,
            'batch_size': 16,
            'name': 'medium_config'
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        model_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"\nProcessing {dataset_name} dataset...")
            
            # Prepare data
            if isinstance(data, tuple):
                X, y = data
            else:
                X, y = prepare_complex_data(data, sequence_length=20)
            
            # Scale data
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            y = scaler.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
            
            # Convert to tensors
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]
            
            print(f"Training data shape: {X_train.shape}")
            
            # Create model
            input_size = X_train.shape[-1]
            output_size = y_train.shape[-1] if len(y_train.shape) > 1 else 1
            
            model = SuperLiquidNetwork(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                output_size=output_size,
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )
            
            # Train and evaluate
            print("\nTraining model...")
            train_results = train_and_evaluate(
                model=model,
                train_data=(X_train, y_train),
                test_data=(X_test, y_test),
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                epochs=50  # Reduced epochs for testing
            )
            
            model_results[dataset_name] = train_results
            
            # Visualize final predictions
            print("\nGenerating visualizations...")
            with torch.no_grad():
                test_pred = model(X_test[:100])
                plt.figure(figsize=(12, 6))
                plt.plot(y_test[:100, 0].numpy(), label='True')
                plt.plot(test_pred[:, 0].numpy(), label='Predicted')
                plt.title(f'{dataset_name} - Predictions vs True Values')
                plt.legend()
                plt.show()
                
                # Plot training history
                plt.figure(figsize=(12, 6))
                plt.plot(train_results['train_losses'], label='Train Loss')
                plt.plot(train_results['test_losses'], label='Test Loss')
                plt.title(f'{dataset_name} - Training History')
                plt.yscale('log')
                plt.legend()
                plt.show()
                
                plt.close('all')  # Clean up
            
        results[config['name']] = model_results
    
    # Print summary
    print("\nTraining Summary:")
    for config_name, model_results in results.items():
        print(f"\n{config_name}:")
        for dataset_name, metrics in model_results.items():
            print(f"  {dataset_name}:")
            print(f"    Final Train Loss: {metrics['final_train_loss']:.6f}")
            print(f"    Final Test Loss: {metrics['final_test_loss']:.6f}")
            print(f"    Best Test Loss: {metrics['best_test_loss']:.6f}")
            print(f"    Epochs Run: {metrics['stopped_epoch']}")

if __name__ == "__main__":
    main()
