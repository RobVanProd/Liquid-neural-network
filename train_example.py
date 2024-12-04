import torch
import sys
sys.path.append('.')  # Add current directory to Python path
from liquid_neural_network import LiquidNeuralNetwork
import matplotlib.pyplot as plt

# Create a more interesting synthetic dataset (a simple sine wave with noise)
X = torch.linspace(0, 10, 1000).reshape(-1, 1)
y = torch.sin(X) + torch.randn_like(X) * 0.1

# Create the model
model = LiquidNeuralNetwork(input_size=1, hidden_size=32, output_size=1)

# Train the model
print("Training the model...")
losses = model.train(X, y, epochs=500, lr=0.01)

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)

# Plot the predictions
model.eval()
with torch.no_grad():
    predictions = model(X)

plt.figure(figsize=(10, 5))
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='True Data')
plt.plot(X.numpy(), predictions.numpy(), 'r-', label='Predictions')
plt.title('Liquid Neural Network Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
