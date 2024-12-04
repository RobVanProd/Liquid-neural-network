from data_loader import get_dataloader
from model import ChatModel, train_model
import torch

def main():
    # Initialize dataloader with a small subset of data
    print("Loading data...")
    dataloader = get_dataloader(batch_size=8, max_length=128, subset_size=100)
    
    # Initialize model
    print("Initializing model...")
    model = ChatModel(n_positions=128)
    
    # Train model
    print("Starting training...")
    train_model(model, dataloader, num_epochs=3)
    
    # Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), 'chat_model.pth')
    print("Training complete!")

if __name__ == "__main__":
    main()
