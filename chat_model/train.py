from model import ChatModel, train_model
from data_loader import get_dataloader
import torch
import os
from datetime import datetime
import torch_directml

def get_device():
    """Get the best available device."""
    try:
        device = torch_directml.device()
        print(f"Successfully initialized DirectML device: {device}")
        return device
    except Exception as e:
        print(f"DirectML initialization failed: {e}")
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def main():
    # Initialize model and dataloader
    print("Initializing model and dataloader...")
    device = get_device()
    print(f"Using device: {device}")
    
    # Use larger batch size for GPU
    if str(device).startswith('privateuseone'):
        print("AMD GPU detected! Optimizing batch size...")
        batch_size = 32
    else:
        batch_size = 16
    
    dataloader, tokenizer = get_dataloader(batch_size=batch_size)
    
    # Initialize model
    model = ChatModel(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_layer=6,
        n_head=8
    )
    
    # Move model to device before creating optimizer
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Train model
    print("Starting training...")
    train_model(model, dataloader, num_epochs=10, device=device)
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join("checkpoints", f"chat_model_final_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_positions': 128,
            'batch_size': batch_size,
            'num_epochs': 10,
            'device': str(device)
        }
    }, final_path)
    print("Training complete!")

if __name__ == "__main__":
    main()
