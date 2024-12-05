from model import ChatModel, train_model
from data_loader import get_dataloader
import torch
import os
from datetime import datetime
import torch_directml
from tqdm import tqdm

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

def train_model(model, train_dataloader, num_epochs=3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    model = model.to(device)
    model.train()
    
    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            try:
                # Format each conversation pair
                conversations = []
                for text in batch['text']:
                    # Split into Q&A pairs
                    pairs = text.split('\n')
                    formatted_text = ""
                    for i in range(0, len(pairs)-1, 2):
                        q = pairs[i].strip()
                        a = pairs[i+1].strip()
                        formatted_text += f"[HUMAN]: {q}\n[SHAKESPEARE]: {a}\n\n"
                    conversations.append(formatted_text)
                
                # Tokenize with padding
                inputs = model.tokenizer(
                    conversations,
                    padding=True,
                    truncation=True,
                    max_length=model.model.config.n_positions,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                continue
        
        # Calculate average loss for epoch
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': model.model.config.to_dict()
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return model

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
