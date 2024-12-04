import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.optim import SGD
from tqdm import tqdm
import os
import re

class ChatModel(nn.Module):
    def __init__(self, vocab_size, n_positions=128, n_layer=6, n_head=8):
        super().__init__()
        configuration = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=512,  # Reduced embedding dimension
            pad_token_id=50256
        )
        print("Initializing model with config:", configuration)
        self.model = GPT2LMHeadModel(configuration)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate_response(self, input_text, tokenizer, max_length=128, device=None):
        if device is None:
            device = next(self.parameters()).device

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def train_model(model, train_dataloader, num_epochs=3, device=None):
    if device is None:
        device = torch.device('cpu')
    
    print(f"Training on device: {device}")
    
    # Use a simpler optimizer for better DirectML compatibility
    optimizer = SGD(model.parameters(), lr=0.01)
    
    best_loss = float('inf')
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / len(progress_bar)
            progress_bar.set_postfix({'loss': avg_loss})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(checkpoint_dir, f'chat_model_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'max_length': model.model.config.n_positions,
                        'vocab_size': model.model.config.vocab_size,
                        'n_layer': model.model.config.n_layer,
                        'n_head': model.model.config.n_head,
                        'device': str(device)
                    }
                }, checkpoint_path)
        
        print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
