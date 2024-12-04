import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from torch.optim import AdamW
from tqdm import tqdm

class ChatModel(nn.Module):
    def __init__(self, vocab_size=50257, n_positions=512, n_layer=6, n_head=8):
        super().__init__()
        # Using a smaller GPT2 configuration for faster training
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_head * 64  # Smaller embedding dimension
        )
        self.model = GPT2LMHeadModel(config)
        
    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)

def train_model(model, train_dataloader, num_epochs=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / len(progress_bar)})
