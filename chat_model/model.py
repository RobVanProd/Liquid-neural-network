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
            pad_token_id=50256,
            bos_token_id=50256,
            eos_token_id=50256,
        )
        print("Initializing model with config:", configuration)
        self.model = GPT2LMHeadModel(configuration)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print model size information
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model size: {total_params:,} parameters")
        print(f"Position embedding size: {n_positions}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate_text(self, input_text, max_length=None):
        """Generate text in Shakespeare's style."""
        try:
            if max_length is None:
                max_length = self.model.config.n_positions
            
            # Shakespearean prompts for different types of inputs
            greetings = ["how are you", "hello", "hi ", "hey", "greetings"]
            questions = ["what", "why", "how", "where", "when"]
            
            # Select appropriate context based on input
            if any(g in input_text.lower() for g in greetings):
                context = (
                    "A gracious greeting deserves a courteous reply. "
                    "The Bard shall respond with warmth and eloquence.\n\n"
                )
            elif any(q in input_text.lower() for q in questions):
                context = (
                    "A thoughtful question deserves a wise answer. "
                    "The Bard shall share knowledge with poetic grace.\n\n"
                )
            else:
                context = (
                    "The Bard shall respond with the wisdom of ages, "
                    "weaving words with artistic flourish.\n\n"
                )
            
            # Format prompt
            prompt = f"{context}[HUMAN]: {input_text}\n[SHAKESPEARE]:"
            
            # Tokenize with careful length handling
            max_input_length = max_length - 100  # Reserve tokens for generation
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
            
            # Generate with carefully tuned parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=20,  # Ensure some minimum response length
                    num_return_sequences=1,
                    no_repeat_ngram_size=4,
                    do_sample=True,
                    top_k=30,
                    top_p=0.85,
                    temperature=0.9,
                    repetition_penalty=1.3,
                    length_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._clean_response(response, context)
            
            # Add Shakespearean flair for short responses
            if len(response.split()) < 10:
                if any(g in input_text.lower() for g in greetings):
                    response = "Hark! " + response + " Prithee, speak thy mind!"
                elif any(q in input_text.lower() for q in questions):
                    response = "Forsooth! " + response + " Let us explore this matter further!"
                else:
                    response = "Verily, " + response + " What say you to that?"
            
            return response
            
        except Exception as e:
            print(f"Error in generate_text: {str(e)}")
            return "Alas, my words fail me. Pray, let us begin anew."

    def _clean_response(self, text, context):
        """Clean up the generated response."""
        # Remove context and prompt markers
        text = text.replace(context, "").strip()
        text = text.replace("[HUMAN]:", "").replace("[SHAKESPEARE]:", "").strip()
        
        # Basic cleanup
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)  # Remove repeated punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Fix spaces before punctuation
        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)  # Add space after punctuation
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        cleaned_sentences = []
        
        for i in range(0, len(sentences)-1, 2):
            sentence = (sentences[i] + sentences[i+1]).strip()
            if sentence and not any(marker in sentence.lower() for marker in ["the bard", "shall", "response shall"]):
                cleaned_sentences.append(sentence)
        
        # Join sentences and capitalize
        text = ' '.join(cleaned_sentences)
        text = text[0].upper() + text[1:] if text else text
        
        return text.strip()

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
