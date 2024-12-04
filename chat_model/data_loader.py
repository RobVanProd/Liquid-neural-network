import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import requests
import os
import re

class ShakespeareDataset(Dataset):
    def __init__(self, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Download TinyShakespeare dataset if not exists
        if not os.path.exists('shakespeare.txt'):
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            r = requests.get(url)
            with open('shakespeare.txt', 'w', encoding='utf-8') as f:
                f.write(r.text)
        
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Preprocess text to create dialogue format
        print("Preprocessing text...")
        dialogues = self._create_dialogues(text)
        
        # Tokenize dialogues
        print("Tokenizing dataset...")
        self.examples = []
        
        for dialogue in dialogues:
            # Tokenize with padding
            encoded = self.tokenizer(
                dialogue,
                truncation=True,
                max_length=self.block_size,
                padding='max_length',
                return_tensors='pt'
            )
            
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })
        
        print(f"Created {len(self.examples)} training examples")
    
    def _create_dialogues(self, text):
        """Create structured dialogues from the text."""
        # Split into scenes
        scenes = re.split(r'Scene|ACT', text)
        
        dialogues = []
        for scene in scenes:
            # Extract character dialogues
            lines = scene.split('\n')
            current_dialogue = []
            
            for i, line in enumerate(lines):
                # Look for character names (in caps) followed by dialogue
                if re.match(r'^[A-Z]{2,}\.?', line.strip()):
                    character = line.strip()
                    # Get next lines until next character or empty line
                    dialogue = []
                    j = i + 1
                    while j < len(lines) and not re.match(r'^[A-Z]{2,}\.?', lines[j].strip()) and lines[j].strip():
                        dialogue.append(lines[j].strip())
                        j += 1
                    
                    if dialogue:
                        # Format as prompt-response pair
                        dialogue_text = ' '.join(dialogue)
                        current_dialogue.append(f"[{character}]: {dialogue_text}")
                
                if len(current_dialogue) >= 2:
                    # Create dialogue pairs
                    dialogue_text = '\n'.join(current_dialogue)
                    dialogues.append(dialogue_text)
                    current_dialogue = current_dialogue[1:]  # Overlap dialogues for context
        
        # Add some common dialogue templates
        templates = [
            "[HAMLET]: What is love?\n[HORATIO]: A most divine and precious feeling, my lord, that doth pierce the heart with sweetness.",
            "[ROMEO]: How fare thee?\n[BENVOLIO]: Most well, dear cousin, though the day grows weary.",
            "[PORTIA]: What think you of justice?\n[SHYLOCK]: 'Tis but a word, unless backed by deed and consequence.",
            "[OTHELLO]: Tell me of honor.\n[IAGO]: A jewel most precious, my lord, yet oft misused by those who claim it.",
        ]
        dialogues.extend(templates)
        
        return dialogues
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return {
            'input_ids': self.examples[i]['input_ids'],
            'attention_mask': self.examples[i]['attention_mask'],
            'labels': self.examples[i]['input_ids'].clone()
        }

def get_dataloader(batch_size=16):
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Creating dataset...")
    dataset = ShakespeareDataset(tokenizer)
    
    print("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return dataloader, tokenizer
