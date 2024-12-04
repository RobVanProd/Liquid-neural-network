from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class OpenWebTextDataset(Dataset):
    def __init__(self, max_length=512, subset_size=1000):
        # Load a small subset of OpenWebText
        self.dataset = load_dataset("openwebtext", split=f"train[:{subset_size}]", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        
        # Tokenize all texts
        self.encoded_texts = []
        for item in tqdm(self.dataset, desc="Tokenizing texts"):
            encoded = self.tokenizer.encode(
                item['text'],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.encoded_texts.append(encoded.squeeze(0))
            
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        # Return input_ids and labels (shifted by 1 for next token prediction)
        tokens = self.encoded_texts[idx]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }

def get_dataloader(batch_size=16, max_length=512, subset_size=1000):
    dataset = OpenWebTextDataset(max_length=max_length, subset_size=subset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
