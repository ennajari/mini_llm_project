import torch
from torch.utils.data import Dataset

LABEL2IDX = {
    "negative": 0,
    "negatif": 0,
    "neutral": 1,
    "neutre": 1,
    "positive": 2,
    "positif": 2,
}

class SentimentDataset(Dataset):
    def __init__(self, path, tokenizer, block_size=32):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                label, text = line.strip().split("\t", 1)
                label = label.strip().lower()
                if label not in LABEL2IDX:
                    continue
                self.samples.append((LABEL2IDX[label], text))
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        ids = self.tokenizer.encode(text)
        original_len = len(ids)
        ids = ids[:self.block_size]  # tronque si trop long
        
        # padding right
        if len(ids) < self.block_size:
            ids = ids + [0]*(self.block_size - len(ids))
        
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        
        # Créer un masque d'attention (1 pour tokens réels, 0 pour padding)
        attention_mask = torch.zeros(self.block_size, dtype=torch.long)
        attention_mask[:original_len] = 1
        
        return x, y, attention_mask

