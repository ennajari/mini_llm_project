import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Gérer le cas avec ou sans attention_mask (rétrocompatibilité)
        if len(batch) == 3:
            x, y, attention_mask = batch
            attention_mask = attention_mask.to(device)
        else:
            x, y = batch
            attention_mask = None
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Déballer la sortie du modèle (avec ou sans attention_mask)
        if attention_mask is not None and hasattr(model, 'forward'):
            # Vérifier si le modèle accepte attention_mask
            try:
                logits, loss = model(x, y, attention_mask=attention_mask)
            except TypeError:
                # Modèle qui n'accepte pas attention_mask
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        
        # GPT renvoie déjà une loss calculée, sinon on la calcule manuellement
        if loss is None:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
