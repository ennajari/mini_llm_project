"""
Script d'entraînement amélioré pour le modèle de classification de sentiment.
Inclut : validation set, métriques d'évaluation, learning rate scheduler, et meilleures pratiques.
"""
from configurator import Config
from tokenizer import CharTokenizer
from sentiment_dataset import SentimentDataset
from dataloader import create_dataloader
from optimizer.adamw import create_optimizer
from checkpoint_manager import save_checkpoint
from model.encoder import Encoder
import torch
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def evaluate_model(model, dataloader, device):
    """Évalue le modèle sur un dataset et retourne les métriques"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, attention_mask = batch
                attention_mask = attention_mask.to(device)
            else:
                x, y = batch
                attention_mask = None
            
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y, attention_mask=attention_mask)
            
            if loss is not None:
                total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    return accuracy, avg_loss, all_preds, all_labels


def train_model_improved(model, train_dl, val_dl, optimizer, scheduler, criterion, 
                        checkpoint_path, epochs, checkpoint_every, device):
    """Entraînement amélioré avec validation et métriques"""
    model = model.to(device)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accs = []
    
    print(f"\n{'='*60}")
    print(f"Début de l'entraînement - {epochs} époques")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_dl:
            if len(batch) == 3:
                x, y, attention_mask = batch
                attention_mask = attention_mask.to(device)
            else:
                x, y = batch
                attention_mask = None
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, loss = model(x, y, attention_mask=attention_mask)
            if loss is None:
                loss = criterion(logits, y)
            
            loss.backward()
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_loss /= len(train_dl)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Phase de validation
        val_acc, val_loss, val_preds, val_labels = evaluate_model(model, val_dl, device)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Affichage des métriques
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        if scheduler is not None:
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Sauvegarder le meilleur modèle (basé sur validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, path=checkpoint_path)
            print(f"  ✓ Meilleur modèle sauvegardé (Val Acc: {val_acc:.4f})")
        
        # Sauvegarde périodique
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, 
                          path=checkpoint_path.replace('.pt', f'_epoch_{epoch+1}.pt'))
        
        print()
    
    # Rapport final de classification
    print(f"\n{'='*60}")
    print("Rapport de classification final (validation)")
    print(f"{'='*60}")
    print(classification_report(val_labels, val_preds, 
                               target_names=['Negative', 'Neutral', 'Positive']))
    
    print(f"\nMatrice de confusion (validation):")
    print(confusion_matrix(val_labels, val_preds))
    
    print(f"\n{'='*60}")
    print(f"Entraînement terminé!")
    print(f"Meilleure validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Meilleure validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return train_losses, val_losses, val_accs


if __name__ == "__main__":
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Charger et préparer les données
    print('\n--- Préparation des données ---')
    text_sent = " ".join([l.split("\t",1)[1] for l in open("data/sentiment_data.txt", encoding="utf-8") if "\t" in l])
    tokenizer_sent = CharTokenizer(text_sent)
    print(f"Vocabulaire: {tokenizer_sent.vocab_size} caractères uniques")
    
    # Créer le dataset complet
    full_ds = SentimentDataset("data/sentiment_data.txt", tokenizer_sent, block_size=cfg.block_size)
    print(f"Total d'échantillons: {len(full_ds)}")
    
    # Split train/validation (80/20)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(42))
    
    print(f"Train: {len(train_ds)} échantillons")
    print(f"Validation: {len(val_ds)} échantillons")
    
    # Créer les dataloaders
    train_dl = create_dataloader(train_ds, cfg.batch_size)
    val_dl = create_dataloader(val_ds, cfg.batch_size)
    
    # Créer le modèle avec plus de couches pour de meilleures performances
    print('\n--- Création du modèle ---')
    model = Encoder(
        vocab_size=tokenizer_sent.vocab_size,
        embed_dim=cfg.embed_dim,
        num_classes=3,
        num_layers=cfg.num_layers,  # Utiliser le nombre de couches de la config
        num_heads=cfg.num_heads,    # Utiliser le nombre de têtes de la config
        ff_hidden_dim=cfg.ff_hidden_dim,
        dropout=cfg.dropout,
        max_len=cfg.block_size
    ).to(device)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    # Optimiseur avec learning rate plus bas pour classification
    optimizer = create_optimizer(model, lr=cfg.lr, weight_decay=getattr(cfg, 'weight_decay', 0.01))
    
    # Learning rate scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Entraînement
    print('\n--- Entraînement du modèle ---')
    train_losses, val_losses, val_accs = train_model_improved(
        model, train_dl, val_dl, optimizer, scheduler, criterion,
        "checkpoints/encoder_ckpt.pt", cfg.epochs, cfg.checkpoint_every, device
    )

