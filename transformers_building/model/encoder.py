# model/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward
from .positional_encoding import PositionalEncoding
from .normalization import LayerNorm


class TransformerEncoderBlock(nn.Module):
    """Bloc encoder transformer avec self-attention"""
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN architecture
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class Encoder(nn.Module):
    """
    Encoder Transformer pour classification de sentiment.
    Utilise les composants transformer existants.
    """
    def __init__(self, vocab_size=128, embed_dim=128, num_classes=3, 
                 num_layers=4, num_heads=4, ff_hidden_dim=512, 
                 dropout=0.1, max_len=512, hidden_dim=None, block_size=None):
        super().__init__()
        
        # Compatibilité avec les anciens paramètres
        if hidden_dim is not None:
            ff_hidden_dim = hidden_dim
        if block_size is not None:
            max_len = block_size
        
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Stack de blocs transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_ln = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Attention pooling pour meilleure représentation
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim // 2, num_classes)
        )
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, y=None, attention_mask=None):
        """
        Args:
            x: (batch, seq_len) indices de tokens
            y: (batch,) labels pour classification (optionnel)
            attention_mask: (batch, seq_len) masque pour ignorer le padding (optionnel)
        Returns:
            logits: (batch, num_classes)
            loss: scalaire si y fourni, sinon None
        """
        # Embeddings + encodage positionnel
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = self.pos_encoding(emb)
        emb = self.dropout(emb)
        
        # Passer par les blocs transformer
        for block in self.blocks:
            emb = block(emb)
        
        emb = self.final_ln(emb)
        
        # Attention pooling (plus efficace que mean pooling)
        # Calculer les scores d'attention
        attn_scores = self.attention_pool(emb)  # (batch, seq_len, 1)
        
        # Appliquer le masque si fourni (ignorer les tokens de padding)
        if attention_mask is not None:
            # attention_mask: 1 pour tokens réels, 0 pour padding
            attn_mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax pour obtenir les poids d'attention
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        
        # Pooling pondéré par attention
        pooled = (emb * attn_weights).sum(dim=1)  # (batch, embed_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits, y)
        
        return logits, loss