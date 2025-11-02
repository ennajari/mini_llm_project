# 📐 Formules Mathématiques - Référence Rapide

## Self-Attention

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Où:**
- Q (Query): matrice de requêtes (n × d_k)
- K (Key): matrice de clés (n × d_k)
- V (Value): matrice de valeurs (n × d_v)
- d_k: dimension des clés
- n: longueur de séquence

### Étapes de calcul:
1. **Scores**: `S = QK^T` → (n × n)
2. **Scaling**: `S_scaled = S / √d_k`
3. **Softmax**: `A = softmax(S_scaled)` → poids d'attention
4. **Output**: `O = AV` → (n × d_v)

### Pourquoi √d_k?
- Variance de QK^T ≈ d_k
- Division par √d_k normalise à variance ≈ 1
- Évite les valeurs extrêmes dans le softmax

---

## Multi-Head Attention

### Formule générale:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Paramètres:**
- h: nombre de têtes
- d_model: dimension du modèle
- d_k = d_v = d_model / h

**Matrices de projection:**
- W^Q_i ∈ ℝ^(d_model × d_k)
- W^K_i ∈ ℝ^(d_model × d_k)
- W^V_i ∈ ℝ^(d_model × d_v)
- W^O ∈ ℝ^(hd_v × d_model)

### Complexité:
- **Temps**: O(n² · d_model)
- **Espace**: O(n² + n · d_model)

---

## Positional Encoding

### Formule sinusoïdale:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Où:**
- pos: position dans la séquence (0 à n-1)
- i: dimension (0 à d_model/2 - 1)
- 2i: indices pairs
- 2i+1: indices impairs

### Propriétés:
1. **Unique**: Chaque position a un encodage distinct
2. **Borné**: Valeurs entre -1 et 1
3. **Distance relative**: PE(pos+k) est une fonction linéaire de PE(pos)

### Fréquences:
```
λ_i = 10000^(2i/d_model)
```
- Dimension basse (i=0): λ = 1, haute fréquence
- Dimension haute (i=d_model/2): λ = 10000, basse fréquence

---

## Layer Normalization

### Formule:
```
LayerNorm(x) = γ ⊙ (x - μ) / σ + β
```

**Où:**
- μ = mean(x): moyenne sur la dimension d_model
- σ = std(x): écart-type sur la dimension d_model
- γ: paramètres d'échelle (appris)
- β: paramètres de décalage (appris)
- ⊙: multiplication élément par élément

### Calcul:
```
μ = (1/d_model) Σ x_i
σ² = (1/d_model) Σ (x_i - μ)²
σ = √(σ² + ε)  # ε pour stabilité numérique
```

---

## Feed-Forward Network

### Formule:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Dimensions:**
- x: (batch, seq_len, d_model)
- W_1: (d_model, d_ff)
- W_2: (d_ff, d_model)
- d_ff typiquement = 4 × d_model

### Équivalent fonctionnel:
```
FFN(x) = ReLU(Linear_1(x))
output = Linear_2(FFN)
```

---

## Transformer Encoder Block

### Architecture complète:
```
Block(x):
  # Sub-layer 1: Multi-Head Attention
  x_1 = MultiHeadAttention(x, x, x)
  x = LayerNorm(x + Dropout(x_1))
  
  # Sub-layer 2: Feed-Forward
  x_2 = FFN(x)
  x = LayerNorm(x + Dropout(x_2))
  
  return x
```

### Avec notation mathématique:
```
x̂ = LayerNorm(x + MultiHeadAttention(x))
output = LayerNorm(x̂ + FFN(x̂))
```

---

## Masques d'Attention

### Padding Mask:
```
M_pad[i, j] = {
  -∞  si token_j est un PAD
  0   sinon
}
```

### Causal Mask (Look-ahead):
```
M_causal[i, j] = {
  -∞  si j > i  (futur)
  0   si j ≤ i  (passé/présent)
}
```

### Application:
```
Attention = softmax((QK^T / √d_k) + Mask)V
```

---

## Softmax

### Formule:
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

### Stable numériquement:
```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

**Propriétés:**
- Sortie entre 0 et 1
- Σ softmax(x_i) = 1
- Différentiable

---

## Complexité Totale

### Par couche Transformer:
- **Multi-Head Attention**: O(n² · d_model + n · d_model²)
- **Feed-Forward**: O(n · d_model · d_ff)
- **Total**: O(n² · d_model + n · d_model · d_ff)

### Pour L couches:
- **Temps**: O(L · (n² · d_model + n · d_model · d_ff))
- **Mémoire**: O(n · d_model + L · n²) pour stockage attention

### Avec paramètres typiques (BERT-base):
- L = 12
- d_model = 768
- d_ff = 3072
- Paramètres totaux ≈ 110M

---

## Fonctions d'Activation

### ReLU:
```
ReLU(x) = max(0, x)
```

### GELU (souvent utilisé):
```
GELU(x) ≈ x · Φ(x)
où Φ(x) = (1/2)[1 + erf(x/√2)]
```

### Approximation GELU:
```
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

---

## Initialisation des Poids

### Xavier/Glorot:
```
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
```

### He/Kaiming (pour ReLU):
```
W ~ N(0, √(2/n_in))
```

---

## Gradient Flow

### Residual Connection:
```
y = x + F(x)
∂y/∂x = 1 + ∂F(x)/∂x
```
- Le gradient peut toujours passer (terme 1)
- Évite le problème de vanishing gradient

### Layer Norm Gradient:
```
∂LayerNorm/∂x = γ/σ · (I - (1/d)(1·1^T + (x-μ)(x-μ)^T/σ²))
```

---

## Learning Rate Schedules

### Warmup puis Decay:
```
lr(t) = d_model^(-0.5) · min(t^(-0.5), t · warmup_steps^(-1.5))
```

### Cosine Annealing:
```
lr(t) = lr_min + (lr_max - lr_min) · (1 + cos(πt/T)) / 2
```

---

## Formules Utiles

### Nombre de Paramètres (Multi-Head Attention):
```
params_MHA = 4 · d_model² + 4 · d_model
          = 4d_model(d_model + 1)
```

### Nombre de Paramètres (FFN):
```
params_FFN = 2 · d_model · d_ff + d_model + d_ff
```

### Paramètres totaux par bloc:
```
params_block ≈ 12d_model² + 13d_model·d_ff
```

Pour d_model=512, d_ff=2048:
```
params_block ≈ 16.8M
```

---

**Note**: Ces formules sont simplifiées pour la clarté. Les implémentations réelles peuvent avoir des variations mineures.
