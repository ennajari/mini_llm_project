# Guide d'entraînement amélioré pour le modèle de sentiment

## Améliorations apportées

Ce guide décrit les améliorations apportées pour obtenir de meilleurs résultats lors de l'entraînement du modèle de classification de sentiment.

### 1. Validation Set
- **Avant** : Entraînement sur toutes les données sans validation
- **Maintenant** : Split 80/20 train/validation pour évaluer les performances en temps réel

### 2. Métriques d'évaluation
- **Accuracy** : Pourcentage de prédictions correctes
- **Classification Report** : Précision, rappel, F1-score par classe
- **Confusion Matrix** : Matrice de confusion pour analyser les erreurs

### 3. Architecture améliorée
- **Attention Pooling** : Remplace le simple mean pooling par un pooling attentionné qui se concentre sur les tokens importants
- **Plus de couches** : Utilise le nombre de couches configuré (num_layers) au lieu de seulement 2
- **Gestion du padding** : Masque d'attention pour ignorer les tokens de padding

### 4. Optimisations d'entraînement
- **Learning Rate Scheduler** : Réduit automatiquement le learning rate si la validation loss stagne (ReduceLROnPlateau)
- **Gradient Clipping** : Limite la norme des gradients pour éviter l'explosion des gradients
- **Meilleur suivi** : Affichage des métriques train et validation à chaque époque

## Utilisation

### Option 1 : Script amélioré (recommandé)

```bash
python train_sentiment_improved.py
```

Ce script inclut toutes les améliorations :
- Split train/validation automatique
- Métriques d'évaluation complètes
- Learning rate scheduler
- Gradient clipping
- Rapports détaillés

### Option 2 : Script original (basique)

```bash
python train.py
```

Le script original entraîne les 3 modèles (GPT, Encoder, TranslationModel) mais sans validation ni métriques détaillées.

## Interprétation des résultats

### Métriques affichées

1. **Train Loss / Val Loss** : La loss d'entraînement doit diminuer. La validation loss doit suivre la même tendance (sinon overfitting).

2. **Train Acc / Val Acc** : L'accuracy doit augmenter. La différence entre train et validation ne doit pas être trop grande (max 5-10%).

3. **Classification Report** : 
   - **Precision** : Parmi les prédictions positives, combien sont vraiment positives ?
   - **Recall** : Parmi les vrais positifs, combien sont détectés ?
   - **F1-score** : Moyenne harmonique de precision et recall

4. **Confusion Matrix** : Montre quelles classes sont confondues entre elles.

### Signes de bon entraînement

- Validation accuracy > 70% (bon)
- Validation accuracy > 80% (très bon)
- Train et validation loss diminuent ensemble
- Pas de grande différence entre train et validation accuracy (< 10%)

### Signes de problèmes

- **Overfitting** : Train accuracy >> Validation accuracy (> 15% de différence)
  - Solution : Augmenter dropout, réduire le nombre de couches, augmenter weight_decay

- **Underfitting** : Train et validation accuracy stagnent à un niveau bas
  - Solution : Augmenter le nombre de couches, augmenter embed_dim, réduire dropout

- **Loss qui augmente** : Learning rate trop élevé
  - Solution : Réduire le learning rate dans config.yaml

## Configuration recommandée

Pour de meilleurs résultats, ajustez `config.yaml` :

```yaml
embed_dim: 256          # Augmenter pour plus de capacité
num_layers: 4           # Plus de couches = plus de capacité
num_heads: 8            # Plus de têtes d'attention
ff_hidden_dim: 1024     # Dimension cachée du feedforward
dropout: 0.2            # Augmenter pour éviter overfitting
lr: 0.0001             # Learning rate plus bas pour classification
epochs: 50              # Plus d'époques si nécessaire
batch_size: 32          # Batch size standard
weight_decay: 0.01      # Régularisation L2
```

## Comparaison des résultats

### Avant les améliorations
- Pas de validation set
- Pas de métriques d'évaluation
- Mean pooling simple
- Seulement 2 couches
- Pas de learning rate scheduling

### Après les améliorations
- Validation set pour évaluer les performances
- Métriques complètes (accuracy, precision, recall, F1)
- Attention pooling pour meilleure représentation
- Nombre de couches configurable
- Learning rate scheduling automatique
- Gradient clipping pour stabilité

## Prochaines étapes

1. **Entraîner avec le script amélioré** : `python train_sentiment_improved.py`
2. **Analyser les résultats** : Vérifier l'accuracy et le classification report
3. **Ajuster les hyperparamètres** : Si nécessaire, modifier config.yaml
4. **Réentraîner** : Jusqu'à obtenir de bons résultats
5. **Tester dans l'interface** : `streamlit run ui_streamlit.py`

## Notes importantes

- Le modèle sauvegarde automatiquement le meilleur checkpoint basé sur la validation accuracy
- Les checkpoints sont sauvegardés dans `checkpoints/encoder_ckpt.pt`
- Le script affiche un rapport de classification complet à la fin de l'entraînement
- Le learning rate est automatiquement réduit si la validation loss stagne pendant 3 époques

