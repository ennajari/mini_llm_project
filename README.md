# Build an LLM from Scratch - Part I: Attention & Transformer Core

## 📚 Description du Workshop

Ce workshop vous guide dans la construction d'un Large Language Model (LLM) **from scratch** en utilisant PyTorch. Vous allez implémenter et comprendre les mécanismes fondamentaux qui alimentent des modèles comme GPT, BERT, et autres Transformers modernes.

## 🎯 Objectifs d'apprentissage

1. ✅ **Self-Attention Mechanism** - Le cœur des Transformers
2. ✅ **Multi-Head Attention** - Apprentissage de patterns multiples
3. ✅ **Positional Encoding** - Encodage de la position des mots
4. ✅ **Transformer Encoder Block** - Architecture complète
5. ✅ **Visualisation** - Comprendre ce que le modèle apprend

## 📦 Structure des Notebooks

### 1️⃣ `1_Self_Attention.ipynb`
**Comprendre le mécanisme d'attention de base**
- Implémentation du scaled dot-product attention
- Visualisation des poids d'attention
- Masque causal pour décodeurs
- Exemple concret avec des phrases

**Ce que vous allez apprendre:**
- Pourquoi l'attention est révolutionnaire
- Comment calculer Q, K, V
- L'importance du scaling par √d_k
- Interprétation des heatmaps d'attention

### 2️⃣ `2_Multi_Head_Attention.ipynb`
**Apprentissage de relations multiples en parallèle**
- Implémentation du multi-head attention
- Comparaison 1 tête vs multi-têtes
- Visualisation des différentes têtes
- Analyse de la complexité

**Ce que vous allez apprendre:**
- Pourquoi utiliser plusieurs têtes
- Comment les têtes capturent différents patterns
- Architecture et implémentation PyTorch
- Trade-offs en termes de paramètres

### 3️⃣ `3_Positional_Encoding.ipynb`
**Ajouter l'information de position**
- Encodage sinusoïdal (original Transformer)
- Alternatives (learned positional encoding)
- Visualisation des patterns
- Propriétés mathématiques

**Ce que vous allez apprendre:**
- Pourquoi les Transformers ont besoin de PE
- Comment fonctionnent les sinusoïdes
- Interprétation des fréquences
- Distance relative entre positions

### 4️⃣ `4_Transformer_Encoder.ipynb`
**Assembler tous les composants**
- Layer Normalization
- Feed-Forward Networks
- Residual Connections
- Bloc Encoder complet
- Stack de N couches

**Ce que vous allez apprendre:**
- Architecture complète du Transformer
- Rôle de chaque composant
- Comment empiler les couches
- Exemple d'application (classification)

### 5️⃣ `5_Complete_Mini_LLM.ipynb` ⭐
**Synthèse finale - Un LLM fonctionnel!**
- Architecture complète from scratch
- Entraînement sur toy dataset
- Visualisation des prédictions
- Attention heatmaps interprétables
- Code production-ready

**Ce que vous allez créer:**
- Un modèle complet de classification de sentiment
- Visualisations interactives
- Pipeline d'entraînement
- Modèle sauvegardable et réutilisable

## 🚀 Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install torch numpy matplotlib seaborn jupyter
```

## 📖 Utilisation

### Option 1: Suivre l'ordre recommandé
```bash
jupyter notebook
```

Puis ouvrez les notebooks dans l'ordre:
1. `1_Self_Attention.ipynb`
2. `2_Multi_Head_Attention.ipynb`
3. `3_Positional_Encoding.ipynb`
4. `4_Transformer_Encoder.ipynb`
5. `5_Complete_Mini_LLM.ipynb`

### Option 2: Aller directement au modèle complet
Si vous êtes pressé, ouvrez directement `5_Complete_Mini_LLM.ipynb` qui contient tout le code avec explications.

## 🎓 Concepts Clés Expliqués

### Self-Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- **Q (Query)**: Ce que je cherche
- **K (Key)**: Ce qui est disponible
- **V (Value)**: L'information réelle
- **Scaling**: Division par √d_k pour stabilité

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```
- Permet d'apprendre plusieurs types de relations
- Même complexité qu'une seule tête
- Parallélisation efficace

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Ajoute l'information de position
- Fréquences variées pour différentes échelles
- Permet la généralisation à des longueurs non vues

### Transformer Block
```
Block(x):
  1. x = x + MultiHeadAttention(x)
  2. x = LayerNorm(x)
  3. x = x + FeedForward(x)
  4. x = LayerNorm(x)
  return x
```

## 📊 Résultats Attendus

Après avoir complété les notebooks, vous aurez:

✅ Un modèle Transformer fonctionnel  
✅ Compréhension profonde de l'attention  
✅ Capacité à visualiser et interpréter les attention maps  
✅ Code modulaire et réutilisable  
✅ Base solide pour explorer des architectures plus complexes  

## 🔍 Visualisations Incluses

- **Attention Heatmaps**: Voir quels mots le modèle regarde
- **Training Curves**: Loss et accuracy au fil du temps
- **Positional Encoding Patterns**: Visualiser les sinusoïdes
- **Multi-head Comparisons**: Comparer différentes têtes
- **Layer Analysis**: Évolution des représentations

## 📚 Pour Aller Plus Loin

### Papers à lire:
1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Le paper original du Transformer
   
2. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
   - Utilisation du Transformer pour le NLP
   
3. **"GPT-3: Language Models are Few-Shot Learners"** (Brown et al., 2020)
   - Scaling laws et émergence

### Ressources additionnelles:
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

## 🛠️ Customisation

Le code est modulaire et peut être facilement adapté pour:
- Différentes tailles de modèles
- Autres tâches (génération, traduction, etc.)
- Datasets personnalisés
- Expérimentations architecturales

Exemple pour changer la configuration:
```python
model = MiniLLM(
    vocab_size=10000,
    d_model=512,      # ← Augmenter pour plus de capacité
    num_heads=8,      # ← Plus de têtes = plus de patterns
    d_ff=2048,        # ← Dimension du FFN
    num_layers=6,     # ← Profondeur du modèle
    num_classes=3     # ← Nombre de classes à prédire
)
```

## 💡 Tips & Best Practices

### Pour l'entraînement:
- Commencer avec un petit modèle pour debugger
- Utiliser un learning rate scheduler
- Monitorer les gradients (gradient clipping si nécessaire)
- Sauvegarder les checkpoints régulièrement

### Pour la visualisation:
- Examiner plusieurs exemples
- Comparer les patterns entre couches
- Vérifier que l'attention a du sens sémantiquement

### Pour le debugging:
- Vérifier les shapes à chaque étape
- Tester avec des batch_size=1 d'abord
- Utiliser des assertions pour valider les dimensions

## 🤝 Contribution

Ce workshop est conçu pour être éducatif. N'hésitez pas à:
- Expérimenter avec le code
- Ajouter vos propres visualisations
- Tester sur vos données
- Partager vos découvertes

## 📝 Notes Importantes

⚠️ **Ce code est à but éducatif**
- Optimisé pour la clarté, pas la performance
- Pour la production, utilisez des bibliothèques comme Hugging Face
- Les toy datasets sont pour la démonstration

✨ **Points forts du code:**
- Annotations détaillées en français
- Explications étape par étape
- Visualisations interactives
- Architecture modulaire

## 🎯 Livrables du Workshop

À la fin de ce workshop, vous devez produire:

1. **Notebook fonctionnel** avec implémentation complète ✓
2. **Visualisations d'attention** montrant les patterns appris ✓
3. **Courte explication** de vos résultats (déjà dans les notebooks) ✓

## 📧 Support

Si vous avez des questions:
1. Consultez d'abord les commentaires dans le code
2. Regardez les cellules de visualisation
3. Expérimentez avec différents paramètres

## 🎉 Félicitations!

Vous avez maintenant les outils pour comprendre et construire des LLMs modernes!

**Next Steps:**
- Part II: Decoder Architecture & Text Generation
- Part III: Training at Scale
- Part IV: Fine-tuning & Applications

---

**Happy Learning! 🚀**

*Build, Learn, Iterate*
