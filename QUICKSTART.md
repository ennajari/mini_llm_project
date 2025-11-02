# 🚀 Guide de Démarrage Rapide

## Installation (5 minutes)

```bash
# 1. Créer un environnement virtuel
python -m venv llm_env
source llm_env/bin/activate  # Linux/Mac
# ou: llm_env\Scripts\activate  # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer Jupyter
jupyter notebook
```

## 📚 Ordre Recommandé

### Pour les débutants:
**Suivez l'ordre numérique** (environ 2-3 heures):

1. **1_Self_Attention.ipynb** (30 min)
   - Comprendre le mécanisme de base
   - Visualiser les attention maps
   
2. **2_Multi_Head_Attention.ipynb** (30 min)
   - Apprendre le multi-head
   - Voir les différents patterns
   
3. **3_Positional_Encoding.ipynb** (20 min)
   - Comprendre l'encodage de position
   - Visualiser les sinusoïdes
   
4. **4_Transformer_Encoder.ipynb** (40 min)
   - Assembler tous les composants
   - Créer un bloc complet
   
5. **5_Complete_Mini_LLM.ipynb** (60 min) ⭐
   - Entraîner un modèle complet
   - Visualiser les résultats

### Pour les pressés:
**Allez directement au notebook 5** (1 heure):
- `5_Complete_Mini_LLM.ipynb` contient tout le code
- Explications complètes incluses
- Modèle fonctionnel from scratch

## 🎯 Ce que vous allez construire

Un mini-LLM capable de:
- ✅ Classifier des sentiments (positif/négatif)
- ✅ Visualiser ses attention patterns
- ✅ Être entraîné sur vos données
- ✅ Être étendu pour d'autres tâches

## 📊 Résultats Attendus

Après ce workshop:
- ✅ Modèle fonctionnel créé from scratch
- ✅ Compréhension profonde de l'attention
- ✅ Visualisations interprétables
- ✅ Base solide pour les LLMs avancés

## 💡 Conseils

1. **Exécutez les cellules dans l'ordre**
2. **Lisez les commentaires** (très détaillés)
3. **Expérimentez** avec les paramètres
4. **Visualisez** à chaque étape

## 🐛 Résolution de Problèmes

### Erreur: "Module not found"
```bash
pip install --upgrade torch numpy matplotlib seaborn
```

### Jupyter ne démarre pas
```bash
pip install --upgrade jupyter ipykernel
python -m ipykernel install --user
```

### Erreur de mémoire
Réduisez les paramètres du modèle:
```python
d_model = 64      # Au lieu de 512
num_layers = 2    # Au lieu de 6
```

## 📝 Structure des Fichiers

```
.
├── README.md                      # Documentation complète
├── QUICKSTART.md                  # Ce fichier
├── requirements.txt               # Dépendances
├── 1_Self_Attention.ipynb        # Attention de base
├── 2_Multi_Head_Attention.ipynb  # Multi-head
├── 3_Positional_Encoding.ipynb   # Encodage position
├── 4_Transformer_Encoder.ipynb   # Bloc encoder
└── 5_Complete_Mini_LLM.ipynb     # Modèle complet ⭐
```

## 🎓 Après le Workshop

Vous serez capable de:
1. Comprendre les papers sur les Transformers
2. Implémenter vos propres variantes
3. Fine-tuner des modèles pré-entraînés
4. Lire le code de GPT, BERT, etc.

## 🚀 Next Steps

Après avoir maîtrisé ces notebooks:
- **Part II**: Decoder & génération de texte
- **Part III**: Training at scale
- **Part IV**: Fine-tuning & applications

## 📚 Ressources Supplémentaires

- [Paper Original: "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course](https://huggingface.co/course)

---

**Prêt à commencer? Lancez Jupyter et ouvrez le premier notebook!** 🎉

```bash
jupyter notebook 1_Self_Attention.ipynb
```
