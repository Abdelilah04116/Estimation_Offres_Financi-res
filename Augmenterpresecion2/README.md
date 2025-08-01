# 🚀 Pipeline ML Avancé pour l'Estimation d'Offres

## 📋 Vue d'ensemble

Ce projet propose une architecture ML avancée pour prédire les résultats d'appels d'offres avec deux objectifs principaux :
- **Classification** : Prédire si une soumission gagnera (probabilité de succès)
- **Régression** : Prédire le montant gagnant

L'architecture a été conçue pour des serveurs avec des ressources importantes et intègre les dernières techniques de Machine Learning.

## 🏗️ Architecture

### 📁 Structure des fichiers

```
Augmenterpresecion2/
├── script.py                          # Générateur de données original
├── train_models.py                    # Entraînement basique (conservé)
├── streamlit_app.py                   # Interface utilisateur améliorée
├── requirements.txt                   # Dépendances complètes
├── advanced_feature_engineering.py    # Feature engineering avancé
├── advanced_model_trainer.py          # Entraînement avec hyperparameter tuning
├── model_evaluation_dashboard.py      # Dashboard d'évaluation complet
├── advanced_ml_pipeline.py           # Pipeline principal orchestré
└── README.md                         # Ce fichier
```

### 🔧 Modules principaux

#### 1. **Advanced Feature Engineering** (`advanced_feature_engineering.py`)
- **Features temporelles** : jour de la semaine, mois, saisonnalité
- **Features d'interaction** : produits et ratios complexes
- **Target Encoding** : encodage avancé des variables catégorielles
- **Features métier** : stratégie de prix, niveau de concurrence
- **Sélection automatique** : combinaison de méthodes statistiques et ML

#### 2. **Advanced Model Trainer** (`advanced_model_trainer.py`)
- **Hyperparameter Optimization** : Optuna avec validation croisée
- **Multiples algorithmes** : XGBoost, LightGBM, CatBoost, Random Forest, SVM, MLP
- **Gestion de l'imbalance** : SMOTE pour équilibrer les classes
- **Ensembles** : VotingClassifier et VotingRegressor automatiques
- **Early stopping** : prévention de l'overfitting

#### 3. **Model Evaluation Dashboard** (`model_evaluation_dashboard.py`)
- **Métriques complètes** : ROC-AUC, Precision-Recall, R², RMSE, MAE
- **Visualisations avancées** : courbes ROC, matrices de confusion, Q-Q plots
- **Dashboard interactif** : Plotly pour l'exploration
- **Analyses par segment** : performance par secteur/catégorie
- **Rapports automatiques** : recommandations et insights

#### 4. **Advanced ML Pipeline** (`advanced_ml_pipeline.py`)
- **Orchestration complète** : 5 étapes automatisées
- **Logging avancé** : suivi détaillé avec timestamps
- **Gestion d'erreurs** : robustesse et récupération
- **Configuration flexible** : paramètres optimisables
- **Sauvegarde automatique** : modèles et résultats

## 🚀 Installation et utilisation

### 1. **Installation des dépendances**

```bash
pip install -r requirements.txt
```

### 2. **Exécution du pipeline complet**

```bash
python advanced_ml_pipeline.py
```

### 3. **Interface utilisateur**

```bash
streamlit run streamlit_app.py
```

## 📊 Fonctionnalités avancées

### 🎯 **Optimisation des performances**

#### **Feature Engineering**
- **50+ features créées** automatiquement
- **Sélection intelligente** des meilleures features
- **Encodage avancé** des variables catégorielles
- **Features temporelles** et saisonnières

#### **Hyperparameter Tuning**
- **Optuna** pour l'optimisation bayésienne
- **150+ essais** par modèle
- **Validation croisée** stratifiée (5-fold)
- **Early stopping** pour éviter l'overfitting

#### **Ensembles de modèles**
- **VotingClassifier** pour la classification
- **VotingRegressor** pour la régression
- **Sélection automatique** des meilleurs modèles
- **Pondération optimisée** des prédictions

### 📈 **Évaluation complète**

#### **Métriques de classification**
- **ROC-AUC** avec intervalles de confiance
- **Precision-Recall curves**
- **Matrice de confusion** avec heatmap
- **Classification report** détaillé

#### **Métriques de régression**
- **R², RMSE, MAE** par segment
- **Scatter plot** prédictions vs réel
- **Distribution des résidus**
- **Q-Q plot** pour la normalité

#### **Analyses avancées**
- **Performance par secteur**
- **Analyse des erreurs**
- **Stabilité temporelle**
- **Détection d'outliers**

### 🎨 **Visualisations**

#### **Graphiques statiques** (Matplotlib/Seaborn)
- Layout en grille 2x3
- Couleurs cohérentes (bleus/marine)
- Métriques annotées
- Export haute résolution

#### **Dashboard interactif** (Plotly)
- Graphiques interactifs
- Zoom et pan
- Hover information
- Export dynamique

## ⚙️ Configuration

### **Configuration par défaut optimisée**

```python
config = {
    'data_generation': {
        'n_aos': 20000,  # Nombre d'appels d'offres
        'locale': 'fr_FR'
    },
    'feature_engineering': {
        'n_features': 60,  # Features à sélectionner
        'use_advanced_features': True
    },
    'model_training': {
        'n_trials': 150,  # Essais d'optimisation
        'test_size': 0.2,
        'random_state': 42
    },
    'evaluation': {
        'create_visualizations': True,
        'create_interactive_dashboard': True,
        'save_models': True
    }
}
```

### **Personnalisation**

Vous pouvez modifier la configuration dans `advanced_ml_pipeline.py` :

```python
# Configuration personnalisée
custom_config = {
    'data_generation': {'n_aos': 30000},  # Plus de données
    'model_training': {'n_trials': 200},   # Plus d'optimisation
    # ... autres paramètres
}

pipeline = AdvancedMLPipeline(custom_config)
```

## 📈 Objectifs de performance

### **Cibles définies**
- **Classification** : AUC-ROC > 0.85
- **Régression** : R² > 0.80
- **Généralisation** : Gap train/val < 5%
- **Stabilité** : Validation croisée stable

### **Métriques business**
- **Précision des prédictions** de gain
- **Estimation précise** des montants
- **ROI des recommandations**
- **Réduction des coûts** d'opportunité

## 🔍 Monitoring et logging

### **Logs détaillés**
- **Timestamps** pour chaque étape
- **Métriques de performance** en temps réel
- **Gestion d'erreurs** avec stack traces
- **Sauvegarde automatique** des logs

### **Fichiers générés**
```
logs/
├── advanced_pipeline_20241201_143022.log
├── models/
│   ├── classification_xgboost.pkl
│   ├── classification_lightgbm.pkl
│   ├── regression_xgboost.pkl
│   └── metadata.pkl
├── classification_evaluation.png
├── regression_evaluation.png
└── pipeline_results.json
```

## 🛠️ Utilisation avancée

### **Entraînement personnalisé**

```python
from advanced_model_trainer import AdvancedModelTrainer

# Créer un trainer personnalisé
trainer = AdvancedModelTrainer(random_state=42)

# Préparer les données
X_train, X_test, y_train, y_test, features = trainer.prepare_data(df)

# Entraîner avec optimisation
results = trainer.train_models(X_train, y_train, X_test, y_test, task='classification')

# Créer un ensemble
ensemble, score = trainer.create_ensemble(X_train, y_train, X_test, y_test, task='classification')
```

### **Évaluation personnalisée**

```python
from model_evaluation_dashboard import ModelEvaluationDashboard

# Créer le dashboard
dashboard = ModelEvaluationDashboard(models_data, X_test, y_test)

# Évaluation complète
report = dashboard.run_complete_evaluation()

# Analyse par segment
segment_analysis = dashboard.analyze_performance_by_segment('secteur')
```

### **Feature Engineering personnalisé**

```python
from advanced_feature_engineering import AdvancedFeatureEngineer

# Créer l'engineer
engineer = AdvancedFeatureEngineer()

# Engineering complet
df_engineered, selected_features = engineer.run_complete_engineering(df)

# Ou étapes individuelles
df = engineer.extract_numeric_features(df)
df = engineer.create_temporal_features(df)
df = engineer.create_interaction_features(df)
```

## 🚨 Dépannage

### **Problèmes courants**

#### **1. Erreur d'import**
```bash
# Solution : installer les dépendances manquantes
pip install category_encoders optuna
```

#### **2. Mémoire insuffisante**
```python
# Réduire la configuration
config['data_generation']['n_aos'] = 10000
config['model_training']['n_trials'] = 50
```

#### **3. Modèles non trouvés**
```bash
# Exécuter d'abord le pipeline
python advanced_ml_pipeline.py
```

### **Optimisation des performances**

#### **Pour serveurs puissants**
```python
config = {
    'data_generation': {'n_aos': 50000},
    'feature_engineering': {'n_features': 100},
    'model_training': {'n_trials': 300}
}
```

#### **Pour serveurs limités**
```python
config = {
    'data_generation': {'n_aos': 5000},
    'feature_engineering': {'n_features': 30},
    'model_training': {'n_trials': 50}
}
```

## 📚 Ressources additionnelles

### **Documentation technique**
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Category Encoders](https://contrib.scikit-learn.org/category_encoders/)

### **Bonnes pratiques ML**
- Validation croisée stratifiée
- Gestion de l'imbalance des classes
- Feature selection automatique
- Monitoring de la dérive

## 🤝 Contribution

Pour contribuer au projet :

1. **Fork** le repository
2. **Créer** une branche feature
3. **Implémenter** les améliorations
4. **Tester** avec le pipeline complet
5. **Soumettre** une pull request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

**🎯 Objectif** : Créer le meilleur système de prédiction d'appels d'offres avec des performances optimales et une interprétabilité maximale. 