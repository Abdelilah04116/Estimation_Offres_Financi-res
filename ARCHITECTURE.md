# Architecture du Projet

## 📁 Structure des Dossiers

```
Estimation_Offres_Financières/
├── src/                          # Scripts Python principaux
│   ├── advanced_feature_engineering.py    # Ingénierie des caractéristiques avancée
│   ├── advanced_ml_pipeline.py            # Pipeline ML avancé avec optimisation
│   ├── advanced_model_trainer.py          # Entraîneur de modèles avancé
│   ├── model_evaluation_dashboard.py      # Dashboard d'évaluation des modèles
│   ├── train_models.py                    # Script d'entraînement principal
│   ├── ml_pipeline.py                     # Pipeline ML de base
│   ├── parse_simple.py                    # Parser simple de données
│   ├── script_safakat.py                  # Script spécifique Safakat
│   ├── reorganize_safakat_data.py         # Réorganisation des données Safakat
│   ├── diagnostic_csv.py                  # Diagnostic des fichiers CSV
│   └── faker/                             # Génération de données simulées
│       ├── requirements.txt               # Dépendances pour Faker
│       └── script.py                      # Script de génération de données
│
├── app/                          # Application Streamlit
│   └── streamlit_app.py                  # Interface utilisateur web
│
├── data/                         # Datasets
│   ├── appels_offres_zero_null.csv       # Dataset principal nettoyé
│   ├── appels_offres_zero_null.json      # Dataset en format JSON
│   ├── appels_offres.csv                 # Dataset original
│   └── faker_data/                       # Données simulées
│       ├── appels_offres_maroc_simules.csv
│       ├── Safakat Test 1000 AOs.csv
│       ├── safakat_aos_organises.csv
│       └── safakat_lots_detailles.csv
│
├── models/                       # Modèles entraînés
│   ├── model_classification.pkl          # Modèle de classification
│   ├── model_regression.pkl              # Modèle de régression
│   ├── scaler.pkl                        # Normaliseur
│   ├── encoders.pkl                      # Encodeurs de catégories
│   └── feature_columns.pkl               # Colonnes de caractéristiques
│
├── extras/                       # Fichiers supplémentaires
│   ├── model_validation_results.png      # Graphiques de validation
│   ├── presentation_data copy.ipynb      # Notebook d'analyse
│   ├── Augmenterpresecion2.zip           # Archive du développement avancé
│   └── Ourti Abdelilah.zip               # Archive d'analyses supplémentaires
│
├── README.md                     # Documentation principale
├── ARCHITECTURE.md               # Documentation de l'architecture
├── .gitignore                    # Fichiers à ignorer par Git
└── requirements.txt              # Dépendances Python
```

## 🔄 Flux de Données

1. **Préparation des données** (`data/`)
   - Les données brutes sont stockées dans `data/`
   - Les données simulées sont dans `data/faker_data/`

2. **Traitement** (`src/`)
   - Les scripts de traitement lisent les données depuis `data/`
   - Les modèles sont entraînés et sauvegardés dans `models/`

3. **Interface utilisateur** (`app/`)
   - L'application Streamlit charge les modèles depuis `models/`
   - Elle lit les données depuis `data/` pour les prédictions

4. **Fichiers supplémentaires** (`extras/`)
   - Contient les analyses, notebooks et archives

## 🛠️ Scripts Principaux

### Entraînement
- `src/train_models.py` : Entraînement de base
- `src/advanced_model_trainer.py` : Entraînement avancé avec optimisation
- `src/advanced_ml_pipeline.py` : Pipeline ML complet

### Interface
- `app/streamlit_app.py` : Application web Streamlit

### Utilitaires
- `src/diagnostic_csv.py` : Diagnostic des données
- `src/faker/script.py` : Génération de données simulées

## 📦 Dépendances

### Principales
- `pandas`, `numpy` : Manipulation de données
- `scikit-learn` : Algorithmes ML de base
- `xgboost`, `lightgbm`, `catboost` : Algorithmes avancés
- `streamlit` : Interface web
- `category_encoders` : Encodage des catégories

### Optionnelles
- `optuna`, `hyperopt` : Optimisation d'hyperparamètres
- `shap`, `lime` : Interprétabilité des modèles
- `faker` : Génération de données simulées

## 🚀 Utilisation

1. **Installation** : `pip install -r requirements.txt`
2. **Entraînement** : `python src/train_models.py`
3. **Application** : `streamlit run app/streamlit_app.py`

## 🔧 Maintenance

- Les modèles sont sauvegardés automatiquement dans `models/`
- Les données sont organisées par type dans `data/`
- Les scripts utilitaires sont dans `src/`
- La documentation est dans les fichiers `.md`
