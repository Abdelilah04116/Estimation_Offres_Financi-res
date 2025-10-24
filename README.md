# Estimation d'Offres Financières

Ce projet utilise l'apprentissage automatique pour prédire les montants des offres financières dans les appels d'offres marocains.

## 📁 Structure du Projet

```
Estimation_Offres_Financières/
├── src/                          # Scripts Python principaux
│   ├── advanced_feature_engineering.py
│   ├── advanced_ml_pipeline.py
│   ├── advanced_model_trainer.py
│   ├── model_evaluation_dashboard.py
│   ├── train_models.py
│   ├── script.py
│   ├── ml_pipeline.py
│   ├── parse_simple.py
│   ├── script_safakat.py
│   ├── reorganize_safakat_data.py
│   ├── diagnostic_csv.py
│   └── faker/                    # Génération de données simulées
│       ├── requirements.txt
│       └── script.py
│
├── app/                          # Application Streamlit
│   └── streamlit_app.py
│
├── data/                         # Datasets
│   ├── appels_offres_zero_null.csv
│   ├── appels_offres_zero_null.json
│   ├── appels_offres.csv
│   ├── safakat_aos_organises.csv
│   ├── safakat_lots_detailles.csv
│   └── faker_data/               # Données simulées
│       ├── appels_offres_maroc_simules.csv
│       └── Safakat Test 1000 AOs.csv
│
├── models/                       # Modèles entraînés
│   ├── encoders.pkl
│   ├── feature_columns.pkl
│   ├── model_classification.pkl
│   ├── model_regression.pkl
│   └── scaler.pkl
│
├── extras/                       # Fichiers supplémentaires
│   ├── model_validation_results.png
│   ├── presentation_data copy.ipynb
│   ├── Augmenterpresecion2.zip
│   └── Ourti Abdelilah.zip
│
├── README.md
├── .gitignore
└── requirements.txt
```

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone <repository-url>
cd Estimation_Offres_Financières
git checkout vps
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
# Sur Windows
venv\Scripts\activate
# Sur Linux/Mac
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📊 Préparation des Données

### 1. Données principales
Les fichiers de données principaux se trouvent dans le dossier `data/` :
- `appels_offres_zero_null.csv` : Dataset principal nettoyé
- `appels_offres.csv` : Dataset original
- `safakat_aos_organises.csv` : Données Safakat organisées
- `safakat_lots_detailles.csv` : Lots détaillés Safakat

### 2. Génération de données simulées (optionnel)
Pour générer des données simulées supplémentaires :
```bash
cd src/faker
pip install -r requirements.txt
python script.py
```

## 🤖 Entraînement des Modèles

### 1. Entraînement sur petit dataset (test rapide)
```bash
python src/train_models.py --quick-test
```

### 2. Entraînement complet
```bash
python src/train_models.py
```

### 3. Pipeline avancé avec optimisation d'hyperparamètres
```bash
python src/advanced_model_trainer.py
```

### 4. Pipeline ML complet
```bash
python src/advanced_ml_pipeline.py
```

## 🎯 Utilisation de l'Application

### Lancer l'application Streamlit
```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## 📈 Évaluation des Modèles

### Dashboard d'évaluation
```bash
python src/model_evaluation_dashboard.py
```

### Validation des résultats
Les résultats de validation sont sauvegardés dans `extras/model_validation_results.png`

## 🔧 Scripts Utilitaires

### Diagnostic des données CSV
```bash
python src/diagnostic_csv.py
```

### Parsing simple de données
```bash
python src/parse_simple.py
```

### Réorganisation des données Safakat
```bash
python src/reorganize_safakat_data.py
```

### Script Safakat
```bash
python src/script_safakat.py
```

## 📋 Fonctionnalités Principales

- **Prédiction de montants** : Prédiction des montants d'offres financières
- **Classification** : Classification des types d'offres
- **Feature Engineering** : Ingénierie des caractéristiques avancée
- **Optimisation d'hyperparamètres** : Recherche automatique des meilleurs paramètres
- **Visualisation** : Dashboard interactif pour l'évaluation des modèles
- **Génération de données** : Création de données simulées pour les tests

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **Scikit-learn** : Algorithmes ML de base
- **XGBoost, LightGBM, CatBoost** : Algorithmes de gradient boosting
- **Streamlit** : Interface utilisateur web
- **Pandas, NumPy** : Manipulation de données
- **Plotly, Matplotlib, Seaborn** : Visualisation
- **Optuna, Hyperopt** : Optimisation d'hyperparamètres
- **SHAP, LIME** : Interprétabilité des modèles
- **Faker** : Génération de données simulées

## 📁 Fichiers Supplémentaires

### Fichiers ZIP
- `Augmenterpresecion2.zip` : Archive du dossier de développement avancé
- `Ourti Abdelilah.zip` : Archive contenant des analyses supplémentaires

### Notebooks Jupyter
- `presentation_data copy.ipynb` : Notebook d'analyse et de présentation des données

### Images
- `model_validation_results.png` : Graphiques de validation des modèles

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajouter nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📝 Notes

- Les modèles entraînés sont sauvegardés dans le dossier `models/`
- Les données sensibles ne doivent pas être commitées (voir `.gitignore`)
- Utilisez l'environnement virtuel pour éviter les conflits de dépendances
- Les données simulées sont générées avec Faker et peuvent être utilisées pour les tests

## 🐛 Dépannage

### Problèmes courants

1. **Erreur de dépendances** : Vérifiez que toutes les dépendances sont installées avec `pip install -r requirements.txt`

2. **Erreur de chemins** : Assurez-vous d'exécuter les scripts depuis la racine du projet

3. **Erreur de modèles** : Vérifiez que les modèles sont entraînés avant de lancer l'application

4. **Problème de mémoire** : Utilisez l'option `--quick-test` pour les tests rapides

## 📞 Support

Pour toute question ou problème, veuillez créer une issue sur le repository GitHub.