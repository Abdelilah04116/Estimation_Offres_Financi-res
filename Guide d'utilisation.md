# Guide d'Utilisation - Projet d'Estimation d'Offres Financières

## Description du Projet

Ce projet développe un système d'estimation intelligente pour les offres financières dans les appels d'offres publics marocains. Il utilise des modèles de machine learning pour prédire la probabilité de succès et recommander des fourchettes de prix optimales.

## Installation

### Prérequis
- Python 3.8 ou version supérieure
- Git installé sur votre système

### Étapes d'installation

1. **Cloner le repository** (si pas déjà fait)
```bash
git clone [URL_DU_REPO]
cd estimation_offres
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## Utilisation du Projet

### Première Configuration

Avant d'utiliser l'application, vous devez entraîner les modèles de machine learning :

```bash
python train_models.py
```

Cette commande va :
- Charger les données d'appels d'offres
- Préparer et nettoyer les données
- Entraîner les modèles de classification et régression
- Sauvegarder les modèles dans des fichiers .pkl

### Lancement de l'Application

Pour utiliser l'interface graphique :

```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira dans votre navigateur web à l'adresse http://localhost:8501

### Utilisation de l'Interface

1. **Saisie des Paramètres**
   - Budget estimé de l'appel d'offres
   - Montant que vous souhaitez proposer
   - Nombre de concurrents attendus
   - Délai d'exécution en mois
   - Votre expérience (en années)
   - Votre notation technique
   - Secteur d'activité
   - Type de procédure
   - Critères d'attribution
   - Niveau de complexité du projet

2. **Analyse des Résultats**
   - Probabilité de succès de votre offre
   - Fourchette de prix recommandée
   - Montant optimal suggéré
   - Graphiques de simulation

3. **Simulation de Prix**
   - Testez différents montants
   - Visualisez l'impact sur la probabilité de succès
   - Analysez la concurrence

### Scripts de Traitement des Données

Si vous devez travailler avec les données brutes :

```bash
# Diagnostic des fichiers CSV
python diagnostic_csv.py

# Réorganisation des données Safakat
python reorganize_safakat_data.py

# Traitement principal des données
python script.py

# Traitement spécifique Safakat
python script_safakat.py
```

## Structure des Données

Le projet utilise plusieurs sources de données :
- `appels_offres.csv` : Données principales d'appels d'offres
- `safakat_aos_organises.csv` : Données organisées de Safakat
- `appels_offres_maroc_simules.csv` : Données simulées pour les tests

## Fonctionnalités Principales

### Prédiction de Succès
Le système calcule la probabilité qu'une offre soit retenue en fonction de :
- Le ratio entre le montant proposé et le budget
- Le nombre de concurrents
- Les caractéristiques techniques
- L'historique des prix gagnants

### Recommandations de Prix
- Fourchette optimale de prix
- Montant recommandé pour maximiser les chances
- Analyse de la compétitivité

### Analyse Concurrentielle
- Comparaison avec les offres historiques
- Évaluation du niveau de concurrence
- Identification des facteurs clés de succès

## Dépannage

### Problèmes Courants

1. **Erreur "Modèles non trouvés"**
   - Solution : Exécuter `python train_models.py`

2. **Erreur de dépendances**
   - Solution : Vérifier l'installation avec `pip install -r requirements.txt`

3. **Port déjà utilisé**
   - Solution : Fermer les autres applications Streamlit ou changer le port

### Messages d'Erreur Fréquents

- **FileNotFoundError** : Les fichiers de données ou modèles sont manquants
- **ImportError** : Dépendances Python non installées
- **MemoryError** : Données trop volumineuses pour la mémoire disponible

## Maintenance

### Mise à Jour des Modèles
Pour retraîner les modèles avec de nouvelles données :
```bash
python train_models.py
```

### Sauvegarde des Données
Les modèles entraînés sont sauvegardés dans :
- `model_classification.pkl`
- `model_regression.pkl`
- `scaler.pkl`
- `encoders.pkl`

## Support Technique

Pour toute question ou problème :
- Consulter le fichier README.md
- Vérifier les logs d'erreur dans la console
- Contacter l'auteur : abdelilahourti@gmail.com

## Commandes de Référence Rapide

```bash
# Installation complète
pip install -r requirements.txt

# Entraînement initial
python train_models.py

# Lancement de l'application
streamlit run streamlit_app.py

# Diagnostic des données
python diagnostic_csv.py
```

## Exemple d'Utilisation Typique

1. **Installation initiale**
   ```bash
   pip install -r requirements.txt
   ```

2. **Entraînement des modèles**
   ```bash
   python train_models.py
   ```

3. **Lancement de l'application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Utilisation dans le navigateur**
   - Saisir les paramètres de l'appel d'offres
   - Consulter les prédictions et recommandations
   - Analyser les graphiques de simulation

Ce guide couvre l'essentiel pour utiliser le projet. L'interface Streamlit est intuitive et guide l'utilisateur à travers les différentes étapes de l'analyse. 