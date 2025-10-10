#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de présentation des données Data_P1.csv
Adaptation du notebook presentation_data.ipynb pour les données spécifiques
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration pour l'affichage
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def print_header():
    """Affiche l'en-tête du script"""
    print("=" * 80)
    print("ESTIMATION DES OFFRES FINANCIÈRES POUR LES APPELS D'OFFRES")
    print("BASÉ SUR LES RÉSULTATS HISTORIQUES - ANALYSE Data_P1.csv")
    print("=" * 80)
    print()

def print_section(title):
    """Affiche une section avec formatage"""
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")

def print_subsection(title):
    """Affiche une sous-section"""
    print(f"\n{title}")
    print("-" * len(title))

def check_available_files():
    """Vérifie les fichiers disponibles dans le répertoire"""
    print("Fichiers disponibles dans le répertoire courant:")
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for i, file in enumerate(files, 1):
        print(f"  {i:2d}. {file}")
    print()

def read_csv_data():
    """Lit le fichier CSV Data_P1.csv"""
    try:
        # Essayer de lire le fichier CSV
        df = pd.read_csv('Data_P1.csv')
        print("✓ Fichier Data_P1.csv chargé avec succès!")
        return df, "csv"
    except FileNotFoundError:
        print("❌ Fichier Data_P1.csv non trouvé dans le répertoire courant")
        print("Vérifiez que le fichier existe et est accessible.")
        return None, None
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier CSV: {str(e)}")
        print("Vérifiez le format et l'encodage du fichier.")
        return None, None

def main():
    print_header()
    
    print("Dans ce script, vous trouverez la présentation des données du fichier Data_P1.csv")
    print("que je souhaite utiliser pendant le projet.\n")
    
    # Section 1: Importation des bibliothèques
    print_section("IMPORTATION DES BIBLIOTHÈQUES")
    print("Bibliothèques importées avec succès:")
    print("✓ numpy - Calculs numériques")
    print("✓ pandas - Manipulation des données")
    print("✓ matplotlib - Visualisations")
    print("✓ seaborn - Graphiques statistiques")
    
    # Section 2: Lecture des données
    print_section("LECTURE DES DONNÉES")
    
    # Vérifier les fichiers disponibles
    check_available_files()
    
    # Lire le fichier CSV
    df, file_type = read_csv_data()
    
    if df is None:
        print("\n❌ Impossible de continuer sans les données.")
        print("Veuillez vérifier que le fichier Data_P1.csv existe et est accessible.")
        return
    
    print(f"✓ Fichier chargé avec succès! (Format: {file_type})")
    print(f"✓ Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    print("\nPremières lignes du dataset:")
    print(df.head())
    
    # Section 3: Description des colonnes
    print_section("DESCRIPTION DES COLONNES DE LA TABLE DE DONNÉES")
    
    print("Note: Les descriptions ci-dessous sont basées sur la structure typique")
    print("des données d'appels d'offres. Ajustez selon les colonnes réelles.\n")
    
    # Analyser automatiquement les colonnes disponibles
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
        
        # Détecter le type de colonne et donner une description appropriée
        if 'id' in col.lower() or 'ref' in col.lower() or 'num' in col.lower():
            print("   🔹 Identifiant unique ou référence")
            print("   🔹 Sert à référencer et différencier chaque enregistrement")
        elif 'titre' in col.lower() or 'nom' in col.lower() or 'libelle' in col.lower():
            print("   🔹 Titre ou nom décrivant l'objet")
            print("   🔹 Permet de cerner rapidement le thème ou le secteur")
        elif 'desc' in col.lower() or 'detail' in col.lower():
            print("   🔹 Description complète ou détails")
            print("   🔹 Utile pour l'analyse sémantique")
        elif 'date' in col.lower() or 'jour' in col.lower() or 'heure' in col.lower():
            print("   🔹 Information temporelle")
            print("   🔹 Sert à analyser la chronologie ou la saisonnalité")
        elif 'prix' in col.lower() or 'montant' in col.lower() or 'cout' in col.lower() or 'budget' in col.lower():
            print("   🔹 Information financière")
            print("   🔹 Montant, prix ou budget du projet")
        elif 'categorie' in col.lower() or 'type' in col.lower() or 'secteur' in col.lower():
            print("   🔹 Classification ou catégorisation")
            print("   🔹 Permet de catégoriser selon la nature ou le domaine")
        elif 'region' in col.lower() or 'ville' in col.lower() or 'pays' in col.lower() or 'local' in col.lower():
            print("   🔹 Information géographique")
            print("   🔹 Localisation ou zone géographique")
        elif 'entreprise' in col.lower() or 'soumission' in col.lower() or 'participant' in col.lower():
            print("   🔹 Information sur les participants")
            print("   🔹 Détails sur les entreprises ou soumissionnaires")
        else:
            print("   🔹 Information spécifique au domaine")
            print("   🔹 À analyser selon le contexte métier")
        print()
    
    # Section 4: Informations générales
    print_section("INFORMATIONS GÉNÉRALES SUR LE DATASET")
    
    print(f"• La dimension de la Table est : {df.shape}")
    print(f"• Le nombre des lignes dupliquées est : {df.duplicated().sum()}")
    
    # Section 5: Analyse des valeurs manquantes
    print_subsection("Le nombre des valeurs nulles dans chaque colonne est :")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Valeurs manquantes': missing_data,
        'Pourcentage': missing_percent
    })
    
    # Afficher seulement les colonnes avec des valeurs manquantes
    missing_columns = missing_df[missing_df['Valeurs manquantes'] > 0]
    if len(missing_columns) > 0:
        print(missing_columns)
    else:
        print("✓ Aucune valeur manquante détectée dans le dataset")
    
    print(f"\nTotal des valeurs manquantes: {missing_data.sum()}")
    
    # Section 6: Analyse des colonnes catégorielles
    print_section("ANALYSE DES COLONNES CATÉGORIELLES")
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_columns) > 0:
        print(f"Nombre de colonnes catégorielles distinctes : {len(categorical_columns)}")
        
        for col in categorical_columns:
            unique_count = df[col].nunique()
            print(f"\n• {col} : {unique_count} valeurs distinctes")
            
            if unique_count <= 20:
                value_counts = df[col].value_counts().head(10)
                print(f"  Valeurs principales : {dict(value_counts)}")
            else:
                print(f"  Premières valeurs : {list(df[col].unique()[:5])}")
    else:
        print("Aucune colonne catégorielle identifiée")
    
    # Section 7: Analyse des colonnes numériques
    print_section("ANALYSE DES COLONNES NUMÉRIQUES")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        print(f"Nombre de colonnes numériques distinctes : {len(numeric_columns)}")
        
        for col in numeric_columns:
            print(f"\n• {col} :")
            print(f"  - Min: {df[col].min():.2f}")
            print(f"  - Max: {df[col].max():.2f}")
            print(f"  - Moyenne: {df[col].mean():.2f}")
            print(f"  - Médiane: {df[col].median():.2f}")
            print(f"  - Écart-type: {df[col].std():.2f}")
            
            # Vérifier les valeurs aberrantes
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
            print(f"  - Valeurs aberrantes: {outliers}")
    else:
        print("Aucune colonne numérique identifiée")
    
    # Section 8: Analyse des colonnes temporelles
    print_section("ANALYSE DES COLONNES TEMPORELLES")
    
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'jour' in col.lower() or 'heure' in col.lower():
            date_columns.append(col)
    
    if date_columns:
        print(f"Colonnes temporelles identifiées : {date_columns}")
        for col in date_columns:
            try:
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"\n• {col} :")
                print(f"  - Plage : {df[col].min()} à {df[col].max()}")
                print(f"  - Valeurs manquantes : {df[col].isna().sum()}")
                
                # Analyse temporelle
                if df[col].notna().sum() > 0:
                    year_counts = df[col].dt.year.value_counts().sort_index()
                    print(f"  - Distribution par année : {dict(year_counts)}")
            except Exception as e:
                print(f"  Impossible de convertir {col} en datetime: {str(e)}")
    else:
        print("Aucune colonne temporelle identifiée automatiquement")
    
    # Section 9: Relations entre variables
    print_section("PRÉSENTATION DE QUELQUES RELATIONS ENTRE LES VARIABLES")
    
    if len(numeric_columns) > 1:
        print("Analyse des corrélations entre variables numériques :")
        
        # Calculer la corrélation
        correlation_matrix = df[numeric_columns].corr()
        
        # Identifier les corrélations fortes
        strong_correlations = []
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append((numeric_columns[i], numeric_columns[j], corr_value))
        
        if strong_correlations:
            print("\nCorrélations fortes (|r| > 0.7) :")
            for var1, var2, corr in strong_correlations:
                print(f"  • {var1} - {var2}: {corr:.3f}")
        else:
            print("\nAucune corrélation forte détectée (|r| > 0.7)")
    else:
        print("Pas assez de variables numériques pour analyser les corrélations")
    
    # Section 10: Résumé et recommandations
    print_section("RÉSUMÉ ET RECOMMANDATIONS")
    
    print(f"• Dataset de {df.shape[0]} lignes avec {df.shape[1]} colonnes")
    print(f"• {len(numeric_columns)} variables numériques, {len(categorical_columns)} variables catégorielles")
    print(f"• {missing_data.sum()} valeurs manquantes au total")
    print(f"• {df.duplicated().sum()} lignes dupliquées")
    
    print("\n=== PROCHAINES ÉTAPES RECOMMANDÉES ===")
    print("1. Nettoyer les données (valeurs manquantes, doublons)")
    print("2. Encoder les variables catégorielles")
    print("3. Normaliser/standardiser les variables numériques")
    print("4. Analyser plus en détail les corrélations identifiées")
    print("5. Préparer un pipeline de prétraitement des données")
    print("6. Diviser les données en ensembles d'entraînement et de test")
    print("7. Appliquer les techniques de feature engineering appropriées")
    
    # Sauvegarder un résumé
    try:
        summary_file = "resume_presentation_data_p1.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== RÉSUMÉ DE LA PRÉSENTATION Data_P1.csv ===\n\n")
            f.write(f"Type de données: {file_type}\n")
            f.write(f"Dimensions: {df.shape}\n")
            f.write(f"Colonnes: {list(df.columns)}\n")
            f.write(f"Types: {dict(df.dtypes)}\n")
            f.write(f"Valeurs manquantes: {dict(missing_data)}\n")
            f.write(f"Doublons: {df.duplicated().sum()}\n")
            f.write(f"Variables numériques: {list(numeric_columns)}\n")
            f.write(f"Variables catégorielles: {list(categorical_columns)}\n")
        
        print(f"\n✓ Résumé sauvegardé dans {summary_file}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {str(e)}")
    
    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("=" * 80)

if __name__ == "__main__":
    main()
