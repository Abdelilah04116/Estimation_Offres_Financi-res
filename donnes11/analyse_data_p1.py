#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'analyse des données Data_P1.xlsx
Adaptation du notebook presentation_data.ipynb pour les données spécifiques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration pour l'affichage
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def main():
    print("=== ANALYSE DES DONNÉES Data_P1.xlsx ===\n")
    
    try:
        # Lecture du fichier Excel
        print("1. Lecture du fichier Excel...")
        df = pd.read_excel('Data_P1.xlsx')
        print(f"✓ Fichier chargé avec succès!")
        print(f"✓ Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes\n")
        
        # Affichage des premières lignes
        print("2. Premières lignes du dataset:")
        print(df.head())
        print()
        
        # Informations générales
        print("3. Informations générales:")
        print(f"   - Nombre total de lignes: {df.shape[0]}")
        print(f"   - Nombre total de colonnes: {df.shape[1]}")
        print(f"   - Mémoire utilisée: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"   - Types de données:")
        print(df.dtypes.value_counts())
        print()
        
        # Noms des colonnes
        print("4. Noms des colonnes:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        print()
        
        # Valeurs manquantes
        print("5. Analyse des valeurs manquantes:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Valeurs manquantes': missing_data,
            'Pourcentage': missing_percent
        })
        print(missing_df[missing_df['Valeurs manquantes'] > 0])
        if missing_df['Valeurs manquantes'].sum() == 0:
            print("   ✓ Aucune valeur manquante détectée")
        print()
        
        # Lignes dupliquées
        print("6. Analyse des doublons:")
        duplicates = df.duplicated().sum()
        print(f"   - Lignes dupliquées: {duplicates}")
        if duplicates > 0:
            print(f"   - Pourcentage de doublons: {(duplicates/len(df))*100:.2f}%")
        print()
        
        # Analyse des colonnes catégorielles
        print("7. Analyse des colonnes catégorielles:")
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        print(f"   - Colonnes catégorielles identifiées: {list(categorical_columns)}")
        
        for col in categorical_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                print(f"   - {col}: {unique_count} valeurs distinctes")
                if unique_count <= 20:
                    print(f"     Valeurs principales: {dict(df[col].value_counts().head(5))}")
        print()
        
        # Analyse des colonnes numériques
        print("8. Analyse des colonnes numériques:")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        print(f"   - Colonnes numériques identifiées: {list(numeric_columns)}")
        
        if len(numeric_columns) > 0:
            print("   - Statistiques descriptives:")
            print(df[numeric_columns].describe())
            
            # Vérifier les valeurs aberrantes
            print("\n   - Vérification des valeurs aberrantes:")
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
                print(f"     {col}: {outliers} valeurs aberrantes")
        print()
        
        # Analyse des colonnes temporelles
        print("9. Analyse des colonnes temporelles:")
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'jour' in col.lower():
                date_columns.append(col)
        
        if date_columns:
            print(f"   - Colonnes temporelles identifiées: {date_columns}")
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"     {col}: {df[col].min()} à {df[col].max()}")
                    print(f"       Valeurs manquantes: {df[col].isna().sum()}")
                except:
                    print(f"     Impossible de convertir {col} en datetime")
        else:
            print("   - Aucune colonne temporelle identifiée automatiquement")
        print()
        
        # Résumé et recommandations
        print("=== RÉSUMÉ ET RECOMMANDATIONS ===")
        print(f"• Dataset de {df.shape[0]} lignes avec {df.shape[1]} colonnes")
        print(f"• {len(numeric_columns)} variables numériques, {len(categorical_columns)} variables catégorielles")
        print(f"• {missing_df['Valeurs manquantes'].sum()} valeurs manquantes au total")
        print(f"• {duplicates} lignes dupliquées")
        
        print("\n=== PROCHAINES ÉTAPES RECOMMANDÉES ===")
        print("1. Nettoyer les données (valeurs manquantes, doublons)")
        print("2. Encoder les variables catégorielles")
        print("3. Normaliser/standardiser les variables numériques")
        print("4. Analyser les corrélations entre variables")
        print("5. Préparer les données pour la modélisation")
        
        # Sauvegarder un résumé
        summary_file = "resume_analyse_data_p1.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== RÉSUMÉ DE L'ANALYSE Data_P1.xlsx ===\n\n")
            f.write(f"Dimensions: {df.shape}\n")
            f.write(f"Colonnes: {list(df.columns)}\n")
            f.write(f"Types: {dict(df.dtypes)}\n")
            f.write(f"Valeurs manquantes: {dict(missing_data)}\n")
            f.write(f"Doublons: {duplicates}\n")
        
        print(f"\n✓ Résumé sauvegardé dans {summary_file}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {str(e)}")
        print("Vérifiez que le fichier Data_P1.xlsx existe et est accessible.")

if __name__ == "__main__":
    main()

