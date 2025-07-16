#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de traitement des appels d'offres Safakat
Auteur: Assistant Claude
Date: 2025

Ce script traite un fichier CSV contenant des données JSON d'appels d'offres,
les aplatit pour avoir une ligne par AO, et exporte le résultat.
"""

import pandas as pd
import json
import csv
import re
from typing import List, Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_json_string(json_str: str) -> str:
    """
    Nettoie la chaîne JSON en corrigeant les échappements de guillemets.
    
    Args:
        json_str (str): Chaîne JSON brute
        
    Returns:
        str: Chaîne JSON nettoyée
    """
    # Enlever les guillemets de début et fin si présents
    if json_str.startswith('"') and json_str.endswith('"'):
        json_str = json_str[1:-1]
    
    # Corriger les échappements de guillemets problématiques
    # Remplacer '\"' par '"' dans les clés et valeurs
    json_str = json_str.replace('\\"', '"')
    
    # Corriger le format des clés qui ont des guillemets échappés
    json_str = re.sub(r"'\"([^']+)\"'", r'"\1"', json_str)
    
    # Remplacer les guillemets simples par des doubles pour les clés
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    
    return json_str

def extract_ao_fields(lot: Dict[str, Any], org_info: str = None) -> Dict[str, Any]:
    """
    Extrait les champs principaux d'un appel d'offres.
    
    Args:
        lot (dict): Dictionnaire contenant les informations du lot/AO
        org_info (str): Information sur l'organisation
        
    Returns:
        dict: Dictionnaire avec les champs extraits et normalisés
    """
    return {
        'idAO': lot.get('lotId'),
        'titreAO': lot.get('lotObject', '').strip(),
        'description': lot.get('lotDesc', '').strip(),
        'numeroLot': lot.get('lotNbr', '').strip(),
        'categorie': lot.get('lotCategory', '').strip(),
        'montantEstime': lot.get('lotEstimation'),
        'prixFinal': lot.get('finalPrice'),
        'cautionnement': lot.get('lotCaution'),
        'reserve': lot.get('lotReserve', '').strip(),
        'qualified': lot.get('isQualified', False),
        'infructueux': lot.get('isInfructueux'),
        'referencesPossibles': lot.get('isRefPossible', False),
        'statut': lot.get('lotResultStatus', '').strip(),
        'gagnant': lot.get('winner'),
        'contractant': lot.get('contractor'),
        'choix': lot.get('choice'),
        'dateJury': lot.get('juryDate'),
        'dateReunionLot': lot.get('lotReunionDate'),
        'descriptionReunion': lot.get('lotReunionDesc', '').strip(),
        'organisation': org_info,
        'nombreVisites': len(lot.get('visites', [])),
        'nombreAgrements': len(lot.get('agrements', [])),
        'nombreExigences': len(lot.get('requirements', [])),
        'nombreQualifications': len(lot.get('qualifications', [])),
        'nombreResultatsEnLigne': len(lot.get('onlineResults', []))
    }

def process_csv_file(input_file: str, output_file: str = "safakat_ao_flat.csv") -> pd.DataFrame:
    """
    Traite le fichier CSV principal et génère un DataFrame aplati.
    
    Args:
        input_file (str): Chemin vers le fichier CSV d'entrée
        output_file (str): Chemin vers le fichier CSV de sortie
        
    Returns:
        pd.DataFrame: DataFrame contenant tous les AO aplatis
    """
    logger.info(f"Début du traitement du fichier: {input_file}")
    
    all_aos = []
    processed_lines = 0
    error_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            for line_num, row in enumerate(csv_reader, 1):
                if not row or not row[0].strip():
                    continue
                    
                try:
                    # Récupérer la chaîne JSON de la première colonne
                    json_string = row[0].strip()
                    
                    # Nettoyer la chaîne JSON
                    cleaned_json = clean_json_string(json_string)
                    
                    # Parser le JSON
                    data = json.loads(cleaned_json)
                    
                    # Vérifier la structure
                    if 'data' not in data or not isinstance(data['data'], list):
                        logger.warning(f"Ligne {line_num}: Structure 'data' manquante ou invalide")
                        continue
                    
                    # Traiter chaque élément dans 'data'
                    for item in data['data']:
                        org_info = item.get('org', '')
                        
                        # Traiter chaque lot dans cet élément
                        lots = item.get('lots', [])
                        if not isinstance(lots, list):
                            continue
                            
                        for lot in lots:
                            if isinstance(lot, dict):
                                ao_data = extract_ao_fields(lot, org_info)
                                all_aos.append(ao_data)
                    
                    processed_lines += 1
                    if processed_lines % 10 == 0:
                        logger.info(f"Traité {processed_lines} lignes...")
                
                except json.JSONDecodeError as e:
                    logger.error(f"Ligne {line_num}: Erreur JSON - {e}")
                    error_lines += 1
                except Exception as e:
                    logger.error(f"Ligne {line_num}: Erreur inattendue - {e}")
                    error_lines += 1
    
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé: {input_file}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier: {e}")
        raise
    
    logger.info(f"Traitement terminé. Lignes traitées: {processed_lines}, Erreurs: {error_lines}")
    logger.info(f"Nombre total d'AO extraits: {len(all_aos)}")
    
    # Créer le DataFrame
    df = pd.DataFrame(all_aos)
    
    if df.empty:
        logger.warning("Aucune donnée extraite !")
        return df
    
    # Trier par idAO
    if 'idAO' in df.columns:
        df = df.sort_values('idAO').reset_index(drop=True)
    
    # Exporter vers CSV
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Données exportées vers: {output_file}")
    except Exception as e:
        logger.error(f"Erreur lors de l'export: {e}")
        raise
    
    return df

def analyze_dataframe(df: pd.DataFrame) -> None:
    """
    Affiche des statistiques sur le DataFrame généré.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    """
    if df.empty:
        print("DataFrame vide - aucune analyse possible.")
        return
    
    print("\n" + "="*60)
    print("ANALYSE DES DONNÉES EXTRAITES")
    print("="*60)
    
    print(f"Nombre total d'appels d'offres: {len(df)}")
    print(f"Nombre de colonnes: {len(df.columns)}")
    
    # Statistiques sur les catégories
    if 'categorie' in df.columns:
        print(f"\nRépartition par catégorie:")
        print(df['categorie'].value_counts().head())
    
    # Statistiques sur les montants
    if 'montantEstime' in df.columns:
        montants = df['montantEstime'].dropna()
        if len(montants) > 0:
            print(f"\nStatistiques des montants estimés:")
            print(f"  - Minimum: {montants.min():,.2f}")
            print(f"  - Maximum: {montants.max():,.2f}")
            print(f"  - Moyenne: {montants.mean():,.2f}")
            print(f"  - Médiane: {montants.median():,.2f}")
    
    # Organisations
    if 'organisation' in df.columns:
        orgs = df['organisation'].value_counts()
        print(f"\nNombre d'organisations uniques: {len(orgs)}")
        print(f"Top 5 organisations:")
        print(orgs.head())
    
    # Données manquantes
    print(f"\nDonnées manquantes par colonne:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"  - {col}: {count} ({percentage:.1f}%)")

def main():
    """
    Fonction principale du script.
    """
    print("="*60)
    print("TRAITEMENT DES APPELS D'OFFRES SAFAKAT")
    print("="*60)
    
    # Configuration des fichiers
    input_file = "Safakat Test 1000 AOs.csv"
    output_file = "safakat_ao_flat.csv"
    
    try:
        # Traitement principal
        df = process_csv_file(input_file, output_file)
        
        # Analyse des résultats
        analyze_dataframe(df)
        
        # Affichage d'un échantillon
        if not df.empty:
            print(f"\nPremières lignes du résultat:")
            print(df.head())
            
            print(f"\nColonnes disponibles:")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i:2d}. {col}")
        
        print(f"\n✅ Traitement terminé avec succès !")
        print(f"📁 Fichier de sortie: {output_file}")
        
    except Exception as e:
        logger.error(f"Erreur dans le traitement principal: {e}")
        print(f"\n❌ Erreur: {e}")
        raise

if __name__ == "__main__":
    main()