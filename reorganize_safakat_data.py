#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour organiser les données Safakat d'un fichier JSON/CSV vers un format CSV structuré
Auteur: Assistant IA - Expert en organisation de données
Dépendances: pandas, json, csv, datetime
"""

import json
import csv
import pandas as pd
from datetime import datetime
import re
import os
import sys

def clean_json_line(line):
    """
    Nettoie et corrige le format JSON d'une ligne
    """
    if not line.strip():
        return None
    
    # Enlever les guillemets externes si présents
    cleaned_line = line.strip()
    if cleaned_line.startswith('"') and cleaned_line.endswith('"'):
        cleaned_line = cleaned_line[1:-1]
    
    # Corriger le format des guillemets échappés
    cleaned_line = cleaned_line.replace('\\"', '"')
    cleaned_line = cleaned_line.replace('{\'"', '{"')
    cleaned_line = cleaned_line.replace('"\'', '"')
    cleaned_line = cleaned_line.replace('\'"', '"')
    cleaned_line = cleaned_line.replace('\': \'', '": "')
    cleaned_line = cleaned_line.replace('\', \'', '", "')
    cleaned_line = cleaned_line.replace('\'}', '"}')
    cleaned_line = cleaned_line.replace('\']', '"]')
    cleaned_line = cleaned_line.replace('[\'', '["')
    
    return cleaned_line

def parse_date(date_string):
    """
    Parse une date ISO et la retourne au format français
    """
    if not date_string:
        return ''
    try:
        date_obj = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return date_obj.strftime('%d/%m/%Y')
    except:
        return date_string

def organize_safakat_data_compact(input_file, output_file='safakat_aos_organises.csv'):
    """
    Organise les données Safakat en format CSV compact (une ligne par AO)
    """
    print("📖 Lecture du fichier source...")
    
    all_aos = []
    total_lines = 0
    processed_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                total_lines += 1
                try:
                    cleaned_line = clean_json_line(line)
                    if cleaned_line:
                        data = json.loads(cleaned_line)
                        if 'data' in data and data['data']:
                            processed_lines += 1
                            
                            for ao in data['data']:
                                # Extraire les informations principales de l'AO
                                transformed_ao = {
                                    # Informations générales
                                    'consId': ao.get('consId', ''),
                                    'reference': ao.get('reference', ''),
                                    'acheteur': ao.get('acheteur', ''),
                                    'org': ao.get('org', ''),
                                    'AchAbr': ao.get('AchAbr', ''),
                                    
                                    # Dates
                                    'publishedDate': parse_date(ao.get('publishedDate')),
                                    'endDate': parse_date(ao.get('endDate')),
                                    'createdAt': parse_date(ao.get('createdAt')),
                                    
                                    # Procédure
                                    'procedureType': ao.get('procedureType', ''),
                                    'reponseType': ao.get('reponseType', ''),
                                    'isConsCancelled': 'Oui' if ao.get('isConsCancelled') else 'Non',
                                    
                                    # Localisation
                                    'provinces': ', '.join(ao.get('provinces', [])),
                                    
                                    # Contact administratif
                                    'administratifName': ao.get('administratifName', ''),
                                    'administratifEmail': ao.get('administratifEmail', ''),
                                    'administratifTel': ao.get('administratifTel', ''),
                                    'administratifFax': ao.get('administratifFax', ''),
                                    
                                    # URL et documents
                                    'detailsUrl': ao.get('detailsUrl', ''),
                                    'consDAO': ao.get('consDAO', ''),
                                    
                                    # Informations sur les lots (premier lot ou résumé)
                                    'nombreLots': len(ao.get('lots', [])),
                                    'premierLotId': ao.get('lots', [{}])[0].get('lotId', '') if ao.get('lots') else '',
                                    'premierLotObject': ao.get('lots', [{}])[0].get('lotObject', '') if ao.get('lots') else '',
                                    'premierLotCategory': ao.get('lots', [{}])[0].get('lotCategory', '') if ao.get('lots') else '',
                                    'premierLotEstimation': ao.get('lots', [{}])[0].get('lotEstimation', 0) if ao.get('lots') else 0,
                                    'premierLotCaution': ao.get('lots', [{}])[0].get('lotCaution', 0) if ao.get('lots') else 0,
                                    'premierLotReserve': ao.get('lots', [{}])[0].get('lotReserve', '') if ao.get('lots') else '',
                                    'premierLotQualified': 'Oui' if (ao.get('lots') and ao.get('lots')[0].get('isQualified')) else 'Non',
                                    
                                    # Estimation totale (somme de tous les lots)
                                    'estimationTotale': sum(lot.get('lotEstimation', 0) for lot in ao.get('lots', [])),
                                    
                                    # Domaines d'activité
                                    'nombreDomaines': len(ao.get('domains', [])),
                                    'domainesPrincipaux': ', '.join(d.get('domain', '') for d in ao.get('domains', [])),
                                    'activitesPrincipales': ', '.join(d.get('activite', '') for d in ao.get('domains', [])),
                                    'sousDomaines': ', '.join(d.get('sousDomain', '') for d in ao.get('domains', [])),
                                    
                                    # Visites (du premier lot)
                                    'nombreVisites': len(ao.get('lots', [{}])[0].get('visites', [])) if ao.get('lots') else 0,
                                    'premiereDateVisite': parse_date(ao.get('lots', [{}])[0].get('visites', [{}])[0].get('date', '')) 
                                                        if (ao.get('lots') and ao.get('lots')[0].get('visites')) else '',
                                    
                                    # Statuts
                                    'isFavoris': 'Oui' if ao.get('isFavoris') else 'Non',
                                    'nombreAvertissements': len(ao.get('avertissements', []))
                                }
                                
                                all_aos.append(transformed_ao)
                                
                except json.JSONDecodeError as e:
                    print(f"⚠️ Erreur JSON ligne {line_num}: {str(e)}")
                except Exception as e:
                    print(f"⚠️ Erreur traitement ligne {line_num}: {str(e)}")
    
    except FileNotFoundError:
        print(f"❌ Fichier '{input_file}' non trouvé!")
        return False
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {str(e)}")
        return False
    
    print(f"✅ {len(all_aos)} AOs extraits de {processed_lines}/{total_lines} lignes")
    
    # Générer le CSV avec pandas
    print("📝 Génération du fichier CSV...")
    try:
        df = pd.DataFrame(all_aos)
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=',', quotechar='"')
        print(f"🎉 Fichier CSV généré: {output_file}")
        print(f"📊 Statistiques: {len(all_aos)} AOs, {len(df.columns)} colonnes")
        
        # Afficher les colonnes
        print("\n📋 Colonnes disponibles dans le CSV:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération CSV: {str(e)}")
        return False

def organize_safakat_data_detailed(input_file, output_file='safakat_lots_detailles.csv'):
    """
    Organise les données Safakat en format CSV détaillé (une ligne par lot)
    """
    print("\n🔄 Version détaillée: Un CSV avec une ligne par lot...")
    
    all_lots = []
    total_lines = 0
    processed_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                total_lines += 1
                try:
                    cleaned_line = clean_json_line(line)
                    if cleaned_line:
                        data = json.loads(cleaned_line)
                        if 'data' in data and data['data']:
                            processed_lines += 1
                            
                            for ao in data['data']:
                                # Pour chaque lot de l'AO
                                if ao.get('lots'):
                                    for lot_index, lot in enumerate(ao['lots']):
                                        lot_record = {
                                            # Infos AO
                                            'consId': ao.get('consId', ''),
                                            'reference': ao.get('reference', ''),
                                            'acheteur': ao.get('acheteur', ''),
                                            'org': ao.get('org', ''),
                                            'procedureType': ao.get('procedureType', ''),
                                            'publishedDate': parse_date(ao.get('publishedDate')),
                                            'endDate': parse_date(ao.get('endDate')),
                                            'provinces': ', '.join(ao.get('provinces', [])),
                                            
                                            # Infos lot
                                            'lotIndex': lot_index + 1,
                                            'lotId': lot.get('lotId', ''),
                                            'lotNbr': lot.get('lotNbr', ''),
                                            'lotObject': lot.get('lotObject', ''),
                                            'lotDesc': lot.get('lotDesc', ''),
                                            'lotCategory': lot.get('lotCategory', ''),
                                            'lotEstimation': lot.get('lotEstimation', 0),
                                            'lotCaution': lot.get('lotCaution', 0),
                                            'lotReserve': lot.get('lotReserve', ''),
                                            'isQualified': 'Oui' if lot.get('isQualified') else 'Non',
                                            'lotVarianteValeur': lot.get('lotVarianteValeur', ''),
                                            
                                            # Résultats
                                            'winner': lot.get('winner', ''),
                                            'contractor': lot.get('contractor', ''),
                                            'finalPrice': lot.get('finalPrice', ''),
                                            'lotResultStatus': lot.get('lotResultStatus', ''),
                                            
                                            # Qualifications
                                            'nombreQualifications': len(lot.get('qualifications', [])),
                                            'qualificationsSecteurs': ', '.join(q.get('secteur', '') for q in lot.get('qualifications', [])),
                                            
                                            # Visites
                                            'nombreVisites': len(lot.get('visites', [])),
                                            'premiereVisite': parse_date(lot.get('visites', [{}])[0].get('date', '')) if lot.get('visites') else '',
                                            
                                            # Échantillons
                                            'lotEchantillonsDate': parse_date(lot.get('lotEchantillonsDate', '')),
                                            'lotEchantillonsDesc': lot.get('lotEchantillonsDesc', ''),
                                            
                                            # Contact
                                            'administratifName': ao.get('administratifName', ''),
                                            'administratifEmail': ao.get('administratifEmail', ''),
                                            'administratifTel': ao.get('administratifTel', '')
                                        }
                                        
                                        all_lots.append(lot_record)
                                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Erreur JSON ligne {line_num}: {str(e)}")
                except Exception as e:
                    print(f"⚠️ Erreur traitement ligne {line_num}: {str(e)}")
    
    except FileNotFoundError:
        print(f"❌ Fichier '{input_file}' non trouvé!")
        return False
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {str(e)}")
        return False
    
    print(f"✅ {len(all_lots)} lots extraits de {processed_lines}/{total_lines} lignes")
    
    # Générer le CSV avec pandas
    print("📝 Génération du fichier CSV détaillé...")
    try:
        df = pd.DataFrame(all_lots)
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=',', quotechar='"')
        print(f"🎉 Fichier CSV détaillé généré: {output_file}")
        print(f"📊 Statistiques: {len(all_lots)} lots, {len(df.columns)} colonnes")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération CSV: {str(e)}")
        return False

def main():
    """
    Fonction principale du script
    """
    print("🚀 Script d'organisation des données Safakat")
    print("=" * 50)
    
    # Fichier d'entrée
    input_file = 'Safakat Test 1000 AOs.csv'
    
    # Vérifier l'existence du fichier
    if not os.path.exists(input_file):
        print(f"❌ Le fichier '{input_file}' n'existe pas!")
        print("📝 Assurez-vous que le fichier est dans le même répertoire que ce script.")
        return
    
    # Demander à l'utilisateur quel format il souhaite
    print("\nChoisissez le format de sortie:")
    print("1. Format compact (une ligne par AO)")
    print("2. Format détaillé (une ligne par lot)")
    print("3. Les deux formats")
    
    try:
        choice = input("\nVotre choix (1, 2 ou 3): ").strip()
        
        if choice in ['1', '3']:
            print("\n" + "="*50)
            print("📊 GÉNÉRATION DU FORMAT COMPACT")
            print("="*50)
            success1 = organize_safakat_data_compact(input_file)
            
        if choice in ['2', '3']:
            print("\n" + "="*50)
            print("📊 GÉNÉRATION DU FORMAT DÉTAILLÉ")
            print("="*50)
            success2 = organize_safakat_data_detailed(input_file)
        
        if choice in ['1', '2', '3']:
            print("\n🏆 Traitement terminé!")
            print("📁 Vérifiez les fichiers CSV générés dans le répertoire courant.")
        else:
            print("❌ Choix invalide. Veuillez relancer le script.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Script interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {str(e)}")

if __name__ == "__main__":
    main()