
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Python pour extraire les données pertinentes de Safakat dans un CSV clair
"""

import csv
import re
import sys
import os
import json
import pandas as pd

def clean_unicode_chars(text):
    """Nettoie les caractères Unicode problématiques"""
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('°', 'deg').replace('€', 'EUR').replace('…', '...')
    return text

def extract_field_exact(text, field_name):
    """Extrait un champ texte avec patterns multiples"""
    patterns = [
        rf"{field_name}'\":\s*'\"([^\"]*?)\"'",  # fieldName'": '"value"'
        rf"'{field_name}':\s*'\"([^\"]*?)\"'",   # 'fieldName': '"value"'
        rf'"{field_name}":\s*"([^"]*)"',
        rf'{field_name}":\s*"([^"]*)"',
        rf'{field_name}[^"]*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            value = match.group(1).strip().replace('\\"', '"').replace("\\'", "'")
            if len(value) > 2:
                return value
    return ""

def extract_numeric_exact(text, field_name):
    """Extrait une valeur numérique"""
    patterns = [
        rf"{field_name}'\":\s*([0-9.]+)",  # fieldName'": number
        rf"'{field_name}':\s*([0-9.]+)",
        rf'"{field_name}":\s*([0-9.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None

def extract_list_exact(text, field_name):
    """Extrait une liste (ex. : provinces, domains, qualifications)"""
    patterns = [
        rf"{field_name}'\":\s*\[\s*([^\]]*)\]",  # fieldName'": [values]
        rf"'{field_name}':\s*\[\s*([^\]]*)\]",
        rf'"{field_name}":\s*\[\s*([^\]]*)\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match and match.group(1).strip():
            items = [item.strip().replace('"', '').replace("'", "") for item in match.group(1).split(',') if item.strip()]
            return items
    return []

def extract_online_results(text):
    """Extrait les soumissionnaires et leurs montants"""
    results = []
    pattern = r'\{[^}]*company[^}]*companyName[^}]*"([^"]+)"[^}]*priceAfterCorrection[^}]*([0-9.]+)[^}]*\}'
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        company_name = match.group(1).strip().replace('\\"', '"').replace("\\'", "'")
        price = float(match.group(2)) if match.group(2) else None
        if company_name and price and price > 0:
            results.append({"companyName": company_name, "priceAfterCorrection": price})
    return results

def extract_lots_and_ao(line):
    """Extrait les données pertinentes d'une ligne"""
    line = clean_unicode_chars(line)
    
    # Extraire les champs de l'AO
    ao_data = {
        'org': extract_field_exact(line, 'org'),
        'consId': extract_numeric_exact(line, 'consId'),
        'reference': extract_field_exact(line, 'reference'),
        'acheteur': extract_field_exact(line, 'acheteur'),
        'procedureType': extract_field_exact(line, 'procedureType'),
        'publishedDate': extract_field_exact(line, 'publishedDate'),
        'endDate': extract_field_exact(line, 'endDate'),
        'provinces': extract_list_exact(line, 'provinces'),
        'isConsCancelled': extract_field_exact(line, 'isConsCancelled') == 'true',
        'domains': extract_list_exact(line, 'domains'),
        'administratifName': extract_field_exact(line, 'administratifName'),
        'administratifEmail': extract_field_exact(line, 'administratifEmail')
    }
    
    # Trouver tous les lots
    lots = []
    lot_pattern = r'lotId[^0-9]*(\d+)'
    matches = re.finditer(lot_pattern, line)
    
    for match in matches:
        lot_id = int(match.group(1))
        position = match.start()
        section_start = max(0, position - 200)
        section_end = min(len(line), position + 3000)
        section = line[section_start:section_end]
        
        lot_data = {
            'lotId': lot_id,
            'lotObject': extract_field_exact(section, 'lotObject'),
            'lotCategory': extract_field_exact(section, 'lotCategory'),
            'lotEstimation': extract_numeric_exact(section, 'lotEstimation'),
            'finalPrice': extract_numeric_exact(section, 'finalPrice'),
            'winner': extract_field_exact(section, 'winner'),
            'lotCaution': extract_numeric_exact(section, 'lotCaution'),
            'lotReserve': extract_field_exact(section, 'lotReserve'),
            'isQualified': extract_field_exact(section, 'isQualified') == 'true',
            'lotResultStatus': extract_field_exact(section, 'lotResultStatus'),
            'juryDate': extract_field_exact(section, 'juryDate'),
            'qualifications': extract_list_exact(section, 'qualifications'),
            'onlineResults': extract_online_results(section)
        }
        lots.append(lot_data)
    
    return ao_data, lots

def process_safakat_file(input_file, output_file):
    """Traite le fichier Safakat et génère un CSV avec les données pertinentes"""
    print(f"🚀 Lecture du fichier: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Erreur lecture: {e}")
        return
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    print(f"📄 Lignes trouvées: {len(lines)}")
    
    # En-têtes CSV
    headers = [
        'ao_id', 'reference', 'acheteur', 'procedure_type', 'published_date', 'end_date',
        'provinces', 'is_cancelled', 'domains', 'administratif_name', 'administratif_email',
        'lot_id', 'lot_object', 'lot_category', 'lot_estimation', 'final_price', 'winner',
        'lot_caution', 'lot_reserve', 'is_qualified', 'lot_result_status', 'jury_date',
        'qualifications', 'soumissionnaires', 'montants_soumis', 'nombre_soumissionnaires'
    ]
    
    rows = []
    for i, line in enumerate(lines):
        try:
            print(f"⚙️ Traitement ligne {i+1}/{len(lines)}", end='\r')
            ao_data, lots = extract_lots_and_ao(line)
            
            for lot in lots:
                soumissionnaires = [res["companyName"] for res in lot['onlineResults']]
                montants_soumis = [res["priceAfterCorrection"] for res in lot['onlineResults']]
                row = {
                    'ao_id': ao_data['consId'],
                    'reference': ao_data['reference'],
                    'acheteur': ao_data['acheteur'],
                    'procedure_type': ao_data['procedureType'],
                    'published_date': ao_data['publishedDate'],
                    'end_date': ao_data['endDate'],
                    'provinces': ';'.join(ao_data['provinces']),
                    'is_cancelled': ao_data['isConsCancelled'],
                    'domains': ';'.join(ao_data['domains']),
                    'administratif_name': ao_data['administratifName'],
                    'administratif_email': ao_data['administratifEmail'],
                    'lot_id': lot['lotId'],
                    'lot_object': lot['lotObject'],
                    'lot_category': lot['lotCategory'],
                    'lot_estimation': lot['lotEstimation'],
                    'final_price': lot['finalPrice'],
                    'winner': lot['winner'],
                    'lot_caution': lot['lotCaution'],
                    'lot_reserve': lot['lotReserve'],
                    'is_qualified': lot['isQualified'],
                    'lot_result_status': lot['lotResultStatus'],
                    'jury_date': lot['juryDate'],
                    'qualifications': ';'.join(lot['qualifications']),
                    'soumissionnaires': ';'.join(soumissionnaires),
                    'montants_soumis': ';'.join([str(m) for m in montants_soumis]),
                    'nombre_soumissionnaires': len(soumissionnaires)
                }
                rows.append(row)
        except Exception as e:
            print(f"\n⚠️ Erreur ligne {i+1}: {e}")
            continue
    
    # Créer le DataFrame
    df = pd.DataFrame(rows)
    
    # Convertir les colonnes numériques
    numeric_cols = ['lot_estimation', 'final_price', 'lot_caution', 'nombre_soumissionnaires']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convertir les colonnes booléennes
    bool_cols = ['is_cancelled', 'is_qualified']
    for col in bool_cols:
        df[col] = df[col].astype(bool)
    
    # Sauvegarder dans un fichier CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n🎉 Données extraites et sauvegardées dans {output_file}")
    print(f"📊 Total lots extraits: {len(rows)}")

def main():
    print("🚀 EXTRACTEUR SAFAKAT - DONNÉES PERTINENTES")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <fichier_entree> [fichier_sortie]")
        print('Exemple: python script.py "Safakat Test 1000 AOs.csv"')
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'AO_reorganized.csv'
    
    if not os.path.exists(input_file):
        print(f'❌ Fichier "{input_file}" introuvable')
        sys.exit(1)
    
    print(f"📂 Fichier d'entrée: {input_file}")
    print(f"📄 Fichier de sortie: {output_file}")
    print("-" * 50)
    
    process_safakat_file(input_file, output_file)
    
    print("\n" + "=" * 50)
    print("✨ Extraction terminée!")

if __name__ == "__main__":
    main()
