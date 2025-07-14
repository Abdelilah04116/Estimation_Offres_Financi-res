#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outil de diagnostic pour fichiers CSV problématiques
"""

import os
import pandas as pd
import json
import re

def analyze_csv_structure(file_path: str):
    """
    Analyse la structure d'un fichier CSV problématique
    """
    print("🔍 ANALYSE DÉTAILLÉE DU FICHIER CSV")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return
    
    # Informations basiques du fichier
    file_size = os.path.getsize(file_path)
    print(f"📏 Taille du fichier: {file_size:,} octets ({file_size/1024/1024:.2f} MB)")
    
    # Essayer différents encodages
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"✅ Encodage détecté: {encoding}")
            break
        except:
            continue
    
    if not content:
        print("❌ Impossible de lire le fichier avec les encodages standards")
        return
    
    # Analyser le contenu
    lines = content.split('\n')
    print(f"📄 Nombre de lignes: {len(lines)}")
    
    # Analyser les premiers caractères
    print(f"🔤 Premiers caractères: {repr(content[:100])}")
    
    # Analyser les séparateurs
    separators = [',', ';', '\t', '|']
    print("\n🔧 ANALYSE DES SÉPARATEURS:")
    for sep in separators:
        count = content.count(sep)
        avg_per_line = count / len(lines) if lines else 0
        print(f"  {repr(sep)}: {count:,} occurrences (moy: {avg_per_line:.1f}/ligne)")
    
    # Analyser les premières lignes
    print("\n📝 PREMIÈRES LIGNES:")
    for i, line in enumerate(lines[:5]):
        print(f"Ligne {i+1} ({len(line)} chars): {line[:150]}...")
    
    # Détecter des patterns JSON
    json_patterns = [
        r'\{[^}]*\}',  # Objets JSON simples
        r'\[[^\]]*\]', # Tableaux JSON
        r'"[^"]*":\s*[^,}]+',  # Paires clé-valeur
    ]
    
    print("\n🔍 DÉTECTION DE STRUCTURES JSON:")
    for pattern in json_patterns:
        matches = re.findall(pattern, content[:10000])  # Analyser les premiers 10K caractères
        print(f"  Pattern {pattern}: {len(matches)} correspondances")
        if matches:
            print(f"    Exemple: {matches[0][:100]}...")
    
    return content

def attempt_csv_parsing(file_path: str):
    """
    Tente différentes méthodes de parsing CSV
    """
    print("\n🧪 TESTS DE PARSING CSV:")
    print("-" * 30)
    
    # Configuration de test
    configs = [
        {'sep': ',', 'encoding': 'utf-8', 'quotechar': '"'},
        {'sep': ';', 'encoding': 'utf-8', 'quotechar': '"'},
        {'sep': ',', 'encoding': 'utf-8', 'quotechar': "'"},
        {'sep': ',', 'encoding': 'latin-1', 'quotechar': '"'},
        {'sep': '\t', 'encoding': 'utf-8', 'quotechar': '"'},
    ]
    
    best_result = None
    best_config = None
    
    for i, config in enumerate(configs, 1):
        try:
            print(f"Test {i}: {config}")
            df = pd.read_csv(file_path, **config, nrows=10, on_bad_lines='skip')
            print(f"  ✅ Succès: {len(df)} lignes, {len(df.columns)} colonnes")
            
            if best_result is None or len(df.columns) > len(best_result.columns):
                best_result = df
                best_config = config
                
        except Exception as e:
            print(f"  ❌ Échec: {str(e)[:100]}")
    
    if best_result is not None:
        print(f"\n🏆 MEILLEURE CONFIGURATION: {best_config}")
        print(f"📊 Résultat: {len(best_result)} lignes, {len(best_result.columns)} colonnes")
        print("\n📋 Aperçu des colonnes:")
        for i, col in enumerate(best_result.columns[:10]):
            print(f"  {i+1}. {col}")
        
        return best_result, best_config
    
    return None, None

def extract_json_data(content: str):
    """
    Essaie d'extraire des données JSON du contenu
    """
    print("\n🔬 EXTRACTION DE DONNÉES JSON:")
    print("-" * 30)
    
    # Chercher des structures JSON complètes
    json_objects = []
    
    # Pattern pour objets JSON complets
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, content)
    
    print(f"🔍 {len(matches)} objets JSON potentiels trouvés")
    
    valid_objects = []
    for i, match in enumerate(matches[:10]):  # Tester les 10 premiers
        try:
            obj = json.loads(match)
            valid_objects.append(obj)
            print(f"  ✅ Objet {i+1}: Valide - {len(obj)} clés")
        except:
            print(f"  ❌ Objet {i+1}: JSON invalide")
    
    if valid_objects:
        print(f"\n📊 {len(valid_objects)} objets JSON valides extraits")
        
        # Analyser les clés communes
        all_keys = set()
        for obj in valid_objects:
            if isinstance(obj, dict):
                all_keys.update(obj.keys())
        
        print(f"🔑 Clés trouvées ({len(all_keys)}): {list(all_keys)[:10]}...")
        
        return valid_objects
    
    return []

def create_clean_csv(file_path: str, output_path: str = None):
    """
    Crée un CSV propre à partir du fichier problématique
    """
    print("\n🧹 CRÉATION D'UN CSV PROPRE:")
    print("-" * 30)
    
    if output_path is None:
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}_clean.csv"
    
    try:
        # Lire le contenu brut
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Essayer d'extraire des données JSON
        json_objects = extract_json_data(content)
        
        if json_objects:
            # Convertir en DataFrame
            df = pd.DataFrame(json_objects)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ CSV propre créé: {output_path}")
            print(f"📊 {len(df)} lignes, {len(df.columns)} colonnes")
            return output_path
        else:
            print("❌ Impossible d'extraire des données structurées")
            return None
            
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None

def main():
    """
    Fonction principale de diagnostic
    """
    print("🎯 OUTIL DE DIAGNOSTIC CSV")
    print("=" * 50)
    
    filename = "Safakat Test 1000 AOs.csv"
    
    if not os.path.exists(filename):
        print(f"❌ Fichier non trouvé: {filename}")
        print("Placez le fichier dans le même dossier que ce script.")
        return
    
    # 1. Analyser la structure
    content = analyze_csv_structure(filename)
    
    # 2. Tester différents parsings CSV
    best_df, best_config = attempt_csv_parsing(filename)
    
    # 3. Extraire des données JSON si possible
    if content:
        json_data = extract_json_data(content)
        
        if json_data:
            # 4. Créer un CSV propre
            clean_file = create_clean_csv(filename)
            
            if clean_file:
                print(f"\n💡 RECOMMANDATION:")
                print(f"Utilisez le fichier nettoyé: {clean_file}")
    
    print("\n🎯 Diagnostic terminé!")

if __name__ == "__main__":
    main()