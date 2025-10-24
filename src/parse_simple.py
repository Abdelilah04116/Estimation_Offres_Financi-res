#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parser simple de secours pour fichier CSV problématique
"""

import json
import re
import pandas as pd
from datetime import datetime

def simple_parse():
    """
    Parsing simple ligne par ligne
    """
    print("🔧 PARSER SIMPLE DE SECOURS")
    print("=" * 40)
    
    filename = "Safakat Test 1000 AOs.csv"
    
    try:
        # Lire comme fichier texte
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ Fichier lu: {len(content)} caractères")
        
        # Chercher tous les objets JSON-like
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(pattern, content)
        
        print(f"🔍 {len(json_matches)} objets trouvés")
        
        # Parser chaque objet
        consultations = []
        
        for i, match in enumerate(json_matches):
            if i % 10 == 0:
                print(f"   Traitement: {i}/{len(json_matches)}")
            
            try:
                # Nettoyer et parser
                clean_match = match.replace('\\"', '"').replace("\\'", "'")
                
                # Essayer de parser comme JSON
                try:
                    obj = json.loads(clean_match)
                except:
                    # Si ça échoue, essayer eval (plus risqué mais parfois nécessaire)
                    obj = eval(clean_match)
                
                # Extraire les infos principales
                if isinstance(obj, dict):
                    consultation = {}
                    
                    # Infos de base
                    consultation['org'] = obj.get('org', '')
                    consultation['consId'] = obj.get('consId', '')
                    consultation['reference'] = obj.get('reference', '')
                    consultation['acheteur'] = obj.get('acheteur', '')
                    consultation['procedureType'] = obj.get('procedureType', '')
                    
                    # Dates
                    consultation['publishedDate'] = obj.get('publishedDate', '')
                    consultation['endDate'] = obj.get('endDate', '')
                    
                    # Contact
                    consultation['administratifName'] = obj.get('administratifName', '')
                    consultation['administratifEmail'] = obj.get('administratifEmail', '')
                    consultation['administratifTel'] = obj.get('administratifTel', '')
                    
                    # Province
                    provinces = obj.get('provinces', [])
                    consultation['provinces'] = ', '.join(provinces) if isinstance(provinces, list) else str(provinces)
                    
                    # Lots
                    lots = obj.get('lots', [])
                    if lots and isinstance(lots, list):
                        lot = lots[0]
                        consultation['lotObject'] = lot.get('lotObject', '')
                        consultation['lotCategory'] = lot.get('lotCategory', '')
                        consultation['lotEstimation'] = lot.get('lotEstimation')
                        consultation['lotCaution'] = lot.get('lotCaution')
                    else:
                        consultation['lotObject'] = ''
                        consultation['lotCategory'] = ''
                        consultation['lotEstimation'] = None
                        consultation['lotCaution'] = None
                    
                    # Domaines
                    domains = obj.get('domains', [])
                    if domains and isinstance(domains, list):
                        domain_names = [d.get('domain', '') for d in domains if isinstance(d, dict)]
                        consultation['domains'] = ', '.join(set(domain_names))
                    else:
                        consultation['domains'] = ''
                    
                    consultations.append(consultation)
                    
            except Exception as e:
                continue
        
        # Créer DataFrame
        df = pd.DataFrame(consultations)
        
        # Supprimer doublons
        if 'consId' in df.columns:
            df = df.drop_duplicates(subset=['consId'])
        
        print(f"✅ {len(df)} consultations extraites")
        
        # Sauvegarder
        output_file = "consultations_extraites.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        excel_file = "consultations_extraites.xlsx"
        df.to_excel(excel_file, index=False)
        
        print(f"📊 Fichiers générés:")
        print(f"  • {output_file}")
        print(f"  • {excel_file}")
        
        # Statistiques rapides
        print(f"\n📈 STATISTIQUES:")
        print(f"  • Total consultations: {len(df)}")
        
        if 'acheteur' in df.columns:
            print(f"  • Acheteurs uniques: {df['acheteur'].nunique()}")
        
        if 'lotEstimation' in df.columns:
            estimations = df['lotEstimation'].dropna()
            if len(estimations) > 0:
                print(f"  • Estimation totale: {estimations.sum():,.2f}")
                print(f"  • Estimation moyenne: {estimations.mean():,.2f}")
        
        if 'lotCategory' in df.columns:
            print(f"\n📋 Top catégories:")
            top_cats = df['lotCategory'].value_counts().head(5)
            for cat, count in top_cats.items():
                print(f"  • {cat}: {count}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None

if __name__ == "__main__":
    simple_parse()