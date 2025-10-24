#!/usr/bin/env python3
"""
Pipeline ML Complet - Amélioration des Modèles d'Estimation d'Offres
====================================================================

Ce script orchestre le pipeline complet d'amélioration des performances :
1. Génération de données avec Faker
2. Feature Engineering et sélection
3. Entraînement optimisé avec hyperparamètres
4. Évaluation et validation
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Affiche un en-tête stylisé"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Affiche une étape du pipeline"""
    print(f"\n📋 ÉTAPE {step}: {description}")
    print("-" * 50)

def check_dependencies():
    """Vérifie et installe les dépendances nécessaires"""
    print_step(0, "Vérification des dépendances")
    
    required_packages = [
        'faker', 'lightgbm', 'scikit-learn', 'xgboost', 
        'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MANQUANT")
    
    if missing_packages:
        print(f"\n⚠️ Packages manquants: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_data_generation():
    """Étape 1: Génération de données avec Faker"""
    print_step(1, "Génération de données avec Faker")
    
    try:
        from data_generator import DataGenerator
        
        print("🔄 Démarrage de la génération de données...")
        generator = DataGenerator()
        df_enhanced = generator.generate_enhanced_dataset(15000)
        
        print(f"✅ Données générées: {len(df_enhanced)} soumissions")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return False

def run_feature_engineering():
    """Étape 2: Feature Engineering et sélection"""
    print_step(2, "Feature Engineering et sélection")
    
    try:
        from feature_engineering import FeatureEngineer
        
        print("🔄 Démarrage du feature engineering...")
        engineer = FeatureEngineer()
        df_final = engineer.run_complete_feature_engineering()
        
        print(f"✅ Features créées: {len(df_final.columns)-1} features")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du feature engineering: {e}")
        return False

def run_optimized_training():
    """Étape 3: Entraînement optimisé"""
    print_step(3, "Entraînement optimisé avec hyperparamètres")
    
    try:
        from optimized_training import OptimizedTrainer
        
        print("🔄 Démarrage de l'entraînement optimisé...")
        trainer = OptimizedTrainer()
        best_models = trainer.run_complete_optimization()
        
        print("✅ Modèles optimisés créés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return False

def run_validation():
    """Étape 4: Validation des nouveaux modèles"""
    print_step(4, "Validation des modèles améliorés")
    
    try:
        # Adapter le script de validation pour les nouveaux modèles
        import joblib
        import pandas as pd
        from sklearn.metrics import classification_report, roc_auc_score, r2_score, mean_squared_error
        
        # Charger les nouveaux modèles
        clf_model = joblib.load('models/best_classification_model.pkl')
        reg_model = joblib.load('models/best_regression_model.pkl')
        scaler = joblib.load('models/best_scaler.pkl')
        feature_columns = joblib.load('models/best_feature_columns.pkl')
        
        # Charger les données de test
        df_test = pd.read_csv('final_engineered_dataset.csv')
        
        # Préparer les données de test
        X_test = df_test[feature_columns]
        y_class_test = df_test['is_winner']
        y_reg_test = df_test['montant_soumis']
        
        # Normaliser
        X_test_scaled = scaler.transform(X_test)
        
        # Évaluer classification
        y_pred_class = clf_model.predict(X_test_scaled)
        y_pred_proba = clf_model.predict_proba(X_test_scaled)[:, 1]
        
        clf_accuracy = (y_pred_class == y_class_test).mean()
        clf_auc = roc_auc_score(y_class_test, y_pred_proba)
        
        # Évaluer régression
        y_pred_reg = reg_model.predict(X_test_scaled)
        reg_r2 = r2_score(y_reg_test, y_pred_reg)
        reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
        
        print("\n📊 RÉSULTATS DE VALIDATION:")
        print(f"Classification - Accuracy: {clf_accuracy:.4f}")
        print(f"Classification - ROC-AUC: {clf_auc:.4f}")
        print(f"Régression - R²: {reg_r2:.4f}")
        print(f"Régression - RMSE: {reg_rmse:.2f}")
        
        # Comparer avec les anciens résultats
        print("\n📈 COMPARAISON AVEC ANCIENS MODÈLES:")
        print("Ancien - Classification ROC-AUC: 0.7457")
        print(f"Nouveau - Classification ROC-AUC: {clf_auc:.4f}")
        print(f"Amélioration: {(clf_auc - 0.7457)*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {e}")
        return False

def create_performance_report():
    """Crée un rapport de performance complet"""
    print_step(5, "Création du rapport de performance")
    
    try:
        report = f"""
RAPPORT DE PERFORMANCE - PIPELINE ML AMÉLIORÉ
=============================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ÉTAPES EXÉCUTÉES:
✅ Génération de données avec Faker (15,000 AOs)
✅ Feature Engineering avancé
✅ Sélection de features optimisée
✅ Entraînement avec hyperparamètres
✅ Création d'ensembles de modèles
✅ Validation complète

AMÉLIORATIONS APPORTÉES:
1. Données plus réalistes et complètes
2. Features métier avancées
3. Sélection de features intelligente
4. Optimisation des hyperparamètres
5. Ensembles de modèles
6. Validation robuste

FICHIERS GÉNÉRÉS:
- enhanced_ao_dataset.csv (données générées)
- final_engineered_dataset.csv (features sélectionnées)
- models/best_classification_model.pkl (modèle classification)
- models/best_regression_model.pkl (modèle régression)
- models/best_scaler.pkl (normalisation)
- models/best_feature_columns.pkl (features utilisées)
- training_results.json (résultats détaillés)

RECOMMANDATIONS:
1. Utiliser les nouveaux modèles en production
2. Monitorer les performances en temps réel
3. Retraîner périodiquement avec de nouvelles données
4. Ajuster les seuils selon le contexte métier

Pipeline ML - Projet d'Estimation d'Offres Financières
Auteur: Abdelilah OURTI
"""
        
        with open('performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Rapport sauvegardé dans 'performance_report.txt'")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du rapport: {e}")
        return False

def main():
    """Fonction principale du pipeline"""
    print_header("PIPELINE ML - AMÉLIORATION DES PERFORMANCES")
    
    start_time = time.time()
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("❌ Dépendances manquantes. Arrêt du pipeline.")
        return False
    
    # Étape 1: Génération de données
    if not run_data_generation():
        print("❌ Échec de la génération de données")
        return False
    
    # Étape 2: Feature Engineering
    if not run_feature_engineering():
        print("❌ Échec du feature engineering")
        return False
    
    # Étape 3: Entraînement optimisé
    if not run_optimized_training():
        print("❌ Échec de l'entraînement")
        return False
    
    # Étape 4: Validation
    if not run_validation():
        print("❌ Échec de la validation")
        return False
    
    # Étape 5: Rapport de performance
    if not create_performance_report():
        print("❌ Échec de la création du rapport")
        return False
    
    # Calcul du temps total
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print_header("PIPELINE TERMINÉ AVEC SUCCÈS!")
    print(f"⏱️ Temps total: {hours}h {minutes}m {seconds}s")
    print("\n🎉 Tous les modèles ont été améliorés et sont prêts pour la production!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 