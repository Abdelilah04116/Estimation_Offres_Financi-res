"""
Pipeline ML Avancé pour les Appels d'Offres
Intègre : feature engineering, hyperparameter tuning, ensembles, évaluation complète
"""

import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import des modules personnalisés
from script import AppelOffreGenerator
from advanced_feature_engineering import AdvancedFeatureEngineer
from advanced_model_trainer import AdvancedModelTrainer
from model_evaluation_dashboard import ModelEvaluationDashboard

class AdvancedMLPipeline:
    """Pipeline ML avancé complet"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.setup_logging()
        self.results = {}
        
    def get_default_config(self):
        """Configuration par défaut optimisée"""
        return {
            'data_generation': {
                'n_aos': 20000,  # Nombre d'appels d'offres
                'locale': 'fr_FR'
            },
            'feature_engineering': {
                'n_features': 50,  # Nombre de features à sélectionner
                'use_advanced_features': True
            },
            'model_training': {
                'n_trials': 100,  # Nombre d'essais pour l'optimisation
                'test_size': 0.2,
                'random_state': 42
            },
            'evaluation': {
                'create_visualizations': True,
                'create_interactive_dashboard': True,
                'save_models': True
            }
        }
    
    def setup_logging(self):
        """Configure le système de logging"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{log_dir}/advanced_pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline ML avance initialise")
    
    def step_1_generate_data(self):
        """Étape 1: Génération de données"""
        self.logger.info("=" * 60)
        self.logger.info("ETAPE 1: GENERATION DE DONNEES")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Générer les données
            generator = AppelOffreGenerator()
            df_json, df_csv = generator.generer_dataset(
                self.config['data_generation']['n_aos']
            )
            
            # Créer le dataset de soumissions
            df = pd.DataFrame(df_csv)
            
            # Préparer les données pour l'analyse
            df = self.prepare_submission_dataset(df)
            
            elapsed_time = time.time() - start_time
            
            self.results['data_generation'] = {
                'status': 'success',
                'n_samples': len(df),
                'n_features': len(df.columns),
                'elapsed_time': elapsed_time
            }
            
            self.logger.info(f"Donnees generees: {len(df)} echantillons en {elapsed_time:.2f}s")
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la generation de donnees: {e}")
            raise
    
    def prepare_submission_dataset(self, df):
        """Prépare le dataset au niveau soumission"""
        print("🔧 Préparation du dataset de soumissions...")
        
        # Créer des soumissions à partir des appels d'offres
        submissions = []
        
        for _, ao in df.iterrows():
            # Extraire les soumissionnaires et montants
            soumissionnaires = ao['soumissionnaires'].split('; ') if isinstance(ao['soumissionnaires'], str) else []
            montants = ao['montants_soumis'].split('; ') if isinstance(ao['montants_soumis'], str) else []
            
            # Créer une soumission pour chaque soumissionnaire
            for i, soumissionnaire in enumerate(soumissionnaires):
                if i < len(montants):
                    montant = float(montants[i].replace(' DH', '').replace(',', '')) if montants[i] else 0
                else:
                    montant = 0
                
                # Déterminer si c'est le gagnant
                is_winner = 1 if soumissionnaire == ao['soumissionnaire_gagnant'] else 0
                
                submission = {
                    'id_appel_offre': ao['id_appel_offre'],
                    'soumissionnaire': soumissionnaire,
                    'montant_soumis': montant,
                    'is_winner': is_winner,
                    'budget_estime': ao['budget_estime'],
                    'montant_gagnant': ao['montant_gagnant'],
                    'montant_moyen': ao['montant_moyen'],
                    'nombre_soumissionnaires': ao['nombre_soumissionnaires'],
                    'experience_gagnant': ao['experience_gagnant'],
                    'notation_technique_gagnant': ao['notation_technique_gagnant'],
                    'delai_execution': ao['delai_execution'],
                    'secteur': ao['secteur'],
                    'type_procedure': ao['type_procedure'],
                    'critere_attribution': ao['critere_attribution'],
                    'complexite_projet': ao['complexite_projet'],
                    'statut_ao': ao['statut_ao'],
                    'date_publication': ao['date_publication'],
                    'date_limite': ao['date_limite'],
                    'date_resultat': ao['date_resultat'],
                    'ville': ao['ville'],
                    'organisme_emetteur': ao['organisme_emetteur']
                }
                submissions.append(submission)
        
        df_submissions = pd.DataFrame(submissions)
        print(f"✅ {len(df_submissions)} soumissions créées")
        
        return df_submissions
    
    def step_2_feature_engineering(self, df):
        """Étape 2: Feature Engineering avancé"""
        self.logger.info("=" * 60)
        self.logger.info("ETAPE 2: FEATURE ENGINEERING AVANCE")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialiser le feature engineer
            feature_engineer = AdvancedFeatureEngineer()
            
            # Appliquer l'ingénierie de features
            df_engineered, selected_features = feature_engineer.run_complete_engineering(
                df, target_col='is_winner'
            )
            
            elapsed_time = time.time() - start_time
            
            self.results['feature_engineering'] = {
                'status': 'success',
                'n_features': len(selected_features),
                'selected_features': selected_features,
                'elapsed_time': elapsed_time
            }
            
            self.logger.info(f"Features creees: {len(selected_features)} features en {elapsed_time:.2f}s")
            return df_engineered, selected_features
            
        except Exception as e:
            self.logger.error(f"Erreur lors du feature engineering: {e}")
            raise
    
    def step_3_model_training(self, df_engineered, selected_features):
        """Étape 3: Entraînement des modèles"""
        self.logger.info("=" * 60)
        self.logger.info("ETAPE 3: ENTRAINEMENT DES MODELES")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialiser le trainer
            trainer = AdvancedModelTrainer(random_state=self.config['model_training']['random_state'])
            
            # Préparer les données
            X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(
                df_engineered, 
                target_col='is_winner',
                test_size=self.config['model_training']['test_size']
            )
            
            # Entraîner les modèles de classification
            classification_results = trainer.train_models(
                X_train, y_train, X_test, y_test, task='classification'
            )
            
            # Créer un ensemble
            ensemble, ensemble_score = trainer.create_ensemble(
                X_train, y_train, X_test, y_test, task='classification'
            )
            
            # Évaluer les modèles
            evaluation_results = trainer.evaluate_models(X_test, y_test, task='classification')
            
            # Sauvegarder les modèles
            if self.config['evaluation']['save_models']:
                trainer.save_models()
            
            elapsed_time = time.time() - start_time
            
            self.results['model_training'] = {
                'status': 'success',
                'classification_results': classification_results,
                'ensemble_score': ensemble_score,
                'evaluation_results': evaluation_results,
                'elapsed_time': elapsed_time
            }
            
            self.logger.info(f"Entrainement termine en {elapsed_time:.2f}s")
            return trainer, evaluation_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entrainement: {e}")
            raise
    
    def step_4_evaluation(self, training_results):
        """Étape 4: Évaluation complète"""
        self.logger.info("=" * 60)
        self.logger.info("ETAPE 4: EVALUATION COMPLETE")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Créer le dashboard d'évaluation
            dashboard = ModelEvaluationDashboard(
                models_data=training_results,
                X_test=None,  # Sera défini plus tard
                y_test=None    # Sera défini plus tard
            )
            
            # Générer les rapports
            evaluation_report = dashboard.generate_comprehensive_report()
            
            # Créer les visualisations
            if self.config['evaluation']['create_visualizations']:
                dashboard.create_all_visualizations()
            
            elapsed_time = time.time() - start_time
            
            self.results['evaluation'] = {
                'status': 'success',
                'evaluation_report': evaluation_report,
                'elapsed_time': elapsed_time
            }
            
            self.logger.info(f"Evaluation terminee en {elapsed_time:.2f}s")
            return evaluation_report
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'evaluation: {e}")
            raise
    
    def step_5_save_results(self, training_results):
        """Étape 5: Sauvegarde des résultats"""
        self.logger.info("=" * 60)
        self.logger.info("ETAPE 5: SAUVEGARDE DES RESULTATS")
        self.logger.info("=" * 60)
        
        try:
            # Créer le dossier de résultats
            results_dir = 'results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Sauvegarder les résultats en JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'{results_dir}/pipeline_results_{timestamp}.json'
            
            # Convertir les résultats en format JSON-serializable
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = str(value)
            
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Resultats sauvegardes: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def run_complete_pipeline(self):
        """Exécute le pipeline complet"""
        self.logger.info("DEMARRAGE DU PIPELINE ML AVANCE COMPLET")
        self.logger.info("=" * 80)
        
        total_start_time = time.time()
        
        try:
            # Étape 1: Génération de données
            df = self.step_1_generate_data()
            
            # Étape 2: Feature Engineering
            df_engineered, selected_features = self.step_2_feature_engineering(df)
            
            # Étape 3: Entraînement des modèles
            trainer, evaluation_results = self.step_3_model_training(df_engineered, selected_features)
            
            # Étape 4: Évaluation
            evaluation_report = self.step_4_evaluation(evaluation_results)
            
            # Étape 5: Sauvegarde
            self.step_5_save_results(evaluation_results)
            
            # Résumé final
            total_time = time.time() - total_start_time
            self.display_summary(evaluation_report)
            
            self.logger.info("=" * 80)
            self.logger.info(f"PIPELINE TERMINE AVEC SUCCES - Temps total: {total_time:.2f}s")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            total_time = time.time() - total_start_time
            self.logger.error(f"ERREUR CRITIQUE DANS LE PIPELINE: {e}")
            self.logger.error(f"Temps ecoule: {total_time:.2f} secondes")
            self.logger.error("")
            self.logger.error("🎯 Résultat final: ÉCHEC")
            self.logger.error(f"⏱️ Temps total: {total_time:.2f} secondes")
            return False
    
    def display_summary(self, evaluation_report):
        """Affiche un résumé des résultats"""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DES PERFORMANCES")
        print("=" * 80)
        
        if 'classification_results' in self.results.get('model_training', {}):
            results = self.results['model_training']['classification_results']
            
            print("\n🏆 TOP 3 MODÈLES DE CLASSIFICATION:")
            sorted_models = sorted(results.items(), key=lambda x: x[1]['test_score'], reverse=True)
            
            for i, (name, result) in enumerate(sorted_models[:3], 1):
                print(f"  {i}. {name.upper()}: {result['test_score']:.4f}")
        
        if 'ensemble_score' in self.results.get('model_training', {}):
            ensemble_score = self.results['model_training']['ensemble_score']
            print(f"\n🤝 ENSEMBLE: {ensemble_score:.4f}")
        
        print("\n✅ Pipeline exécuté avec succès!")
        print("📁 Résultats sauvegardés dans le dossier 'results/'")
        print("🎯 Modèles sauvegardés dans le dossier 'models/'")

def main():
    """Fonction principale"""
    # Créer et exécuter le pipeline
    pipeline = AdvancedMLPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Pipeline terminé avec succès!")
    else:
        print("\n❌ Pipeline échoué!")

if __name__ == "__main__":
    main() 