"""
Module d'Entraînement Avancé pour les Modèles ML
Inclut : hyperparameter tuning, validation croisée, ensembles, early stopping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                            mean_squared_error, r2_score, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """Classe pour l'entraînement avancé des modèles"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def prepare_data(self, df, target_col='is_winner', test_size=0.2):
        """Prépare les données pour l'entraînement"""
        print("📊 Préparation des données...")
        
        # Séparer features et target
        feature_cols = [col for col in df.columns if col not in [target_col, 'id_appel_offre']]
        X = df[feature_cols].copy()
        
        # Nettoyer les données - convertir les colonnes non numériques
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                except:
                    # Si la conversion échoue, exclure la colonne
                    X = X.drop(columns=[col])
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                # Exclure les colonnes de dates
                X = X.drop(columns=[col])
        
        X = X.fillna(0)
        y = df[target_col]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Gérer l'imbalance des classes
        if target_col == 'is_winner':
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"✅ Données équilibrées: {len(y_train)} -> {len(y_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        return X_train_balanced, X_test, y_train_balanced, y_test, X.columns.tolist()
    
    def create_classification_models(self):
        """Crée les modèles de classification"""
        models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'xgboost': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'lightgbm': LGBMClassifier(random_state=self.random_state, verbose=-1),
            'catboost': CatBoostClassifier(random_state=self.random_state, verbose=False),
            'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'svm': SVC(random_state=self.random_state, probability=True),
            'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000)
        }
        return models
    
    def create_regression_models(self):
        """Crée les modèles de régression"""
        models = {
            'random_forest': RandomForestRegressor(random_state=self.random_state),
            'xgboost': XGBRegressor(random_state=self.random_state),
            'lightgbm': LGBMRegressor(random_state=self.random_state, verbose=-1),
            'catboost': CatBoostRegressor(random_state=self.random_state, verbose=False),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic': ElasticNet(random_state=self.random_state),
            'svr': SVR(),
            'mlp': MLPRegressor(random_state=self.random_state, max_iter=1000)
        }
        return models
    
    def optimize_hyperparameters(self, model_name, model, X_train, y_train, task='classification', n_trials=100):
        """Optimise les hyperparamètres avec Optuna"""
        print(f"🎯 Optimisation des hyperparamètres pour {model_name}...")
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
                }
            elif model_name == 'logistic':
                params = {
                    'C': trial.suggest_float('C', 0.1, 10, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'liblinear'
                }
            elif model_name == 'mlp':
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                        [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True)
                }
            else:
                return 0.0
            
            # Validation croisée
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model_cv = type(model)(**params)
                model_cv.fit(X_cv_train, y_cv_train)
                
                if task == 'classification':
                    y_pred_proba = model_cv.predict_proba(X_cv_val)[:, 1]
                    score = roc_auc_score(y_cv_val, y_pred_proba)
                else:
                    y_pred = model_cv.predict(X_cv_val)
                    score = r2_score(y_cv_val, y_pred)
                
                scores.append(score)
            
            return np.mean(scores)
        
        # Optimisation
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
    
    def train_models(self, X_train, y_train, X_test, y_test, task='classification'):
        """Entraîne tous les modèles avec optimisation"""
        print(f"🚀 Entraînement des modèles de {task}...")
        
        if task == 'classification':
            models = self.create_classification_models()
        else:
            models = self.create_regression_models()
        
        results = {}
        
        for name, model in models.items():
            print(f"\n📈 Entraînement de {name}...")
            
            # Optimisation des hyperparamètres
            best_params, best_score = self.optimize_hyperparameters(
                name, model, X_train, y_train, task, n_trials=50
            )
            
            # Entraînement avec les meilleurs paramètres
            best_model = type(model)(**best_params, random_state=self.random_state)
            best_model.fit(X_train, y_train)
            
            # Évaluation
            if task == 'classification':
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                y_pred = best_model.predict(X_test)
                test_score = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred = best_model.predict(X_test)
                test_score = r2_score(y_test, y_pred)
            
            # Sauvegarder les résultats
            results[name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': best_score,
                'test_score': test_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba if task == 'classification' else None
            }
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[name] = best_model.feature_importances_
            
            print(f"✅ {name}: CV={best_score:.4f}, Test={test_score:.4f}")
        
        self.models[task] = results
        return results
    
    def create_ensemble(self, X_train, y_train, X_test, y_test, task='classification'):
        """Crée un ensemble de modèles"""
        print(f"🤝 Création de l'ensemble de {task}...")
        
        if task not in self.models:
            print("❌ Modèles non entraînés. Exécutez d'abord train_models()")
            return None
        
        # Sélectionner les meilleurs modèles
        model_results = self.models[task]
        best_models = []
        
        for name, result in model_results.items():
            if result['test_score'] > 0.7:  # Seuil de performance
                best_models.append((name, result['model']))
        
        if len(best_models) < 2:
            print("⚠️ Pas assez de modèles performants pour l'ensemble")
            return None
        
        # Créer l'ensemble
        if task == 'classification':
            ensemble = VotingClassifier(
                estimators=best_models,
                voting='soft'
            )
        else:
            ensemble = VotingRegressor(
                estimators=best_models
            )
        
        # Entraîner l'ensemble
        ensemble.fit(X_train, y_train)
        
        # Évaluer
        if task == 'classification':
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
            y_pred = ensemble.predict(X_test)
            ensemble_score = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred = ensemble.predict(X_test)
            ensemble_score = r2_score(y_test, y_pred)
        
        print(f"✅ Ensemble: {ensemble_score:.4f}")
        
        return ensemble, ensemble_score
    
    def evaluate_models(self, X_test, y_test, task='classification'):
        """Évalue tous les modèles en détail"""
        print(f"📊 Évaluation détaillée des modèles de {task}...")
        
        if task not in self.models:
            print("❌ Modèles non entraînés")
            return
        
        evaluation_results = {}
        
        for name, result in self.models[task].items():
            model = result['model']
            y_pred = result['predictions']
            y_pred_proba = result.get('probabilities')
            
            # Métriques de base
            if task == 'classification':
                metrics = {
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'accuracy': (y_pred == y_test).mean(),
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
            else:
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred)
                }
            
            evaluation_results[name] = metrics
            
            print(f"\n📈 {name}:")
            for metric, value in metrics.items():
                if metric not in ['classification_report', 'confusion_matrix']:
                    print(f"  {metric}: {value:.4f}")
        
        return evaluation_results
    
    def save_models(self, filepath_prefix='models/'):
        """Sauvegarde tous les modèles"""
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        
        print("💾 Sauvegarde des modèles...")
        
        for task, models in self.models.items():
            for name, result in models.items():
                filename = f"{filepath_prefix}{task}_{name}.pkl"
                joblib.dump(result['model'], filename)
                print(f"✅ {filename}")
        
        # Sauvegarder les métadonnées
        metadata = {
            'best_params': {task: {name: result['best_params'] 
                                 for name, result in models.items()}
                           for task, models in self.models.items()},
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(metadata, f"{filepath_prefix}metadata.pkl")
        print("✅ metadata.pkl")
    
    def load_models(self, filepath_prefix='models/'):
        """Charge les modèles sauvegardés"""
        print("📂 Chargement des modèles...")
        
        import glob
        model_files = glob.glob(f"{filepath_prefix}*.pkl")
        
        for file in model_files:
            if 'metadata' not in file:
                model = joblib.load(file)
                task_name = file.split('/')[-1].replace('.pkl', '')
                task, name = task_name.split('_', 1)
                
                if task not in self.models:
                    self.models[task] = {}
                
                self.models[task][name] = {'model': model}
        
        # Charger les métadonnées
        try:
            metadata = joblib.load(f"{filepath_prefix}metadata.pkl")
            self.best_params = metadata['best_params']
            self.feature_importance = metadata['feature_importance']
        except:
            print("⚠️ Métadonnées non trouvées")
        
        print("✅ Modèles chargés")

if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module d'entraînement avancé...") 