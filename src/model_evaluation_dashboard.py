"""
Dashboard d'Évaluation Complet des Modèles ML
Inclut : métriques détaillées, visualisations avancées, analyses par segment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                            classification_report, mean_squared_error, r2_score,
                            mean_absolute_error)
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationDashboard:
    """Dashboard complet d'évaluation des modèles"""
    
    def __init__(self, models_data, X_test, y_test, feature_names=None):
        self.models_data = models_data
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_test.shape[1])]
        self.results = {}
        
    def evaluate_classification_models(self):
        """Évalue tous les modèles de classification"""
        print("📊 Évaluation des modèles de classification...")
        
        classification_results = {}
        
        for name, model_data in self.models_data.items():
            if 'classification' in name.lower() or 'clf' in name.lower():
                model = model_data['model']
                y_pred = model_data['predictions']
                y_pred_proba = model_data.get('probabilities')
                
                # Métriques de base
                metrics = {
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                    'accuracy': (y_pred == self.y_test).mean(),
                    'precision': classification_report(self.y_test, y_pred, output_dict=True)['1']['precision'],
                    'recall': classification_report(self.y_test, y_pred, output_dict=True)['1']['recall'],
                    'f1_score': classification_report(self.y_test, y_pred, output_dict=True)['1']['f1-score']
                }
                
                # Courbes ROC et PR
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
                
                # Matrice de confusion
                cm = confusion_matrix(self.y_test, y_pred)
                
                classification_results[name] = {
                    'metrics': metrics,
                    'roc_curve': (fpr, tpr),
                    'pr_curve': (precision, recall),
                    'confusion_matrix': cm,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✅ {name}: AUC={metrics['roc_auc']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        self.results['classification'] = classification_results
        return classification_results
    
    def evaluate_regression_models(self):
        """Évalue tous les modèles de régression"""
        print("📊 Évaluation des modèles de régression...")
        
        regression_results = {}
        
        for name, model_data in self.models_data.items():
            if 'regression' in name.lower() or 'reg' in name.lower():
                model = model_data['model']
                y_pred = model_data['predictions']
                
                # Métriques de base
                metrics = {
                    'r2': r2_score(self.y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'mae': mean_absolute_error(self.y_test, y_pred),
                    'mse': mean_squared_error(self.y_test, y_pred)
                }
                
                # Résidus
                residuals = self.y_test - y_pred
                
                regression_results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'residuals': residuals,
                    'actual': self.y_test
                }
                
                print(f"✅ {name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        self.results['regression'] = regression_results
        return regression_results
    
    def create_classification_visualizations(self):
        """Crée les visualisations pour la classification"""
        print("📈 Création des visualisations de classification...")
        
        if 'classification' not in self.results:
            print("❌ Résultats de classification non disponibles")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Dashboard d\'Évaluation - Classification', fontsize=16, fontweight='bold')
        
        # 1. Courbes ROC
        ax = axes[0, 0]
        for name, result in self.results['classification'].items():
            fpr, tpr = result['roc_curve']
            auc = result['metrics']['roc_auc']
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbes ROC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Courbes Precision-Recall
        ax = axes[0, 1]
        for name, result in self.results['classification'].items():
            precision, recall = result['pr_curve']
            ax.plot(recall, precision, label=name)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Courbes Precision-Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Comparaison des métriques
        ax = axes[0, 2]
        metrics_df = pd.DataFrame([
            result['metrics'] for result in self.results['classification'].values()
        ], index=self.results['classification'].keys())
        
        metrics_df[['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']].plot(
            kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
        ax.set_title('Comparaison des Métriques')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 4. Matrices de confusion (premier modèle)
        ax = axes[1, 0]
        first_model = list(self.results['classification'].keys())[0]
        cm = self.results['classification'][first_model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Matrice de Confusion - {first_model}')
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Valeurs Réelles')
        
        # 5. Distribution des probabilités
        ax = axes[1, 1]
        for name, result in self.results['classification'].items():
            probs = result['probabilities']
            ax.hist(probs, bins=30, alpha=0.7, label=name, density=True)
        
        ax.set_xlabel('Probabilité Prédite')
        ax.set_ylabel('Densité')
        ax.set_title('Distribution des Probabilités')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Feature Importance (premier modèle)
        ax = axes[1, 2]
        first_model = list(self.results['classification'].keys())[0]
        model = self.models_data[first_model]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_features = np.argsort(importance)[-10:]
            
            ax.barh(range(len(top_features)), importance[top_features])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([self.feature_names[i] for i in top_features])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance - {first_model}')
        
        plt.tight_layout()
        plt.savefig('classification_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_regression_visualizations(self):
        """Crée les visualisations pour la régression"""
        print("📈 Création des visualisations de régression...")
        
        if 'regression' not in self.results:
            print("❌ Résultats de régression non disponibles")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Dashboard d\'Évaluation - Régression', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot prédictions vs réel
        ax = axes[0, 0]
        for name, result in self.results['regression'].items():
            ax.scatter(result['actual'], result['predictions'], alpha=0.6, label=name)
        
        # Ligne de régression parfaite
        min_val = min([min(result['actual']) for result in self.results['regression'].values()])
        max_val = max([max(result['actual']) for result in self.results['regression'].values()])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('Valeurs Réelles')
        ax.set_ylabel('Prédictions')
        ax.set_title('Prédictions vs Réel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distribution des résidus
        ax = axes[0, 1]
        for name, result in self.results['regression'].items():
            residuals = result['residuals']
            ax.hist(residuals, bins=30, alpha=0.7, label=name, density=True)
        
        ax.set_xlabel('Résidus')
        ax.set_ylabel('Densité')
        ax.set_title('Distribution des Résidus')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Comparaison des métriques
        ax = axes[0, 2]
        metrics_df = pd.DataFrame([
            result['metrics'] for result in self.results['regression'].values()
        ], index=self.results['regression'].keys())
        
        metrics_df[['r2', 'rmse', 'mae']].plot(
            kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        ax.set_title('Comparaison des Métriques')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plot des résidus (premier modèle)
        ax = axes[1, 0]
        first_model = list(self.results['regression'].keys())[0]
        residuals = self.results['regression'][first_model]['residuals']
        
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot des Résidus - {first_model}')
        ax.grid(True, alpha=0.3)
        
        # 5. Résidus vs Prédictions
        ax = axes[1, 1]
        for name, result in self.results['regression'].items():
            ax.scatter(result['predictions'], result['residuals'], alpha=0.6, label=name)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Résidus')
        ax.set_title('Résidus vs Prédictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Feature Importance (premier modèle)
        ax = axes[1, 2]
        first_model = list(self.results['regression'].keys())[0]
        model = self.models_data[first_model]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_features = np.argsort(importance)[-10:]
            
            ax.barh(range(len(top_features)), importance[top_features])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([self.feature_names[i] for i in top_features])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance - {first_model}')
        
        plt.tight_layout()
        plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_interactive_dashboard(self):
        """Crée un dashboard interactif avec Plotly"""
        print("🎨 Création du dashboard interactif...")
        
        # Dashboard de classification
        if 'classification' in self.results:
            fig_clf = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Courbes ROC', 'Precision-Recall', 'Métriques',
                              'Distribution Probabilités', 'Comparaison Modèles', 'Feature Importance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Courbes ROC
            for name, result in self.results['classification'].items():
                fpr, tpr = result['roc_curve']
                auc = result['metrics']['roc_auc']
                fig_clf.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={auc:.3f})', mode='lines'),
                    row=1, col=1
                )
            
            # Ligne de référence
            fig_clf.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', 
                          line=dict(dash='dash', color='black')),
                row=1, col=1
            )
            
            # Distribution des probabilités
            for name, result in self.results['classification'].items():
                probs = result['probabilities']
                fig_clf.add_trace(
                    go.Histogram(x=probs, name=name, opacity=0.7, nbinsx=30),
                    row=2, col=1
                )
            
            # Comparaison des métriques
            metrics_data = []
            for name, result in self.results['classification'].items():
                metrics_data.append({
                    'Model': name,
                    'ROC-AUC': result['metrics']['roc_auc'],
                    'Accuracy': result['metrics']['accuracy'],
                    'Precision': result['metrics']['precision'],
                    'Recall': result['metrics']['recall'],
                    'F1-Score': result['metrics']['f1_score']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig_clf.add_trace(
                go.Bar(x=metrics_df['Model'], y=metrics_df['ROC-AUC'], name='ROC-AUC'),
                row=1, col=3
            )
            
            fig_clf.update_layout(
                title_text="Dashboard d'Évaluation - Classification",
                showlegend=True,
                height=800
            )
            
            fig_clf.show()
        
        # Dashboard de régression
        if 'regression' in self.results:
            fig_reg = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Prédictions vs Réel', 'Résidus', 'Métriques',
                              'Q-Q Plot', 'Résidus vs Prédictions', 'Feature Importance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Scatter plot prédictions vs réel
            for name, result in self.results['regression'].items():
                fig_reg.add_trace(
                    go.Scatter(x=result['actual'], y=result['predictions'], 
                              mode='markers', name=name, opacity=0.6),
                    row=1, col=1
                )
            
            # Distribution des résidus
            for name, result in self.results['regression'].items():
                fig_reg.add_trace(
                    go.Histogram(x=result['residuals'], name=name, opacity=0.7, nbinsx=30),
                    row=1, col=2
                )
            
            # Comparaison des métriques
            metrics_data = []
            for name, result in self.results['regression'].items():
                metrics_data.append({
                    'Model': name,
                    'R²': result['metrics']['r2'],
                    'RMSE': result['metrics']['rmse'],
                    'MAE': result['metrics']['mae']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig_reg.add_trace(
                go.Bar(x=metrics_df['Model'], y=metrics_df['R²'], name='R²'),
                row=1, col=3
            )
            
            fig_reg.update_layout(
                title_text="Dashboard d'Évaluation - Régression",
                showlegend=True,
                height=800
            )
            
            fig_reg.show()
    
    def analyze_performance_by_segment(self, segment_col):
        """Analyse la performance par segment"""
        print(f"📊 Analyse de performance par {segment_col}...")
        
        if segment_col not in self.X_test.columns:
            print(f"❌ Colonne {segment_col} non trouvée")
            return
        
        segment_analysis = {}
        
        for segment in self.X_test[segment_col].unique():
            mask = self.X_test[segment_col] == segment
            y_segment = self.y_test[mask]
            
            if len(y_segment) < 10:  # Trop peu d'échantillons
                continue
            
            segment_metrics = {}
            
            # Métriques de classification
            if 'classification' in self.results:
                for name, result in self.results['classification'].items():
                    y_pred_segment = result['predictions'][mask]
                    y_pred_proba_segment = result['probabilities'][mask]
                    
                    segment_metrics[f'{name}_auc'] = roc_auc_score(y_segment, y_pred_proba_segment)
                    segment_metrics[f'{name}_accuracy'] = (y_pred_segment == y_segment).mean()
            
            # Métriques de régression
            if 'regression' in self.results:
                for name, result in self.results['regression'].items():
                    y_pred_segment = result['predictions'][mask]
                    
                    segment_metrics[f'{name}_r2'] = r2_score(y_segment, y_pred_segment)
                    segment_metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_segment, y_pred_segment))
            
            segment_analysis[segment] = segment_metrics
        
        return segment_analysis
    
    def generate_comprehensive_report(self):
        """Génère un rapport complet"""
        print("📋 Génération du rapport complet...")
        
        report = {
            'summary': {
                'total_models': len(self.models_data),
                'classification_models': len(self.results.get('classification', {})),
                'regression_models': len(self.results.get('regression', {}))
            },
            'classification_results': {},
            'regression_results': {},
            'recommendations': []
        }
        
        # Résultats de classification
        if 'classification' in self.results:
            best_clf = None
            best_clf_score = 0
            
            for name, result in self.results['classification'].items():
                metrics = result['metrics']
                report['classification_results'][name] = metrics
                
                if metrics['roc_auc'] > best_clf_score:
                    best_clf_score = metrics['roc_auc']
                    best_clf = name
            
            if best_clf:
                report['recommendations'].append(f"Meilleur modèle de classification: {best_clf} (AUC={best_clf_score:.4f})")
        
        # Résultats de régression
        if 'regression' in self.results:
            best_reg = None
            best_reg_score = 0
            
            for name, result in self.results['regression'].items():
                metrics = result['metrics']
                report['regression_results'][name] = metrics
                
                if metrics['r2'] > best_reg_score:
                    best_reg_score = metrics['r2']
                    best_reg = name
            
            if best_reg:
                report['recommendations'].append(f"Meilleur modèle de régression: {best_reg} (R²={best_reg_score:.4f})")
        
        # Recommandations générales
        if 'classification' in self.results:
            avg_auc = np.mean([result['metrics']['roc_auc'] for result in self.results['classification'].values()])
            if avg_auc > 0.85:
                report['recommendations'].append("Performance excellente en classification (>0.85 AUC)")
            elif avg_auc > 0.75:
                report['recommendations'].append("Performance bonne en classification (>0.75 AUC)")
            else:
                report['recommendations'].append("Performance à améliorer en classification (<0.75 AUC)")
        
        if 'regression' in self.results:
            avg_r2 = np.mean([result['metrics']['r2'] for result in self.results['regression'].values()])
            if avg_r2 > 0.80:
                report['recommendations'].append("Performance excellente en régression (>0.80 R²)")
            elif avg_r2 > 0.60:
                report['recommendations'].append("Performance bonne en régression (>0.60 R²)")
            else:
                report['recommendations'].append("Performance à améliorer en régression (<0.60 R²)")
        
        return report
    
    def run_complete_evaluation(self):
        """Exécute l'évaluation complète"""
        print("🚀 Démarrage de l'évaluation complète...")
        
        # Évaluer les modèles
        self.evaluate_classification_models()
        self.evaluate_regression_models()
        
        # Créer les visualisations
        self.create_classification_visualizations()
        self.create_regression_visualizations()
        
        # Créer le dashboard interactif
        self.create_interactive_dashboard()
        
        # Générer le rapport
        report = self.generate_comprehensive_report()
        
        print("✅ Évaluation complète terminée!")
        return report

if __name__ == "__main__":
    # Test du module
    print("🧪 Test du dashboard d'évaluation...") 