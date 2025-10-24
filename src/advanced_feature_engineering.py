"""
Module de Feature Engineering Avancé pour les Appels d'Offres
Inclut : interactions, features temporelles, target encoding, feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from category_encoders import TargetEncoder, CatBoostEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Classe pour l'ingénierie de features avancée"""
    
    def __init__(self, target_col='is_winner', amount_col='montant_soumis'):
        self.target_col = target_col
        self.amount_col = amount_col
        self.encoders = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
        
    def extract_numeric_features(self, df):
        """Extrait et nettoie les features numériques"""
        print("🔧 Extraction des features numériques...")
        
        # Nettoyer les colonnes numériques
        df['budget_estime_num'] = df['budget_estime'].apply(self._extract_numeric_value)
        df['montant_gagnant_num'] = df['montant_gagnant'].apply(self._extract_numeric_value)
        df['montant_moyen_num'] = df['montant_moyen'].apply(self._extract_numeric_value)
        
        # S'assurer que montant_soumis est numérique
        if 'montant_soumis' in df.columns:
            df['montant_soumis'] = pd.to_numeric(df['montant_soumis'], errors='coerce').fillna(0)
        
        # Features de base (avec protection division par zéro)
        df['ratio_prix_budget'] = df['montant_soumis'] / df['budget_estime_num'].replace(0, 1)
        df['ecart_prix_budget'] = df['montant_soumis'] - df['budget_estime_num']
        df['ratio_prix_moyen'] = df['montant_soumis'] / df['montant_moyen_num'].replace(0, 1)
        
        # Features de positionnement
        df['position_prix'] = (df['montant_soumis'] - df['montant_moyen_num']) / df['montant_moyen_num'].replace(0, 1)
        df['ecart_budget_pourcentage'] = (df['budget_estime_num'] - df['montant_soumis']) / df['budget_estime_num'].replace(0, 1)
        
        return df
    
    def create_temporal_features(self, df):
        """Crée des features temporelles"""
        print("📅 Création des features temporelles...")
        
        # Convertir les dates
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce')
        df['date_limite'] = pd.to_datetime(df['date_limite'], errors='coerce')
        
        # Features temporelles
        df['jour_semaine'] = df['date_publication'].dt.dayofweek
        df['mois'] = df['date_publication'].dt.month
        df['trimestre'] = df['date_publication'].dt.quarter
        df['semaine_annee'] = df['date_publication'].dt.isocalendar().week
        df['jour_annee'] = df['date_publication'].dt.dayofyear
        
        # Saisonnalité
        df['saison'] = df['mois'].map({
            12: 'hiver', 1: 'hiver', 2: 'hiver',
            3: 'printemps', 4: 'printemps', 5: 'printemps',
            6: 'ete', 7: 'ete', 8: 'ete',
            9: 'automne', 10: 'automne', 11: 'automne'
        })
        
        # Délai entre publication et limite
        df['delai_soumission'] = (df['date_limite'] - df['date_publication']).dt.days
        
        return df
    
    def create_interaction_features(self, df):
        """Crée des features d'interaction"""
        print("🔗 Création des features d'interaction...")
        
        # S'assurer que les colonnes sont numériques
        numeric_cols = ['budget_estime_num', 'experience_gagnant', 'nombre_soumissionnaires', 
                       'notation_technique_gagnant', 'delai_execution']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Interactions numériques (avec vérification)
        if 'budget_estime_num' in df.columns and 'experience_gagnant' in df.columns:
            df['budget_experience'] = df['budget_estime_num'] * df['experience_gagnant']
        
        if 'budget_estime_num' in df.columns and 'nombre_soumissionnaires' in df.columns:
            df['budget_concurrence'] = df['budget_estime_num'] * df['nombre_soumissionnaires']
        
        if 'experience_gagnant' in df.columns and 'nombre_soumissionnaires' in df.columns:
            df['experience_concurrence'] = df['experience_gagnant'] * df['nombre_soumissionnaires']
        
        if 'notation_technique_gagnant' in df.columns and 'nombre_soumissionnaires' in df.columns:
            df['notation_concurrence'] = df['notation_technique_gagnant'] * df['nombre_soumissionnaires']
        
        # Features polynomiales
        if 'budget_estime_num' in df.columns:
            df['budget_squared'] = df['budget_estime_num'] ** 2
        
        if 'experience_gagnant' in df.columns:
            df['experience_squared'] = df['experience_gagnant'] ** 2
        
        if 'nombre_soumissionnaires' in df.columns:
            df['concurrence_squared'] = df['nombre_soumissionnaires'] ** 2
        
        # Ratios complexes (avec protection division par zéro)
        if 'experience_gagnant' in df.columns and 'delai_execution' in df.columns:
            df['ratio_experience_delai'] = df['experience_gagnant'] / df['delai_execution'].replace(0, 1)
        
        if 'notation_technique_gagnant' in df.columns and 'budget_estime_num' in df.columns:
            df['ratio_notation_budget'] = df['notation_technique_gagnant'] / df['budget_estime_num'].replace(0, 1)
        
        return df
    
    def create_categorical_features(self, df):
        """Crée des features catégorielles avancées"""
        print("🏷️ Création des features catégorielles...")
        
        # Encodage simple
        categorical_cols = ['secteur', 'type_procedure', 'critere_attribution', 
                           'complexite_projet', 'statut_ao', 'saison']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                self.encoders[col] = le
        
        # Target Encoding pour les variables importantes
        if self.target_col in df.columns:
            target_encoder = TargetEncoder(cols=['secteur', 'type_procedure'])
            df_encoded = target_encoder.fit_transform(df[['secteur', 'type_procedure']], df[self.target_col])
            df['secteur_target_encoded'] = df_encoded['secteur']
            df['procedure_target_encoded'] = df_encoded['type_procedure']
            self.encoders['target_encoder'] = target_encoder
        
        # Features de fréquence
        for col in ['secteur', 'type_procedure']:
            if col in df.columns:
                freq_encoding = df[col].value_counts(normalize=True)
                df[f'{col}_freq'] = df[col].map(freq_encoding)
        
        return df
    
    def create_business_features(self, df):
        """Crée des features métier spécifiques"""
        print("💼 Création des features métier...")
        
        # Stratégie de prix
        df['strategie_prix'] = np.where(
            df['ratio_prix_budget'] < 0.9, 'agressive',
            np.where(df['ratio_prix_budget'] < 1.1, 'equilibree', 'elevee')
        )
        
        # Niveau de concurrence
        df['niveau_concurrence'] = np.where(
            df['nombre_soumissionnaires'] <= 3, 'faible',
            np.where(df['nombre_soumissionnaires'] <= 7, 'moyenne', 'elevee')
        )
        
        # Catégories de délai
        df['delai_categorie'] = np.where(
            df['delai_execution'] <= 30, 'court',
            np.where(df['delai_execution'] <= 90, 'moyen', 'long')
        )
        
        # Catégories de prix
        df['prix_categorie'] = np.where(
            df['montant_soumis'] <= 100000, 'petit',
            np.where(df['montant_soumis'] <= 1000000, 'moyen', 'gros')
        )
        
        # Expérience catégorisée
        df['experience_categorie'] = np.where(
            df['experience_gagnant'] <= 5, 'junior',
            np.where(df['experience_gagnant'] <= 15, 'senior', 'expert')
        )
        
        # Notation catégorisée
        df['notation_categorie'] = np.where(
            df['notation_technique_gagnant'] <= 60, 'faible',
            np.where(df['notation_technique_gagnant'] <= 80, 'moyenne', 'excellente')
        )
        
        return df
    
    def create_statistical_features(self, df):
        """Crée des features statistiques"""
        print("📊 Création des features statistiques...")
        
        # Percentiles par secteur
        for col in ['montant_soumis', 'ratio_prix_budget', 'experience_gagnant']:
            if col in df.columns:
                df[f'{col}_percentile'] = df.groupby('secteur')[col].rank(pct=True)
        
        # Moyennes mobiles par secteur
        for col in ['montant_soumis', 'ratio_prix_budget']:
            if col in df.columns:
                df[f'{col}_moyenne_secteur'] = df.groupby('secteur')[col].transform('mean')
                df[f'{col}_ecart_secteur'] = df[col] - df[f'{col}_moyenne_secteur']
        
        # Features de dispersion
        df['concurrence_prix'] = df.groupby('id_appel_offre')['montant_soumis'].transform('std')
        df['concurrence_ratio'] = df.groupby('id_appel_offre')['ratio_prix_budget'].transform('std')
        
        return df
    
    def select_features(self, df, target_col, n_features=50):
        """Sélection automatique des meilleures features"""
        print(f"🎯 Sélection des {n_features} meilleures features...")
        
        # Préparer les données - exclure les colonnes de dates et autres types non numériques
        exclude_cols = [target_col, 'id_appel_offre', 'date_publication', 'date_limite', 'date_resultat',
                       'soumissionnaire', 'secteur', 'type_procedure', 'critere_attribution', 
                       'complexite_projet', 'statut_ao', 'ville', 'organisme_emetteur']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Nettoyer les données avant la sélection
        X = df[feature_cols].copy()
        
        # Convertir toutes les colonnes numériques et exclure les dates
        numeric_cols = []
        for col in X.columns:
            if X[col].dtype == 'object':
                # Essayer de convertir en numérique
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    numeric_cols.append(col)
                except:
                    # Si la conversion échoue, exclure la colonne
                    X = X.drop(columns=[col])
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                # Exclure les colonnes de dates
                X = X.drop(columns=[col])
            else:
                numeric_cols.append(col)
        
        X = X.fillna(0)
        y = df[target_col]
        
        # Sélection statistique
        f_selector = SelectKBest(score_func=f_classif, k=min(n_features, len(X.columns)))
        f_selector.fit(X, y)
        f_scores = f_selector.scores_
        
        # Information mutuelle
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(X.columns)))
        mi_selector.fit(X, y)
        mi_scores = mi_selector.scores_
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Combiner les scores
        combined_scores = (f_scores / f_scores.max() + 
                           mi_scores / mi_scores.max() + 
                           rf_importance / rf_importance.max()) / 3
        
        # Sélectionner les meilleures features
        top_indices = np.argsort(combined_scores)[-min(n_features, len(X.columns)):]
        selected_features = [X.columns[i] for i in top_indices]
        
        self.feature_importance = {
            'features': list(X.columns),
            'scores': combined_scores,
            'selected': selected_features
        }
        
        return selected_features
    
    def scale_features(self, df, feature_cols):
        """Normalise les features numériques"""
        print("📏 Normalisation des features...")
        
        # Séparer les features numériques et catégorielles
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
        categorical_features = df[feature_cols].select_dtypes(include=['object']).columns
        
        # Normaliser les features numériques
        scaler = RobustScaler()
        df_scaled = df.copy()
        df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])
        self.scalers['robust'] = scaler
        
        return df_scaled
    
    def _extract_numeric_value(self, value):
        """Extrait la valeur numérique d'une chaîne"""
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            import re
            # Nettoyer la chaîne en supprimant les espaces et caractères non numériques
            cleaned_value = value.replace(' ', '').replace('DH', '').replace('dirhams', '').replace('dh', '')
            # Extraire tous les chiffres et points décimaux
            numbers = re.findall(r'[\d,\.]+', cleaned_value)
            if numbers:
                try:
                    return float(numbers[0].replace(',', ''))
                except (ValueError, IndexError):
                    return np.nan
        try:
            return float(value) if value else np.nan
        except (ValueError, TypeError):
            return np.nan
    
    def run_complete_engineering(self, df, target_col='is_winner'):
        """Exécute l'ingénierie de features complète"""
        print("🚀 Démarrage de l'ingénierie de features avancée...")
        
        # Appliquer toutes les transformations
        df = self.extract_numeric_features(df)
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        df = self.create_categorical_features(df)
        df = self.create_business_features(df)
        df = self.create_statistical_features(df)
        
        # Sélectionner les meilleures features
        selected_features = self.select_features(df, target_col)
        
        # Normaliser
        df_final = self.scale_features(df, selected_features)
        
        # Préparer le dataset final
        final_features = selected_features + [target_col]
        if 'id_appel_offre' in df.columns:
            final_features.append('id_appel_offre')
        
        df_final = df_final[final_features]
        
        print(f"✅ Ingénierie terminée: {len(selected_features)} features créées")
        
        return df_final, selected_features

if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module de feature engineering...") 