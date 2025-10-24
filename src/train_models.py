import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def extract_numeric_value(value):
    """Extrait la valeur numérique d'une chaîne contenant 'DH' ou autres unités"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Extraire les nombres de la chaîne
        import re
        numbers = re.findall(r'[\d,\.]+', value.replace(' ', ''))
        if numbers:
            return float(numbers[0].replace(',', ''))
    return float(value) if value else np.nan

def prepare_data(df):
    """Prépare les données pour l'entraînement"""
    # Nettoyer les colonnes numériques
    df['budget_estime_num'] = df['budget_estime'].apply(extract_numeric_value)
    df['montant_gagnant_num'] = df['montant_gagnant'].apply(extract_numeric_value)
    df['montant_moyen_num'] = df['montant_moyen'].apply(extract_numeric_value)
    
    # Créer les features d'ingénierie
    df['ratio_prix_budget'] = df['montant_gagnant_num'] / df['budget_estime_num']
    df['ecart_prix_budget'] = df['montant_gagnant_num'] - df['budget_estime_num']
    df['ratio_prix_moyen'] = df['montant_gagnant_num'] / df['montant_moyen_num']
    
    # Encoder les variables catégorielles
    le_secteur = LabelEncoder()
    le_procedure = LabelEncoder()
    le_critere = LabelEncoder()
    le_complexite = LabelEncoder()
    le_statut = LabelEncoder()
    
    df['secteur_encoded'] = le_secteur.fit_transform(df['secteur'].fillna('Unknown'))
    df['type_procedure_encoded'] = le_procedure.fit_transform(df['type_procedure'].fillna('Unknown'))
    df['critere_attribution_encoded'] = le_critere.fit_transform(df['critere_attribution'].fillna('Unknown'))
    df['complexite_projet_encoded'] = le_complexite.fit_transform(df['complexite_projet'].fillna('Unknown'))
    df['statut_ao_encoded'] = le_statut.fit_transform(df['statut_ao'].fillna('Unknown'))
    
    return df, {
        'secteur': le_secteur,
        'procedure': le_procedure,
        'critere': le_critere,
        'complexite': le_complexite,
        'statut': le_statut
    }

def create_submission_dataset(df):
    """Crée un dataset avec une ligne par soumission"""
    submissions = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['montants_soumis']) and pd.notna(row['soumissionnaires']):
            try:
                # Parse les montants soumis
                montants = [float(x.strip()) for x in str(row['montants_soumis']).split(';')]
                soumissionnaires = [x.strip() for x in str(row['soumissionnaires']).split(';')]
                
                if len(montants) == len(soumissionnaires):
                    gagnant = str(row['soumissionnaire_gagnant']).strip() if pd.notna(row['soumissionnaire_gagnant']) else ''
                    
                    for i, (soum, montant) in enumerate(zip(soumissionnaires, montants)):
                        submission = {
                            'id_appel_offre': row['id_appel_offre'],
                            'budget_estime_num': row['budget_estime_num'],
                            'montant_soumis': montant,
                            'nombre_soumissionnaires': row['nombre_soumissionnaires'],
                            'delai_execution': row['delai_execution'],
                            'experience_gagnant': row['experience_gagnant'],
                            'notation_technique_gagnant': row['notation_technique_gagnant'],
                            'secteur_encoded': row['secteur_encoded'],
                            'type_procedure_encoded': row['type_procedure_encoded'],
                            'critere_attribution_encoded': row['critere_attribution_encoded'],
                            'complexite_projet_encoded': row['complexite_projet_encoded'],
                            'statut_ao_encoded': row['statut_ao_encoded'],
                            'ratio_prix_budget': montant / row['budget_estime_num'] if row['budget_estime_num'] > 0 else 0,
                            'ecart_prix_budget': montant - row['budget_estime_num'],
                            'is_winner': 1 if soum == gagnant else 0,
                            'montant_gagnant_num': row['montant_gagnant_num']
                        }
                        submissions.append(submission)
            except:
                continue
    
    return pd.DataFrame(submissions)

def train_models(file_path):
    """Entraîne les modèles de classification et régression"""
    print("🚀 Démarrage de l'entraînement des modèles...")
    
    # Charger les données
    df = pd.read_csv(file_path)
    print(f"📊 Données chargées: {len(df)} lignes")
    
    # Préparer les données
    df, encoders = prepare_data(df)
    
    # Créer le dataset de soumissions
    submissions_df = create_submission_dataset(df)
    print(f"📋 Dataset de soumissions créé: {len(submissions_df)} soumissions")
    
    # Filtrer les données valides
    valid_submissions = submissions_df.dropna(subset=['montant_soumis', 'budget_estime_num'])
    print(f"✅ Soumissions valides: {len(valid_submissions)}")
    
    # Définir les features
    feature_columns = [
        'budget_estime_num', 'montant_soumis', 'nombre_soumissionnaires',
        'delai_execution', 'experience_gagnant', 'notation_technique_gagnant',
        'secteur_encoded', 'type_procedure_encoded', 'critere_attribution_encoded',
        'complexite_projet_encoded', 'ratio_prix_budget', 'ecart_prix_budget'
    ]
    
    X = valid_submissions[feature_columns]
    y_classification = valid_submissions['is_winner']
    y_regression = valid_submissions['montant_gagnant_num']
    
    # Diviser les données
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Normaliser les features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n🎯 Entraînement du modèle de classification...")
    # Modèle de classification
    clf_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    clf_model.fit(X_train_scaled, y_class_train)
    
    # Prédictions classification
    y_pred_class = clf_model.predict(X_test_scaled)
    y_pred_proba = clf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Évaluation classification
    print("\n📊 Résultats Classification:")
    print(classification_report(y_class_test, y_pred_class))
    print(f"AUC-ROC: {roc_auc_score(y_class_test, y_pred_proba):.4f}")
    
    print("\n🎯 Entraînement du modèle de régression...")
    # Modèle de régression
    reg_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    reg_model.fit(X_train_scaled, y_reg_train)
    
    # Prédictions régression
    y_pred_reg = reg_model.predict(X_test_scaled)
    
    # Évaluation régression
    print("\n📊 Résultats Régression:")
    print(f"R²: {r2_score(y_reg_test, y_pred_reg):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_pred_reg)):.2f}")
    
    # Sauvegarder les modèles
    print("\n💾 Sauvegarde des modèles...")
    joblib.dump(clf_model, 'models/model_classification.pkl')
    joblib.dump(reg_model, 'models/model_regression.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    
    print("✅ Modèles sauvegardés avec succès!")
    
    return clf_model, reg_model, scaler, encoders, feature_columns

def predict_win_probability(budget, montant_propose, nombre_concurrents, 
                          delai, experience, notation, secteur, procedure, 
                          critere, complexite, models_data):
    """Prédit la probabilité de gain pour une offre"""
    
    clf_model, reg_model, scaler, encoders, feature_columns = models_data
    
    # Encoder les variables catégorielles
    try:
        secteur_enc = encoders['secteur'].transform([secteur])[0]
        procedure_enc = encoders['procedure'].transform([procedure])[0]
        critere_enc = encoders['critere'].transform([critere])[0]
        complexite_enc = encoders['complexite'].transform([complexite])[0]
    except:
        # Valeurs par défaut si nouvelles catégories
        secteur_enc = 0
        procedure_enc = 0
        critere_enc = 0
        complexite_enc = 0
    
    # Créer les features
    features = np.array([[
        budget,
        montant_propose,
        nombre_concurrents,
        delai,
        experience,
        notation,
        secteur_enc,
        procedure_enc,
        critere_enc,
        complexite_enc,
        montant_propose / budget if budget > 0 else 0,
        montant_propose - budget
    ]])
    
    # Normaliser
    features_scaled = scaler.transform(features)
    
    # Prédire
    probability = clf_model.predict_proba(features_scaled)[0, 1]
    predicted_amount = reg_model.predict(features_scaled)[0]
    
    return probability, predicted_amount

if __name__ == "__main__":
    # Entraîner les modèles
    models = train_models('data/appels_offres_zero_null.csv')
    
    print("\n🎉 Entraînement terminé avec succès!")
    print("Les fichiers suivants ont été créés:")
    print("- models/model_classification.pkl")
    print("- models/model_regression.pkl") 
    print("- models/scaler.pkl")
    print("- models/encoders.pkl")
    print("- models/feature_columns.pkl")