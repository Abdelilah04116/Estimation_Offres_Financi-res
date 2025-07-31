import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur d'Appels d'Offres",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Charge les modèles entraînés"""
    try:
        clf_model = joblib.load('model_classification.pkl')
        reg_model = joblib.load('model_regression.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return clf_model, reg_model, scaler, encoders, feature_columns
    except FileNotFoundError:
        st.error("❌ Modèles non trouvés. Veuillez d'abord exécuter le script d'entraînement.")
        return None, None, None, None, None

def predict_probability(budget, montant_propose, nombre_concurrents, delai, 
                       experience, notation, secteur, procedure, critere, complexite):
    """Prédit la probabilité de gain"""
    clf_model, reg_model, scaler, encoders, feature_columns = load_models()
    
    if clf_model is None:
        return None, None
    
    # Encoder les variables catégorielles
    try:
        secteur_enc = encoders['secteur'].transform([secteur])[0]
        procedure_enc = encoders['procedure'].transform([procedure])[0]
        critere_enc = encoders['critere'].transform([critere])[0]
        complexite_enc = encoders['complexite'].transform([complexite])[0]
    except:
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

def simulate_price_range(budget, nombre_concurrents, delai, experience, notation, 
                        secteur, procedure, critere, complexite, min_pct=0.8, max_pct=1.2):
    """Simule différents prix pour trouver la fourchette optimale"""
    price_range = np.linspace(budget * min_pct, budget * max_pct, 50)
    probabilities = []
    
    for price in price_range:
        prob, _ = predict_probability(
            budget, price, nombre_concurrents, delai, experience, 
            notation, secteur, procedure, critere, complexite
        )
        probabilities.append(prob if prob is not None else 0)
    
    return price_range, probabilities

def main():
    st.title("🎯 Prédicteur d'Appels d'Offres")
    st.markdown("---")
    
    # Vérifier si les modèles sont chargés
    clf_model, reg_model, scaler, encoders, feature_columns = load_models()
    if clf_model is None:
        st.stop()
    
    # Sidebar pour les paramètres
    st.sidebar.header("📋 Paramètres de l'Appel d'Offres")
    
    # Paramètres de base
    budget = st.sidebar.number_input("💰 Budget estimé (DH)", min_value=1000, value=1000000, step=10000)
    montant_propose = st.sidebar.number_input("💵 Montant proposé (DH)", min_value=1000, value=int(budget*0.95), step=1000)
    nombre_concurrents = st.sidebar.slider("👥 Nombre de concurrents", 1, 20, 5)
    delai = st.sidebar.slider("⏱️ Délai d'exécution (jours)", 30, 365, 120)
    
    # Paramètres de l'entreprise
    st.sidebar.subheader("🏢 Profil de l'entreprise")
    experience = st.sidebar.slider("📈 Expérience (années)", 0, 30, 10)
    notation = st.sidebar.slider("⭐ Notation technique", 0, 100, 75)
    
    # Paramètres catégoriels
    st.sidebar.subheader("🏷️ Caractéristiques du marché")
    secteur = st.sidebar.selectbox("🏭 Secteur", [
        "Informatique", "BTP", "Maintenance", "Agriculture", "Santé", 
        "Éducation", "Transport", "Énergie", "Télécommunications"
    ])
    
    procedure = st.sidebar.selectbox("📋 Type de procédure", [
        "Appel d'offres ouvert", "Appel d'offres restreint", "Concours", "Négociation"
    ])
    
    critere = st.sidebar.selectbox("🎯 Critère d'attribution", [
        "Prix seul", "Rapport qualité-prix", "Critères multiples", "Meilleure offre économiquement"
    ])
    
    complexite = st.sidebar.selectbox("🔧 Complexité du projet", [
        "Faible", "Moyenne", "Élevée"
    ])
    
    # Colonnes principales
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎲 Prédiction de Probabilité")
        
        # Prédiction pour le montant proposé
        probability, predicted_amount = predict_probability(
            budget, montant_propose, nombre_concurrents, delai, experience, 
            notation, secteur, procedure, critere, complexite
        )
        
        if probability is not None:
            # Affichage de la probabilité
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: white; margin: 0;">Probabilité de gain</h3>
                <h1 style="color: white; margin: 10px 0; font-size: 3em;">{probability:.1%}</h1>
                <p style="color: white; margin: 0;">pour {montant_propose:,.0f} DH</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Indicateur de couleur
            if probability > 0.6:
                st.success("🎉 Excellente chance de gagner!")
            elif probability > 0.4:
                st.warning("⚠️ Chance moyenne de gagner")
            else:
                st.error("❌ Faible chance de gagner")
            
            # Simulation de la fourchette de prix
            st.subheader("📊 Analyse de la fourchette de prix optimale")
            
            price_range, probabilities = simulate_price_range(
                budget, nombre_concurrents, delai, experience, notation, 
                secteur, procedure, critere, complexite
            )
            
            # Graphique des probabilités
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_range,
                y=probabilities,
                mode='lines+markers',
                name='Probabilité de gain',
                line=dict(color='#667eea', width=3),
                marker=dict(size=4)
            ))
            
            # Ligne de référence à 60%
            fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                         annotation_text="Seuil recommandé (60%)")
            
            # Marquer le prix proposé
            fig.add_vline(x=montant_propose, line_dash="dash", line_color="green",
                         annotation_text=f"Prix proposé: {montant_propose:,.0f} DH")
            
            fig.update_layout(
                title="Probabilité de gain selon le prix proposé",
                xaxis_title="Prix proposé (DH)",
                yaxis_title="Probabilité de gain",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Fourchette recommandée
            optimal_prices = []
            for i, prob in enumerate(probabilities):
                if prob >= 0.6:
                    optimal_prices.append(price_range[i])
            
            if optimal_prices:
                min_optimal = min(optimal_prices)
                max_optimal = max(optimal_prices)
                st.success(f"💡 **Fourchette recommandée (≥60% de chance):** {min_optimal:,.0f} - {max_optimal:,.0f} DH")
            else:
                st.warning("⚠️ Aucun prix dans cette plage n'atteint 60% de probabilité")
    
    with col2:
        st.header("📈 Statistiques")
        
        # Métriques clés
        ratio_prix_budget = montant_propose / budget
        ecart_budget = montant_propose - budget
        
        st.metric("Ratio Prix/Budget", f"{ratio_prix_budget:.2f}", 
                 f"{(ratio_prix_budget-1)*100:+.1f}%")
        
        st.metric("Écart au Budget", f"{ecart_budget:,.0f} DH", 
                 f"{ecart_budget/budget*100:+.1f}%")
        
        if predicted_amount:
            st.metric("Montant Gagnant Prédit", f"{predicted_amount:,.0f} DH")
        
        # Graphique radial des facteurs
        st.subheader("🎯 Profil de l'offre")
        
        categories = ['Expérience', 'Notation', 'Compétitivité Prix', 'Délai']
        values = [
            experience/30*100,
            notation,
            100 - abs(ratio_prix_budget - 0.95) * 500,  # Optimum à 95% du budget
            min(100, 365-delai)/365*100
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Profil',
            line=dict(color='#667eea')
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Conseils
        st.subheader("💡 Conseils")
        
        if ratio_prix_budget > 1.1:
            st.warning("Prix élevé par rapport au budget")
        elif ratio_prix_budget < 0.85:
            st.info("Prix très compétitif")
        
        if nombre_concurrents > 10:
            st.warning("Concurrence élevée")
        
        if notation < 70:
            st.error("Notation technique faible")
        elif notation > 90:
            st.success("Excellente notation technique")
    
    # Section d'analyse avancée
    st.header("🔍 Analyse Avancée")
    
    tabs = st.tabs(["Simulation Multi-Prix", "Analyse Concurrentielle", "Historique"])
    
    with tabs[0]:
        st.subheader("💰 Simulation de plusieurs prix")
        
        # Permettre à l'utilisateur de tester plusieurs prix
        test_prices = st.text_input("Prix à tester (séparés par des virgules)", 
                                   value=f"{int(budget*0.9)}, {int(budget*0.95)}, {int(budget*1.0)}, {int(budget*1.05)}")
        
        if test_prices:
            try:
                prices = [int(p.strip()) for p in test_prices.split(',')]
                results = []
                
                for price in prices:
                    prob, pred_amount = predict_probability(
                        budget, price, nombre_concurrents, delai, experience, 
                        notation, secteur, procedure, critere, complexite
                    )
                    results.append({
                        'Prix': f"{price:,} DH",
                        'Probabilité': f"{prob:.1%}" if prob else "N/A",
                        'Ratio Budget': f"{price/budget:.2f}",
                        'Écart': f"{price-budget:,} DH"
                    })
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
            except ValueError:
                st.error("Format invalide. Utilisez des nombres séparés par des virgules.")
    
    with tabs[1]:
        st.subheader("👥 Impact du nombre de concurrents")
        
        competitor_range = range(1, 21)
        competitor_probs = []
        
        for n_comp in competitor_range:
            prob, _ = predict_probability(
                budget, montant_propose, n_comp, delai, experience, 
                notation, secteur, procedure, critere, complexite
            )
            competitor_probs.append(prob if prob else 0)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=list(competitor_range),
            y=competitor_probs,
            mode='lines+markers',
            name='Probabilité selon concurrents',
            line=dict(color='#764ba2', width=3)
        ))
        
        fig_comp.update_layout(
            title="Impact du nombre de concurrents",
            xaxis_title="Nombre de concurrents",
            yaxis_title="Probabilité de gain",
            height=400
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with tabs[2]:
        st.subheader("📊 Données de référence")
        st.info("Cette section pourrait afficher l'historique des AO similaires dans le secteur sélectionné")
        
        # Simulation de données historiques
        hist_data = pd.DataFrame({
            'Secteur': [secteur] * 10,
            'Budget Moyen': np.random.normal(budget, budget*0.2, 10),
            'Taux de Réussite': np.random.uniform(0.3, 0.8, 10),
            'Nombre Concurrents': np.random.randint(3, 15, 10)
        })
        
        st.dataframe(hist_data, use_container_width=True)

if __name__ == "__main__":
    main()