import json
import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import uuid
from typing import Dict, List, Any, Optional
import logging

# Configuration du logging pour tracer les problèmes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du générateur de données fictives en français
fake = Faker('fr_FR')

class AppelOffreGenerator:
    def __init__(self):
        """Initialise le générateur avec toutes les constantes nécessaires"""
        self.villes_maroc = [
            "Rabat", "Casablanca", "Marrakech", "Fès", "Tanger", 
            "Agadir", "Oujda", "Kenitra", "Tétouan", "Safi",
            "Mohammedia", "Khouribga", "Beni Mellal", "El Jadida", "Nador",
            "Settat", "Berrechid", "Meknès", "Salé", "Temara"
        ]
        
        self.regions_maroc = [
            "Rabat-Salé-Kénitra", "Casablanca-Settat", "Marrakech-Safi",
            "Fès-Meknès", "Tanger-Tétouan-Al Hoceïma", "Souss-Massa",
            "Oriental", "Béni Mellal-Khénifra", "Drâa-Tafilalet",
            "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab", "Guelmim-Oued Noun"
        ]
        
        self.organismes_publics = [
            "Ministère de l'Intérieur", "Ministère de la Santé", "Ministère de l'Éducation",
            "Ministère de l'Équipement", "Ministère de l'Agriculture", "Ministère de la Justice",
            "Office National de l'Électricité et de l'Eau Potable", "Office National des Chemins de Fer",
            "Caisse Nationale de Sécurité Sociale", "Barid Al-Maghrib", "Royal Air Maroc",
            "Office National des Aéroports", "Agence Nationale de la Conservation Foncière",
            "Office National des Hydrocarbures", "Caisse de Dépôt et de Gestion",
            "Agence Nationale de Réglementation des Télécommunications", "Office National de Sécurité Sanitaire",
            "Agence Nationale de l'Assurance Maladie", "Office de la Formation Professionnelle",
            "Agence de Développement Social"
        ]
        
        self.secteurs = [
            "Informatique", "Énergie", "Transport", "Médical", "BTP", 
            "Fournitures scolaires", "Télécommunications", "Environnement",
            "Agriculture", "Sécurité", "Formation", "Conseil", "Maintenance"
        ]
        
        self.categories_marche = ["Travaux", "Fournitures", "Services"]
        
        self.types_procedure = [
            "Appel d'offres ouvert", "Appel d'offres restreint", 
            "Concours", "Gré à gré", "Procédure négociée"
        ]
        
        self.criteres_attribution = [
            "Meilleure offre économiquement", "Prix seul", 
            "Rapport qualité-prix", "Critères multiples"
        ]
        
        self.statuts_ao = ["Attribué", "Annulé", "Sans suite"]
        
        self.motifs_annulation = [
            "Non-conformité des offres", "Manque de budget", 
            "Spécifications techniques inadéquates", "Problème procédural",
            "Absence d'offres", "Offres trop élevées", "Changement de priorités",
            "Contraintes réglementaires", "Insuffisance technique", "Budget insuffisant"
        ]
        
        self.entreprises_maroc = [
            "INGEMA", "SOGEA", "Novec", "SOMAGEC", "TGCC", "SGTM",
            "CIMAR", "TRAVOCÉAN", "ATLAS HOSPITALITY", "COSUMAR",
            "MAROC TELECOM", "INWI", "ORANGE MAROC", "LAFARGE",
            "HOLCIM", "MANAGEM", "OCP", "BMCE", "ATTIJARI",
            "CDG", "WAFA", "SAHAM", "AFRIQUIA", "TOTAL MAROC",
            "SHELL MAROC", "SOREAD", "LYDEC", "REDAL", "AMENDIS",
            "VEOLIA", "SUEZ", "COVEC", "CHINA RAILWAY", "BOUYGUES",
            "VINCI", "EIFFAGE", "SOGEC", "DELTA HOLDING", "MAGHREB STEEL",
            "TECHNO PARK", "MAJOREL", "CAPGEMINI MAROC", "IBM MAROC",
            "MICROSOFT MAROC", "CISCO MAROC", "DELL MAROC", "HP MAROC"
        ]
        
        self.tailles_entreprise = ["PME", "Grande entreprise", "Moyenne entreprise", "ETI"]
        
        self.niveaux_complexite = ["Faible", "Moyenne", "Élevée", "Très élevée"]
        
        self.normes_techniques = ['ISO 9001', 'ISO 14001', 'ISO 45001', 'ISO 27001', 'ISO 22000']
        
        self.technologies = ['modernes', 'éprouvées', 'innovantes', 'certifiées', 'avancées']
        
        # Valeurs par défaut pour éviter les None
        self.valeurs_defaut = {
            'string': "Non spécifié",
            'int': 0,
            'float': 0.0,
            'list': [],
            'dict': {}
        }

    def _valider_et_corriger_valeur(self, valeur: Any, type_attendu: str = 'string') -> Any:
        """Valide et corrige une valeur pour éviter les None"""
        if valeur is None or valeur == "":
            return self.valeurs_defaut.get(type_attendu, "Non spécifié")
        
        if isinstance(valeur, list) and len(valeur) == 0:
            return ["Non spécifié"]
            
        return valeur

    def _generer_dates_coherentes(self) -> Dict[str, str]:
        """Génère des dates cohérentes pour l'appel d'offre"""
        try:
            date_publication = fake.date_between(start_date='-6m', end_date='today')
            date_limite = date_publication + timedelta(days=random.randint(21, 60))
            date_resultat = date_limite + timedelta(days=random.randint(3, 15))
            
            return {
                'date_publication': date_publication.strftime("%Y-%m-%d"),
                'date_limite': date_limite.strftime("%Y-%m-%d"),
                'date_resultat': date_resultat.strftime("%Y-%m-%d")
            }
        except Exception as e:
            logger.error(f"Erreur génération dates: {e}")
            # Dates par défaut en cas d'erreur
            aujourd_hui = datetime.now()
            return {
                'date_publication': (aujourd_hui - timedelta(days=30)).strftime("%Y-%m-%d"),
                'date_limite': (aujourd_hui - timedelta(days=1)).strftime("%Y-%m-%d"),
                'date_resultat': aujourd_hui.strftime("%Y-%m-%d")
            }

    def _generer_soumissionnaires_et_montants(self, budget_estime: int) -> Dict[str, Any]:
        """Génère les soumissionnaires et leurs montants de manière cohérente"""
        try:
            nb_soumissionnaires = max(3, random.randint(3, min(8, len(self.entreprises_maroc))))
            soumissionnaires = random.sample(self.entreprises_maroc, nb_soumissionnaires)
            
            # Génération des montants avec variation réaliste
            montants_soumis = []
            for _ in range(nb_soumissionnaires):
                variation = random.uniform(0.8, 1.2)
                montant = max(100000, round(budget_estime * variation, 2))
                montants_soumis.append(montant)
            
            # Génération des notes techniques
            notes_techniques = [random.randint(60, 100) for _ in range(nb_soumissionnaires)]
            
            # Génération des données d'expérience
            experiences = [random.randint(3, 25) for _ in range(nb_soumissionnaires)]
            
            # Génération des capacités financières
            capacites_financieres = [random.randint(10000000, 500000000) for _ in range(nb_soumissionnaires)]
            
            # Génération des localisations
            localisations = [random.choice(self.villes_maroc) for _ in range(nb_soumissionnaires)]
            
            # Génération des tailles d'entreprises
            tailles = [random.choice(self.tailles_entreprise) for _ in range(nb_soumissionnaires)]
            
            # Génération des historiques
            historiques = [random.randint(0, 15) for _ in range(nb_soumissionnaires)]
            
            return {
                'nb_soumissionnaires': nb_soumissionnaires,
                'soumissionnaires': soumissionnaires,
                'montants_soumis': montants_soumis,
                'notes_techniques': notes_techniques,
                'experiences': experiences,
                'capacites_financieres': capacites_financieres,
                'localisations': localisations,
                'tailles': tailles,
                'historiques': historiques
            }
        except Exception as e:
            logger.error(f"Erreur génération soumissionnaires: {e}")
            # Données par défaut en cas d'erreur
            return {
                'nb_soumissionnaires': 3,
                'soumissionnaires': self.entreprises_maroc[:3],
                'montants_soumis': [budget_estime] * 3,
                'notes_techniques': [75] * 3,
                'experiences': [10] * 3,
                'capacites_financieres': [50000000] * 3,
                'localisations': [self.villes_maroc[0]] * 3,
                'tailles': [self.tailles_entreprise[0]] * 3,
                'historiques': [5] * 3
            }

    def _determiner_gagnant(self, donnees_soumissionnaires: Dict[str, Any]) -> Dict[str, Any]:
        """Détermine le gagnant et les données associées"""
        try:
            nb_soumissionnaires = donnees_soumissionnaires['nb_soumissionnaires']
            soumissionnaires = donnees_soumissionnaires['soumissionnaires']
            montants_soumis = donnees_soumissionnaires['montants_soumis']
            notes_techniques = donnees_soumissionnaires['notes_techniques']
            experiences = donnees_soumissionnaires['experiences']
            
            # Calcul des scores combinés
            scores_combines = []
            for i in range(nb_soumissionnaires):
                score_technique = notes_techniques[i] * 0.6
                score_prix = (min(montants_soumis) / montants_soumis[i]) * 40
                scores_combines.append(score_technique + score_prix)
            
            gagnant_index = scores_combines.index(max(scores_combines))
            
            # Séparation des données gagnant/non-gagnants
            indices_non_gagnants = [i for i in range(nb_soumissionnaires) if i != gagnant_index]
            
            return {
                'gagnant_index': gagnant_index,
                'soumissionnaire_gagnant': soumissionnaires[gagnant_index],
                'montant_gagnant': montants_soumis[gagnant_index],
                'notation_technique_gagnant': notes_techniques[gagnant_index],
                'experience_gagnant': experiences[gagnant_index],
                'nombre_ao_gagnes_par_gagnant': random.randint(1, 15),
                'historique_gagnant_dans_les_3_derniers_ao': random.choice(["Oui", "Non"]),
                'notes_techniques_non_gagnants': [notes_techniques[i] for i in indices_non_gagnants],
                'experiences_non_gagnants': [experiences[i] for i in indices_non_gagnants],
                'historiques_non_gagnants': [donnees_soumissionnaires['historiques'][i] for i in indices_non_gagnants]
            }
        except Exception as e:
            logger.error(f"Erreur détermination gagnant: {e}")
            # Données par défaut
            return {
                'gagnant_index': 0,
                'soumissionnaire_gagnant': donnees_soumissionnaires['soumissionnaires'][0],
                'montant_gagnant': donnees_soumissionnaires['montants_soumis'][0],
                'notation_technique_gagnant': donnees_soumissionnaires['notes_techniques'][0],
                'experience_gagnant': donnees_soumissionnaires['experiences'][0],
                'nombre_ao_gagnes_par_gagnant': 5,
                'historique_gagnant_dans_les_3_derniers_ao': "Non",
                'notes_techniques_non_gagnants': donnees_soumissionnaires['notes_techniques'][1:],
                'experiences_non_gagnants': donnees_soumissionnaires['experiences'][1:],
                'historiques_non_gagnants': donnees_soumissionnaires['historiques'][1:]
            }

    def generer_donnees_appel_offre(self, id_ao: int) -> Dict[str, Any]:
        """Génère un appel d'offre complet avec GARANTIE zéro valeur null"""
        try:
            # Sélection sécurisée des données de base
            ville = self._valider_et_corriger_valeur(random.choice(self.villes_maroc))
            region = self._valider_et_corriger_valeur(random.choice(self.regions_maroc))
            secteur = self._valider_et_corriger_valeur(random.choice(self.secteurs))
            organisme = self._valider_et_corriger_valeur(random.choice(self.organismes_publics))
            categorie = self._valider_et_corriger_valeur(random.choice(self.categories_marche))
            
            # Génération des textes
            titre = self._valider_et_corriger_valeur(f"Appel d'offre pour {secteur.lower()} à {ville}")
            description = self._valider_et_corriger_valeur(f"Marché public concernant le secteur {secteur.lower()} pour {organisme}")
            
            # Génération des dates
            dates = self._generer_dates_coherentes()
            
            # Budget et montants
            budget_estime = max(500000, random.randint(500000, 50000000))
            
            # Génération des soumissionnaires et données associées
            donnees_soumissionnaires = self._generer_soumissionnaires_et_montants(budget_estime)
            
            # Détermination du gagnant
            donnees_gagnant = self._determiner_gagnant(donnees_soumissionnaires)
            
            # Statut et motif d'annulation
            statut = self._valider_et_corriger_valeur(random.choice(self.statuts_ao))
            motif_annulation = "Non applicable" if statut == "Attribué" else self._valider_et_corriger_valeur(random.choice(self.motifs_annulation))
            
            # Calculs dérivés
            montant_moyen = sum(donnees_soumissionnaires['montants_soumis']) / len(donnees_soumissionnaires['montants_soumis'])
            ecart_montant_vs_budget = (donnees_gagnant['montant_gagnant'] - budget_estime) / budget_estime
            
            # Données techniques et contextuelles
            delai_execution = max(30, random.randint(30, 365))
            type_procedure = self._valider_et_corriger_valeur(random.choice(self.types_procedure))
            critere_attribution = self._valider_et_corriger_valeur(random.choice(self.criteres_attribution))
            complexite_projet = self._valider_et_corriger_valeur(random.choice(self.niveaux_complexite))
            
            # Exigences techniques
            norme = self._valider_et_corriger_valeur(random.choice(self.normes_techniques))
            technologie = self._valider_et_corriger_valeur(random.choice(self.technologies))
            exigences_techniques = f"Normes {norme}, Technologies {technologie}"
            
            # Contexte économique
            inflation = round(random.uniform(1.0, 4.0), 1)
            croissance = round(random.uniform(2.0, 5.0), 1)
            contexte_economique = f"Inflation: {inflation}%, Croissance: {croissance}%"
            
            # Poids des critères
            poids_technique = random.randint(40, 80)
            poids_prix = 100 - poids_technique
            poids_criteres = f"Technique: {poids_technique}%, Prix: {poids_prix}%"
            
            # Historique et statistiques
            nb_ao_similaires = random.randint(5, 50)
            historique_ao_similaires = f"Derniers 12 mois: {nb_ao_similaires} AO similaires"
            
            taux_annulation = random.randint(5, 20)
            details_annulations = f"Taux d'annulation secteur: {taux_annulation}%"
            
            # Construction de l'objet final avec validation complète
            appel_offre = {
                "id_appel_offre": self._valider_et_corriger_valeur(id_ao, 'int'),
                "titre": titre,
                "description": description,
                "categorie_marche": categorie,
                "secteur": secteur,
                "organisme_emetteur": organisme,
                "pays": "Maroc",
                "region": region,
                "ville": ville,
                "date_publication": dates['date_publication'],
                "date_limite": dates['date_limite'],
                "date_resultat": dates['date_resultat'],
                "budget_estime": f"{budget_estime} DH",
                "soumissionnaires": donnees_soumissionnaires['soumissionnaires'],
                "montants_soumis": donnees_soumissionnaires['montants_soumis'],
                "soumissionnaire_gagnant": donnees_gagnant['soumissionnaire_gagnant'],
                "montant_gagnant": f"{donnees_gagnant['montant_gagnant']} DH",
                "montant_moyen": f"{montant_moyen:.2f} DH",
                "ecart_montant_vs_budget": round(ecart_montant_vs_budget, 4),
                "delai_execution": delai_execution,
                "type_procedure": type_procedure,
                "critere_attribution": critere_attribution,
                "nombre_soumissionnaires": donnees_soumissionnaires['nb_soumissionnaires'],
                "historique_gagnant_dans_les_3_derniers_ao": donnees_gagnant['historique_gagnant_dans_les_3_derniers_ao'],
                "nombre_ao_gagnes_par_gagnant": donnees_gagnant['nombre_ao_gagnes_par_gagnant'],
                "experience_gagnant": donnees_gagnant['experience_gagnant'],
                "notation_technique_gagnant": donnees_gagnant['notation_technique_gagnant'],
                "statut_ao": statut,
                "motif_annulation": motif_annulation,
                "notes_techniques_non_gagnants": donnees_gagnant['notes_techniques_non_gagnants'],
                "capacites_financieres": donnees_soumissionnaires['capacites_financieres'],
                "experiences_non_gagnants": donnees_gagnant['experiences_non_gagnants'],
                "historiques_non_gagnants": donnees_gagnant['historiques_non_gagnants'],
                "complexite_projet": complexite_projet,
                "exigences_techniques_detaillees": exigences_techniques,
                "localisations_soumissionnaires": donnees_soumissionnaires['localisations'],
                "tailles_entreprises": donnees_soumissionnaires['tailles'],
                "contexte_economique": contexte_economique,
                "poids_criteres_attribution": poids_criteres,
                "historique_ao_similaires": historique_ao_similaires,
                "details_annulations": details_annulations
            }
            
            # Validation finale - vérification exhaustive
            for key, value in appel_offre.items():
                if value is None:
                    logger.warning(f"Valeur None détectée pour la clé '{key}' dans l'AO {id_ao}")
                    appel_offre[key] = self.valeurs_defaut['string']
                elif isinstance(value, list) and len(value) == 0:
                    logger.warning(f"Liste vide détectée pour la clé '{key}' dans l'AO {id_ao}")
                    appel_offre[key] = ["Non spécifié"]
                elif isinstance(value, str) and value.strip() == "":
                    logger.warning(f"Chaîne vide détectée pour la clé '{key}' dans l'AO {id_ao}")
                    appel_offre[key] = "Non spécifié"
            
            return appel_offre
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'AO {id_ao}: {e}")
            # Retourner un objet minimal mais complet en cas d'erreur
            return self._generer_ao_minimal(id_ao)

    def _generer_ao_minimal(self, id_ao: int) -> Dict[str, Any]:
        """Génère un AO minimal mais complet en cas d'erreur"""
        return {
            "id_appel_offre": id_ao,
            "titre": "Appel d'offre standard",
            "description": "Description non disponible",
            "categorie_marche": "Services",
            "secteur": "Général",
            "organisme_emetteur": "Organisme public",
            "pays": "Maroc",
            "region": "Rabat-Salé-Kénitra",
            "ville": "Rabat",
            "date_publication": "2024-01-01",
            "date_limite": "2024-02-01",
            "date_resultat": "2024-02-15",
            "budget_estime": "1000000 DH",
            "soumissionnaires": ["Entreprise A", "Entreprise B", "Entreprise C"],
            "montants_soumis": [1000000, 1100000, 1200000],
            "soumissionnaire_gagnant": "Entreprise A",
            "montant_gagnant": "1000000 DH",
            "montant_moyen": "1100000.00 DH",
            "ecart_montant_vs_budget": 0.0,
            "delai_execution": 180,
            "type_procedure": "Appel d'offres ouvert",
            "critere_attribution": "Prix seul",
            "nombre_soumissionnaires": 3,
            "historique_gagnant_dans_les_3_derniers_ao": "Non",
            "nombre_ao_gagnes_par_gagnant": 1,
            "experience_gagnant": 10,
            "notation_technique_gagnant": 75,
            "statut_ao": "Attribué",
            "motif_annulation": "Non applicable",
            "notes_techniques_non_gagnants": [70, 65],
            "capacites_financieres": [50000000, 45000000, 40000000],
            "experiences_non_gagnants": [8, 6],
            "historiques_non_gagnants": [3, 2],
            "complexite_projet": "Moyenne",
            "exigences_techniques_detaillees": "Normes ISO 9001, Technologies éprouvées",
            "localisations_soumissionnaires": ["Rabat", "Casablanca", "Fès"],
            "tailles_entreprises": ["PME", "Grande entreprise", "Moyenne entreprise"],
            "contexte_economique": "Inflation: 2.5%, Croissance: 3.2%",
            "poids_criteres_attribution": "Technique: 60%, Prix: 40%",
            "historique_ao_similaires": "Derniers 12 mois: 25 AO similaires",
            "details_annulations": "Taux d'annulation secteur: 10%"
        }

    def aplatir_donnees(self, donnees: Dict[str, Any]) -> Dict[str, str]:
        """Aplatit la structure JSON pour l'export CSV avec validation complète"""
        ligne = {}
        
        try:
            for key, value in donnees.items():
                if isinstance(value, list):
                    # Validation et conversion des listes
                    if len(value) == 0:
                        ligne[key] = "Non spécifié"
                    else:
                        # Conversion sécurisée en chaîne
                        valeurs_str = []
                        for v in value:
                            if v is None:
                                valeurs_str.append("Non spécifié")
                            else:
                                valeurs_str.append(str(v))
                        ligne[key] = '; '.join(valeurs_str)
                elif value is None:
                    ligne[key] = "Non spécifié"
                elif isinstance(value, str) and value.strip() == "":
                    ligne[key] = "Non spécifié"
                else:
                    ligne[key] = str(value)
            
            return ligne
            
        except Exception as e:
            logger.error(f"Erreur lors de l'aplatissement des données: {e}")
            # Retourner une ligne par défaut
            return {key: "Erreur de conversion" for key in donnees.keys()}

    def verifier_donnees_completes(self, donnees_json: List[Dict[str, Any]]) -> bool:
        """Vérifie de manière exhaustive qu'il n'y a aucune valeur manquante"""
        print("🔍 Vérification approfondie de l'intégrité des données...")
        
        valeurs_problematiques = 0
        total_champs = 0
        problemes_detectes = []
        
        for i, ao in enumerate(donnees_json):
            for key, value in ao.items():
                total_champs += 1
                
                # Vérifications multiples
                if value is None:
                    valeurs_problematiques += 1
                    problemes_detectes.append(f"AO {i+1}, champ '{key}': None")
                elif isinstance(value, str) and value.strip() == "":
                    valeurs_problematiques += 1
                    problemes_detectes.append(f"AO {i+1}, champ '{key}': chaîne vide")
                elif isinstance(value, list) and len(value) == 0:
                    valeurs_problematiques += 1
                    problemes_detectes.append(f"AO {i+1}, champ '{key}': liste vide")
                elif isinstance(value, list):
                    # Vérifier les éléments de la liste
                    for j, item in enumerate(value):
                        if item is None:
                            valeurs_problematiques += 1
                            problemes_detectes.append(f"AO {i+1}, champ '{key}[{j}]': None dans liste")
        
        # Affichage des résultats
        if problemes_detectes:
            print(f"⚠️  {len(problemes_detectes)} problèmes détectés:")
            for probleme in problemes_detectes[:10]:  # Limiter l'affichage
                print(f"   - {probleme}")
            if len(problemes_detectes) > 10:
                print(f"   ... et {len(problemes_detectes) - 10} autres problèmes")
        
        print(f"📊 Résultats de la vérification:")
        print(f"   - Total des champs: {total_champs:,}")
        print(f"   - Valeurs problématiques: {valeurs_problematiques}")
        print(f"   - Taux de complétude: {((total_champs - valeurs_problematiques) / total_champs * 100):.2f}%")
        
        if valeurs_problematiques == 0:
            print("✅ AUCUNE valeur problématique détectée!")
            return True
        else:
            print("❌ Des valeurs problématiques ont été détectées!")
            return False

    def generer_dataset(self, nombre_appels_offre: int = 10000) -> tuple:
        """Génère un dataset complet d'appels d'offres avec GARANTIE zéro valeur null"""
        print(f"🚀 Génération de {nombre_appels_offre:,} appels d'offres (GARANTIE zéro valeur null)...")
        
        donnees_json = []
        donnees_csv = []
        erreurs_generation = 0
        
        try:
            # Génération avec barre de progression et gestion d'erreurs
            for i in range(1, nombre_appels_offre + 1):
                try:
                    appel_offre = self.generer_donnees_appel_offre(i)
                    
                    # Double validation avant ajout
                    if self._valider_ao_individual(appel_offre, i):
                        donnees_json.append(appel_offre)
                        donnees_csv.append(self.aplatir_donnees(appel_offre))
                    else:
                        erreurs_generation += 1
                        logger.warning(f"AO {i} rejeté pour valeurs invalides")
                        
                except Exception as e:
                    erreurs_generation += 1
                    logger.error(f"Erreur génération AO {i}: {e}")
                    # Ajouter un AO minimal en cas d'erreur
                    ao_minimal = self._generer_ao_minimal(i)
                    donnees_json.append(ao_minimal)
                    donnees_csv.append(self.aplatir_donnees(ao_minimal))
                
                # Affichage du progrès
                if i % 1000 == 0:
                    pourcentage = (i / nombre_appels_offre) * 100
                    print(f"✅ Progrès: {i:,}/{nombre_appels_offre:,} ({pourcentage:.1f}%) - Erreurs: {erreurs_generation}")
            
            print(f"📋 Génération terminée - Total: {len(donnees_json):,} AO générés")
            print(f"⚠️  Erreurs corrigées: {erreurs_generation}")
            
            # Vérification complète de l'intégrité
            print("\n🔍 Vérification finale de l'intégrité...")
            if self.verifier_donnees_completes(donnees_json):
                print("✅ Validation réussie - AUCUNE valeur null détectée!")
            else:
                print("❌ Problèmes détectés - Correction en cours...")
                donnees_json, donnees_csv = self._corriger_donnees_problematiques(donnees_json)
            
            # Vérification post-correction
            if self.verifier_donnees_completes(donnees_json):
                print("✅ Dataset final validé - GARANTIE zéro valeur null!")
            else:
                raise Exception("Impossible de garantir l'absence de valeurs null")
            
            # Sauvegarde sécurisée
            self._sauvegarder_fichiers(donnees_json, donnees_csv)
            
            # Statistiques finales
            self._afficher_statistiques_finales(donnees_json)
            
            return donnees_json, donnees_csv
            
        except Exception as e:
            logger.error(f"Erreur critique lors de la génération: {e}")
            return [], []

    def _valider_ao_individual(self, ao: Dict[str, Any], id_ao: int) -> bool:
        """Valide un AO individuel pour s'assurer qu'il ne contient aucune valeur null"""
        try:
            for key, value in ao.items():
                if value is None:
                    logger.warning(f"AO {id_ao}: valeur None pour '{key}'")
                    return False
                elif isinstance(value, str) and value.strip() == "":
                    logger.warning(f"AO {id_ao}: chaîne vide pour '{key}'")
                    return False
                elif isinstance(value, list):
                    if len(value) == 0:
                        logger.warning(f"AO {id_ao}: liste vide pour '{key}'")
                        return False
                    for item in value:
                        if item is None:
                            logger.warning(f"AO {id_ao}: None dans liste '{key}'")
                            return False
            return True
        except Exception as e:
            logger.error(f"Erreur validation AO {id_ao}: {e}")
            return False

    def _corriger_donnees_problematiques(self, donnees_json: List[Dict[str, Any]]) -> tuple:
        """Corrige automatiquement les données problématiques détectées"""
        print("🔧 Correction automatique des données problématiques...")
        
        donnees_corrigees = []
        corrections_effectuees = 0
        
        for i, ao in enumerate(donnees_json):
            ao_corrige = {}
            
            for key, value in ao.items():
                if value is None:
                    ao_corrige[key] = self.valeurs_defaut['string']
                    corrections_effectuees += 1
                elif isinstance(value, str) and value.strip() == "":
                    ao_corrige[key] = "Non spécifié"
                    corrections_effectuees += 1
                elif isinstance(value, list):
                    if len(value) == 0:
                        ao_corrige[key] = ["Non spécifié"]
                        corrections_effectuees += 1
                    else:
                        # Corriger les éléments None dans les listes
                        liste_corrigee = []
                        for item in value:
                            if item is None:
                                liste_corrigee.append("Non spécifié")
                                corrections_effectuees += 1
                            else:
                                liste_corrigee.append(item)
                        ao_corrige[key] = liste_corrigee
                else:
                    ao_corrige[key] = value
            
            donnees_corrigees.append(ao_corrige)
        
        # Régénération des données CSV
        donnees_csv_corrigees = [self.aplatir_donnees(ao) for ao in donnees_corrigees]
        
        print(f"✅ Corrections effectuées: {corrections_effectuees}")
        return donnees_corrigees, donnees_csv_corrigees

    def _sauvegarder_fichiers(self, donnees_json: List[Dict[str, Any]], donnees_csv: List[Dict[str, str]]):
        """Sauvegarde sécurisée des fichiers avec gestion d'erreurs"""
        try:
            # Sauvegarde JSON avec validation
            print("💾 Sauvegarde du fichier JSON...")
            with open('appels_offres_zero_null.json', 'w', encoding='utf-8') as f:
                json.dump(donnees_json, f, ensure_ascii=False, indent=2, default=str)
            print("✅ Fichier JSON sauvegardé")
            
            # Sauvegarde CSV avec validation
            print("💾 Sauvegarde du fichier CSV...")
            if donnees_csv and len(donnees_csv) > 0:
                with open('appels_offres_zero_null.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=donnees_csv[0].keys())
                    writer.writeheader()
                    writer.writerows(donnees_csv)
                print("✅ Fichier CSV sauvegardé")
            else:
                print("❌ Erreur: données CSV vides")
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            print(f"❌ Erreur de sauvegarde: {e}")

    def _afficher_statistiques_finales(self, donnees_json: List[Dict[str, Any]]):
        """Affiche des statistiques détaillées du dataset généré"""
        print(f"\n📊 STATISTIQUES FINALES DU DATASET:")
        print("=" * 60)
        
        if not donnees_json:
            print("❌ Aucune donnée à analyser")
            return
        
        # Statistiques de base
        nb_ao = len(donnees_json)
        nb_colonnes = len(donnees_json[0].keys()) if donnees_json else 0
        
        print(f"📈 Informations générales:")
        print(f"   • Nombre total d'AO: {nb_ao:,}")
        print(f"   • Nombre de colonnes: {nb_colonnes}")
        print(f"   • Taille estimée: {nb_ao * nb_colonnes:,} cellules")
        
        # Répartition par secteur
        secteurs = {}
        for ao in donnees_json:
            secteur = ao.get('secteur', 'Non spécifié')
            secteurs[secteur] = secteurs.get(secteur, 0) + 1
        
        print(f"\n🏢 Répartition par secteur:")
        for secteur, count in sorted(secteurs.items(), key=lambda x: x[1], reverse=True)[:10]:
            pourcentage = (count / nb_ao) * 100
            print(f"   • {secteur}: {count:,} ({pourcentage:.1f}%)")
        
        # Répartition par statut
        statuts = {}
        for ao in donnees_json:
            statut = ao.get('statut_ao', 'Non spécifié')
            statuts[statut] = statuts.get(statut, 0) + 1
        
        print(f"\n📋 Répartition par statut:")
        for statut, count in statuts.items():
            pourcentage = (count / nb_ao) * 100
            print(f"   • {statut}: {count:,} ({pourcentage:.1f}%)")
        
        # Répartition géographique
        villes = {}
        for ao in donnees_json:
            ville = ao.get('ville', 'Non spécifié')
            villes[ville] = villes.get(ville, 0) + 1
        
        print(f"\n🌍 Top 10 des villes:")
        for ville, count in sorted(villes.items(), key=lambda x: x[1], reverse=True)[:10]:
            pourcentage = (count / nb_ao) * 100
            print(f"   • {ville}: {count:,} ({pourcentage:.1f}%)")
        
        # Organismes émetteurs
        organismes = {}
        for ao in donnees_json:
            organisme = ao.get('organisme_emetteur', 'Non spécifié')
            organismes[organisme] = organismes.get(organisme, 0) + 1
        
        print(f"\n🏛️  Top 10 des organismes émetteurs:")
        for organisme, count in sorted(organismes.items(), key=lambda x: x[1], reverse=True)[:10]:
            pourcentage = (count / nb_ao) * 100
            print(f"   • {organisme[:40]}{'...' if len(organisme) > 40 else ''}: {count:,} ({pourcentage:.1f}%)")
        
        print("=" * 60)
        print("🎉 DATASET GÉNÉRÉ AVEC SUCCÈS - GARANTIE ZÉRO VALEUR NULL!")

    def afficher_exemple(self):
        """Affiche un exemple d'appel d'offre généré avec validation"""
        print("🧪 EXEMPLE D'APPEL D'OFFRE GÉNÉRÉ:")
        print("=" * 60)
        
        try:
            exemple = self.generer_donnees_appel_offre(1)
            
            # Validation de l'exemple
            if not self._valider_ao_individual(exemple, 1):
                print("❌ L'exemple contient des valeurs invalides!")
                return
            
            # Affichage formaté des informations principales
            print(f"🆔 ID: {exemple['id_appel_offre']}")
            print(f"📝 Titre: {exemple['titre']}")
            print(f"🏢 Organisme: {exemple['organisme_emetteur']}")
            print(f"🔧 Secteur: {exemple['secteur']}")
            print(f"📍 Ville: {exemple['ville']}")
            print(f"💰 Budget estimé: {exemple['budget_estime']}")
            print(f"👥 Soumissionnaires: {exemple['nombre_soumissionnaires']}")
            print(f"🏆 Gagnant: {exemple['soumissionnaire_gagnant']}")
            print(f"💵 Montant gagnant: {exemple['montant_gagnant']}")
            print(f"📊 Statut: {exemple['statut_ao']}")
            print(f"📅 Date publication: {exemple['date_publication']}")
            print(f"⏰ Délai d'exécution: {exemple['delai_execution']} jours")
            
            # Validation finale de l'exemple
            print(f"\n✅ Validation de l'exemple: AUCUNE valeur null détectée!")
            
        except Exception as e:
            logger.error(f"Erreur génération exemple: {e}")
            print(f"❌ Erreur lors de la génération de l'exemple: {e}")
        
        print("=" * 60)

    def tester_robustesse(self, nb_tests: int = 100):
        """Teste la robustesse du générateur avec de multiples tentatives"""
        print(f"🧪 TEST DE ROBUSTESSE - {nb_tests} générations:")
        print("-" * 50)
        
        succes = 0
        echecs = 0
        
        for i in range(1, nb_tests + 1):
            try:
                ao = self.generer_donnees_appel_offre(i)
                if self._valider_ao_individual(ao, i):
                    succes += 1
                else:
                    echecs += 1
                    print(f"❌ Test {i}: validation échouée")
            except Exception as e:
                echecs += 1
                print(f"❌ Test {i}: erreur - {e}")
        
        taux_succes = (succes / nb_tests) * 100
        print(f"\n📊 Résultats du test de robustesse:")
        print(f"   • Succès: {succes}/{nb_tests} ({taux_succes:.1f}%)")
        print(f"   • Échecs: {echecs}/{nb_tests}")
        
        if taux_succes >= 99:
            print("✅ Générateur TRÈS ROBUSTE!")
        elif taux_succes >= 95:
            print("✅ Générateur robuste")
        else:
            print("⚠️  Générateur nécessite des améliorations")
        
        return taux_succes


# Fonction principale améliorée
def main():
    """Fonction principale pour exécuter le générateur amélioré"""
    print("🚀 GÉNÉRATEUR D'APPELS D'OFFRES - VERSION ZÉRO VALEUR NULL")
    print("=" * 70)
    
    try:
        generator = AppelOffreGenerator()
        
        # Test de robustesse
        print("1️⃣  Test de robustesse du générateur...")
        taux_succes = generator.tester_robustesse(50)
        
        if taux_succes < 95:
            print("❌ Le générateur n'est pas assez robuste. Arrêt du processus.")
            return
        
        # Affichage d'un exemple
        print("\n2️⃣  Génération d'un exemple:")
        generator.afficher_exemple()
        
        # Confirmation pour la génération complète
        print("\n3️⃣  Génération du dataset complet:")
        while True:
            try:
                nombre = input("Nombre d'AO à générer (défaut: 10000): ").strip()
                if not nombre:
                    nombre = 10000
                else:
                    nombre = int(nombre)
                
                if nombre <= 0:
                    print("❌ Le nombre doit être positif!")
                    continue
                elif nombre > 100000:
                    print("⚠️  Attention: génération de plus de 100,000 AO peut prendre du temps.")
                    confirmation = input("Continuer? (o/n): ").strip().lower()
                    if confirmation not in ['o', 'oui', 'y', 'yes']:
                        continue
                break
                
            except ValueError:
                print("❌ Veuillez entrer un nombre valide!")
        
        print(f"\n🚀 Lancement de la génération de {nombre:,} appels d'offres...")
        print("⏳ Cette opération peut prendre plusieurs minutes...")
        
        # Génération avec mesure du temps
        import time
        debut = time.time()
        
        donnees_json, donnees_csv = generator.generer_dataset(nombre)
        
        fin = time.time()
        duree = fin - debut
        
        if donnees_json and donnees_csv:
            print(f"\n🎉 GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
            print(f"⏱️  Temps d'exécution: {duree:.2f} secondes")
            print(f"⚡ Vitesse: {nombre/duree:.0f} AO/seconde")
            print(f"📁 Fichiers générés:")
            print(f"   • appels_offres_zero_null.json")
            print(f"   • appels_offres_zero_null.csv")
            print(f"🔒 GARANTIE: AUCUNE valeur null dans le dataset!")
        else:
            print("❌ Erreur lors de la génération du dataset!")
    
    except KeyboardInterrupt:
        print("\n⏹️  Génération interrompue par l'utilisateur.")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        print(f"❌ Erreur critique: {e}")


if __name__ == "__main__":
    main()