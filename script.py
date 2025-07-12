import json
import csv
import random
from datetime import datetime, timedelta
from faker import Faker

# Configuration du générateur de données fictives en français
fake = Faker('fr_FR')

class AppelOffreGenerator:
    def __init__(self):
        self.villes_maroc = [
            "Rabat", "Casablanca", "Marrakech", "Fès", "Tanger", 
            "Agadir", "Oujda", "Kenitra", "Tétouan", "Safi",
            "Mohammedia", "Khouribga", "Beni Mellal", "El Jadida", "Nador"
        ]
        
        self.organismes_publics = [
            "Ministère de l'Intérieur", "Ministère de la Santé", "Ministère de l'Éducation",
            "Office National de l'Électricité et de l'Eau Potable", "Office National des Chemins de Fer",
            "Caisse Nationale de Sécurité Sociale", "Barid Al-Maghrib", "Royal Air Maroc",
            "Office National des Aéroports", "Agence Nationale de la Conservation Foncière"
        ]
        
        self.objets_marche = [
            "Fourniture et installation de matériel informatique",
            "Travaux de construction d'un centre de santé",
            "Prestation de services de nettoyage",
            "Fourniture de mobilier de bureau",
            "Maintenance des équipements électriques",
            "Étude technique pour l'aménagement urbain",
            "Fourniture de véhicules administratifs",
            "Services de sécurité et gardiennage",
            "Travaux de rénovation de bâtiments",
            "Fourniture de produits pharmaceutiques"
        ]
        
        self.documents_admin = [
            "Certificat de qualification et classification",
            "Attestation fiscale",
            "Attestation CNSS",
            "Registre de commerce",
            "Procuration (si applicable)",
            "Caution provisoire"
        ]
        
        self.documents_technique = [
            "Note technique détaillée",
            "Planning d'exécution",
            "Références similaires",
            "CV des intervenants",
            "Attestations de formation",
            "Certificats de conformité"
        ]
        
        self.documents_financier = [
            "Acte d'engagement",
            "Devis quantitatif et estimatif",
            "Bordereau des prix unitaires",
            "Bilan des 3 dernières années",
            "Attestation bancaire",
            "Chiffre d'affaires des 3 dernières années"
        ]
        
        self.conditions_participation = [
            "Être inscrit au registre de commerce",
            "Avoir un chiffre d'affaires minimum de 5M DH",
            "Justifier d'une expérience de 5 ans minimum",
            "Disposer d'un personnel qualifié",
            "Être en règle avec les obligations fiscales",
            "Fournir une caution provisoire de 2% du montant estimé"
        ]
        
        self.competences = [
            "Gestion de projet", "Analyse technique", "Contrôle qualité",
            "Normes ISO", "Management d'équipe", "Planification",
            "Expertise sectorielle", "Audit technique", "Formation"
        ]
        
        self.diplomes = [
            "Ingénieur d'État", "Master spécialisé", "Doctorat",
            "Diplôme d'ingénieur", "Master en management", "MBA"
        ]

    def generer_donnees_appel_offre(self):
        """Génère un appel d'offre avec des données réalistes"""
        
        # Génération de la référence
        annee = datetime.now().year
        numero = random.randint(1000, 9999)
        reference = f"AO-{annee}-{numero}"
        
        # Sélection aléatoire des données
        ville = random.choice(self.villes_maroc)
        organisme = random.choice(self.organismes_publics)
        objet = random.choice(self.objets_marche)
        
        # Génération du budget (en DH)
        budget = f"{random.randint(500000, 50000000):,} DH".replace(',', ' ')
        
        # Date de publication (entre aujourd'hui et 30 jours)
        date_publication = (datetime.now() + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        
        # Délai d'exécution
        delai_total = random.randint(60, 365)
        delai_phase1 = random.randint(20, delai_total//2)
        delai_phase2 = delai_total - delai_phase1
        
        # Documents requis (sélection aléatoire)
        docs_admin = random.sample(self.documents_admin, random.randint(3, 5))
        docs_technique = random.sample(self.documents_technique, random.randint(3, 5))
        docs_financier = random.sample(self.documents_financier, random.randint(3, 4))
        
        # Conditions de participation
        conditions = random.sample(self.conditions_participation, random.randint(3, 5))
        
        # Composition de l'équipe
        def generer_expert():
            return {
                "diplome": random.choice(self.diplomes),
                "experience": f"{random.randint(5, 20)} ans",
                "competences": random.sample(self.competences, random.randint(3, 5))
            }
        
        # Critères de sélection
        note_technique = random.randint(60, 80)
        note_financiere = 100 - note_technique
        
        # Structure complète de l'appel d'offre
        appel_offre = {
            "reference": reference,
            "objet_marche": objet,
            "maitre_ouvrage": {
                "nom": organisme,
                "adresse": f"{fake.street_address()}, {ville}",
                "telephone": fake.phone_number(),
                "fax": fake.phone_number()
            },
            "budget": budget,
            "date_publication": date_publication,
            "delai_execution": {
                "total": f"{delai_total} jours",
                "phase1": f"{delai_phase1} jours",
                "phase2": f"{delai_phase2} jours"
            },
            "documents_requis": {
                "administratif": docs_admin,
                "technique": docs_technique,
                "financier": docs_financier
            },
            "conditions_participation": conditions,
            "composition_equipe": {
                "chef_de_mission": generer_expert(),
                "expert_exploitation": generer_expert(),
                "expert_maintenance": generer_expert()
            },
            "critere_selection": {
                "note_technique": f"{note_technique}%",
                "note_financiere": f"{note_financiere}%",
                "note_minimale": "60%",
                "ponderation": {
                    "technique": f"{note_technique}%",
                    "financiere": f"{note_financiere}%"
                }
            },
            "mode_soumission": {
                "electronique": True,
                "support_portail": "www.marchespublics.gov.ma"
            }
        }
        
        return appel_offre

    def aplatir_donnees(self, donnees):
        """Aplatit la structure JSON pour l'export CSV"""
        ligne = {}
        
        # Champs simples
        ligne['reference'] = donnees['reference']
        ligne['objet_marche'] = donnees['objet_marche']
        ligne['budget'] = donnees['budget']
        ligne['date_publication'] = donnees['date_publication']
        
        # Maître d'ouvrage
        ligne['maitre_ouvrage_nom'] = donnees['maitre_ouvrage']['nom']
        ligne['maitre_ouvrage_adresse'] = donnees['maitre_ouvrage']['adresse']
        ligne['maitre_ouvrage_telephone'] = donnees['maitre_ouvrage']['telephone']
        ligne['maitre_ouvrage_fax'] = donnees['maitre_ouvrage']['fax']
        
        # Délai d'exécution
        ligne['delai_execution_total'] = donnees['delai_execution']['total']
        ligne['delai_execution_phase1'] = donnees['delai_execution']['phase1']
        ligne['delai_execution_phase2'] = donnees['delai_execution']['phase2']
        
        # Documents requis (convertis en chaînes)
        ligne['documents_administratif'] = '; '.join(donnees['documents_requis']['administratif'])
        ligne['documents_technique'] = '; '.join(donnees['documents_requis']['technique'])
        ligne['documents_financier'] = '; '.join(donnees['documents_requis']['financier'])
        
        # Conditions de participation
        ligne['conditions_participation'] = '; '.join(donnees['conditions_participation'])
        
        # Composition équipe - Chef de mission
        ligne['chef_mission_diplome'] = donnees['composition_equipe']['chef_de_mission']['diplome']
        ligne['chef_mission_experience'] = donnees['composition_equipe']['chef_de_mission']['experience']
        ligne['chef_mission_competences'] = '; '.join(donnees['composition_equipe']['chef_de_mission']['competences'])
        
        # Expert exploitation
        ligne['expert_exploitation_diplome'] = donnees['composition_equipe']['expert_exploitation']['diplome']
        ligne['expert_exploitation_experience'] = donnees['composition_equipe']['expert_exploitation']['experience']
        ligne['expert_exploitation_competences'] = '; '.join(donnees['composition_equipe']['expert_exploitation']['competences'])
        
        # Expert maintenance
        ligne['expert_maintenance_diplome'] = donnees['composition_equipe']['expert_maintenance']['diplome']
        ligne['expert_maintenance_experience'] = donnees['composition_equipe']['expert_maintenance']['experience']
        ligne['expert_maintenance_competences'] = '; '.join(donnees['composition_equipe']['expert_maintenance']['competences'])
        
        # Critères de sélection
        ligne['note_technique'] = donnees['critere_selection']['note_technique']
        ligne['note_financiere'] = donnees['critere_selection']['note_financiere']
        ligne['note_minimale'] = donnees['critere_selection']['note_minimale']
        ligne['ponderation_technique'] = donnees['critere_selection']['ponderation']['technique']
        ligne['ponderation_financiere'] = donnees['critere_selection']['ponderation']['financiere']
        
        # Mode de soumission
        ligne['mode_soumission_electronique'] = donnees['mode_soumission']['electronique']
        ligne['support_portail'] = donnees['mode_soumission']['support_portail']
        
        return ligne

    def generer_dataset(self, nombre_appels_offre=100):
        """Génère un dataset complet d'appels d'offres"""
        print(f"Génération de {nombre_appels_offre} appels d'offres...")
        
        # Génération des données
        donnees_json = []
        donnees_csv = []
        
        for i in range(nombre_appels_offre):
            appel_offre = self.generer_donnees_appel_offre()
            donnees_json.append(appel_offre)
            donnees_csv.append(self.aplatir_donnees(appel_offre))
            
            if (i + 1) % 10 == 0:
                print(f"Générés: {i + 1}/{nombre_appels_offre}")
        
        # Sauvegarde JSON
        with open('appels_offres.json', 'w', encoding='utf-8') as f:
            json.dump(donnees_json, f, ensure_ascii=False, indent=2)
        
        # Sauvegarde CSV
        if donnees_csv:
            with open('appels_offres.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=donnees_csv[0].keys())
                writer.writeheader()
                writer.writerows(donnees_csv)
        
        print(f"✅ Génération terminée!")
        print(f"📁 Fichiers créés:")
        print(f"   - appels_offres.json ({len(donnees_json)} entrées)")
        print(f"   - appels_offres.csv ({len(donnees_csv)} lignes)")
        
        return donnees_json, donnees_csv

# Fonction principale
def main():
    """Fonction principale pour exécuter le générateur"""
    generator = AppelOffreGenerator()
    
    # Génération d'un exemple pour test
    print("🧪 Génération d'un exemple d'appel d'offre:")
    exemple = generator.generer_donnees_appel_offre()
    print(json.dumps(exemple, ensure_ascii=False, indent=2))
    
    # Génération du dataset complet
    print("\n" + "="*50)
    nombre = int(input("Combien d'appels d'offres voulez-vous générer? (défaut: 100): ") or "100")
    
    donnees_json, donnees_csv = generator.generer_dataset(nombre)
    
    print(f"\n🎉 Dataset généré avec succès!")
    print(f"📊 Statistiques:")
    print(f"   - Nombre total d'appels d'offres: {len(donnees_json)}")
    print(f"   - Nombre de colonnes CSV: {len(donnees_csv[0].keys()) if donnees_csv else 0}")

if __name__ == "__main__":
    main()