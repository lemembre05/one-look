# One Look : détecter ce qui compte

Piscine Makers (Intelligence Lab), projet individuel, rendu le 16 novembre 2026.

Modèle d'IA léger, embarquable à bord d'un satellite, qui analyse une image
en une seule passe et décide si elle mérite une alerte inondation, sans
déclencher de fausses alertes.

## Structure

- `notebooks/` : exploration, entraînement, évaluation (Google Colab)
- `src/` : code réutilisable (chargement des données, modèles)
- `docs/` : carnet de données (sources, licences, biais, limites)
- `data/` : données locales, non versionnées

## Données

Sen1Floods11 (Bonafilia et al., 2020), licence CC BY 4.0 :
https://github.com/cloudtostreet/Sen1Floods11

Détails dans `docs/carnet_donnees.md`.

## Démarrer

Ouvrir `notebooks/01_exploration.ipynb` dans Google Colab : le notebook
installe ses dépendances et télécharge les données lui-même.

En local : `pip install -r requirements.txt`

## Avancement

- [x] Structure du dépôt
- [ ] Exploration des données
- [ ] Définition de l'alerte (ce qui doit et ne doit pas alerter)
- [ ] Découpage train / validation / test sans fuite
- [ ] Baseline simple (indice d'eau NDWI + seuil)
- [ ] Modèle léger pré-entraîné
- [ ] Gestion des fausses alertes et de l'inconnu
- [ ] Export ONNX, taille et temps d'inférence
- [ ] Démonstration
