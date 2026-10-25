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

## Résultats

| Modèle | Partie | Précision | Rappel | F1 | Taux de fausses alertes | Temps / image | Taille |
|---|---|---|---|---|---|---|---|
| Baseline NDWI (t_ndwi = 0, t_alerte = 3 %) | Validation | 0,714 | 0,556 | 0,625 | 0,077 | 14 ms (CPU Colab) | 2 paramètres |
| Baseline NDWI | Test | 0,920 | 0,958 | 0,939 | 0,027 | 14 ms | 2 paramètres |

Seuils choisis sur l'entraînement uniquement (F1 = 0,802). Détails et cas
d'échec dans `notebooks/02_baseline_ndwi.ipynb`.

## Avancement

- [x] Structure du dépôt
- [x] Exploration des données (446 images, 11 pays, voir `docs/carnet_donnees.md`)
- [x] Définition de l'alerte : au moins 5 % de surface inondée hors eau permanente
- [x] Découpage train / validation / test par pays
- [x] Baseline simple (indice d'eau NDWI + seuil)
- [ ] Modèle léger pré-entraîné
- [ ] Gestion des fausses alertes et de l'inconnu
- [ ] Export ONNX, taille et temps d'inférence
- [ ] Démonstration
