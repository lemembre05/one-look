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

Alerte si au moins 5 % de l'image est inondée (hors eau permanente). Seuils de décision
choisis sur l'entraînement ; validation = Sri Lanka + Pakistan (18 alertes sur 70 images),
test = Bolivie + Espagne + Ghana (24 alertes sur 98 images).

| Modèle | Paramètres | Taille | Temps / image (1 cœur CPU) | IoU eau (val.) | F1 val. | Fausses alertes val. | F1 test | Fausses alertes test |
|---|---|---|---|---|---|---|---|---|
| Baseline NDWI | 2 | ~0 | 5 ms | 0,665 | 0,625 | 7,7 % | 0,939 | 2,7 % |
| Petit U-Net (de zéro) | 484 177 | 1,9 Mo | 408 ms | 0,794 | 0,727 | 5,8 % | 0,913 | 1,4 % |
| U-Net MobileNetV2 (pré-entraîné ImageNet) | 4 137 937 | 16,6 Mo | 324 ms | 0,778 | 0,789 | 9,6 % | 0,936 | 1,4 % |

- Les deux réseaux détectent l'eau **bien mieux** que le NDWI (IoU de validation 0,78-0,79
  contre 0,67) et améliorent nettement le F1 de validation.
- Sur le test, les trois modèles sont proches : avec 24 alertes, un écart d'une ou deux
  images n'est pas significatif.
- Limite principale restante : **les nuages**. Sur Ghana_866994 (40 % d'eau annotée, image
  très nuageuse), tous les modèles optiques voient moins de 5 % d'eau.

Détails : `notebooks/02_baseline_ndwi.ipynb` et `notebooks/03_reseaux_legers.ipynb`.

## Avancement

- [x] Structure du dépôt
- [x] Exploration des données (446 images, 11 pays, voir `docs/carnet_donnees.md`)
- [x] Définition de l'alerte : au moins 5 % de surface inondée hors eau permanente
- [x] Découpage train / validation / test par pays
- [x] Baseline simple (indice d'eau NDWI + seuil)
- [x] Réseaux légers : petit U-Net et U-Net MobileNetV2 pré-entraîné
- [ ] Gestion des fausses alertes et de l'inconnu
- [ ] Export ONNX, taille et temps d'inférence
- [ ] Démonstration
