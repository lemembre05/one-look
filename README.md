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

### Ablation du petit U-Net (notebook 04)

Même graine, même recette ; seuls changent les bandes d'entrée et la durée d'entraînement.

| Expérience | Bandes | Époques | IoU eau val. | F1 val. | Fausses alertes val. | IoU eau test | F1 test | IoU eau, images nuageuses (val. + test) |
|---|---|---|---|---|---|---|---|---|
| A | Sentinel-2 (13) | 30 | 0,788 | 0,686 | 9,6 % | 0,792 | 0,913 | 0,637 |
| B | Sentinel-2 (13) | 60 | 0,803 | 0,727 | 5,8 % | 0,793 | 0,909 | 0,647 |
| C | Sentinel-2 + radar Sentinel-1 (15) | 30 | **0,828** | 0,727 | 5,8 % | **0,797** | 0,894 | **0,688** |

Taille identique pour les trois (1,9 Mo ; le radar ajoute 288 paramètres), environ 0,5 s par image
sur un cœur de CPU.

- **Entraîner deux fois plus longtemps (B) n'apporte presque rien** : +0,015 d'IoU en validation,
  rien en test. Le temps d'entraînement n'était pas la limite.
- **Le radar (C) améliore les cartes d'eau partout**, surtout sous les nuages (+0,05 d'IoU sur les
  58 images nuageuses). Sur Ghana_866994 (40 % inondé sous les nuages), la part d'eau détectée
  passe de 2,4 % à 7,5 % : l'alerte se déclenche enfin.
- **Le F1 bouge peu et de façon bruitée** : la même configuration A donne un F1 de validation de
  0,727 dans le notebook 03 et 0,686 ici (variabilité d'un entraînement sur GPU à l'autre).
  Avec 18 alertes, l'IoU par pixel est la mesure fiable pour comparer.
- **Limite du radar pour l'embarqué** : Sentinel-1 et Sentinel-2 sont deux satellites différents.
  Le modèle C suppose qu'une image radar récente de la même zone est disponible (traitement au sol,
  ou constellation qui partage ses données). Pour un satellite optique seul, c'est le modèle A ou B.

## Avancement

- [x] Structure du dépôt
- [x] Exploration des données (446 images, 11 pays, voir `docs/carnet_donnees.md`)
- [x] Définition de l'alerte : au moins 5 % de surface inondée hors eau permanente
- [x] Découpage train / validation / test par pays
- [x] Baseline simple (indice d'eau NDWI + seuil)
- [x] Réseaux légers : petit U-Net et U-Net MobileNetV2 pré-entraîné
- [x] Ablation : entraînement plus long et ajout du radar Sentinel-1
- [ ] Gestion des fausses alertes et de l'inconnu
- [ ] Export ONNX, taille et temps d'inférence
- [ ] Démonstration
