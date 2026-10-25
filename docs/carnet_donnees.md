# Carnet de données

## Sources

| Source | Contenu | Licence | Accès |
|---|---|---|---|
| Sen1Floods11 (v1.1), sous-ensemble annoté à la main | Images Sentinel-1 et Sentinel-2 (512 x 512 px, 10 m) de 11 inondations, masques d'eau annotés, masques d'eau permanente (JRC) | CC BY 4.0 | `gs://sen1floods11` (bucket public Google Cloud) |

Référence : Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020).
Sen1Floods11: a georeferenced dataset to train and test deep learning flood
algorithms for Sentinel-1. CVPR Workshops.

## Composition

À compléter après l'exploration (notebook 01) : nombre d'images par pays,
part d'eau par image.

## Découpage entraînement / validation / test

À définir. Principe : découper par événement (pays) et non au hasard, pour
évaluer le modèle sur des zones jamais vues et éviter les fuites.

## Biais et limites

À compléter.
