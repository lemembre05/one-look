# Carnet de données

## Sources

| Source | Contenu | Licence | Accès |
|---|---|---|---|
| Sen1Floods11 (v1.1), sous-ensemble annoté à la main | Images Sentinel-1 et Sentinel-2 (512 x 512 px, 10 m) de 11 inondations, masques d'eau annotés, masques d'eau permanente (JRC) | CC BY 4.0 | `gs://sen1floods11` (bucket public Google Cloud) |

Référence : Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020).
Sen1Floods11: a georeferenced dataset to train and test deep learning flood
algorithms for Sentinel-1. CVPR Workshops.

## Composition

Sous-ensemble annoté à la main : **446 images** Sentinel-2 (13 bandes, 512 x 512 px,
entiers int16), réparties sur **11 événements d'inondation** :

| Pays | Images | Part d'eau moyenne | Part d'eau hors eau permanente |
|---|---|---|---|
| Bolivie | 15 | 0,170 | 0,158 |
| Ghana | 53 | 0,064 | 0,053 |
| Inde | 68 | 0,128 | 0,103 |
| Mékong | 30 | 0,236 | 0,161 |
| Nigeria | 18 | 0,223 | 0,203 |
| Pakistan | 28 | 0,076 | 0,058 |
| Paraguay | 67 | 0,094 | 0,066 |
| Somalie | 26 | 0,062 | 0,062 |
| Espagne | 30 | 0,141 | 0,079 |
| Sri Lanka | 42 | 0,112 | 0,036 |
| USA | 69 | 0,046 | 0,031 |

Observation : la majorité des images contient très peu d'eau. Le jeu contient
donc naturellement des exemples « normaux » (pas d'alerte) en plus des images
inondées.

Masques : `LabelHand` vaut -1 (pas de donnée, souvent des nuages), 0 (sec) ou
1 (eau). `JRCWaterHand` marque l'eau permanente.

## Découpage entraînement / validation / test

À définir. Principe : découper par événement (pays) et non au hasard, pour
évaluer le modèle sur des zones jamais vues et éviter les fuites.

## Biais et limites

À compléter.
