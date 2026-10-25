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

## Définition de l'alerte

Une image déclenche une alerte si **au moins 5 %** de sa surface annotée est
inondée, l'eau permanente (JRC) étant exclue (paramètre `SEUIL_ALERTE`,
notebook 02). 5 images entièrement couvertes de nuages n'ont aucune annotation :
elles sont comptées comme « pas d'alerte ».

## Découpage entraînement / validation / test

Découpage **par événement (pays)**, jamais au hasard : une image d'un pays
n'apparaît que dans une seule partie, ce qui évite les fuites (images voisines
d'une même inondation) et mesure la généralisation à des zones jamais vues.

| Partie | Pays | Images | Alertes (seuil 5 %) |
|---|---|---|---|
| Entraînement | Inde, Mékong, Nigeria, Paraguay, Somalie, USA | 278 | 97 |
| Validation | Sri Lanka, Pakistan | 70 | 18 |
| Test | Bolivie, Espagne, Ghana | 98 | 24 |

Choix du test : la Bolivie est l'événement que les auteurs de Sen1Floods11
gardent eux-mêmes à part ; l'Espagne est le seul événement européen ; le Ghana
contient beaucoup d'images peu inondées, utiles pour mesurer les fausses alertes.

## Biais et limites

- **Peu d'exemples** : 18 alertes en validation et 24 en test. Une ou deux
  erreurs changent fortement les scores ; les résultats sont à lire avec prudence.
- **Nuages** : en moyenne 14 % de chaque image est sans annotation. Les nuages
  cachent l'eau et leurs ombres peuvent ressembler à de l'eau.
- **Hypothèse embarquée** : la carte d'eau permanente est supposée connue à bord
  (carte fixe chargée avant le lancement).
- **Couverture géographique** : 11 événements seulement, aucun en zone froide
  (neige, glace), qui pourrait tromper un indice d'eau.
