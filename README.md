# One Look : détecter ce qui compte

Piscine Makers (Intelligence Lab), projet individuel, rendu le 16 novembre 2026.

Modèle d'IA léger, embarquable à bord d'un satellite, qui analyse une image
en une seule passe et décide si elle mérite une alerte inondation, sans
déclencher de fausses alertes.

## Structure

- `notebooks/` : exploration, entraînement, évaluation (Google Colab)
- `src/` : code réutilisable (données, modèles, entraînement, export embarqué, décision, démo)
- `docs/` : carnet de données (sources, licences, biais, limites)
- `modeles/` : modèle final embarqué (ONNX 8 bits et 32 bits) et seuils de décision (`config.json`)
- `app/` : application de démonstration (Gradio)
- `data/` : données locales, non versionnées

## Données

Sen1Floods11 (Bonafilia et al., 2020), licence CC BY 4.0 :
https://github.com/cloudtostreet/Sen1Floods11

Détails dans `docs/carnet_donnees.md`.

## Démarrer

Ouvrir `notebooks/01_exploration.ipynb` dans Google Colab : le notebook
installe ses dépendances et télécharge les données lui-même.

Pour la démonstration seule : `notebooks/06_demo.ipynb` (environ 2 minutes, sans GPU) et
`notebooks/07_image_inconnue.ipynb` (inondations en Italie, 2023), ou en local
`pip install -r requirements.txt` puis `python app/app.py`.

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

## Modèle optique, devenu modèle de secours (notebook 05)

Premier modèle final : **petit U-Net, Sentinel-2 seul, 60 époques** (expérience B). Depuis que
l'usage de plusieurs satellites est validé, le modèle principal est le modèle fusionné optique +
radar (section suivante) ; ce modèle optique reste **le secours**, utilisé quand aucune image radar
récente n'est disponible.

### Reproductibilité et choix de la graine

Deux premiers entraînements identiques sur le papier avaient donné des IoU de validation de 0,822
et 0,809 : les poids de départ du réseau étaient tirés sans graine. Tout est maintenant fixé
(poids de départ, découpes, cuDNN déterministe) et trois graines sont entraînées ; on garde la
meilleure **sur la validation** (le test ne sert pas au choix).

| Graine | Meilleure époque | IoU validation | IoU test |
|---|---|---|---|
| 0 | 35 | 0,807 | 0,780 |
| **1 (retenue)** | 18 | **0,824** | 0,788 |
| 2 | 37 | 0,823 | 0,801 |
| Moyenne ± écart-type | | 0,818 ± 0,010 | 0,790 ± 0,011 |

D'une graine à l'autre, l'IoU varie d'environ ±0,01 : c'est l'ordre de grandeur des écarts entre
modèles dans les tableaux précédents, qu'il faut donc lire avec prudence.

### Versions du modèle retenu

| Version | Taille | Temps / image 512×512 (1 cœur CPU) | IoU val. | IoU test | F1 val. | F1 test | Fausses alertes val. | Fausses alertes test |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 1,94 Mo | 435 ms | 0,824 | 0,788 | 0,857 | 0,880 | 11,5 % | 5,4 % |
| ONNX FP32 | 1,94 Mo | 226 ms | 0,824 | 0,788 | 0,857 | 0,880 | 11,5 % | 5,4 % |
| **ONNX INT8** (embarqué) | **0,66 Mo** | 235 ms | 0,821 | 0,789 | 0,878 | 0,880 | 9,6 % | 5,4 % |

- La compression en 8 bits divise la taille par 3 sans dégrader les alertes.
- ONNX Runtime est 2 fois plus rapide que PyTorch ; le 8 bits n'accélère pas sur le CPU de Colab
  (il le ferait sur un processeur doté d'instructions entières dédiées).
- Seuil d'alerte choisi sur l'entraînement : 3 % d'eau détectée (l'alerte « vraie » reste définie
  à 5 % d'eau annotée).

### Savoir dire « je ne sais pas »

Quatre réponses : **ALERTE**, **PAS D'ALERTE**, **INCERTAIN** (trop de nuages ou carte d'eau
ambiguë) et **INCONNU** (image hors de ce qui a été vu à l'entraînement). Tous les indicateurs
sont calculés à bord, sans annotation ; leurs seuils sont fixés sur l'entraînement.

| Partie | Décision | ALERTE | PAS D'ALERTE | INCERTAIN | Couverture | Précision | Rappel | Inondations manquées sans prévenir |
|---|---|---|---|---|---|---|---|---|
| Validation | 2 issues | 23 | 47 | 0 | 100 % | 0,783 | 1,000 | 0 |
| Validation | 4 issues | 23 | 35 | 12 | 83 % | 0,783 | 1,000 | 0 |
| Test | 2 issues | 26 | 72 | 0 | 100 % | 0,846 | 0,917 | 2 |
| Test | 4 issues | 26 | 57 | 15 | 85 % | 0,846 | 0,917 | 2 |

- Validation : **aucune inondation manquée**. Test : 2 manquées sur 24.
- Avec ce modèle, les 2 inondations manquées du test ne sont pas sous les nuages : la case
  INCERTAIN ne les rattrape pas. Elle évite en revanche d'affirmer « pas d'inondation » sur
  15 images où le sol est mal visible.

**Images anormales** (notebook 05 : 3 images de validation × 4 transformations ; notebook 06 :
une image de test × 3 transformations) : **toutes classées INCONNU**, les originales gardent
leur décision. Détecteur : écart de la réflectance moyenne de chaque bande à l'entraînement
(seuil = 1,2 × 99e centile), contrôle des bandes mortes et contrôle de saturation.

## Modèle principal : fusion optique + radar (notebooks 08 et 10)

Mêmes 15 bandes que l'expérience C (13 bandes Sentinel-2 + radar Sentinel-1 VV et VH), même petit
U-Net, 3 graines, choix de la graine sur la validation. Deux durées d'entraînement :

| Entraînement | Graine 0 (val. / test) | Graine 1 | Graine 2 | Moyenne val. | Moyenne test |
|---|---|---|---|---|---|
| 60 époques (notebook 08) | 0,835 / 0,803 | 0,835 / 0,802 | 0,842 / 0,809 | 0,837 ± 0,004 | 0,805 ± 0,004 |
| **150 époques (notebook 10)** | 0,855 / 0,822 | **0,859** / 0,818 | 0,849 / 0,805 | **0,854 ± 0,005** | **0,815 ± 0,009** |

Règle fixée avant l'entraînement long : il remplace le modèle de 60 époques seulement si son IoU
de validation est meilleure. C'est le cas (+0,017, au-delà de la variabilité entre graines) :
**modèle retenu : 150 époques, graine 1**, fichiers `modeles/fusion/`. Contrairement au modèle
optique (expérience B), le modèle fusionné profite d'un entraînement plus long : ses meilleures
époques tombent entre 103 et 133 sur 150.

Version embarquée : **ONNX INT8, 0,66 Mo, 240 ms par image 512 × 512 sur un cœur de CPU**
(IoU 0,857 en validation, 0,816 en test).

### Seuil d'alerte : réglé sur la validation

Avec le seuil d'alerte choisi sur l'entraînement comme pour le modèle optique, le modèle fusionné
de 60 époques manquait 5 inondations de validation sans prévenir (2 pour celui de 150 époques).
Explication : un modèle colle mieux à ses propres images d'entraînement, il y estime très bien la
surface inondée, et le seuil qui en sort (5 % d'eau détectée) est trop strict sur des pays
nouveaux. **Règle retenue pour le système final : le seuil d'alerte est choisi sur la validation**
(pays jamais vus à l'entraînement) ; le test ne sert qu'à mesurer. Cette règle a été adoptée
après avoir vu les résultats du notebook 08 ; le test n'a pas servi à la choisir. Pour le modèle
optique, elle donne le même seuil qu'avant (3 %).

### Fusion contre optique seul (décision à 4 issues, chacun avec son seuil réglé sur la validation)

| | Optique seul (secours) | **Fusion optique + radar** |
|---|---|---|
| IoU eau, validation | 0,821 | **0,857** |
| IoU eau, test | 0,789 | **0,816** |
| IoU eau, images nuageuses, validation (22 images) | 0,641 | **0,701** |
| IoU eau, images nuageuses, test (36 images) | **0,720** | 0,716 |
| F1 des alertes, validation | 0,878 | **0,923** |
| F1 des alertes, test | 0,880 | **0,906** |
| Inondations manquées sans prévenir, validation (18) | 0 | 0 |
| Inondations manquées sans prévenir, test (24) | 2 | **0** |
| Fausses alertes, validation | 12,5 % | **7,9 %** |
| Fausses alertes, test | **6,8 %** | 8,6 % |
| INCERTAIN, validation / test | 12 / 15 | 14 / 15 |

- **Aucune inondation manquée** en validation ni en test : les 24 inondations du test sont toutes
  signalées en ALERTE.
- Cartes d'eau meilleures partout, nettement sous les nuages en validation ; en test, l'IoU
  sous les nuages est la même que l'optique seul : le radar n'y fait pas de miracle.
- Le prix : un peu plus de fausses alertes en test (8,6 % contre 6,8 %).
- Images anormales (capteur saturé, bandes perdues, radar perdu, neige, bruit) : toutes INCONNU.
  Si le radar manque ou est illisible, la démonstration bascule sur le modèle optique de secours.

## Démonstration (notebook 06)

Le système complet tel qu'il tournerait à bord : `modeles/one_look_int8.onnx` (0,66 Mo) et
`modeles/config.json`, sans PyTorch ni GPU. Images de pays jamais vus à l'entraînement :

| Image | Décision | Eau détectée | Inondé (annotation) | Nuages | Verdict |
|---|---|---|---|---|---|
| Bolivia_103757 | ALERTE | 37,8 % | 40,1 % | 3 % | juste |
| Bolivia_360519 | PAS D'ALERTE | 1,0 % | 1,1 % | 1 % | juste |
| Spain_7370579 | ALERTE | 32,1 % | 26,7 % | 1 % | juste |
| Ghana_313799 | PAS D'ALERTE | 2,6 % | 6,1 % | 1 % | **manquée** (cas limite) |
| Ghana_866994 | ALERTE | 3,6 % | 40,0 % | 52 % | juste, de justesse |
| Ghana_26376 | INCERTAIN | 0,0 % | non annotée | 81 % | juste : image presque entièrement nuageuse |

Environ 220 ms par image sur un cœur de processeur. Une application Gradio (`app/app.py`) permet
de déposer n'importe quelle image Sentinel-2 à 13 bandes et de simuler des pannes.

### Image vraiment inconnue : Émilie-Romagne, mai 2023 (notebook 07)

Images Sentinel-2 brutes téléchargées directement (copie publique Copernicus sur Google Cloud),
hors de Sen1Floods11 : un pays absent (Italie), une autre année (2023) et une autre version de
traitement (décalage radiométrique de +1000 depuis 2022, retiré automatiquement). Zone de
20 km × 20 km entre Lugo et Conselice, avant et après les crues des 16 et 17 mai 2023.

Radar Sentinel-1 des mêmes jours, traité comme dans Sen1Floods11 (`src/radar.py` : étalonnage
sigma0, retrait du bruit thermique, dB, mise sur la grille de l'image optique). Acquisitions
les plus proches : 28 avril 2023 (5 jours après l'image optique d'avant, orbite descendante) et
23 mai 2023 (le jour même, orbite montante). Sur l'image radar du 23 mai, les deux zones inondées
du nord sont noires (l'eau calme renvoie très peu de signal) ; elles n'existent pas le 28 avril.

| Date | Modèle | Zone entière | Eau anormale détectée | Vignettes de 5 km (ALERTE / INCERTAIN / PAS D'ALERTE) |
|---|---|---|---|---|
| 23 avril 2023 (avant) | optique seul | PAS D'ALERTE | 0,7 % | 1 / 2 / 13 |
| 23 avril 2023 (avant) | **fusion optique + radar** | PAS D'ALERTE | 0,7 % | **0** / 2 / 14 |
| 23 mai 2023 (après) | optique seul | **ALERTE** | 8,9 % | 7 / 0 / 9 |
| 23 mai 2023 (après) | **fusion optique + radar** | **ALERTE** | 8,5 % | 7 / 0 / 9 |

- Après les crues, les deux grandes zones inondées au nord de la zone sont détectées et les
  7 vignettes en ALERTE sont toutes dans la moitié nord, avec ou sans radar. Aucune annotation
  n'existe pour cette image : la vérification est visuelle (les zones détectées correspondent
  aux étendues d'eau visibles sur les images optique et radar). Une comparaison avec les cartes
  Copernicus EMSR664 reste à faire.
- Avant les crues, la vignette passée à tort en ALERTE par le modèle optique (ombres de nuages,
  voir l'analyse des échecs) **repasse en PAS D'ALERTE avec le radar**. Réserve : l'image radar
  d'avant a 5 jours d'écart avec l'image optique.
- Les 2 vignettes très nuageuses d'avant restent INCERTAIN.
- Le score d'inconnu reste bas : malgré le changement de pays, d'année et de traitement, les
  images restent dans le domaine connu des modèles.
- Limite du traitement radar : pas de correction du relief (valable en plaine, pas en montagne).

## Analyse des échecs

- **Seuil d'inconnu trop permissif (corrigé).** La première version fixait le seuil sur le
  *maximum* des scores d'entraînement (7,6, une seule image atypique), soit un seuil de 9,2 : un
  capteur saturé (score 8,5) et des bandes perdues (score 4,5) déclenchaient de fausses ALERTES.
  Corrections : seuil au 99e centile (5,9), contrôle des bandes mortes, et exclusion d'une image
  d'entraînement entièrement vide (Paraguay_34417) qui faussait ce contrôle.
- **Erreur de gain sur une scène sombre (corrigé).** Dans la démo, Bolivia_103757 (scène très
  inondée, donc sombre) multipliée par 3 gardait un score d'inconnu de 5,3 et déclenchait une
  ALERTE. Règle physique ajoutée : plus de 2 % des pixels avec une réflectance ≥ 1 → INCONNU.
  Sur les 446 vraies images, cette part ne dépasse jamais 0,9 % ; sur Bolivia ×3 elle est de 24 %.
- **Ombres de nuages (corrigé par le radar sur l'exemple italien).** Sur l'image italienne
  d'avril, le modèle optique passe une vignette sans inondation en ALERTE avec 3,4 % d'« eau »
  pour un seuil de 3 % : les ombres des nuages sont sombres dans toutes les bandes, comme l'eau.
  Le modèle fusionné la classe PAS D'ALERTE : le radar, qui ne voit pas les ombres, ne confirme
  pas cette eau. Un seul exemple : l'étude bande par bande (notebook 09) doit mesurer l'effet
  sur tout le jeu de test.
- **Inondations faibles (non corrigé).** Ghana_313799 (6,1 % inondé, 2,6 % détecté) est juste
  sous le seuil : les petites inondations dispersées restent le point faible.
- **Nuages (limite).** Sous 52 % de nuages, Ghana_866994 montre 3,6 % d'eau pour 40 % réels
  avec l'optique seul (ALERTE de justesse), 4,8 % avec la fusion. La fusion ne suffit pas à
  retrouver toute l'eau cachée sous ces nuages.
- **Variabilité de l'entraînement.** ±0,01 d'IoU d'une graine à l'autre : avec 18 à 24 alertes
  par partie, une ou deux images changent nettement le F1.

## Avancement

- [x] Structure du dépôt
- [x] Exploration des données (446 images, 11 pays, voir `docs/carnet_donnees.md`)
- [x] Définition de l'alerte : au moins 5 % de surface inondée hors eau permanente
- [x] Découpage train / validation / test par pays
- [x] Baseline simple (indice d'eau NDWI + seuil)
- [x] Réseaux légers : petit U-Net et U-Net MobileNetV2 pré-entraîné
- [x] Ablation : entraînement plus long et ajout du radar Sentinel-1
- [x] Gestion des fausses alertes et de l'inconnu
- [x] Export ONNX, compression 8 bits, taille et temps d'inférence
- [x] Démonstration sur des images inconnues (notebook 06 et application Gradio)
- [x] Démonstration sur une image hors jeu de données : Émilie-Romagne, mai 2023 (notebook 07)
- [x] Modèle principal fusionné optique + radar, entraînement long, seuil d'alerte réglé sur la validation (notebooks 08 et 10)
- [x] Radar Sentinel-1 sur la démonstration italienne (notebook 07)
- [ ] Étude bande par bande : RGB, infrarouge, radar (notebook 09, en attente de GPU)
