"""Chargement et préparation des données Sen1Floods11 (sous-ensemble annoté à la main)."""

import glob
import os
import subprocess

import numpy as np
import rasterio

BUCKET = "gs://sen1floods11/v1.1/data/flood_events/HandLabeled"
DOSSIERS = ["S2Hand", "LabelHand", "JRCWaterHand"]

# Découpage par événement (pays), pour évaluer sur des zones jamais vues.
# - Bolivie : événement que les auteurs de Sen1Floods11 gardent eux-mêmes à part pour le test.
# - Espagne : seul événement européen.
# - Ghana : Afrique de l'Ouest, beaucoup d'images peu inondées (bon test des fausses alertes).
# - Validation : Sri Lanka (beaucoup d'eau permanente) et Pakistan.
PAYS_TEST = ["Bolivia", "Spain", "Ghana"]
PAYS_VALIDATION = ["Sri-Lanka", "Pakistan"]


def telecharger(dossier_data="data", radar=False):
    """Télécharge les images Sentinel-2, les masques d'eau et l'eau permanente (une seule fois).
    Avec radar=True, télécharge aussi les images radar Sentinel-1 (dossier S1Hand)."""
    os.makedirs(dossier_data, exist_ok=True)
    for nom in DOSSIERS + (["S1Hand"] if radar else []):
        if not os.path.isdir(os.path.join(dossier_data, nom)):
            subprocess.run(["gsutil", "-m", "-q", "cp", "-r", f"{BUCKET}/{nom}", dossier_data + "/"], check=True)
    return liste_images(dossier_data)


def liste_images(dossier_data="data"):
    """Noms des images, par exemple 'Bolivia_103757'."""
    fichiers = sorted(glob.glob(os.path.join(dossier_data, "S2Hand", "*_S2Hand.tif")))
    return [os.path.basename(f).replace("_S2Hand.tif", "") for f in fichiers]


def pays(nom):
    return nom.split("_")[0]


def partie(nom):
    """'train', 'validation' ou 'test' selon le pays de l'image."""
    p = pays(nom)
    if p in PAYS_TEST:
        return "test"
    if p in PAYS_VALIDATION:
        return "validation"
    return "train"


def _lire(chemin):
    with rasterio.open(chemin) as f:
        return f.read().astype("float32")


def lire_image(nom, dossier_data="data"):
    """Renvoie (s2, label, permanente).

    s2         : (13, H, W) réflectances Sentinel-2
    label      : (H, W) -1 = pas de donnée (souvent nuages), 0 = sec, 1 = eau
    permanente : (H, W) booléen, eau permanente (rivières, lacs)
    """
    s2 = _lire(os.path.join(dossier_data, "S2Hand", f"{nom}_S2Hand.tif"))
    label = _lire(os.path.join(dossier_data, "LabelHand", f"{nom}_LabelHand.tif"))[0]
    perm = _lire(os.path.join(dossier_data, "JRCWaterHand", f"{nom}_JRCWaterHand.tif"))[0] == 1
    return s2, label, perm


def lire_radar(nom, dossier_data="data"):
    """Radar Sentinel-1, forme (2, H, W) : polarisations VV et VH, en décibels.

    Le radar traverse les nuages ; l'eau calme renvoie peu de signal et apparaît très sombre
    (environ -20 dB et moins). Les pixels sans donnée (NaN) sont mis à -50 dB.
    """
    s1 = _lire(os.path.join(dossier_data, "S1Hand", f"{nom}_S1Hand.tif"))
    return np.clip(np.nan_to_num(s1, nan=-50.0, posinf=5.0, neginf=-50.0), -50.0, 5.0)


def ndwi(s2):
    """Indice d'eau de McFeeters : (B3 - B8) / (B3 + B8). Proche de +1 sur l'eau."""
    vert, pir = s2[2], s2[7]
    return (vert - pir) / (vert + pir + 1e-6)


def rgb(s2):
    """Image en vraies couleurs (B4, B3, B2), contraste étiré entre les percentiles 2 et 98."""
    img = s2[[3, 2, 1]].transpose(1, 2, 0)
    bas, haut = np.percentile(img, [2, 98])
    return np.clip((img - bas) / (haut - bas + 1e-6), 0, 1)


def part_inondee_vraie(label, perm):
    """Part de l'image (pixels annotés) couverte d'eau qui n'est pas de l'eau permanente."""
    valide = label >= 0
    if valide.sum() == 0:
        return np.nan
    inonde = (label == 1) & ~perm & valide
    return inonde.sum() / valide.sum()


def pixels_utiles(s2, perm):
    """Pixels où le satellite a vu le sol (pas de bord noir) et qui ne sont pas de l'eau permanente.

    Hypothèse : à bord, la carte d'eau permanente est connue à l'avance (c'est une carte fixe).
    """
    vu = s2.max(axis=0) > 0
    return vu & ~perm
