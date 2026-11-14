"""Extension aux feux de forêt : détecter et détourer les zones brûlées, avec la même logique de
décision que pour les inondations (ALERTE, PAS D'ALERTE, INCERTAIN, INCONNU).

Pas de réseau entraîné ici : faute de jeu de données annoté sur les feux dans le temps du projet,
on utilise l'indice physique de référence, le NBR (Normalized Burn Ratio) :

    NBR = (PIR - IRM) / (PIR + IRM),  PIR = bande B8 (842 nm), IRM = bande B12 (2190 nm)

La végétation saine renvoie fortement le proche infrarouge et peu l'infrarouge moyen (NBR élevé) ;
une zone brûlée fait l'inverse (NBR bas). La différence avant / après, dNBR = NBR_avant - NBR_après,
mesure la sévérité du feu. Seuils de sévérité de l'USGS (Key et Benson, 2006) :

    dNBR < 0,10 non brûlé ; 0,10-0,27 faible ; 0,27-0,44 modérée-faible ;
    0,44-0,66 modérée-forte ; > 0,66 forte.

Le calcul ne demande que deux bandes et quelques opérations par pixel : il tient sans difficulté
à bord d'un satellite.
"""

import numpy as np
from scipy import ndimage

SEUILS_SEVERITE = [0.10, 0.27, 0.44, 0.66]
NOMS_SEVERITE = ["non brûlé", "faible", "modérée-faible", "modérée-forte", "forte"]
SEUIL_BRULE = 0.27            # à partir de la sévérité « modérée-faible » : zone brûlée certaine
SURFACE_MIN_PIXELS = 100      # 1 ha à 10 m : en dessous, bruit ou petit changement agricole
PART_ALERTE = 0.05            # même définition que pour les inondations : 5 % de la zone
NUAGES_MAX = 0.30             # même seuil d'incertitude que pour les inondations

# Classes de la carte de scène (SCL) des produits Sentinel-2 L2A
SCL_INVALIDES = {0, 1, 3, 8, 9, 10}   # sans donnée, saturé, ombre de nuage, nuages, cirrus
SCL_EAU = 6


def nbr(pir, irm):
    pir = pir.astype(np.float32); irm = irm.astype(np.float32)
    return (pir - irm) / np.maximum(pir + irm, 1e-6)


def dnbr(pir_avant, irm_avant, pir_apres, irm_apres):
    return nbr(pir_avant, irm_avant) - nbr(pir_apres, irm_apres)


def severite(d):
    """Classe de sévérité 0 (non brûlé) à 4 (forte) pour chaque pixel."""
    return np.digitize(d, SEUILS_SEVERITE)


def carte_brulee(d, valides, seuil=SEUIL_BRULE, surface_min=SURFACE_MIN_PIXELS):
    """Masque des zones brûlées : dNBR au-dessus du seuil, trous bouchés, petites taches retirées."""
    m = (d >= seuil) & valides
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=1) & valides
    m = ndimage.binary_fill_holes(m)
    etiquettes, n = ndimage.label(m)
    if n == 0:
        return m
    tailles = ndimage.sum(m, etiquettes, index=np.arange(1, n + 1))
    garder = np.zeros(n + 1, bool); garder[1:] = tailles >= surface_min
    return garder[etiquettes]


def decider_feu(part_brulee, part_invalide, part_alerte=PART_ALERTE, nuages_max=NUAGES_MAX):
    """Même ordre de décision que pour les inondations (l'inconnu est traité à part)."""
    if part_brulee >= part_alerte:
        return "ALERTE"
    if part_invalide >= nuages_max:
        return "INCERTAIN"
    return "PAS D'ALERTE"


def contours(masque, transform, crs, taille_pixel_m=10.0, surface_min_ha=1.0):
    """Détourage : polygones des zones brûlées (GeoJSON), avec leur surface en hectares."""
    from rasterio.features import shapes

    entites = []
    for geom, val in shapes(masque.astype(np.uint8), mask=masque, transform=transform):
        if val != 1:
            continue
        # surface d'après les pixels de l'anneau extérieur moins les trous (formule du lacet)
        def aire(anneau):
            x, y = np.array(anneau).T
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        anneaux = geom["coordinates"]
        ha = (aire(anneaux[0]) - sum(aire(a) for a in anneaux[1:])) / 1e4
        if ha >= surface_min_ha:
            entites.append({"type": "Feature", "geometry": geom, "properties": {"surface_ha": round(float(ha), 1)}})
    entites.sort(key=lambda e: -e["properties"]["surface_ha"])
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": str(crs)}},
            "features": entites}
