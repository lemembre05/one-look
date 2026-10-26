"""Passage d'une carte d'eau à une décision d'alerte (oui / non) pour une image."""

import numpy as np
import pandas as pd

from src.donnees import ndwi, pixels_utiles
from src.metriques import scores_alerte

GRILLE_ALERTE = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)

# Seuils de la baseline, choisis sur l'entraînement dans le notebook 02
BASELINE_T_NDWI = 0.0
BASELINE_T_ALERTE = 0.03


def part_eau(eau, s2, perm):
    """Part des pixels vus détectés comme eau, hors eau permanente."""
    vu = s2.max(axis=0) > 0
    return (eau & pixels_utiles(s2, perm)).sum() / max(vu.sum(), 1)


def part_eau_ndwi(s2, perm, t_ndwi=BASELINE_T_NDWI):
    return part_eau(ndwi(s2) > t_ndwi, s2, perm)


def part_eau_reseau(prob, s2, perm, seuil=0.5):
    return part_eau(prob > seuil, s2, perm)


def choisir_seuil_alerte(vrai, part, seuil_alerte, grille=GRILLE_ALERTE):
    """Seuil t_alerte qui maximise le F1 ; à égalité, moins de fausses alertes,
    puis le seuil le plus proche de la définition de l'alerte."""
    essais = pd.DataFrame([{"t_alerte": t, **scores_alerte(vrai, np.asarray(part) >= t)} for t in grille])
    essais["ecart"] = (essais["t_alerte"] - seuil_alerte).abs()
    essais = essais.sort_values(["f1", "taux_fausse_alerte", "ecart"], ascending=[False, True, True])
    return float(essais.iloc[0]["t_alerte"]), essais
