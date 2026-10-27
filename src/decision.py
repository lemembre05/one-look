"""Décision finale : ALERTE, PAS D'ALERTE, ou « je ne sais pas » (INCERTAIN, INCONNU).

Un système d'alerte fiable doit savoir dire qu'il ne sait pas plutôt que d'affirmer
« pas d'inondation » quand il n'a pas pu voir le sol. Quatre réponses possibles :

- INCONNU      : l'image ne ressemble à rien de ce qui a été vu à l'entraînement
                 (capteur défaillant, phénomène inconnu) : aucune conclusion.
- ALERTE       : assez d'eau anormale est visible, même si une partie de l'image est cachée.
- INCERTAIN    : pas assez d'eau visible, mais trop de nuages ou une carte d'eau trop ambiguë
                 pour conclure qu'il n'y a pas d'inondation.
- PAS D'ALERTE : la scène est bien visible et ne contient pas d'inondation.
"""

import numpy as np
import pandas as pd

from src.donnees import pixels_utiles
from src.metriques import scores_alerte

ETATS = ["ALERTE", "PAS D'ALERTE", "INCERTAIN", "INCONNU"]

# Pixel nuageux : bande bleue (B2) plus réfléchissante que 0,2 (les nuages sont très brillants,
# l'eau et la végétation sont sombres dans le bleu). Règle simple et explicable.
SEUIL_BLEU_NUAGE = 2000


def part_nuages(s2):
    """Part des pixels vus estimés nuageux, d'après la seule image (pas d'annotation à bord)."""
    vu = s2.max(axis=0) > 0
    return ((s2[1] > SEUIL_BLEU_NUAGE) & vu).sum() / max(vu.sum(), 1)


def ambiguite(prob, s2, perm, bas=0.2, haut=0.8):
    """Part des pixels utiles où le réseau hésite (probabilité d'eau entre `bas` et `haut`)."""
    utiles = pixels_utiles(s2, perm)
    p = prob.astype(np.float32)[utiles]
    return float(((p > bas) & (p < haut)).mean()) if p.size else 1.0


def signature(s2):
    """Réflectance moyenne de chaque bande sur les pixels vus : résumé de l'image."""
    vu = s2.max(axis=0) > 0
    if not vu.any():
        return np.zeros(s2.shape[0])
    return s2[:, vu].mean(axis=1)


class DetecteurInconnu:
    """Repère les images très différentes de l'entraînement (hors distribution).

    Score = plus grand écart, en nombre d'écarts-types, entre la réflectance moyenne d'une bande
    et ce qui a été vu à l'entraînement. Une image est « inconnue » si son score dépasse de
    `marge` (20 % par défaut) le score le plus élevé observé sur les images d'entraînement.
    """

    def __init__(self, signatures_entrainement, marge=1.2):
        sig = np.asarray(signatures_entrainement, dtype=np.float64)
        self.moyenne = sig.mean(axis=0)
        self.ecart = sig.std(axis=0) + 1e-6
        self.seuil = marge * max(self.score_signature(s) for s in sig)

    def score_signature(self, sig):
        return float(np.max(np.abs((np.asarray(sig) - self.moyenne) / self.ecart)))

    def score(self, s2):
        return self.score_signature(signature(s2))

    def config(self):
        return {"moyenne": self.moyenne.round(2).tolist(), "ecart": self.ecart.round(2).tolist(),
                "seuil": round(self.seuil, 3)}


def decider(part_eau, nuages, amb, score_inconnu, seuils):
    """seuils : dict avec les clés « alerte », « nuages », « ambiguite », « inconnu »."""
    if score_inconnu > seuils["inconnu"]:
        return "INCONNU"
    if part_eau >= seuils["alerte"]:
        return "ALERTE"
    if nuages >= seuils["nuages"] or amb >= seuils["ambiguite"]:
        return "INCERTAIN"
    return "PAS D'ALERTE"


def bilan(decisions, vrai):
    """Résumé d'un ensemble de décisions face à la vérité (True = inondation)."""
    d = pd.Series(list(decisions))
    v = np.asarray(vrai, dtype=bool)
    sortie = {"images": len(d)}
    for etat in ETATS:
        sortie[etat] = int((d == etat).sum())
    decide = d.isin(["ALERTE", "PAS D'ALERTE"]).to_numpy()
    s = scores_alerte(v[decide], (d[decide] == "ALERTE").to_numpy())
    sortie.update({
        "couverture": round(float(decide.mean()), 3),
        "precision": s["precision"], "rappel": s["rappel"], "f1": s["f1"],
        "taux_fausse_alerte": s["taux_fausse_alerte"],
        "inondations_manquees_sans_prevenir": int(((d == "PAS D'ALERTE").to_numpy() & v).sum()),
        "inondations_signalees_incertaines": int((d.isin(["INCERTAIN", "INCONNU"]).to_numpy() & v).sum()),
    })
    return sortie
