"""Métriques d'évaluation des alertes (une décision oui / non par image)."""

import numpy as np


def scores_alerte(vrai, predit):
    """vrai, predit : tableaux de booléens (True = alerte).

    - precision          : parmi les alertes déclenchées, part de vraies inondations
    - rappel             : parmi les vraies inondations, part détectée
    - taux_fausse_alerte : parmi les images normales, part qui déclenche une alerte
    """
    vrai = np.asarray(vrai, dtype=bool)
    predit = np.asarray(predit, dtype=bool)
    vp = int((vrai & predit).sum())
    fp = int((~vrai & predit).sum())
    fn = int((vrai & ~predit).sum())
    vn = int((~vrai & ~predit).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    taux_fa = fp / (fp + vn) if fp + vn else 0.0
    return {
        "images": len(vrai),
        "inondees": vp + fn,
        "vrais_positifs": vp,
        "fausses_alertes": fp,
        "manquees": fn,
        "precision": round(precision, 3),
        "rappel": round(rappel, 3),
        "f1": round(f1, 3),
        "taux_fausse_alerte": round(taux_fa, 3),
    }
