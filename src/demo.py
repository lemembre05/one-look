"""Analyse d'une image Sentinel-2 quelconque avec le modèle embarqué (démonstration).

Tout ce qui est utilisé ici tiendrait à bord : le fichier ONNX en 8 bits (0,66 Mo), les seuils de
config.json et quelques calculs simples. Aucune annotation n'est nécessaire.
"""

import json
import os
import time

import numpy as np
import rasterio

from src import alerte, decision, embarque
from src.donnees import pixels_utiles, rgb

COULEURS = {"ALERTE": "#c62828", "PAS D'ALERTE": "#2e7d32", "INCERTAIN": "#ef6c00", "INCONNU": "#6a1b9a"}

EXPLICATIONS = {
    "ALERTE": "assez d'eau anormale est visible : l'image doit être envoyée au sol en priorité.",
    "PAS D'ALERTE": "la scène est bien visible et ne montre pas d'inondation.",
    "INCERTAIN": "pas assez d'eau visible, mais trop de nuages ou une carte d'eau trop ambiguë "
                 "pour affirmer qu'il n'y a pas d'inondation.",
    "INCONNU": "l'image ne ressemble à rien de ce qui a été vu à l'entraînement "
               "(capteur défaillant, scène inhabituelle) : aucune conclusion.",
}


class OneLook:
    """Le système complet : modèle ONNX + seuils + détecteur d'inconnu."""

    def __init__(self, dossier_modele="modeles", fils=1):
        with open(os.path.join(dossier_modele, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.chemin_modele = os.path.join(dossier_modele, self.config["modele"])
        self.sess = embarque.session(self.chemin_modele, fils=fils)
        self.seuils = self.config["seuils_decision"]
        self.seuil_eau = self.config["seuil_probabilite_eau"]
        self.detecteur = decision.DetecteurInconnu.depuis_config(self.config["detecteur_inconnu"])

    def analyser(self, s2, perm=None):
        """s2 : (13, H, W) réflectances Sentinel-2 L1C. perm : carte d'eau permanente (H, W),
        facultative ; sans elle, rivières et lacs comptent comme de l'eau anormale."""
        s2 = np.nan_to_num(np.asarray(s2, dtype=np.float32))
        if s2.ndim != 3 or s2.shape[0] != 13:
            raise ValueError(f"image attendue de forme (13, H, W), reçue {s2.shape}")
        perm_connue = perm is not None
        perm = np.zeros(s2.shape[1:], bool) if perm is None else np.asarray(perm) == 1

        debut = time.perf_counter()
        h, w = s2.shape[1:]
        H, W = -(-h // 8) * 8, -(-w // 8) * 8          # le réseau veut des multiples de 8
        x = np.pad(s2, ((0, 0), (0, H - h), (0, W - w)))
        prob = embarque.probabilite_eau(self.sess, x)[:h, :w]
        temps_reseau = (time.perf_counter() - debut) * 1000

        part = alerte.part_eau_reseau(prob, s2, perm, self.seuil_eau)
        nuages = decision.part_nuages(s2)
        amb = decision.ambiguite(prob, s2, perm)
        score = self.detecteur.score(s2)
        d = decision.decider(part, nuages, amb, score, self.seuils)
        return {
            "decision": d, "explication": EXPLICATIONS[d],
            "part_eau": float(part), "nuages": float(nuages), "ambiguite": float(amb),
            "score_inconnu": float(score), "seuils": self.seuils,
            "temps_ms": float((time.perf_counter() - debut) * 1000), "temps_reseau_ms": float(temps_reseau),
            "eau_permanente_connue": perm_connue,
            "prob": prob, "eau": (prob > self.seuil_eau) & pixels_utiles(s2, perm),
        }


def lire_tif(chemin):
    with rasterio.open(chemin) as f:
        return f.read().astype(np.float32)


def resume(res):
    """Texte court expliquant la décision."""
    s = res["seuils"]
    lignes = [
        f"Décision : {res['decision']} : {res['explication']}",
        f"Eau anormale détectée : {res['part_eau']:.1%} de l'image (alerte à partir de {s['alerte']:.0%})",
        f"Nuages estimés : {res['nuages']:.0%} (incertain à partir de {s['nuages']:.0%})",
        f"Ambiguïté de la carte d'eau : {res['ambiguite']:.1%} (incertain à partir de {s['ambiguite']:.1%})",
        f"Score d'inconnu : {res['score_inconnu']:.1f} (inconnu au-delà de {s['inconnu']:.1f})",
        f"Temps de calcul : {res['temps_ms']:.0f} ms (réseau {res['temps_reseau_ms']:.0f} ms)",
    ]
    if not res["eau_permanente_connue"]:
        lignes.append("Attention : carte d'eau permanente non fournie ; rivières et lacs comptent comme eau anormale.")
    return "\n".join(lignes)


def figure(s2, res, titre=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(rgb(np.nan_to_num(s2))); ax[0].set_title("Image (vraies couleurs)")
    im = ax[1].imshow(res["prob"].astype(np.float32), cmap="Blues", vmin=0, vmax=1)
    ax[1].set_title("Probabilité d'eau"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(rgb(np.nan_to_num(s2)))
    ax[2].imshow(np.ma.masked_where(~res["eau"], res["eau"]), cmap="autumn", alpha=0.6)
    ax[2].set_title(f"Eau anormale : {res['part_eau']:.1%}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{titre}  →  {res['decision']}".strip(" →"), color=COULEURS[res["decision"]],
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig
