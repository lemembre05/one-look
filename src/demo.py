"""Analyse d'une image Sentinel-2 (et, si on l'a, de l'image radar Sentinel-1) avec le modèle embarqué.

Tout ce qui est utilisé ici tiendrait à bord : les fichiers ONNX en 8 bits, les seuils de
config.json et quelques calculs simples. Aucune annotation n'est nécessaire.

Deux modèles : le modèle fusionné optique + radar (modeles/fusion) est utilisé quand l'image radar
est disponible ; sinon, le modèle optique seul (modeles) sert de secours.
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


class _Modele:
    """Un modèle ONNX + ses seuils + son détecteur d'inconnu (contenu d'un dossier avec config.json)."""

    def __init__(self, dossier, fils=1):
        with open(os.path.join(dossier, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.canaux = int(self.config.get("canaux", 13))
        self.chemin_modele = os.path.join(dossier, self.config["modele"])
        self.sess = embarque.session(self.chemin_modele, fils=fils)
        self.seuils = self.config["seuils_decision"]
        self.seuil_eau = self.config["seuil_probabilite_eau"]
        self.detecteur = decision.DetecteurInconnu.depuis_config(self.config["detecteur_inconnu"])


def radar_utilisable(s1):
    """Image radar présente et exploitable : 2 bandes (VV, VH) non constantes, au moins la moitié des pixels valides."""
    if s1 is None:
        return False
    s1 = np.asarray(s1, dtype=np.float32)
    if s1.ndim != 3 or s1.shape[0] != 2 or np.isfinite(s1).all(axis=0).mean() < 0.5:
        return False
    return bool((np.nanstd(s1.reshape(2, -1), axis=1) > 0.01).all())  # bande constante = capteur muet


def preparer_radar(s1):
    """Radar Sentinel-1 en dB (VV, VH), comme Sen1Floods11, puis en centièmes de dB pour le réseau."""
    s1 = np.clip(np.nan_to_num(np.asarray(s1, dtype=np.float32), nan=-50.0, posinf=5.0, neginf=-50.0), -50.0, 5.0)
    return s1 * 100


class OneLook:
    """Le système complet : modèle fusionné optique + radar, modèle optique en secours."""

    def __init__(self, dossier_modele="modeles", fils=1, dossier_fusion=None):
        self.optique = _Modele(dossier_modele, fils)
        dossier_fusion = dossier_fusion or os.path.join(dossier_modele, "fusion")
        self.fusion = _Modele(dossier_fusion, fils) if os.path.exists(os.path.join(dossier_fusion, "config.json")) else None
        # compatibilité avec l'ancienne interface
        self.config, self.seuils, self.detecteur = self.optique.config, self.optique.seuils, self.optique.detecteur

    def analyser(self, s2, perm=None, s1=None):
        """s2 : (13, H, W) réflectances Sentinel-2 L1C. perm : carte d'eau permanente (H, W),
        facultative ; sans elle, rivières et lacs comptent comme de l'eau anormale.
        s1 : (2, H, W) radar Sentinel-1 VV et VH en dB, facultatif ; s'il manque ou est inexploitable,
        le modèle optique seul est utilisé."""
        s2 = np.nan_to_num(np.asarray(s2, dtype=np.float32))
        if s2.ndim != 3 or s2.shape[0] != 13:
            raise ValueError(f"image attendue de forme (13, H, W), reçue {s2.shape}")
        if s1 is not None and np.asarray(s1).shape[1:] != s2.shape[1:]:
            raise ValueError(f"image radar de taille {np.asarray(s1).shape[1:]}, image optique {s2.shape[1:]}")
        perm_connue = perm is not None
        perm = np.zeros(s2.shape[1:], bool) if perm is None else np.asarray(perm) == 1

        avec_radar = self.fusion is not None and radar_utilisable(s1)
        m = self.fusion if avec_radar else self.optique
        x = np.concatenate([s2, preparer_radar(s1)]) if avec_radar else s2

        debut = time.perf_counter()
        h, w = s2.shape[1:]
        H, W = -(-h // 8) * 8, -(-w // 8) * 8          # le réseau veut des multiples de 8
        xp = np.pad(x, ((0, 0), (0, H - h), (0, W - w)))
        prob = embarque.probabilite_eau(m.sess, xp)[:h, :w]
        temps_reseau = (time.perf_counter() - debut) * 1000

        part = alerte.part_eau_reseau(prob, s2, perm, m.seuil_eau)
        nuages = decision.part_nuages(s2)
        amb = decision.ambiguite(prob, s2, perm)
        score = m.detecteur.score(x)
        d = decision.decider(part, nuages, amb, score, m.seuils)
        return {
            "decision": d, "explication": EXPLICATIONS[d],
            "modele": "fusion optique + radar" if avec_radar else "optique seul",
            "radar_fourni": s1 is not None, "radar_utilise": avec_radar,
            "part_eau": float(part), "nuages": float(nuages), "ambiguite": float(amb),
            "score_inconnu": float(score), "seuils": m.seuils,
            "temps_ms": float((time.perf_counter() - debut) * 1000), "temps_reseau_ms": float(temps_reseau),
            "eau_permanente_connue": perm_connue,
            "prob": prob, "eau": (prob > m.seuil_eau) & pixels_utiles(s2, perm),
        }


def lire_tif(chemin):
    with rasterio.open(chemin) as f:
        return f.read().astype(np.float32)


def resume(res):
    """Texte court expliquant la décision."""
    s = res["seuils"]
    lignes = [
        f"Décision : {res['decision']} : {res['explication']}",
        f"Modèle : {res.get('modele', 'optique seul')}",
        f"Eau anormale détectée : {res['part_eau']:.1%} de l'image (alerte à partir de {s['alerte']:.0%})",
        f"Nuages estimés : {res['nuages']:.0%} (incertain à partir de {s['nuages']:.0%})",
        f"Ambiguïté de la carte d'eau : {res['ambiguite']:.1%} (incertain à partir de {s['ambiguite']:.1%})",
        f"Score d'inconnu : {res['score_inconnu']:.1f} (inconnu au-delà de {s['inconnu']:.1f})",
        f"Temps de calcul : {res['temps_ms']:.0f} ms (réseau {res['temps_reseau_ms']:.0f} ms)",
    ]
    if res.get("radar_fourni") and not res.get("radar_utilise"):
        lignes.append("Radar fourni mais inexploitable (ou modèle fusionné absent) : modèle optique de secours.")
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
