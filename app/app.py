"""Démonstration One Look : déposer une image Sentinel-2 (GeoTIFF, 13 bandes), si possible l'image radar
Sentinel-1 (GeoTIFF, VV et VH en dB), et obtenir la décision.

Lancement : python app/app.py  (depuis la racine du dépôt, avec modeles/config.json et le modèle ONNX)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np

from src import demo

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
systeme = demo.OneLook(os.path.join(RACINE, "modeles"))

TRANSFORMATIONS = {
    "aucune": lambda x: x,
    "capteur saturé (× 3)": lambda x: x * 3,
    "bandes infrarouges perdues": lambda x: np.concatenate([x[:7], np.zeros_like(x[7:])]),
    "bruit": lambda x: np.random.default_rng(0).uniform(0, 10000, x.shape).astype(np.float32),
}


def _chemin(f):
    return None if f is None else (f if isinstance(f, str) else f.name)


def analyser(fichier_s2, fichier_s1, fichier_perm, transformation, radar_perdu):
    if fichier_s2 is None:
        return None, "Déposer une image Sentinel-2 (GeoTIFF à 13 bandes)."
    chemin = _chemin(fichier_s2)
    try:
        s2 = TRANSFORMATIONS[transformation](demo.lire_tif(chemin))
        perm = None if fichier_perm is None else demo.lire_tif(_chemin(fichier_perm))[0]
        s1 = None if fichier_s1 is None else demo.lire_tif(_chemin(fichier_s1))
        if radar_perdu and s1 is not None:
            s1 = np.full_like(s1, np.nan)          # liaison radar coupée : le système passe en secours
        res = systeme.analyser(s2, perm, s1)
    except Exception as e:
        return None, f"Image refusée : {e}"
    return demo.figure(s2, res, os.path.basename(chemin)), demo.resume(res)


exemples_dir = os.path.join(RACINE, "exemples")
exemples = []
if os.path.isdir(exemples_dir):
    for f in sorted(os.listdir(exemples_dir)):
        if f.endswith("_S2Hand.tif"):
            perm = os.path.join(exemples_dir, f.replace("_S2Hand", "_JRCWaterHand"))
            s1 = os.path.join(exemples_dir, f.replace("_S2Hand", "_S1Hand"))
            exemples.append([os.path.join(exemples_dir, f), s1 if os.path.exists(s1) else None,
                             perm if os.path.exists(perm) else None, "aucune", False])

with gr.Blocks(title="One Look") as app:
    gr.Markdown("# One Look : alerte inondation embarquée\n"
                "Optique + radar quand le radar est là, optique seul sinon. Une seule passe. Réponses possibles : **ALERTE**, **PAS D'ALERTE**, "
                "**INCERTAIN** (nuages, carte ambiguë) ou **INCONNU** (image anormale).")
    with gr.Row():
        with gr.Column(scale=1):
            entree = gr.File(label="Image Sentinel-2 (GeoTIFF, 13 bandes L1C)", file_types=[".tif", ".tiff"])
            radar = gr.File(label="Image radar Sentinel-1 (facultative, VV et VH en dB)", file_types=[".tif", ".tiff"])
            perm = gr.File(label="Carte d'eau permanente (facultative)", file_types=[".tif", ".tiff"])
            transfo = gr.Radio(list(TRANSFORMATIONS), value="aucune", label="Simuler une anomalie optique")
            perdu = gr.Checkbox(value=False, label="Simuler la perte du radar")
            bouton = gr.Button("Analyser", variant="primary")
        with gr.Column(scale=2):
            sortie_fig = gr.Plot(label="Résultat")
            sortie_txt = gr.Textbox(label="Explication", lines=9)
    bouton.click(analyser, [entree, radar, perm, transfo, perdu], [sortie_fig, sortie_txt])
    if exemples:
        gr.Examples(exemples, [entree, radar, perm, transfo, perdu], label="Images de test (pays jamais vus à l'entraînement)")

if __name__ == "__main__":
    app.launch(share="--partage" in sys.argv)
