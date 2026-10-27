"""Démonstration One Look : déposer une image Sentinel-2 (GeoTIFF, 13 bandes) et obtenir la décision.

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


def analyser(fichier_s2, fichier_perm, transformation):
    if fichier_s2 is None:
        return None, "Déposer une image Sentinel-2 (GeoTIFF à 13 bandes)."
    chemin = fichier_s2 if isinstance(fichier_s2, str) else fichier_s2.name
    try:
        s2 = TRANSFORMATIONS[transformation](demo.lire_tif(chemin))
        perm = None
        if fichier_perm is not None:
            perm = demo.lire_tif(fichier_perm if isinstance(fichier_perm, str) else fichier_perm.name)[0]
        res = systeme.analyser(s2, perm)
    except Exception as e:
        return None, f"Image refusée : {e}"
    return demo.figure(s2, res, os.path.basename(chemin)), demo.resume(res)


exemples_dir = os.path.join(RACINE, "exemples")
exemples = []
if os.path.isdir(exemples_dir):
    for f in sorted(os.listdir(exemples_dir)):
        if f.endswith("_S2Hand.tif"):
            perm = os.path.join(exemples_dir, f.replace("_S2Hand", "_JRCWaterHand"))
            exemples.append([os.path.join(exemples_dir, f), perm if os.path.exists(perm) else None, "aucune"])

with gr.Blocks(title="One Look") as app:
    gr.Markdown("# One Look : alerte inondation embarquée\n"
                "Modèle de 0,66 Mo, une seule passe. Réponses possibles : **ALERTE**, **PAS D'ALERTE**, "
                "**INCERTAIN** (nuages, carte ambiguë) ou **INCONNU** (image anormale).")
    with gr.Row():
        with gr.Column(scale=1):
            entree = gr.File(label="Image Sentinel-2 (GeoTIFF, 13 bandes L1C)", file_types=[".tif", ".tiff"])
            perm = gr.File(label="Carte d'eau permanente (facultative)", file_types=[".tif", ".tiff"])
            transfo = gr.Radio(list(TRANSFORMATIONS), value="aucune", label="Simuler une anomalie")
            bouton = gr.Button("Analyser", variant="primary")
        with gr.Column(scale=2):
            sortie_fig = gr.Plot(label="Résultat")
            sortie_txt = gr.Textbox(label="Explication", lines=7)
    bouton.click(analyser, [entree, perm, transfo], [sortie_fig, sortie_txt])
    if exemples:
        gr.Examples(exemples, [entree, perm, transfo], label="Images de test (pays jamais vus à l'entraînement)")

if __name__ == "__main__":
    app.launch(share="--partage" in sys.argv)
