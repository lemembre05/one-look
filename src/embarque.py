"""Version embarquée du modèle : export ONNX, compression en 8 bits, inférence sur processeur."""

import os
import time

import numpy as np
import torch


def exporter_onnx(modele, chemin, canaux=13, taille=512):
    """Exporte le modèle (normalisation comprise) au format ONNX.

    Entrée « reflectances » : (lot, canaux, hauteur, largeur), réflectances brutes de Sentinel-2.
    Sortie « logits » : (lot, 1, hauteur, largeur) ; probabilité d'eau = sigmoïde(logits).
    Hauteur et largeur doivent être des multiples de 8.
    """
    m = modele.cpu().eval()
    x = torch.rand(1, canaux, taille, taille) * 3000
    options = dict(input_names=["reflectances"], output_names=["logits"], opset_version=17,
                   dynamic_axes={"reflectances": {0: "lot", 2: "hauteur", 3: "largeur"},
                                 "logits": {0: "lot", 2: "hauteur", 3: "largeur"}})
    try:
        torch.onnx.export(m, x, chemin, dynamo=False, **options)
    except TypeError:  # anciennes versions de PyTorch, sans le paramètre dynamo
        torch.onnx.export(m, x, chemin, **options)
    return chemin


def quantifier_int8(chemin_fp32, chemin_int8, images_calibration):
    """Compression en entiers 8 bits (quantification statique, format QDQ).

    Seules les convolutions sont quantifiées : la normalisation d'entrée reste en virgule
    flottante, pour ne pas perdre la précision des réflectances brutes.
    """
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)

    source = chemin_fp32
    try:  # optimisation préalable recommandée (fusion BatchNorm + Conv, inférence des formes)
        from onnxruntime.quantization.shape_inference import quant_pre_process
        source = chemin_fp32.replace(".onnx", "_pre.onnx")
        quant_pre_process(chemin_fp32, source, skip_symbolic_shape=True)
    except Exception as e:
        print("pré-traitement ignoré :", e)
        source = chemin_fp32

    class LecteurCalibration(CalibrationDataReader):
        """Fournit quelques images réelles pour mesurer la plage des valeurs internes du réseau."""

        def __init__(self, nom_entree, images):
            self.suite = iter([{nom_entree: img[None].astype(np.float32)} for img in images])

        def get_next(self):
            return next(self.suite, None)

    import onnx
    nom_entree = onnx.load(source).graph.input[0].name
    lecteur = LecteurCalibration(nom_entree, images_calibration)
    quantize_static(source, chemin_int8, lecteur, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv"])
    if source != chemin_fp32 and os.path.exists(source):
        os.remove(source)
    return chemin_int8


def session(chemin, fils=1):
    """Session ONNX Runtime sur processeur, limitée à `fils` cœur(s)."""
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.intra_op_num_threads = fils
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(chemin, o, providers=["CPUExecutionProvider"])


def probabilite_eau(sess, s2):
    """Carte de probabilité d'eau (H, W) pour une image (13, H, W)."""
    nom = sess.get_inputs()[0].name
    logits = sess.run(None, {nom: s2[None].astype(np.float32)})[0][0, 0]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def predire(sess, images, indices):
    return {k: probabilite_eau(sess, images[k]).astype(np.float16) for k in indices}


def latence_ms(sess, canaux=13, taille=512, repetitions=5):
    x = (np.random.rand(1, canaux, taille, taille) * 3000).astype(np.float32)
    nom = sess.get_inputs()[0].name
    sess.run(None, {nom: x})  # échauffement
    debut = time.perf_counter()
    for _ in range(repetitions):
        sess.run(None, {nom: x})
    return (time.perf_counter() - debut) / repetitions * 1000


def taille_fichier_mo(chemin):
    return os.path.getsize(chemin) / 1e6
