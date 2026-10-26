"""Entraînement et prédiction des réseaux de segmentation de l'eau."""

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def statistiques_bandes(images, indices, pas=8):
    """Moyenne et écart-type de chaque bande, sur les images d'entraînement uniquement
    (un pixel sur `pas` dans chaque direction, pixels vus seulement)."""
    echantillons = []
    for k in indices:
        x = images[k][:, ::pas, ::pas].reshape(images[k].shape[0], -1).astype(np.float32)
        echantillons.append(x[:, x.max(axis=0) > 0])
    x = np.concatenate(echantillons, axis=1)
    return x.mean(axis=1), x.std(axis=1) + 1e-6


class Decoupes(Dataset):
    """Découpes aléatoires de taille `taille` x `taille`, avec retournements et rotations."""

    def __init__(self, images, labels, indices, taille=256, par_image=4):
        self.images, self.labels = images, labels
        self.indices = list(indices)
        self.taille, self.par_image = taille, par_image

    def __len__(self):
        return len(self.indices) * self.par_image

    def __getitem__(self, i):
        k = self.indices[i % len(self.indices)]
        img, lab = self.images[k], self.labels[k]
        t = self.taille
        h, w = lab.shape
        y0 = int(torch.randint(0, h - t + 1, (1,)))
        x0 = int(torch.randint(0, w - t + 1, (1,)))
        x = torch.from_numpy(img[:, y0:y0 + t, x0:x0 + t].astype(np.float32))
        y = torch.from_numpy(lab[y0:y0 + t, x0:x0 + t].astype(np.float32))
        if torch.rand(1) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        if torch.rand(1) < 0.5:
            x, y = x.flip(-2), y.flip(-2)
        r = int(torch.randint(0, 4, (1,)))
        return torch.rot90(x, r, (-2, -1)).contiguous(), torch.rot90(y, r, (-2, -1)).contiguous()


def perte(logits, y):
    """Entropie croisée + Dice, calculées uniquement sur les pixels annotés (y >= 0).

    y : -1 = pas de donnée, 0 = sec, 1 = eau. Le Dice compense le déséquilibre
    (l'eau ne couvre que quelques pourcents des pixels).
    """
    valide = (y >= 0).float()
    cible = (y == 1).float()
    n = valide.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(logits, cible, reduction="none") * valide).sum() / n
    p = torch.sigmoid(logits) * valide
    dice = 1 - (2 * (p * cible).sum() + 1) / (p.sum() + (cible * valide).sum() + 1)
    return bce + dice


@torch.no_grad()
def predire(modele, images, indices, appareil, lot=8):
    """Probabilité d'eau pour chaque image complète. Renvoie {indice: tableau (H, W) float16}."""
    modele.eval()
    probs = {}
    indices = list(indices)
    for i in range(0, len(indices), lot):
        ks = indices[i:i + lot]
        x = torch.from_numpy(np.stack([images[k] for k in ks]).astype(np.float32)).to(appareil)
        p = torch.sigmoid(modele(x)).squeeze(1).float().cpu().numpy()
        for k, pk in zip(ks, p):
            probs[k] = pk.astype(np.float16)
    return probs


def iou_eau(probs, labels, indices, seuil=0.5):
    """IoU de l'eau (pixels annotés seulement), toutes images confondues."""
    inter = union = 0
    for k in indices:
        v = labels[k] >= 0
        p = probs[k] > seuil
        t = labels[k] == 1
        inter += int((p & t & v).sum())
        union += int(((p | t) & v).sum())
    return inter / union if union else 0.0


def entrainer(modele, images, labels, idx_train, idx_val, appareil,
              epoques=30, lot=16, taille=256, lr=1e-3, graine=0, journal=print):
    """Entraîne `modele` sur des découpes des images d'entraînement.

    À chaque époque, l'IoU de l'eau est mesurée sur la validation ; on garde les poids
    de la meilleure époque. Renvoie l'historique (perte, IoU) sous forme de DataFrame.
    """
    torch.manual_seed(graine)
    np.random.seed(graine)
    modele.to(appareil)
    charge = DataLoader(Decoupes(images, labels, idx_train, taille), batch_size=lot, shuffle=True,
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epoques * len(charge))
    amp = appareil == "cuda"
    echelle = torch.amp.GradScaler("cuda", enabled=amp)

    meilleur, meilleur_etat, histo = -1.0, None, []
    debut = time.time()
    for ep in range(epoques):
        modele.train()
        total = 0.0
        for x, y in charge:
            x, y = x.to(appareil, non_blocking=True), y.to(appareil, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = modele(x).squeeze(1)
            l = perte(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            echelle.scale(l).backward()
            echelle.step(opt)
            echelle.update()
            plan.step()
            total += l.item()
        iou = iou_eau(predire(modele, images, idx_val, appareil), labels, idx_val)
        histo.append({"epoque": ep + 1, "perte": total / len(charge), "iou_validation": iou})
        if iou > meilleur:
            meilleur, meilleur_etat = iou, copy.deepcopy(modele.state_dict())
        journal(f"époque {ep + 1:2d}/{epoques} | perte {total / len(charge):.3f} | "
                f"IoU eau validation {iou:.3f} | {time.time() - debut:.0f} s")
    modele.load_state_dict(meilleur_etat)
    return pd.DataFrame(histo)


def latence_cpu_ms(modele, taille=512, canaux=13, repetitions=5, fils=1):
    """Temps moyen d'inférence d'une image complète sur processeur, avec `fils` cœur(s)."""
    anciens = torch.get_num_threads()
    torch.set_num_threads(fils)
    m = copy.deepcopy(modele).cpu().eval()
    x = torch.rand(1, canaux, taille, taille) * 3000
    with torch.no_grad():
        m(x)  # échauffement
        debut = time.perf_counter()
        for _ in range(repetitions):
            m(x)
    torch.set_num_threads(anciens)
    return (time.perf_counter() - debut) / repetitions * 1000
