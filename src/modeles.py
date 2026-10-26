"""Réseaux de segmentation de l'eau (une probabilité d'eau par pixel)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalise(nn.Module):
    """Enveloppe un réseau et lui ajoute la normalisation des bandes.

    Le modèle exporté prend directement les réflectances brutes de Sentinel-2 (13 bandes) :
    rien à recalculer à bord. Sortie : logits d'eau, forme (B, 1, H, W).
    """

    def __init__(self, reseau, moyenne, ecart):
        super().__init__()
        self.reseau = reseau
        self.register_buffer("moyenne", torch.as_tensor(moyenne, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("ecart", torch.as_tensor(ecart, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return self.reseau((x - self.moyenne) / self.ecart)


def _bloc(entree, sortie):
    return nn.Sequential(
        nn.Conv2d(entree, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
        nn.Conv2d(sortie, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie), nn.ReLU(inplace=True),
    )


class PetitUNet(nn.Module):
    """U-Net minimal, entraîné de zéro (environ 0,5 million de paramètres avec base=16)."""

    def __init__(self, canaux=13, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1, self.enc2, self.enc3 = _bloc(canaux, c1), _bloc(c1, c2), _bloc(c2, c3)
        self.milieu = _bloc(c3, c4)
        self.haut3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), _bloc(c4, c3)
        self.haut2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), _bloc(c3, c2)
        self.haut1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), _bloc(c2, c1)
        self.sortie = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        m = self.milieu(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.haut3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.haut2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.haut1(d2), e1], dim=1))
        return self.sortie(d1)


def unet_mobilenet(canaux=13, pre_entraine=True):
    """U-Net dont l'encodeur est un MobileNetV2 pré-entraîné sur ImageNet (bibliothèque
    segmentation_models_pytorch). La première couche est adaptée de 3 à 13 bandes par la
    bibliothèque. Taille de l'image : multiple de 32.
    """
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pre_entraine else None,
        in_channels=canaux,
        classes=1,
        decoder_channels=(128, 64, 32, 16, 16),
    )


def nb_parametres(modele):
    return sum(p.numel() for p in modele.parameters())


def taille_mo(modele):
    """Taille des poids en float32, en mégaoctets."""
    return nb_parametres(modele) * 4 / 1e6
