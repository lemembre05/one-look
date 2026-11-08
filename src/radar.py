"""Images radar Sentinel-1 hors de Sen1Floods11 : recherche, étalonnage et mise sur la grille Sentinel-2.

Sen1Floods11 fournit le radar déjà traité par Google Earth Engine : coefficient de rétrodiffusion
sigma0 en dB, bruit thermique retiré, géoréférencé à 10 m. Pour une image nouvelle (démo Italie), on
refait ce traitement à partir du produit GRD brut de Copernicus (copie publique de Microsoft
Planetary Computer) :

1. sigma0 = (DN² - bruit) / A², avec A la table d'étalonnage « sigmaNought » et le bruit thermique
   des tables du produit (dossier annotation/) ;
2. passage de la géométrie radar à la grille de l'image Sentinel-2 grâce aux points d'appui (GCP)
   du produit. Sans correction de relief : valable en plaine (Émilie-Romagne, quelques mètres
   d'altitude), pas en montagne ;
3. conversion en dB, bornée à [-50, 5] comme dans src.donnees.lire_radar.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def chercher(lon, lat, date, jours=6, collection="sentinel-1-grd"):
    """Acquisitions Sentinel-1 IW couvrant le point (lon, lat), triées par écart à `date` (AAAA-MM-JJ)."""
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    d = np.datetime64(date)
    fenetre = f"{d - np.timedelta64(jours, 'D')}/{d + np.timedelta64(jours, 'D')}"
    items = list(cat.search(collections=[collection], intersects={"type": "Point", "coordinates": [lon, lat]},
                            datetime=fenetre).items())
    items = [i for i in items if i.properties.get("sar:instrument_mode", "IW") == "IW"]
    cible = datetime.fromisoformat(date)
    return sorted(items, key=lambda i: abs((i.datetime.replace(tzinfo=None) - cible).total_seconds()))


def _asset(item, mot, pol):
    for cle, a in item.assets.items():
        if mot in cle.lower() and pol in cle.lower():
            return a.href
    return None


def _texte_xml(url):
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return ET.fromstring(r.read())


def _nombres(t):
    return np.array(t.split(), dtype=np.float64)


def _grille(vecteurs, lignes, colonnes):
    """Interpole une table définie par vecteurs (ligne, pixels, valeurs) sur la fenêtre lignes x colonnes."""
    vecteurs = sorted(vecteurs, key=lambda v: v[0])
    l_ref = np.array([v[0] for v in vecteurs], dtype=np.float64)
    par_ligne = np.stack([np.interp(colonnes, v[1], v[2]) for v in vecteurs])   # (n_vecteurs, colonnes)
    sortie = np.empty((len(lignes), len(colonnes)), np.float64)
    for j in range(len(colonnes)):
        sortie[:, j] = np.interp(lignes, l_ref, par_ligne[:, j])
    return sortie


def etalonnage(racine, lignes, colonnes):
    vec = [(int(v.findtext("line")), _nombres(v.findtext("pixel")), _nombres(v.findtext("sigmaNought")))
           for v in racine.iter("calibrationVector")]
    return _grille(vec, lignes, colonnes)


def bruit(racine, lignes, colonnes):
    """Bruit thermique (format IPF >= 2.9 : table en distance x table en azimut). None si absent."""
    vr = [(int(v.findtext("line")), _nombres(v.findtext("pixel")), _nombres(v.findtext("noiseRangeLut")))
          for v in racine.iter("noiseRangeVector")]
    if not vr:
        return None
    n = _grille(vr, lignes, colonnes)
    az = np.ones_like(n)
    for b in racine.iter("noiseAzimuthVector"):
        c0, c1 = int(b.findtext("firstRangeSample")), int(b.findtext("lastRangeSample"))
        l0, l1 = int(b.findtext("firstAzimuthLine")), int(b.findtext("lastAzimuthLine"))
        cols = (colonnes >= c0) & (colonnes <= c1)
        rows = (lignes >= l0) & (lignes <= l1)
        if not cols.any() or not rows.any():
            continue
        lut = _nombres(b.findtext("noiseAzimuthLut"))
        lig = _nombres(b.findtext("line")) if b.find("line") is not None else np.array([l0], float)
        val = np.interp(lignes[rows], lig, lut) if lut.size > 1 else np.full(rows.sum(), lut[0])
        az[np.ix_(rows, cols)] = val[:, None]
    return n * az


def sur_grille(item, crs, transform, taille, marge_m=15000):
    """Image (2, taille, taille) VV, VH en dB sur la grille (crs, transform) de l'image Sentinel-2."""
    import rasterio
    from pyproj import Transformer
    from rasterio.control import GroundControlPoint
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window

    x0, y0 = transform.c, transform.f
    x1, y1 = x0 + taille * transform.a, y0 + taille * transform.e
    vers_s2 = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    bandes = []
    for pol in ["vv", "vh"]:
        href = item.assets[pol].href
        with rasterio.open(href) as f:
            gcps, gcp_crs = f.gcps
            px = np.array([g.col for g in gcps]); ln = np.array([g.row for g in gcps])
            gx, gy = vers_s2.transform(np.array([g.x for g in gcps]), np.array([g.y for g in gcps]))
            dedans = ((gx > min(x0, x1) - marge_m) & (gx < max(x0, x1) + marge_m)
                      & (gy > min(y0, y1) - marge_m) & (gy < max(y0, y1) + marge_m))
            if dedans.sum() < 4:
                raise RuntimeError("zone hors de l'image radar")
            c0, c1 = int(max(px[dedans].min(), 0)), int(min(px[dedans].max(), f.width - 1))
            l0, l1 = int(max(ln[dedans].min(), 0)), int(min(ln[dedans].max(), f.height - 1))
            dn = f.read(1, window=Window(c0, l0, c1 - c0 + 1, l1 - l0 + 1)).astype(np.float64)
        lignes = np.arange(l0, l1 + 1, dtype=np.float64); colonnes = np.arange(c0, c1 + 1, dtype=np.float64)
        a = etalonnage(_texte_xml(_asset(item, "calibration", pol)), lignes, colonnes)
        url_bruit = _asset(item, "noise", pol)
        n = bruit(_texte_xml(url_bruit), lignes, colonnes) if url_bruit else None
        puissance = dn ** 2 - (n if n is not None else 0)
        sigma0 = np.where(dn > 0, np.maximum(puissance, 0) / a ** 2, np.nan).astype(np.float32)
        locaux = [GroundControlPoint(row=g.row - l0, col=g.col - c0, x=g.x, y=g.y, z=g.z)
                  for g, ok in zip(gcps, dedans) if ok]
        sortie = np.full((taille, taille), np.nan, np.float32)
        reproject(sigma0, sortie, gcps=locaux, src_crs=gcp_crs, dst_crs=crs, dst_transform=transform,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        bandes.append(sortie)
    s1 = np.stack(bandes)
    db = 10 * np.log10(np.maximum(s1, 1e-5))
    return np.where(np.isfinite(s1), np.clip(db, -50.0, 5.0), np.nan).astype(np.float32)
