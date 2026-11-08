"""Seuil d'alerte choisi sur la VALIDATION plutôt que sur l'entraînement (à lancer avec %run -i one-look/src/seuil_validation.py
à la fin du notebook 08 ou 10, qui définissent df, SEUILS, eau_opt, amb_opt, inc_opt, cfg_opt).

Pourquoi : un modèle qui colle bien à ses images d'entraînement y estime très bien la surface
inondée ; le seuil t_alerte choisi sur ces images est alors trop optimiste pour des pays nouveaux.
La validation (pays jamais vus à l'entraînement) donne un réglage plus réaliste ; le test reste
réservé à la mesure finale.
"""
import numpy as np
import pandas as pd

from src import alerte, decision

val = (df["partie"] == "validation").to_numpy()
modeles_ = {
    "optique seul": (eau_opt, amb_opt, inc_opt, cfg_opt["seuils_decision"]),
    "fusion optique + radar": (df["part_eau"].to_numpy(), df["ambiguite"].to_numpy(),
                               df["score_inconnu"].to_numpy(), SEUILS),
}
lignes = []
for nom, (eau, amb, inc, seuils) in modeles_.items():
    t_train = seuils["alerte"]
    t_val, _ = alerte.choisir_seuil_alerte(df.loc[val, "alerte_vraie"], eau[val], SEUIL_ALERTE)
    for regle, t in [("seuil choisi sur l'entraînement", t_train), ("seuil choisi sur la validation", t_val)]:
        s = {**seuils, "alerte": t}
        dec = np.array([decision.decider(e, n, a, i, s) for e, n, a, i in zip(eau, df["nuages"], amb, inc)])
        for part in ["validation", "test"]:
            sel = (df["partie"] == part).to_numpy()
            b = decision.bilan(dec[sel], df.loc[sel, "alerte_vraie"])
            lignes.append({"modele": nom, "regle": regle, "t_alerte": t, "partie": part,
                           **{c: b[c] for c in ["ALERTE", "PAS D'ALERTE", "INCERTAIN", "INCONNU", "f1", "rappel",
                                                "taux_fausse_alerte", "inondations_manquees_sans_prevenir",
                                                "inondations_signalees_incertaines"]}})
seuils_val = pd.DataFrame(lignes)
pd.set_option("display.width", 250)
print(seuils_val.to_string(index=False))
seuils_val.to_csv("resultats/seuil_validation.csv", index=False)
