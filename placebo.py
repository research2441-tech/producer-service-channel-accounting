"""Placebo size experiment on the real outcomes: synthetic producer-service paths
with no effect by construction are regressed on the observed channels, and the
rejection rates of competing inference procedures are recorded.
Usage: python placebo.py <results_dir>   (reads panel_analysis.csv, writes placebo.json)"""
import sys, os, json
import numpy as np
import pandas as pd
import estimators as E

RES = sys.argv[1]
NREP = int(os.environ.get("NREP", "300"))
BB = int(os.environ.get("NBOOT_P", "199"))
df = pd.read_csv(os.path.join(RES, "panel_analysis.csv")).sort_values(["province", "year"]).reset_index(drop=True)
plist = sorted(df.province.unique()); N = len(plist); T = df.year.nunique()
g = df.province.values


def design(d):
    Wm = pd.concat([pd.get_dummies(d.province, dtype=float), pd.get_dummies(d.year, drop_first=True, dtype=float)], axis=1)
    Wm["gov"], Wm["lnpop"] = d.gov.values, d.lnpop.values
    yd = pd.get_dummies(d.year, drop_first=True, dtype=float)
    for v in ["lngdppc0", "s20"]:
        Wm = pd.concat([Wm, yd.mul(d[v] - d[v].mean(), axis=0).add_prefix(v + "_")], axis=1)
    return Wm.values


W = design(df)
CK = ["c_scale", "c_structure", "c_mfg_intensity", "c_nonmfg_loading", "c_coal_dependence", "c_conversion"]
outs = {"total": df.lnE.values, "structure": df.c_structure.values,
        "coal_pair": (df.c_coal_dependence + df.c_conversion).values,
        "within": (df.c_mfg_intensity + df.c_coal_dependence + df.c_conversion).values}
rng = np.random.default_rng(404)
rej = {k: {"cr1": 0, "wcr": 0} for k in outs}
rej_ind = {"coal_pair": 0, "within": 0}
for r in range(NREP):
    # province-specific trend plus AR(1) noise, mimicking a trending service index
    x = np.zeros((N, T))
    tr = rng.normal(0, 0.15, N)
    e = rng.normal(0, 0.3, (N, T))
    for t in range(T):
        x[:, t] = (x[:, t - 1] * 0.8 if t else 0) + e[:, t] + tr * t
    xp = ((x - x.mean()) / x.std()).reshape(-1)          # order matches sorted province, year
    for k, y in outs.items():
        o = E.ols(y, xp, W, g)
        rej[k]["cr1"] += o["p"][0] < 0.05
        rej[k]["wcr"] += E.wcr_ols(y, xp, W, g, B=BB, seed=r * 7 + len(k)) < 0.05
    # independence-based s.e. for merged groups
    comp = {c: E.ols(df[c].values, xp, W, g) for c in CK}
    for k, members in [("coal_pair", ["c_coal_dependence", "c_conversion"]),
                       ("within", ["c_mfg_intensity", "c_coal_dependence", "c_conversion"])]:
        b = sum(comp[c]["beta"][0] for c in members)
        se = np.sqrt(sum(comp[c]["se"][0] ** 2 for c in members))
        rej_ind[k] += abs(b / se) > 2.045
out = {k: {m: v / NREP for m, v in d.items()} for k, d in rej.items()}
for k in rej_ind:
    out[k]["independent"] = rej_ind[k] / NREP
out["nrep"] = NREP
out["B"] = BB
json.dump(out, open(os.path.join(RES, "placebo.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
