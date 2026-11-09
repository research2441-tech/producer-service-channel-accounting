"""Exact channel accounting of producer-service deepening and energy-related CO2
across 30 Chinese provinces, 2011-2021.  All estimates in the manuscript are
produced by this script.
Usage: python analysis.py <data_dir> <results_dir>
Environment: NBOOT (wild bootstrap draws, default 1999), NPERM (permutations, 999),
             NSPLIT (cross-fitting repetitions, 20)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
import estimators as E

DATA, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
B = int(os.environ.get("NBOOT", "1999"))
NPERM = int(os.environ.get("NPERM", "999"))
NSPLIT = int(os.environ.get("NSPLIT", "20"))
t_start = time.time()

df = pd.read_csv(os.path.join(DATA, "panel_raw.csv")).sort_values(["province", "year"]).reset_index(drop=True)
years = sorted(df.year.unique())
T0 = years[0]
plist = sorted(df.province.unique())
N, TT = len(plist), len(years)

# ------------------------------------------------------------------ channel identity
df["s2"] = df.secondary_industry / df.gdp_nominal
df["lnE"] = np.log(df.carbon_emissions_mt)
COMP = {
    "scale": np.log(df.gdp_real),
    "structure": np.log(df.s2),
    "mfg_intensity": np.log(df.en_mfg / (df.s2 * df.gdp_real)),
    "nonmfg_loading": np.log(df.en_total / df.en_mfg),
    "coal_dependence": np.log(df.coal_related_tce / df.en_total),
    "conversion": np.log(df.carbon_emissions_mt / df.coal_related_tce),
}
KEYS = list(COMP)
for k, v in COMP.items():
    df["c_" + k] = v
CK = ["c_" + k for k in KEYS]
R = {"identity_max_abs_gap": float(np.max(np.abs(df[CK].sum(axis=1) - df.lnE))),
     "n_obs": int(len(df)), "n_prov": N, "years": [int(years[0]), int(years[-1])]}
MERGED = {"substitution": ["structure"], "within_industry": ["mfg_intensity", "coal_dependence", "conversion"],
          "coal_plus_conversion": ["coal_dependence", "conversion"], "scale": ["scale"],
          "nonmfg": ["nonmfg_loading"]}
for m, ks in MERGED.items():
    df["m_" + m] = df[["c_" + k for k in ks]].sum(axis=1)

# ------------------------------------------------------------------ treatment
df["TS"] = np.log(df.tech_turnover / df.gdp_nominal * 1e4)
df["FS"] = np.log(df.digital_finance)
zs = lambda s: (s - s.mean()) / s.std()
df["PSI"] = 0.5 * (zs(df.TS) + zs(df.FS))
R["psi_sd_raw"] = float(df.PSI.std())
df["PSI"] = zs(df.PSI)                      # coefficients read per one standard deviation
R["sd_TS"], R["sd_FS"] = float(df.TS.std()), float(df.FS.std())

# ------------------------------------------------------------------ controls and baseline paths
base = df[df.year == T0].set_index("province")
df["lngdppc0"] = df.province.map(np.log(base.gdp_real / base.population))
df["s20"] = df.province.map(base.secondary_industry / base.gdp_nominal)
df["mfg0"] = df.province.map(base.en_mfg / base.en_total)
df["coal0"] = df.province.map(base.coal_related_tce / base.en_total)
df["gov"] = df.fiscal_expenditure / df.gdp_nominal
df["lnpop"] = np.log(df.population)
df["urban"] = df.urban_population / df.population
df["open"] = df.trade / df.gdp_nominal / 1e4 * 6.5          # trade (10k USD) as share of GDP, approx. rate
df["t"] = df.year - T0


def design(d, level=3, extra=None):
    """level 1: two-way FE; 2: + time-varying controls; 3: + baseline-path controls."""
    Wm = pd.concat([pd.get_dummies(d.province, dtype=float),
                    pd.get_dummies(d.year, drop_first=True, dtype=float)], axis=1)
    if level >= 2:
        Wm["gov"], Wm["lnpop"] = d.gov.values, d.lnpop.values
    if level >= 3:
        yd = pd.get_dummies(d.year, drop_first=True, dtype=float)
        for v in ["lngdppc0", "s20"]:
            Wm = pd.concat([Wm, yd.mul(d[v] - d[v].mean(), axis=0).add_prefix(v + "_")], axis=1)
    if extra is not None:
        for c in extra:
            Wm[c] = d[c].values
    return Wm.values


W = design(df)
g = df.province.values
kW = np.linalg.matrix_rank(W)
R["k_controls"] = int(kW)

# ------------------------------------------------------------------ descriptives
dcols = ["lnE"] + CK + ["TS", "FS", "PSI", "gov", "lnpop", "urban"]
desc = df[dcols].describe().T[["mean", "std", "min", "max"]]
desc["within_sd"] = [float((df[c] - df.groupby("province")[c].transform("mean")).std()) for c in dcols]
desc.to_csv(os.path.join(RES, "descriptives.csv"))
R["desc"] = {k: {kk: float(vv) for kk, vv in r.items()} for k, r in desc.iterrows()}
nat = df.groupby("year").agg(E=("carbon_emissions_mt", "sum"), Y=("gdp_real", "sum"),
                             S=("secondary_industry", "sum"), Yn=("gdp_nominal", "sum"),
                             M=("en_mfg", "sum"), EN=("en_total", "sum"), C=("coal_related_tce", "sum"))
nat["s2"] = nat.S / nat.Yn
natc = pd.DataFrame({"scale": np.log(nat.Y), "structure": np.log(nat.s2),
                     "mfg_intensity": np.log(nat.M / (nat.s2 * nat.Y)), "nonmfg_loading": np.log(nat.EN / nat.M),
                     "coal_dependence": np.log(nat.C / nat.EN), "conversion": np.log(nat.E / nat.C)})
natc = natc - natc.iloc[0]
natc["total"] = natc.sum(axis=1)
natc.to_csv(os.path.join(RES, "national_components.csv"))
R["national_change"] = {k: float(v) for k, v in natc.iloc[-1].items()}
# variance of within-province changes by component (share of variance of d lnE)
dd = df.groupby("province")[["lnE"] + CK].diff().dropna()
R["dlnE_var"] = float(dd.lnE.var())
R["dcomp_cov_with_dlnE"] = {k: float(np.cov(dd["c_" + k], dd.lnE)[0, 1] / dd.lnE.var()) for k in KEYS}

# ------------------------------------------------------------------ main estimator: TWFE + baseline paths
def system(d, Wd, x, outcomes, gg):
    res = {}
    rr = [E.ols(d[o], d[x], Wd, gg) for o in outcomes]
    for o, r in zip(outcomes, rr):
        res[o] = {"b": float(r["beta"][0]), "se": float(r["se"][0]), "p": float(r["p"][0])}
    U = np.column_stack([r["u"] for r in rr])
    xh = rr[0]["Xh"][:, 0]
    meat, G = E._cluster_meat(xh[:, None] * U, gg)
    V = rr[0]["dof"] * rr[0]["bread"][0, 0] ** 2 * meat
    return res, V


outs = ["lnE"] + CK
main, Vsys = system(df, W, "PSI", outs, g)
Vc = Vsys[1:, 1:]
R["main"] = {}
for o, nm in zip(outs, ["total"] + KEYS):
    m = main[o]
    m["p_wcr"] = E.wcr_ols(df[o], df.PSI, W, g, B=B, seed=hash(nm) % 10000)
    m["ci"] = [m["b"] - stats.t.ppf(0.975, N - 1) * m["se"], m["b"] + stats.t.ppf(0.975, N - 1) * m["se"]]
    m.update(E.sensitivity(m["b"] / m["se"], len(df) - kW - 1))
    R["main"][nm] = m
R["additivity_gap_main"] = float(abs(sum(R["main"][k]["b"] for k in KEYS) - R["main"]["total"]["b"]))
R["component_corr"] = (Vc / np.sqrt(np.outer(np.diag(Vc), np.diag(Vc)))).tolist()
R["component_se"] = np.sqrt(np.diag(Vc)).tolist()
R["merged"] = {}
for mname, ks in MERGED.items():
    a = np.array([1.0 if k in ks else 0.0 for k in KEYS])
    r = E.ols(df["m_" + mname], df.PSI, W, g)
    se_sys = float(np.sqrt(a @ Vc @ a))
    se_naive = float(np.sqrt(sum(Vc[KEYS.index(k), KEYS.index(k)] for k in ks)))
    b = float(r["beta"][0])
    R["merged"][mname] = {"b": b, "se": float(r["se"][0]), "se_system": se_sys, "se_if_independent": se_naive,
                          "p": float(r["p"][0]), "p_wcr": E.wcr_ols(df["m_" + mname], df.PSI, W, g, B=B, seed=99),
                          "ci": [b - stats.t.ppf(0.975, N - 1) * float(r["se"][0]),
                                 b + stats.t.ppf(0.975, N - 1) * float(r["se"][0])],
                          **E.sensitivity(b / float(r["se"][0]), len(df) - kW - 1)}

# ------------------------------------------------------------------ specification ladder ("ablation" of design layers)
ladder = {}
for lvl, lab in [(1, "two-way FE"), (2, "+ time-varying controls"), (3, "+ baseline development paths")]:
    res, _ = system(df, design(df, lvl), "PSI", ["lnE", "m_substitution", "m_within_industry", "m_scale", "m_nonmfg"], g)
    ladder[lab] = res
R["ladder"] = ladder

# ------------------------------------------------------------------ cross-fitted debiased estimator with a common linear smoother
def twoway_demean(a):
    a = np.asarray(a, float).reshape(N, TT, -1)
    return (a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean((0, 1), keepdims=True)).reshape(N * TT, -1)


feat_raw = df[["gov", "lnpop", "urban", "open"]].copy()
for v in ["lngdppc0", "s20", "mfg0", "coal0"]:
    c = df[v] - df[v].mean()
    feat_raw[v + "_t"] = c * df.t
    feat_raw[v + "_t2"] = c * df.t ** 2 / 10
Xf = ((feat_raw - feat_raw.mean()) / feat_raw.std()).values
rng0 = np.random.default_rng(5)
D = 300
med = np.median(np.sqrt(((Xf[:, None, :] - Xf[None, :200, :]) ** 2).sum(-1)))
Om = rng0.normal(0, 1 / med, (Xf.shape[1], D))
ph = rng0.uniform(0, 2 * np.pi, D)
Phi = np.column_stack([Xf, np.sqrt(2 / D) * np.cos(Xf @ Om + ph)])
Phi_w = twoway_demean(Phi)
Ydm = twoway_demean(df[outs].values)
xdm = twoway_demean(df.PSI.values)[:, 0]
prov_idx = np.repeat(np.arange(N), TT)


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    return Xte @ np.linalg.solve(A, Xtr.T @ ytr)


def cv_alpha(yv, grid=(0.1, 1, 3, 10, 30, 100, 300)):
    best, ba = np.inf, None
    for a in grid:
        err = 0.0
        for f in range(5):
            te = np.isin(prov_idx, np.arange(N)[f::5])
            p = ridge_fit_predict(Phi_w[~te], yv[~te], Phi_w[te], a)
            err += np.sum((yv[te] - p) ** 2)
        if err < best:
            best, ba = err, a
    return ba


alpha_y = cv_alpha(Ydm[:, 0])
alpha_x = cv_alpha(xdm)
R["dml_alpha"] = {"outcome": alpha_y, "treatment": alpha_x}


def dml(mode, nsplit=NSPLIT, seed=0):
    rng = np.random.default_rng(seed)
    thetas, ses = [], []
    alphas = {j: (alpha_y if mode == "common" else cv_alpha(Ydm[:, j])) for j in range(Ydm.shape[1])} \
        if mode in ("common", "tuned") else None
    for s in range(nsplit):
        perm = rng.permutation(N)
        folds = [perm[f::5] for f in range(5)]
        Ry = np.zeros_like(Ydm)
        Rx = np.zeros_like(xdm)
        for fo in folds:
            te = np.isin(prov_idx, fo)
            Rx[te] = xdm[te] - ridge_fit_predict(Phi_w[~te], xdm[~te], Phi_w[te], alpha_x)
            for j in range(Ydm.shape[1]):
                if mode == "gbm":
                    mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                                        random_state=s)
                    mdl.fit(Phi_w[~te][:, :Xf.shape[1]], Ydm[~te, j])
                    Ry[te, j] = Ydm[te, j] - mdl.predict(Phi_w[te][:, :Xf.shape[1]])
                else:
                    Ry[te, j] = Ydm[te, j] - ridge_fit_predict(Phi_w[~te], Ydm[~te, j], Phi_w[te], alphas[j])
        th = (Rx @ Ry) / (Rx @ Rx)
        U = Ry - np.outer(Rx, th)
        meat, G = E._cluster_meat(Rx[:, None] * U, g)
        V = (G / (G - 1)) * meat / (Rx @ Rx) ** 2
        thetas.append(th)
        ses.append(np.sqrt(np.diag(V)))
    th = np.mean(thetas, axis=0)                     # mean aggregation preserves additivity
    se = np.sqrt(np.mean(np.array(ses) ** 2 + (np.array(thetas) - th) ** 2, axis=0))
    out = {nm: {"b": float(th[j]), "se": float(se[j])} for j, nm in enumerate(["total"] + KEYS)}
    out["additivity_gap"] = float(abs(th[1:].sum() - th[0]))
    out["within_industry"] = float(sum(th[1 + KEYS.index(k)] for k in MERGED["within_industry"]))
    return out


R["dml_common"] = dml("common")
R["dml_tuned"] = dml("tuned", nsplit=max(4, NSPLIT // 5))
R["dml_gbm"] = dml("gbm", nsplit=max(4, NSPLIT // 5))

# ------------------------------------------------------------------ service types jointly
types = {}
for o, nm in zip(outs + ["m_within_industry"], ["total"] + KEYS + ["within_industry"]):
    r = E.ols(df[o], df[["TS", "FS"]], W, g)
    types[nm] = {"b_TS": float(r["beta"][0]), "se_TS": float(r["se"][0]), "p_TS": float(r["p"][0]),
                 "b_FS": float(r["beta"][1]), "se_FS": float(r["se"][1]), "p_FS": float(r["p"][1])}
R["types"] = types
R["corr_TS_FS_within"] = float(np.corrcoef(E.resid(df[["TS"]].values, W)[:, 0],
                                           E.resid(df[["FS"]].values, W)[:, 0])[0, 1])

# ------------------------------------------------------------------ dynamics: distributed lags
for L in (1, 2):
    df[f"PSI_l{L}"] = df.groupby("province").PSI.shift(L)
dl = df.dropna(subset=["PSI_l2"]).reset_index(drop=True)
Wl = design(dl)
lag = {}
for o, nm in [("lnE", "total"), ("m_substitution", "substitution"), ("m_within_industry", "within_industry")]:
    r = E.ols(dl[o], dl[["PSI", "PSI_l1", "PSI_l2"]], Wl, dl.province.values)
    a = np.ones(3)
    cum, sec = float(r["beta"].sum()), float(np.sqrt(a @ r["V"] @ a))
    lag[nm] = {"b": r["beta"].tolist(), "se": r["se"].tolist(), "cum": cum, "se_cum": sec,
               "p_cum": float(2 * stats.t.sf(abs(cum / sec), N - 1))}
R["lags"] = lag

# ------------------------------------------------------------------ neighbour exposure (SLX)
edges = pd.read_csv(os.path.join(DATA, "contiguity_edges.csv"))
pi_ = {p: i for i, p in enumerate(plist)}
Wsp = np.zeros((N, N))
for _, e in edges.iterrows():
    if e.province_i in pi_ and e.province_j in pi_:
        Wsp[pi_[e.province_i], pi_[e.province_j]] = 1
Wsp = Wsp / Wsp.sum(1, keepdims=True)
co = df[df.year == T0].set_index("province").loc[plist, ["longitude", "latitude"]].values
lon, lat = np.radians(co[:, 0]), np.radians(co[:, 1])
dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat[:, None] - lat[None]) / 2) ** 2 +
                                      np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2))
Wd = np.where(dist > 0, 1 / np.maximum(dist, 1) ** 2, 0)
Wd = Wd / Wd.sum(1, keepdims=True)
for lab, M in [("contiguity", Wsp), ("inverse-distance", Wd)]:
    col = "W_PSI_" + lab
    mat = df.pivot(index="province", columns="year", values="PSI").loc[plist].values
    lagged = M @ mat
    df[col] = [lagged[pi_[p], years.index(y)] for p, y in zip(df.province, df.year)]
spat = {}
for lab in ["contiguity", "inverse-distance"]:
    spat[lab] = {}
    for o, nm in [("lnE", "total"), ("m_substitution", "substitution"), ("m_within_industry", "within_industry"),
                  ("m_scale", "scale"), ("m_nonmfg", "nonmfg")]:
        r = E.ols(df[o], df[["PSI", "W_PSI_" + lab]], W, g)
        a = np.ones(2)
        net, sen = float(r["beta"].sum()), float(np.sqrt(a @ r["V"] @ a))
        spat[lab][nm] = {"own": float(r["beta"][0]), "se_own": float(r["se"][0]), "p_own": float(r["p"][0]),
                         "nb": float(r["beta"][1]), "se_nb": float(r["se"][1]), "p_nb": float(r["p"][1]),
                         "net": net, "se_net": sen}
R["spatial"] = spat

# ------------------------------------------------------------------ manufacturing-base heterogeneity
df["mfg0c"] = df.mfg0 - df.mfg0.mean()
df["PSIxM"] = df.PSI * df.mfg0c
grid_m = np.linspace(df.mfg0.min(), df.mfg0.max(), 41)
het, cur = {}, pd.DataFrame({"mfg0": grid_m})
for o, nm in [("lnE", "total"), ("m_substitution", "substitution"), ("m_within_industry", "within_industry"),
              ("c_mfg_intensity", "mfg_intensity")]:
    r = E.ols(df[o], df[["PSI", "PSIxM"]], W, g)
    het[nm] = {"b": float(r["beta"][0]), "se": float(r["se"][0]), "bx": float(r["beta"][1]),
               "se_x": float(r["se"][1]), "p_x": float(r["p"][1])}
    m = grid_m - df.mfg0.mean()
    cur[nm + "_b"] = r["beta"][0] + r["beta"][1] * m
    cur[nm + "_se"] = np.sqrt(r["V"][0, 0] + 2 * m * r["V"][0, 1] + m ** 2 * r["V"][1, 1])
cur.to_csv(os.path.join(RES, "het_curves.csv"), index=False)
R["het_mfg"] = het
R["mfg0_range"] = [float(df.mfg0.min()), float(df.mfg0.max()), float(df.mfg0.mean())]

# ------------------------------------------------------------------ manufacturing-quality indicators
df["ln_patent_pc"] = np.log(df.patent_grants / df.population)
df["ln_hitech_share"] = np.log(df.en_hitech / df.en_mfg)
df["ln_mfg_coalshare"] = np.log(df.coal_mfg / df.en_mfg)
df["ln_mfg_energy_prod"] = -df.c_mfg_intensity
up = {}
for v in ["ln_patent_pc", "ln_hitech_share", "ln_mfg_coalshare", "ln_mfg_energy_prod"]:
    r = E.ols(df[v], df.PSI, W, g)
    up[v] = {"b": float(r["beta"][0]), "se": float(r["se"][0]), "p": float(r["p"][0]),
             "p_wcr": E.wcr_ols(df[v], df.PSI, W, g, B=B, seed=31)}
R["upgrading"] = up

# ------------------------------------------------------------------ robustness
def spec(d, Wd, x="PSI"):
    out = {}
    for o, nm in [("lnE", "total"), ("m_substitution", "substitution"), ("m_within_industry", "within_industry")]:
        r = E.ols(d[o], d[x], Wd, d.province.values)
        out[nm] = {"b": float(r["beta"][0]), "se": float(r["se"][0]), "p": float(r["p"][0])}
    out["n"] = int(len(d))
    return out


rob = {"Baseline": spec(df, W)}
mun = ~df.province.isin(["北京", "天津", "上海", "重庆"])
d2 = df[mun].reset_index(drop=True); rob["Excluding municipalities"] = spec(d2, design(d2))
d3 = df[df.year <= 2019].reset_index(drop=True); rob["2011-2019 only"] = spec(d3, design(d3))
d6 = df[df.year >= 2012].reset_index(drop=True); rob["2012-2021 only"] = spec(d6, design(d6))
d4 = df.copy()
d4["east_t"] = (d4.region == "东部").astype(float) * d4.t
d4["west_t"] = (d4.region == "西部").astype(float) * d4.t
rob["Region-specific trends"] = spec(d4, design(d4, extra=["east_t", "west_t"]))
d5 = df.copy()
lo, hi = d5.TS.quantile([0.025, 0.975])
d5["PSIw"] = zs(0.5 * (zs(d5.TS.clip(lo, hi)) + zs(d5.FS)))
rob["Winsorised technology services"] = spec(d5, W, x="PSIw")
d7 = df.copy()
d7["PSIpc"] = zs(pd.Series(np.linalg.svd(np.column_stack([zs(df.TS), zs(df.FS)]), full_matrices=False)[0][:, 0]))
d7["PSIpc"] *= np.sign(np.corrcoef(d7.PSIpc, df.PSI)[0, 1])
rob["Principal-component index"] = spec(d7, W, x="PSIpc")
d8 = df[df.province != "湖北"].reset_index(drop=True); rob["Excluding Hubei"] = spec(d8, design(d8))
R["robustness"] = rob

# ------------------------------------------------------------------ randomisation inference (permute whole PSI paths)
rng = np.random.default_rng(2026)
mat = df.pivot(index="province", columns="year", values="PSI").loc[plist]
obs = {nm: R["merged"][nm]["b"] for nm in ["substitution", "within_industry"]}
obs["total"] = R["main"]["total"]["b"]
perm = {k: [] for k in obs}
Yt = {k: E.resid(df[[c]].values, W)[:, 0] for k, c in
      [("total", "lnE"), ("substitution", "m_substitution"), ("within_industry", "m_within_industry")]}
for b in range(NPERM):
    pm = rng.permutation(N)
    xp = mat.values[pm]
    xcol = np.array([xp[pi_[p], years.index(y)] for p, y in zip(df.province, df.year)])
    xt = E.resid(xcol[:, None], W)[:, 0]
    for k in obs:
        perm[k].append(float(xt @ Yt[k] / (xt @ xt)))
R["permutation"] = {k: {"obs": obs[k], "p": float((np.sum(np.abs(np.array(v)) >= abs(obs[k])) + 1) / (len(v) + 1))}
                    for k, v in perm.items()}
pd.DataFrame(perm).to_csv(os.path.join(RES, "permutation.csv"), index=False)

# ------------------------------------------------------------------ leave-one-province-out
loo = []
for p in plist:
    d_ = df[df.province != p].reset_index(drop=True)
    Wp = design(d_)
    row = {"province": p}
    for o, nm in [("lnE", "total"), ("m_substitution", "substitution"), ("m_within_industry", "within_industry")]:
        r = E.ols(d_[o], d_.PSI, Wp, d_.province.values)
        row[nm] = float(r["beta"][0]); row[nm + "_se"] = float(r["se"][0])
    loo.append(row)
loo = pd.DataFrame(loo)
loo["province_en"] = loo.province.map(dict(zip(df.province, df.province_en)))
loo.to_csv(os.path.join(RES, "leave_one_out.csv"), index=False, encoding="utf-8")
R["loo"] = {nm: {"min": float(loo[nm].min()), "max": float(loo[nm].max()),
                 "argmin": loo.province_en[loo[nm].idxmin()], "argmax": loo.province_en[loo[nm].idxmax()]}
            for nm in ["total", "substitution", "within_industry"]}

# ------------------------------------------------------------------ auxiliary IV route
df["TS0"] = df.province.map(base.TS) if "TS" in base else df.province.map(
    np.log(base.tech_turnover / base.gdp_nominal * 1e4))
df["lnd"] = np.log1p(df.dist_hz_km)
nat_t, nat_y = df.groupby("year").tech_turnover.sum(), df.groupby("year").gdp_nominal.sum()
nat_f = df.groupby("year").digital_finance.sum()
loo_t = np.log((df.year.map(nat_t) - df.tech_turnover) / (df.year.map(nat_y) - df.gdp_nominal))
loo_t0 = df.province.map(lambda p: np.log((nat_t[T0] - base.loc[p, "tech_turnover"]) /
                                          (nat_y[T0] - base.loc[p, "gdp_nominal"])))
df["Z_T"] = (df.TS0 - df.TS0.mean()) * (loo_t - loo_t0)
df["Z_F"] = (df.lnd - df.lnd.mean()) * np.log((df.year.map(nat_f) - df.digital_finance) / (N - 1))
ZC = ["Z_T", "Z_F"]
iv = {"first_stage": {}}
for lvl, lab in [(1, "two-way FE"), (2, "+ time-varying controls"), (3, "+ baseline development paths")]:
    f = E.first_stage(df.PSI, df[ZC], design(df, lvl), g)
    iv["first_stage"][lab] = {"pi": f["pi"].tolist(), "se": f["se"].tolist(), "F": f["F"], "F_eff": f["F_eff"]}
grid = np.linspace(-3, 3, 1201)
for lvl, lab in [(2, "+ time-varying controls"), (3, "+ baseline development paths")]:
    Wv = design(df, lvl)
    r = E.tsls(df.lnE, df.PSI, df[ZC], Wv, g)
    pv, ci = E.anderson_rubin(df.lnE, df.PSI, df[ZC], Wv, g, grid)
    J = E.hansen_j(df.lnE, df.PSI, df[ZC], Wv, g)
    iv[lab] = {"b": float(r["beta"][0]), "se": float(r["se"][0]), "ar_lo": float(ci[0]), "ar_hi": float(ci[1]),
               "ar_unbounded": bool(pv[0] > 0.05 or pv[-1] > 0.05), "J": float(J[0]), "J_p": float(J[2])}
    if lvl == 3:
        pd.DataFrame({"beta0": grid, "p": pv}).to_csv(os.path.join(RES, "ar_total.csv"), index=False)
    else:
        pd.DataFrame({"beta0": grid, "p": pv}).to_csv(os.path.join(RES, "ar_total_lvl2.csv"), index=False)
R["iv"] = iv

# ------------------------------------------------------------------ proxy validation
va = pd.read_csv(os.path.join(DATA, "validation_va.csv"))
va["core_share"] = (va.va_transport + va.va_finance) / va.gdp
va["broad_share"] = (va.va_transport + va.va_finance + va.va_wholesale + va.va_other) / va.gdp
vm = df[["province", "province_en", "year", "PSI", "TS", "FS"]].merge(va, on=["province", "year"])
vv = {}
for c in ["core_share", "broad_share"]:
    for x in ["PSI", "TS", "FS"]:
        pr = stats.pearsonr(vm[x], np.log(vm[c]))
        sr = stats.spearmanr(vm[x], vm[c])
        vv[f"{x}|{c}"] = {"pearson": float(pr[0]), "p": float(pr[1]), "spearman": float(sr[0]),
                          "p_s": float(sr[1])}
vv["n"] = int(len(vm)); vv["years"] = sorted(vm.year.unique().tolist())
R["validation"] = vv
vm.to_csv(os.path.join(RES, "validation_merged.csv"), index=False, encoding="utf-8")

# ------------------------------------------------------------------ timing
R["runtime_s"] = float(time.time() - t_start)
df.to_csv(os.path.join(RES, "panel_analysis.csv"), index=False, encoding="utf-8")
json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("identity gap", R["identity_max_abs_gap"], "additivity", R["additivity_gap_main"])
for nm in ["total"] + KEYS:
    m = R["main"][nm]
    print(f"{nm:16s} {m['b']:+.4f} ({m['se']:.4f}) p={m['p']:.3f} wcr={m['p_wcr']:.3f} RV={m['RV']:.3f}")
for nm, m in R["merged"].items():
    print(f"[merged] {nm:22s} {m['b']:+.4f} ({m['se']:.4f}) sys={m['se_system']:.4f} indep={m['se_if_independent']:.4f} p={m['p']:.3f} wcr={m['p_wcr']:.3f}")
print("DML common", {k: (round(v['b'], 4), round(v['se'], 4)) if isinstance(v, dict) else round(v, 6) for k, v in R['dml_common'].items()})
print("DML tuned gap", R["dml_tuned"]["additivity_gap"], "gbm gap", R["dml_gbm"]["additivity_gap"])
print("runtime", R["runtime_s"])
