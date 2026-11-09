"""Grayscale publication figures generated from results/*.  Usage:
python figures.py <results_dir> <fig_dir>"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RES, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(RES, "results.json")))
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#222222",
                     "legend.frameon": True, "legend.edgecolor": "#888888", "legend.fancybox": False,
                     "legend.framealpha": 1.0, "mathtext.fontset": "stix", "savefig.dpi": 600})
K = ["scale", "structure", "mfg_intensity", "nonmfg_loading", "coal_dependence", "conversion"]
LAB = {"scale": "Scale", "structure": "Industrial\ncomposition", "mfg_intensity": "Manufacturing\nenergy intensity",
       "nonmfg_loading": "Non-mfg.\nenergy loading", "coal_dependence": "Coal\ndependence",
       "conversion": "Conversion\nresidual", "total": "Total\nln CO$_2$"}
GRAYS = ["#000000", "#555555", "#8c8c8c", "#b5b5b5"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 framework
def box(ax, x, y, w, h, text, fs=7.2, fill="#ffffff", lw=0.8, style="round,pad=0.008", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec="#111111", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y), x=x, y=y, w=w, h=h)


def arrow(ax, p, q, dashed=False, text=None, toff=(0, 0), rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#111111",
                                                     linestyle=(0, (4, 2.5)) if dashed else "-",
                                                     shrinkA=1.5, shrinkB=1.5, mutation_scale=9,
                                                     connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, fontsize=6.4,
                ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


fig, ax = plt.subplots(figsize=(7.16, 3.9))
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.02, 1.0); ax.axis("off")
srcs = ["CEADs sector-by-fuel\nenergy inventories", "CEADs apparent\nCO$_2$ accounts", "NBS output, structure,\npopulation, finance",
        "Capital coordinates,\ncontiguity", "Technology-market\nturnover", "Digital financial\ninclusion index"]
sb = []
h, gap, top = 0.105, 0.05, 0.965
for i, s in enumerate(srcs):
    sb.append(box(ax, 0.0, top - (i + 1) * h - i * gap, 0.16, h, s, fs=6.6, fill="#f2f2f2"))
panel = box(ax, 0.215, 0.60, 0.15, 0.17, "Province-year panel\n30 provinces\n2011-2021", fs=7.0)
psi = box(ax, 0.215, 0.13, 0.15, 0.17, "Producer-service\nindex  $\\mathit{PSI}$\n(technology, finance)", fs=7.0)
ident = box(ax, 0.42, 0.60, 0.16, 0.17, "Six-term exact\nemission identity\n$\\ln E=\\Sigma_k\\,c_k$", fs=7.0)
smooth = box(ax, 0.42, 0.24, 0.16, 0.18, "Common linear\nsmoother: TWFE with\nbaseline paths;\ncross-fitted ridge", fs=6.8)
system = box(ax, 0.63, 0.42, 0.15, 0.17, "Stacked cluster-\nrobust channel\ncovariance", fs=7.0)
merged = box(ax, 0.83, 0.64, 0.155, 0.17, "Merged channels:\nsubstitution vs.\nwithin-industry", fs=7.0, fill="#e6e6e6")
infer = box(ax, 0.83, 0.20, 0.155, 0.17, "Inference: WCR\nbootstrap, permutation,\nrobustness values", fs=6.6, fill="#e6e6e6")
audit = box(ax, 0.42, 0.0, 0.16, 0.12, "Identification audit:\nexposure IV, AR sets", fs=6.6, fill="#f2f2f2")
for i, s in enumerate(sb):
    if i < 4:
        arrow(ax, s["r"], (panel["x"], panel["y"] + panel["h"] * (0.85 - 0.23 * i)))
    else:
        arrow(ax, s["r"], (psi["x"], psi["y"] + psi["h"] * (0.72 - 0.44 * (i - 4))))
arrow(ax, panel["r"], ident["l"])
arrow(ax, (panel["x"] + panel["w"] * 0.8, panel["y"]), (smooth["x"] + 0.012, smooth["y"] + smooth["h"]), text="controls", toff=(-0.055, 0.01))
arrow(ax, (psi["x"] + psi["w"], psi["y"] + 0.12), (smooth["x"], smooth["y"] + 0.04))
arrow(ax, ident["b"], smooth["t"], text="components $c_k$", toff=(0.055, 0.0))
arrow(ax, (smooth["x"] + smooth["w"], smooth["y"] + 0.13), (system["x"], system["y"] + 0.04))
arrow(ax, (system["x"] + system["w"], system["y"] + 0.13), (merged["x"], merged["y"] + 0.05))
arrow(ax, (system["x"] + system["w"], system["y"] + 0.04), (infer["x"], infer["y"] + 0.12))
arrow(ax, (psi["x"] + psi["w"] * 0.5, psi["y"]), (audit["x"], audit["y"] + audit["h"] * 0.5), rad=0.3)
arrow(ax, (audit["x"] + audit["w"] * 0.75, audit["y"] + audit["h"]),
      (smooth["x"] + smooth["w"] * 0.75, smooth["y"]), dashed=True)
ax.text(audit["x"] + audit["w"] * 0.75 + 0.012, (audit["y"] + audit["h"] + smooth["y"]) / 2, "design revision",
        fontsize=6.2, style="italic", ha="left", va="center")
save(fig, "fig1_framework")

# ---------------------------------------------------------------- Fig. 2 national trajectories
nc = pd.read_csv(os.path.join(RES, "national_components.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(3.5, 2.55))
styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "v"), ("-", "D"), ("--", "x")]
for (k, (ls, mk)), col in zip(zip(K, styles), ["#000", "#333", "#555", "#777", "#999", "#444"]):
    ax.plot(nc.index, nc[k], ls=ls, marker=mk, ms=3.2, lw=1.0, color=col, label=LAB[k].replace("\n", " "),
            mfc="white" if k in ("structure", "coal_dependence") else col)
ax.plot(nc.index, nc["total"], color="#000", lw=2.2, label="Total ln CO$_2$")
ax.axhline(0, color="#999", lw=0.5)
ax.set_xlabel("Year"); ax.set_ylabel("Log change relative to 2011")
ax.set_xticks(nc.index[::2])
ax.legend(fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), columnspacing=1.0,
          handlelength=2.4)
ax.grid(axis="y", color="#e0e0e0", lw=0.4)
save(fig, "fig2_national")

# ---------------------------------------------------------------- Fig. 3 channel coefficients
main, dml = R["main"], R["dml_common"]
names = ["total"] + K
fig, ax = plt.subplots(figsize=(3.5, 2.7))
y = np.arange(len(names))[::-1]
for i, nm in enumerate(names):
    b, se = main[nm]["b"], main[nm]["se"]
    ax.errorbar(b, y[i] + 0.15, xerr=1.96 * se, fmt="o", color="#000", ms=3.6, capsize=2, lw=0.9,
                label="TWFE with baseline paths" if i == 0 else None)
    b2, se2 = dml[nm]["b"], dml[nm]["se"]
    ax.errorbar(b2, y[i] - 0.15, xerr=1.96 * se2, fmt="s", color="#777", mfc="white", ms=3.4, capsize=2, lw=0.9,
                label="Cross-fitted ridge (common smoother)" if i == 0 else None)
ax.axvline(0, color="#555", lw=0.6, ls="--")
ax.set_yticks(y); ax.set_yticklabels([LAB[n] for n in names], fontsize=6.6)
ax.set_xlabel("Effect of a 1-SD rise in $\\mathit{PSI}$ (log points)")
ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=1)
ax.grid(axis="x", color="#e5e5e5", lw=0.4)
save(fig, "fig3_channels")

# ---------------------------------------------------------------- Fig. 4 correlation + precision
C = np.array(R["component_corr"])
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.75), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.55))
ax = axs[0]
im = ax.imshow(C, cmap="Greys", vmin=-1, vmax=1)
short = ["Scale", "Comp.", "Mfg int.", "Non-mfg", "Coal dep.", "Conv."]
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(short, rotation=40, ha="right", fontsize=6.6); ax.set_yticklabels(short, fontsize=6.6)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{C[i, j]:+.2f}", ha="center", va="center", fontsize=5.8,
                color="white" if C[i, j] > 0.1 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)
ax.set_title("(a) Sampling correlation of channel estimates", fontsize=7.4)
ax = axs[1]
mg = R["merged"]
items = [("within_industry", "Within-industry\n(3 channels)"), ("coal_plus_conversion", "Coal dependence\n+ conversion")]
xs = np.arange(len(items))
sys_ = [mg[k]["se_system"] for k, _ in items]
ind = [mg[k]["se_if_independent"] for k, _ in items]
ax.bar(xs - 0.18, ind, 0.34, color="white", ec="#000", hatch="////", lw=0.7, label="If channels were independent")
ax.bar(xs + 0.18, sys_, 0.34, color="#555", ec="#000", lw=0.7, label="Stacked system covariance")
for i in range(len(items)):
    ax.text(xs[i] - 0.18, ind[i] + 0.004, f"{ind[i]:.3f}", ha="center", fontsize=6.2)
    ax.text(xs[i] + 0.18, sys_[i] + 0.004, f"{sys_[i]:.3f}", ha="center", fontsize=6.2)
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in items], fontsize=6.8)
ax.set_ylabel("Standard error of merged coefficient")
ax.set_ylim(0, max(ind) * 1.45)
ax.legend(fontsize=6.2, loc="upper right")
ax.set_title("(b) Precision gained by merging channels", fontsize=7.4)
save(fig, "fig4_covariance")

# ---------------------------------------------------------------- Fig. 5 heterogeneity
hc = pd.read_csv(os.path.join(RES, "het_curves.csv"))
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(wspace=0.3))
for ax, (nm, ttl) in zip(axs, [("substitution", "(a) Industrial-composition channel"),
                               ("within_industry", "(b) Within-industry channels")]):
    b, se = hc[nm + "_b"], hc[nm + "_se"]
    ax.fill_between(hc.mfg0, b - 1.96 * se, b + 1.96 * se, color="#d9d9d9", lw=0, label="95% interval")
    ax.plot(hc.mfg0, b, color="#000", lw=1.3, label="Marginal effect")
    ax.axhline(0, color="#555", lw=0.6, ls="--")
    ax.set_xlabel("Manufacturing share of final energy in 2011")
    ax.set_ylabel("Effect of 1-SD $\\mathit{PSI}$")
    ax.set_title(ttl, fontsize=7.4)
    ax.grid(color="#ececec", lw=0.4)
axs[0].legend(fontsize=6.2, loc="upper left")
pa = pd.read_csv(os.path.join(RES, "panel_analysis.csv"))
m0 = pa.groupby("province").mfg0.first().values
for ax in axs:
    lo = ax.get_ylim()[0]
    ax.plot(m0, np.full_like(m0, lo + 0.02 * (ax.get_ylim()[1] - lo)), "|", color="#333", ms=5)
save(fig, "fig5_heterogeneity")

# ---------------------------------------------------------------- Fig. 6 robustness forest + leave-one-out
rob = R["robustness"]
specs = list(rob)
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True, gridspec_kw=dict(wspace=0.12))
yy = np.arange(len(specs))[::-1]
for ax, (nm, ttl) in zip(axs, [("total", "(a) Total ln CO$_2$"), ("substitution", "(b) Composition"),
                               ("within_industry", "(c) Within-industry")]):
    for i, s in enumerate(specs):
        b, se = rob[s][nm]["b"], rob[s][nm]["se"]
        ax.errorbar(b, yy[i], xerr=1.96 * se, fmt="o" if i else "D", color="#000" if i else "#000",
                    mfc="white" if i else "#000", ms=3.4, capsize=1.8, lw=0.8)
    ax.axvline(0, color="#555", lw=0.6, ls="--")
    ax.set_title(ttl, fontsize=7.4)
    ax.grid(axis="x", color="#ececec", lw=0.4)
    ax.tick_params(axis="x", labelsize=6.4)
axs[0].set_yticks(yy); axs[0].set_yticklabels(specs, fontsize=6.4)
axs[1].set_xlabel("Coefficient on 1-SD $\\mathit{PSI}$ with 95% interval")
save(fig, "fig6_robustness")

# ---------------------------------------------------------------- Fig. 7 identification audit
fs = R["iv"]["first_stage"]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw=dict(width_ratios=[0.9, 1.25], wspace=0.32))
ax = axs[0]
labs = list(fs)
xs = np.arange(len(labs))
ax.bar(xs - 0.18, [fs[l]["F"] for l in labs], 0.34, color="#555", ec="#000", lw=0.6, label="Cluster-robust Wald F")
ax.bar(xs + 0.18, [fs[l]["F_eff"] for l in labs], 0.34, color="white", ec="#000", hatch="\\\\\\\\", lw=0.6,
       label="Effective F")
ax.axhline(10, color="#000", ls=":", lw=0.8)
ax.text(2.45, 10.6, "F = 10", fontsize=6.2, ha="right")
ax.set_xticks(xs); ax.set_xticklabels(["Two-way\nFE", "+ controls", "+ baseline\npaths"], fontsize=6.6)
ax.set_ylabel("First-stage strength"); ax.set_ylim(0, 17)
ax.legend(fontsize=6.0, loc="upper right")
ax.set_title("(a) Exposure instruments", fontsize=7.4)
ax = axs[1]
a2 = pd.read_csv(os.path.join(RES, "ar_total_lvl2.csv")); a3 = pd.read_csv(os.path.join(RES, "ar_total.csv"))
ax.plot(a2.beta0, a2.p, color="#000", lw=1.1, label="Controls, no baseline paths")
ax.plot(a3.beta0, a3.p, color="#666", lw=1.1, ls="--", label="With baseline paths")
ax.axhline(0.05, color="#000", ls=":", lw=0.8, label="5% level")
ax.set_xlabel("Hypothesised effect $\\beta_0$ on total ln CO$_2$"); ax.set_ylabel("Anderson-Rubin p-value")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=6.0, loc="center left", bbox_to_anchor=(0.02, 0.6))
ax.set_title("(b) Weak-instrument-robust confidence sets", fontsize=7.4)
save(fig, "fig7_iv_audit")

# ---------------------------------------------------------------- Fig. 8 validation + permutation
vm = pd.read_csv(os.path.join(RES, "validation_merged.csv"))
perm = pd.read_csv(os.path.join(RES, "permutation.csv"))
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
for yr, mk in zip(sorted(vm.year.unique()), ["o", "s", "^"]):
    s = vm[vm.year == yr]
    ax.scatter(s.PSI, np.log(s.broad_share), marker=mk, s=12, facecolor="white" if yr != 2021 else "#555",
               edgecolor="#000", lw=0.6, label=str(yr))
cf = np.polyfit(vm.PSI, np.log(vm.broad_share), 1)
xx = np.linspace(vm.PSI.min(), vm.PSI.max(), 10)
ax.plot(xx, np.polyval(cf, xx), color="#000", lw=0.9)
v = R["validation"]["PSI|broad_share"]
ax.text(0.03, 0.95, f"r = {v['pearson']:.2f}, n = {R['validation']['n']}", transform=ax.transAxes,
        fontsize=6.6, va="top")
ax.set_xlabel("Producer-service index $\\mathit{PSI}$ (SD units)")
ax.set_ylabel("ln producer-service value-added share")
ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, title=None)
ax.set_title("(a) External validation, 2019-2021", fontsize=7.4)
ax = axs[1]
for k, ls, lab in [("total", "-", "Total"), ("within_industry", "--", "Within-industry"),
                   ("substitution", ":", "Composition")]:
    vals = perm[k].values
    hist, edges = np.histogram(vals, bins=40, density=True)
    ax.step(edges[:-1], hist, where="post", color="#000" if k == "total" else "#555", ls=ls, lw=0.9,
            label=f"{lab} (p = {R['permutation'][k]['p']:.2f})")
    ax.axvline(R["permutation"][k]["obs"], color="#000", ls=ls, lw=1.4)
ax.set_xlabel("Coefficient under permuted $\\mathit{PSI}$ paths"); ax.set_ylabel("Density")
ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=0.8,
          handlelength=2.0)
ax.set_title("(b) Randomisation inference", fontsize=7.4)
save(fig, "fig8_validation_permutation")
print("figures written to", FIG)

# ---------------------------------------------------------------- Fig. 9 provincial decomposition
pa = pd.read_csv(os.path.join(RES, "panel_analysis.csv"))
ck = ["c_" + k for k in K]
a = pa[pa.year == pa.year.min()].set_index("province_en")[ck]
b = pa[pa.year == pa.year.max()].set_index("province_en")[ck]
dlt = (b - a)
dlt["total"] = dlt.sum(axis=1)
dlt = dlt.sort_values("total")
fig, ax = plt.subplots(figsize=(7.16, 3.3))
x = np.arange(len(dlt))
fills = ["#1a1a1a", "#ffffff", "#7f7f7f", "#d0d0d0", "#ffffff", "#4d4d4d"]
hatches = [None, "////", None, None, "....", None]
pos = np.zeros(len(dlt)); neg = np.zeros(len(dlt))
for k, fc, ht in zip(K, fills, hatches):
    v = dlt["c_" + k].values
    bottom = np.where(v >= 0, pos, neg)
    ax.bar(x, v, 0.72, bottom=bottom, color=fc, edgecolor="#000", lw=0.4, hatch=ht,
           label=LAB[k].replace("\n", " "))
    pos = np.where(v >= 0, pos + v, pos); neg = np.where(v < 0, neg + v, neg)
ax.plot(x, dlt.total.values, "D", color="white", mec="#000", ms=3.6, mew=0.9, label="Total ln CO$_2$", zorder=5)
ax.axhline(0, color="#000", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(dlt.index, rotation=60, ha="right", fontsize=6.2)
ax.set_ylabel("Log change, 2011-2021")
ax.set_xlim(-0.7, len(dlt) - 0.3)
ax.set_ylim(-2.0, 2.0)
bj = dlt.loc["Beijing"]
ax.annotate(f"Beijing truncated: coal dependence {bj['c_coal_dependence']:.2f},\nconversion residual {bj['c_conversion']:+.2f}".replace("-", "−"),
            xy=(0, 1.95), xytext=(1.2, 1.62), fontsize=6.2, ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#000", mutation_scale=7))
ax.grid(axis="y", color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
ax.legend(fontsize=6.0, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), columnspacing=1.0, handlelength=1.6)
save(fig, "fig9_provinces")
print("fig9 done")
