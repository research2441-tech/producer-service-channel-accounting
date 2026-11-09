"""Linear IV machinery written directly in NumPy so every step is inspectable.

Conventions: y (n,), X endogenous (n,p), Z excluded instruments (n,q), W exogenous
controls (n,r) including all fixed-effect dummies, g cluster labels (n,).
"""
import numpy as np
from scipy import stats


def resid(A, W):
    """Residualise columns of A on W (Frisch-Waugh-Lovell)."""
    coef, *_ = np.linalg.lstsq(W, A, rcond=None)
    return A - W @ coef


def _cluster_meat(S, g):
    """Sum over clusters of outer products of within-cluster score sums. S: (n,k)."""
    ug = np.unique(g)
    M = np.zeros((S.shape[1], S.shape[1]))
    for c in ug:
        s = S[g == c].sum(axis=0)
        M += np.outer(s, s)
    return M, len(ug)


def tsls(y, X, Z, W, g, liml=False):
    """2SLS (or LIML) with CR1 cluster-robust covariance. Returns dict."""
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float).reshape(len(y), -1)
    Z = np.asarray(Z, float).reshape(len(y), -1)
    yt, Xt, Zt = resid(y[:, None], W)[:, 0], resid(X, W), resid(Z, W)
    n, p = Xt.shape
    k_w = np.linalg.matrix_rank(W)
    Pz = Zt @ np.linalg.pinv(Zt.T @ Zt) @ Zt.T
    kappa = 1.0
    if liml:
        Yb = np.column_stack([yt, Xt])
        Mz = np.eye(n) - Pz
        A = Yb.T @ Mz @ Yb
        B = Yb.T @ Yb
        kappa = np.min(np.real(np.linalg.eigvals(np.linalg.solve(A, B))))
    Mk = np.eye(n) - kappa * (np.eye(n) - Pz)
    XtMk = Xt.T @ Mk
    beta = np.linalg.solve(XtMk @ Xt, XtMk @ yt)
    u = yt - Xt @ beta
    Xh = Pz @ Xt if not liml else Mk @ Xt
    bread = np.linalg.inv(Xh.T @ Xt)
    meat, G = _cluster_meat(Xh * u[:, None], g)
    dof = (G / (G - 1)) * ((n - 1) / (n - k_w - p))
    V = dof * bread @ meat @ bread.T
    se = np.sqrt(np.diag(V))
    return dict(beta=beta, se=se, V=V, u=u, Xh=Xh, bread=bread, n=n, G=G, dof=dof,
                t=beta / se, p=2 * stats.t.sf(np.abs(beta / se), G - 1), kappa=kappa)


def ols(y, X, W, g):
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float).reshape(len(y), -1)
    return tsls(y, X, X, W, g)


def first_stage(X1, Z, W, g):
    """Single endogenous regressor: coefficients, cluster-robust Wald F and
    the Montiel Olea-Pflueger effective F."""
    xt = resid(np.asarray(X1, float).reshape(-1, 1), W)[:, 0]
    Zt = resid(np.asarray(Z, float).reshape(len(xt), -1), W)
    n, q = Zt.shape
    ZZ = Zt.T @ Zt
    pi = np.linalg.solve(ZZ, Zt.T @ xt)
    v = xt - Zt @ pi
    bread = np.linalg.inv(ZZ)
    meat, G = _cluster_meat(Zt * v[:, None], g)
    dof = (G / (G - 1)) * ((n - 1) / (n - np.linalg.matrix_rank(W) - q))
    V = dof * bread @ meat @ bread
    F_rob = float(pi @ np.linalg.solve(V, pi)) / q
    Sig = dof * meat / n            # variance of n^{-1/2} Z'v
    Q = ZZ / n
    # MOP effective F: pi' Q pi / tr(Omega_pipi Q) with Omega_pipi = Q^-1 Sig Q^-1 / n
    Om = np.linalg.inv(Q) @ Sig @ np.linalg.inv(Q) / n
    F_eff = float(pi @ Q @ pi) / float(np.trace(Om @ Q))
    r2p = 1 - (v @ v) / (xt @ xt)
    return dict(pi=pi, se=np.sqrt(np.diag(V)), F=F_rob, F_eff=F_eff, partial_r2=r2p)


def hansen_j(y, X, Z, W, g):
    """Cluster-robust Hansen J (two-step GMM) for overidentification."""
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float).reshape(len(y), -1)
    Z = np.asarray(Z, float).reshape(len(y), -1)
    yt, Xt, Zt = resid(y[:, None], W)[:, 0], resid(X, W), resid(Z, W)
    r = tsls(y, X, Z, W, g)
    u = r["u"]
    S, G = _cluster_meat(Zt * u[:, None], g)
    S /= len(y)
    Sinv = np.linalg.pinv(S)
    ZX, Zy = Zt.T @ Xt / len(y), Zt.T @ yt / len(y)
    b = np.linalg.solve(ZX.T @ Sinv @ ZX, ZX.T @ Sinv @ Zy)
    gbar = Zt.T @ (yt - Xt @ b) / len(y)
    J = len(y) * float(gbar @ Sinv @ gbar)
    df = Zt.shape[1] - Xt.shape[1]
    return J, df, stats.chi2.sf(J, df)


def anderson_rubin(y, x, Z, W, g, grid):
    """Cluster-robust AR test inverted over a grid (single endogenous regressor).
    Returns p-values over grid and the accepted set at 5%."""
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    Zt = resid(np.asarray(Z, float).reshape(len(y), -1), W)
    yt, xt = resid(y[:, None], W)[:, 0], resid(x[:, None], W)[:, 0]
    n, q = Zt.shape
    ZZinv = np.linalg.inv(Zt.T @ Zt)
    G = len(np.unique(g))
    dof = (G / (G - 1)) * ((n - 1) / (n - np.linalg.matrix_rank(W) - q))
    pv = []
    for b0 in grid:
        e = yt - b0 * xt
        d = ZZinv @ Zt.T @ e
        v = e - Zt @ d
        meat, _ = _cluster_meat(Zt * v[:, None], g)
        V = dof * ZZinv @ meat @ ZZinv
        F = float(d @ np.linalg.solve(V, d)) / q
        pv.append(stats.f.sf(F, q, G - 1))
    pv = np.array(pv)
    acc = grid[pv > 0.05]
    return pv, (acc.min() if len(acc) else np.nan, acc.max() if len(acc) else np.nan)


WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wre_bootstrap(y, x, Z, W, g, beta0=0.0, B=1999, seed=0):
    """Wild restricted efficient (WRE) cluster bootstrap for one endogenous
    regressor (Davidson and MacKinnon, 2010) with Webb six-point weights.
    Returns the symmetric bootstrap p-value of H0: beta = beta0."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float).reshape(-1)
    x = np.asarray(x, float).reshape(-1)
    Z = np.asarray(Z, float).reshape(len(y), -1)
    yt, xt, Zt = resid(y[:, None], W)[:, 0], resid(x[:, None], W)[:, 0], resid(Z, W)
    ones = np.ones((len(y), 1))
    obs = tsls(yt, xt, Zt, ones, g)          # same degrees-of-freedom convention as draws
    t_obs = (obs["beta"][0] - beta0) / obs["se"][0]
    ur = yt - beta0 * xt                                   # restricted structural residual
    coef, *_ = np.linalg.lstsq(np.column_stack([Zt, ur]), xt, rcond=None)
    pi_t = coef[: Zt.shape[1]]                              # efficient first stage
    v = xt - Zt @ pi_t
    clusters, inv = np.unique(g, return_inverse=True)
    tb = np.empty(B)
    for b in range(B):
        e = rng.choice(WEBB, size=len(clusters))[inv]
        xs = Zt @ pi_t + v * e
        ys = beta0 * xs + ur * e
        r = tsls(ys, xs, Zt, ones, g)
        tb[b] = (r["beta"][0] - beta0) / r["se"][0]
    return float(np.mean(np.abs(tb) >= abs(t_obs))), float(t_obs)


def wild_system(Y, x, Z, W, g, B=1999, seed=1):
    """Unrestricted wild cluster bootstrap of a system of 2SLS equations that
    share regressors and instruments; one weight draw per cluster is applied to
    every equation so cross-equation dependence is preserved. Returns (B,K)."""
    rng = np.random.default_rng(seed)
    Yt = resid(np.asarray(Y, float), W)
    xt = resid(np.asarray(x, float).reshape(-1, 1), W)
    Zt = resid(np.asarray(Z, float).reshape(len(xt), -1), W)
    Pz = Zt @ np.linalg.pinv(Zt.T @ Zt) @ Zt.T
    pi, *_ = np.linalg.lstsq(Zt, xt, rcond=None)
    v = xt - Zt @ pi
    xh = Pz @ xt
    beta = np.linalg.solve(xh.T @ xt, xh.T @ Yt)        # (1,K)
    U = Yt - xt @ beta
    clusters, inv = np.unique(g, return_inverse=True)
    out = np.empty((B, Yt.shape[1]))
    for b in range(B):
        e = rng.choice(WEBB, size=len(clusters))[inv][:, None]
        xs = Zt @ pi + v * e
        Ys = xs @ beta + U * e
        xhs = Pz @ xs
        out[b] = np.linalg.solve(xhs.T @ xs, xhs.T @ Ys)[0]
    return beta[0], out


def wcr_ols(y, x, W, g, B=1999, seed=0):
    """Wild cluster restricted (WCR) bootstrap-t for H0: coefficient on x is zero
    in an OLS regression with controls W (Cameron, Gelbach and Miller, 2008),
    Webb six-point weights. Returns two-sided p-value."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float).reshape(-1)
    xt = resid(np.asarray(x, float).reshape(-1, 1), W)[:, 0]
    yt = resid(y[:, None], W)[:, 0]
    ones = np.ones((len(y), 1))
    r0 = tsls(yt, xt, xt, ones, g)
    t_obs = r0["beta"][0] / r0["se"][0]
    clusters, inv = np.unique(g, return_inverse=True)
    tb = np.empty(B)
    for b in range(B):
        e = rng.choice(WEBB, size=len(clusters))[inv]
        ys = yt * e                       # restricted model: yt = u under H0
        r = tsls(ys, xt, xt, ones, g)
        tb[b] = r["beta"][0] / r["se"][0]
    return float(np.mean(np.abs(tb) >= abs(t_obs)))


def sensitivity(t, dof):
    """Cinelli-Hazlett (2020) partial R2 of treatment with outcome and robustness
    values RV (estimate driven to zero) and RV_alpha (loss of 5% significance)."""
    f = abs(t) / np.sqrt(dof)
    r2 = t ** 2 / (t ** 2 + dof)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    tcrit = stats.t.ppf(0.975, dof - 1)
    fa = max(f - tcrit / np.sqrt(dof - 1), 0.0)
    rva = 0.5 * (np.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2) if fa > 0 else 0.0
    return dict(partial_r2=float(r2), RV=float(rv), RV_alpha=float(rva))
