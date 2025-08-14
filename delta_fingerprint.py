import numpy as np

"""δ-fingerprint time-series steering detector.

Implements four statistical checks to distinguish steering/bias from chaos or noise.

Use `delta_fingerprint(y)` to compute a verdict, per-test pass flags, and metrics.
"""


def _block_means(y, m):
    n = len(y) // m
    if n == 0:
        return np.array([])
    yy = y[: n * m].reshape(n, m).mean(axis=1)
    return yy


def _dfa_hurst(y, window_sizes=(8, 16, 24, 32, 48, 64)):
    # classic DFA-1
    y = np.asarray(y, float)
    y = y - y.mean()
    x = np.cumsum(y)  # profile
    Fs, Ss = [], []
    for s in window_sizes:
        n = len(x) // s
        if n < 2:
            continue
        X = x[: n * s].reshape(n, s)
        t = np.arange(s)
        # detrend each window with a line
        detr = []
        for row in X:
            A = np.vstack([t, np.ones_like(t)]).T
            a, b = np.linalg.lstsq(A, row, rcond=None)[0]
            detr.append(row - (a * t + b))
        detr = np.array(detr)
        F = np.sqrt((detr ** 2).mean())
        Fs.append(F)
        Ss.append(s)
    if len(Ss) < 2:
        return np.nan
    Ss, Fs = np.array(Ss, float), np.array(Fs, float)
    H = np.polyfit(np.log(Ss), np.log(Fs + 1e-12), 1)[0]
    return float(H)


def _psd_slope(y, lo=0.02, hi=0.25):
    y = np.asarray(y, float)
    y = y - y.mean()
    Y = np.fft.rfft(y)
    P = (np.abs(Y) ** 2)
    f = np.fft.rfftfreq(len(y), d=1.0)
    # bandpass region for slope fit
    mask = (f >= lo) & (f <= hi) & (f > 0)
    if mask.sum() < 5:
        return np.nan
    slope = np.polyfit(np.log(f[mask]), np.log(P[mask] + 1e-12), 1)[0]
    return float(slope)


def _multifractal_width_proxy(
    y, qs=(0.5, 1.0, 2.0, 3.0), scales=(4, 8, 16, 24, 32)
):
    """Lightweight proxy via structure functions.

    S_q(s) = mean( |y_{t+s} - y_t|^q ).
    Fit zeta(q) ~ alpha*q + 0.5*beta*q*(q-1). Width ~ |beta|.
    Narrow (|beta| small) ~ steering; broad (|beta| large) ~ chaos.
    """
    y = np.asarray(y, float)
    deltas = {}
    for s in scales:
        if s >= len(y):
            continue
        diff = np.abs(y[s:] - y[:-s])
        deltas[s] = [(diff ** q).mean() + 1e-12 for q in qs]
    if len(deltas) < 3:
        return np.nan
    # estimate zeta(q) slope vs log s, then fit quadratic in q
    zetas = []
    for i, q in enumerate(qs):
        xs = np.log(np.array(list(deltas.keys()), float))
        ys = np.log(
            np.array([vals[i] for vals in deltas.values()], float)
        )
        zetas.append(np.polyfit(xs, ys, 1)[0])  # slope vs scale
    zetas = np.array(zetas)
    # fit zeta(q) ≈ a*q + 0.5*b*q*(q-1)  → width proxy = |b|
    q = np.array(qs, float)
    A = np.vstack([q, 0.5 * q * (q - 1), np.ones_like(q)]).T
    a, b, c = np.linalg.lstsq(A, zetas, rcond=None)[0]
    width = abs(b)
    return float(width)


def delta_fingerprint(
    y,
    mean_scales=(8, 16, 24, 32, 48, 64),
    hurst_scales=(8, 16, 24, 32, 48, 64),
    mf_qs=(0.5, 1.0, 2.0, 3.0),
    mf_scales=(4, 8, 16, 24, 32),
    psd_band=(0.02, 0.25),
    # thresholds (tune to your data):
    mean_pos_frac=0.7,  # ≥70% block-means > 0
    hurst_tol=0.10,  # |H-0.5| ≤ 0.10
    mf_width_max=0.12,  # narrow spectrum if width ≤ 0.12
    psd_slope_tol=0.2,  # |slope| ≤ 0.2 ~ white-like
):
    y = np.asarray(y, float)
    y = y - y.mean()  # center for A/B/D; C uses increments

    # A) scale-persistent mean
    pos = []
    for m in mean_scales:
        bm = _block_means(y, m)
        if bm.size == 0:
            continue
        pos.append((bm > 0).mean())
    A_pass = (np.mean(pos) >= mean_pos_frac) if pos else False

    # B) DFA Hurst after mean removal
    H = _dfa_hurst(y, window_sizes=hurst_scales)
    B_pass = (not np.isnan(H)) and (abs(H - 0.5) <= hurst_tol)

    # C) multifractal width (proxy)
    Cw = _multifractal_width_proxy(y, qs=mf_qs, scales=mf_scales)
    C_pass = (not np.isnan(Cw)) and (Cw <= mf_width_max)

    # D) spectral slope near-white after mean removal
    slope = _psd_slope(y, lo=psd_band[0], hi=psd_band[1])
    D_pass = (not np.isnan(slope)) and (abs(slope) <= psd_slope_tol)

    passes = dict(
        A_scale_persistent_mean=A_pass,
        B_hurst_white_like=B_pass,
        C_multifractal_narrow=C_pass,
        D_psd_slope_near0=D_pass,
    )
    score = np.mean(list(passes.values()))  # δ-score in [0,1]
    verdict = (
        "STEERING/BIAS (δ-fingerprint)" if score >= 0.75 else "CHAOS/NOISE (no δ)"
    )
    metrics = dict(
        H=H,
        mf_width=Cw,
        psd_slope=slope,
        pos_frac=np.mean(pos) if pos else np.nan,
        delta_score=score,
    )
    return verdict, passes, metrics
