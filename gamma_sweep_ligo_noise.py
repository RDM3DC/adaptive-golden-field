# gamma_sweep_ligo_noise.py
# Sweep odd coupling γ, inject LIGO-shaped noise into observables, report KL-rate & ROC gaps.

import os, csv, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- grid & helpers ----------
def make_grid(N, ex, ey):
    x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
    dx, dy = x[1]-x[0], y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X*X + Y*Y); TH = np.arctan2(Y, X)
    return x,y,dx,dy,X,Y,R,TH

def norm01(Z):
    zmin, zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if zmax<=zmin else (Z - zmin)/(zmax - zmin + 1e-12)

# ---------- fields ----------
def Phi_binary_oriented(X,Y,M1,M2,d,ang,soft=0.18):
    x1, y1 = +0.5*d*np.cos(ang), +0.5*d*np.sin(ang)
    x2, y2 = -x1, -y1
    r1 = np.sqrt((X-x1)**2 + (Y-y1)**2 + soft**2)
    r2 = np.sqrt((X-x2)**2 + (Y-y2)**2 + soft**2)
    return -M1/r1 - M2/r2

def Phi0(R, M=1.0, soft=0.18):
    return -M/np.sqrt(R*R + soft*soft)

def Phi_ringdown_multi(R,TH,t, modes, M=1.0):
    base = Phi0(R, M=M); dsum = 0.0
    for (m, omg, tau, eps, rpk, sig, ph, drift) in modes:
        W = np.exp(- (R - rpk)**2 / (2*sig*sig))
        dsum += eps * W * base * np.exp(-t/tau) * np.cos(omg*t + m*TH + ph + drift*t)
    return base + dsum

def Phi_total(X,Y,R,TH,M1,M2,d,theta, ring_modes, t):
    return Phi_binary_oriented(X,Y,M1,M2,d,theta) + (Phi_ringdown_multi(R,TH,t, ring_modes, M=1.0) - Phi0(R, M=1.0))

def sigma_grad_from_Phi(Phi, dx, dy):
    Aopt = (1 - 2*Phi) / np.clip(1 + 2*Phi, 1e-6, None)
    Aopt = np.clip(Aopt, 1e-6, 1e6)
    sigma = 0.5*np.log(Aopt)
    sx = np.gradient(sigma, dx, axis=1)
    sy = np.gradient(sigma, dy, axis=0)
    return sigma, sx, sy

def phi_a_from_Phi(Phi, dx, dy):
    Px = np.gradient(Phi, dx, axis=1); Py = np.gradient(Phi, dy, axis=0)
    Pxx = np.gradient(Px, dx, axis=1); Pyy = np.gradient(Py, dy, axis=0)
    Pxy = np.gradient(Px, dy, axis=0)
    Lap = Pxx + Pyy; T2  = Pxx**2 + 2*Pxy**2 + Pyy**2
    Rn, Tn = norm01(-Lap), norm01(T2)
    r = 0.45 + 0.40*Rn + 0.25*Tn
    return 0.5*(1 + np.sqrt(1 + 4*r))

# ---------- ray tracing ----------
def bilinear(grid, xv, yv, x, y):
    N = grid.shape[0]
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(np.floor(i), 0, N-2)); j0 = int(np.clip(np.floor(j), 0, N-2))
    di, dj = float(i-i0), float(j-j0)
    G00 = grid[j0, i0]; G10 = grid[j0, i0+1]
    G01 = grid[j0+1, i0]; G11 = grid[j0+1, i0+1]
    return (1-di)*(1-dj)*G00 + di*(1-dj)*G10 + (1-di)*dj*G01 + di*dj*G11

def trace_rays(sx, sy, x, y, ex, ey, y_launch, h=0.016, steps=360):
    start_x = -ex + 0.2; crossings = []
    for y0 in y_launch:
        xc, yc = start_x, y0; vx, vy = 1.0, 0.0
        for _ in range(steps):
            if not (x[0] <= xc <= x[-1] and y[0] <= yc <= y[-1]):
                break
            sxi = bilinear(sx, xc, yc, x, y); syi = bilinear(sy, xc, yc, x, y)
            ax  = - (sxi*(vx*vx - vy*vy) + 2*syi*vx*vy)
            ay  = - (syi*(vy*vy - vx*vx) + 2*sxi*vx*vy)
            vxm = vx + 0.5*h*ax; vym = vy + 0.5*h*ay
            xnew = xc + h*vxm;  ynew = yc + h*vym
            if (xc <= 0.0 <= xnew) or (xnew <= 0.0 <= xc):
                t = (0.0 - xc) / (xnew - xc + 1e-12)
                crossings.append(yc + t*(ynew - yc))
            sxi2 = bilinear(sx, xnew, ynew, x, y); syi2 = bilinear(sy, xnew, ynew, x, y)
            ax2  = - (sxi2*(vxm*vxm - vym*vym) + 2*syi2*vxm*vym)
            ay2  = - (syi2*(vym*vym - vxm*vxm) + 2*sxi2*vxm*vym)
            vx   = vxm + 0.5*h*ax2; vy = vym + 0.5*h*ay2
            xc, yc = xnew, ynew
    return crossings

def focus_from_crossings(crosses):
    return 0.0 if len(crosses) < 3 else 1.0/(np.std(crosses)+1e-3)

# ---------- stochastic dynamics ----------
def ou_update(val, mu, tau, sigma, dt):
    return val + (-(val-mu)/tau)*dt + sigma*np.sqrt(dt)*np.random.randn()

def thermal_barrier_jump(Tk=1.0, attempt_prob=0.22, dG_range=(5.0, 15.0)):
    if np.random.rand() > attempt_prob: return 0.0
    dG = np.random.uniform(*dG_range)
    if np.random.rand() < np.exp(-dG/max(Tk,1e-9)):
        mag = (0.5 + 0.5*np.random.rand()) * (1.0 + 0.1*(dG-10.0))
        return mag if np.random.rand() < 0.5 else -mag
    return 0.0

# odd (nonreciprocal) coupling: breaks detailed balance
def odd_coupled_update(d, theta, eps, mu_d, mu_th, mu_eps, tau_d, tau_th, tau_eps, sig_d, sig_th, sig_eps, dt, odd):
    dd  = (-(d     - mu_d )/tau_d )*dt + sig_d * np.sqrt(dt)*np.random.randn()
    dth = (-(theta - mu_th)/tau_th)*dt + sig_th* np.sqrt(dt)*np.random.randn()
    de  = (-(eps   - mu_eps)/tau_eps)*dt + sig_eps*np.sqrt(dt)*np.random.randn()
    d      += dd
    theta  += dth + odd * eps * dt
    eps    += de  - odd * d   * dt
    return d, theta, eps

# ---------- LIGO-like colored noise ----------
def colored_noise_like_ligo(T, amp=0.25, seed=None):
    """
    Make colored noise with PSD ~ (f0/f)^4 + c0 + (f/f1)^2 (seismic + mid + quantum).
    Returns zero-mean unit-std series scaled by 'amp'.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=T)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(T, d=1.0)
    eps = 1e-6
    f0, f1 = 0.08, 0.35
    S = (f0/np.maximum(f,eps))**4 + 0.02 + (f/np.maximum(f1,eps))**2
    S[0] = S[1]
    Wc = W * np.sqrt(S)
    n = np.fft.irfft(Wc, n=T)
    n = (n - n.mean())/(n.std()+1e-12)
    return amp * n

# ---------- metrics ----------
def roc_from_thresholds(phi_med, labels, lo=1.30, hi=1.60, K=61):
    grid = np.linspace(lo, hi, K); TPR=[]; FPR=[]; PREC=[]; RECALL=[]
    for th in grid:
        pred = (phi_med < th).astype(int)
        TP = int(np.sum((pred==1) & (labels==1)))
        FP = int(np.sum((pred==1) & (labels==0)))
        FN = int(np.sum((pred==0) & (labels==1)))
        TN = int(np.sum((pred==0) & (labels==0)))
        tpr = TP/(TP+FN+1e-9); fpr = FP/(FP+TN+1e-9); prec = TP/(TP+FP+1e-9); rec=tpr
        TPR.append(tpr); FPR.append(fpr); PREC.append(prec); RECALL.append(rec)
    order = np.argsort(FPR); auc = float(np.trapz(np.array(TPR)[order], np.array(FPR)[order]))
    order_pr = np.argsort(RECALL); auprc = float(np.trapz(np.array(PREC)[order_pr], np.array(RECALL)[order_pr]))
    return grid, np.array(FPR), np.array(TPR), np.array(PREC), np.array(RECALL), auc, auprc

def pearson_r(a,b):
    a=(a-a.mean())/(a.std()+1e-9); b=(b-b.mean())/(b.std()+1e-9)
    return float(np.mean(a*b))

def path_KL_gaussian(dX):
    mu = dX.mean(axis=0); Xc = dX - mu
    Sig = (Xc.T @ Xc) / (len(dX)-1 + 1e-9)
    try: inv = np.linalg.inv(Sig + 1e-9*np.eye(Sig.shape[0]))
    except np.linalg.LinAlgError: inv = np.linalg.pinv(Sig + 1e-9*np.eye(Sig.shape[0]))
    return float(2.0 * mu.T @ inv @ mu)

def lag_by_xcorr(a, b, maxlag=10):
    a = (a - a.mean())/(a.std()+1e-9); b = (b - b.mean())/(b.std()+1e-9)
    best_lag = 0; best_val = -1e9
    for L in range(-maxlag, maxlag+1):
        if L<0: v = np.dot(a[-L:], b[:len(b)+L]) / (len(b)+L)
        elif L>0: v = np.dot(a[:len(a)-L], b[L:]) / (len(b)-L)
        else: v = np.dot(a, b) / len(a)
        if v > best_val: best_val, best_lag = v, L
    return best_lag, best_val

# ---------- simulation ----------
def run_sim(T=100, N=101, ex=6.5, ey=4.0, rays=13, steps=340, noise_sigma=0.05, Tk=1.0, odd=0.10, seed=123):
    np.random.seed(seed)
    x,y,dx,dy,X,Y,R,TH = make_grid(N, ex, ey)
    M1, M2 = 1.05, 0.95
    d, theta, eps = 2.45, 0.0, 0.12
    mu_d, mu_th, mu_eps = 2.45, 0.0, 0.12
    tau_d, tau_th, tau_eps = 28.0, 18.0, 12.0
    sig_d, sig_th, sig_eps = 0.06*noise_sigma, 0.08*noise_sigma, 0.05*noise_sigma

    modes0 = [
        (2, 1.18, 13.0, 0.16, 2.9, 0.8, 0.0, 0.02),
        (2, 1.34,  8.0, 0.12, 3.0, 0.9, 0.6, 0.00),
        (3, 1.40,  9.5, 0.08, 3.1, 1.0, 0.9, 0.00),
    ]

    y_launch = np.linspace(-0.7*ey, 0.7*ey, rays)
    phi_med, focus, snaps = [], [], []
    prev_mult = None

    for t in range(T):
        d, theta, eps = odd_coupled_update(d, theta, eps,
                                           mu_d, mu_th, mu_eps,
                                           tau_d, tau_th, tau_eps,
                                           sig_d, sig_th, sig_eps,
                                           1.0, odd)
        d   += 0.05 * thermal_barrier_jump(Tk=Tk, attempt_prob=0.22)
        eps += 0.03 * thermal_barrier_jump(Tk=Tk, attempt_prob=0.14)

        modes = [(m,om,tau,base_eps, rpk,sig,ph,dr) for (m,om,tau,base_eps,rpk,sig,ph,dr) in modes0]
        Phi = Phi_total(X,Y,R,TH,M1, M2, d, theta, modes, t)
        sigma, sx, sy = sigma_grad_from_Phi(Phi, dx, dy)
        phi = phi_a_from_Phi(Phi, dx, dy)

        crosses = trace_rays(sx, sy, x, y, ex, ey, y_launch, steps=steps)
        mult = len(crosses); foc = focus_from_crossings(crosses)

        phi_med.append(float(np.median(phi[R<2.0]))); focus.append(foc)
        snap = (prev_mult is not None) and (abs(mult - prev_mult) >= 2)
        snaps.append(snap); prev_mult = mult

    return np.array(phi_med), np.array(focus), np.array(snaps, bool)

def run_once_with_ligo_noise(T, N, ex, ey, rays, steps, noise_sigma, Tk, odd, seed, amp_phi, amp_focus):
    # forward sim (clean)
    phi, foc, snaps = run_sim(T,N,ex,ey,rays,steps,noise_sigma,Tk,odd,seed)
    # inject detector noise in observables
    n_phi = colored_noise_like_ligo(len(phi), amp=amp_phi, seed=seed+11)
    n_foc = colored_noise_like_ligo(len(foc), amp=amp_focus, seed=seed+23)
    phiN = phi + n_phi*np.std(phi)
    focN = foc + n_foc*np.std(foc)
    labels = snaps.astype(int)

    # ROC/PR on noisy phi
    grid, FPR, TPR, PREC, RECALL, auc, auprc = roc_from_thresholds(phiN, labels)
    # reversed null
    phiR = phiN[::-1]; labelsR = labels[::-1]
    _, FPRr, TPRr, PRECr, RECALLr, aucr, auprcr = roc_from_thresholds(phiR, labelsR)

    # KL **rate** using increments (per step)
    dphi = np.diff(phiN); dfoc = np.diff(focN)
    kl = path_KL_gaussian(np.vstack([dphi, dfoc]).T) / max(len(dphi),1)

    # lag (ϕ minima leading focus peaks → use -phi vs focus)
    lag, xc = lag_by_xcorr(-phiN, focN, maxlag=8)

    return {
        "AUC_fwd": auc, "AUC_rev": aucr, "AUPRC_fwd": auprc, "AUPRC_rev": auprcr,
        "KL_rate": kl, "lag_phi_to_focus": lag, "xcorr": xc,
        "corr_phi_focus": pearson_r(phiN, focN)
    }

# ---------- sweep ----------
def sweep_gamma(outdir, gammas, T=100, N=101, ex=6.5, ey=4.0, rays=13, steps=340,
                noise_sigma=0.05, Tk=1.0, seed=123, amp_phi=0.25, amp_focus=0.25):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for g in gammas:
        m = run_once_with_ligo_noise(T,N,ex,ey,rays,steps,noise_sigma,Tk,g,seed, amp_phi, amp_focus)
        row = {"gamma": g, **m}
        rows.append(row)
        print(f"γ={g:.3f} → KL_rate={m['KL_rate']:.3f}, ΔAUC={(m['AUC_fwd']-m['AUC_rev']):.3f}, lag={m['lag_phi_to_focus']:.2f}")

    # CSV
    csv_path = os.path.join(outdir, "gamma_sweep_ligo.csv")
    keys = ["gamma","KL_rate","AUC_fwd","AUC_rev","AUPRC_fwd","AUPRC_rev","lag_phi_to_focus","xcorr","corr_phi_focus"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in keys})

    # Plots
    gam = np.array([r["gamma"] for r in rows], float)
    KLr = np.array([r["KL_rate"] for r in rows], float)
    dAUC= np.array([r["AUC_fwd"]-r["AUC_rev"] for r in rows], float)
    lag = np.array([r["lag_phi_to_focus"] for r in rows], float)

    # KL_rate vs gamma
    plt.figure(figsize=(5.8,4.2)); plt.plot(gam, KLr, marker='o')
    plt.xlabel("gamma"); plt.ylabel("KL rate (per step)"); plt.title("Entropy production rate vs gamma (LIGO-shaped noise)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"gamma_vs_KLrate.png"), dpi=200); plt.close()

    # ΔAUC vs gamma
    plt.figure(figsize=(5.8,4.2)); plt.plot(gam, dAUC, marker='o')
    plt.xlabel("gamma"); plt.ylabel("\u0394AUC (fwd - rev)"); plt.title("ROC gap vs gamma (noisy)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"gamma_vs_dAUC_noisy.png"), dpi=200); plt.close()

    # lag vs gamma
    plt.figure(figsize=(5.8,4.2)); plt.plot(gam, lag, marker='o')
    plt.xlabel("gamma"); plt.ylabel("lag (frames)"); plt.title("\u03d5_a\u2192focus lag vs gamma (noisy)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"gamma_vs_lag_noisy.png"), dpi=200); plt.close()

    return csv_path

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs/gamma_ligo")
    ap.add_argument("--gammas", type=str, default="0,0.02,0.05,0.10,0.15,0.20")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--rays", type=int, default=13)
    ap.add_argument("--steps", type=int, default=340)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--Tk", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--amp_phi", type=float, default=0.25, help="LIGO-noise amplitude as fraction of std(phi)")
    ap.add_argument("--amp_focus", type=float, default=0.25, help="LIGO-noise amplitude as fraction of std(focus)")
    return ap.parse_args()

def main():
    args = parse_args()
    ex, ey = 6.5, 4.0
    gammas = [float(s) for s in args.gammas.split(",") if s.strip()!=""]
    csv_path = sweep_gamma(args.outdir, gammas,
                           T=args.T, N=args.grid, ex=ex, ey=ey,
                           rays=args.rays, steps=args.steps,
                           noise_sigma=args.noise, Tk=args.Tk, seed=args.seed,
                           amp_phi=args.amp_phi, amp_focus=args.amp_focus)
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()
