"""
Time-Reversal Symmetry (TRS) Test in Adaptive Golden Fields
-----------------------------------------------------------
Simulates a binary lens with multi-mode ringdown, OU noise, *and* an "odd" (nonreciprocal)
coupling between control variables (d, theta, eps) to break detailed balance.
Computes:
  • Forward vs time-reversed ROC (AUC) using ray-traced caustic snaps.
  • Entropy-production proxy via Gaussian path-KL on increments Δ[phi_med, focus].
  • Lag between phi_med minima and focus peaks (sign-consistent lag ⇒ arrow).
  • A simple time-irreversibility skew metric m3(τ) = ⟨(x(t+τ)-x(t))^3⟩.

Usage (example):
  python time_irreversibility_phi_a.py --T 120 --grid 121 --rays 13 --steps 360 --noise 0.05 \
      --Tk 1.0 --odd 0.05 --outdir outputs/trs_run1

Outputs: forward/reversed ROC+PR plots, time-series with lag, KL/metrics CSV.
"""

import os, argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--extent", type=float, default=6.8)
    ap.add_argument("--rays", type=int, default=13)
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--Tk", type=float, default=1.0)
    ap.add_argument("--odd", type=float, default=0.05, help="odd (nonreciprocal) coupling strength")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--outdir", type=str, default="outputs/trs_run1")
    return ap.parse_args()

# ---------------- Core math ----------------
def make_grid(N, ex, ey):
    x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
    dx, dy = x[1]-x[0], y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X*X + Y*Y); TH = np.arctan2(Y, X)
    return x,y,dx,dy,X,Y,R,TH

def norm01(Z):
    zmin, zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if zmax<=zmin else (Z - zmin)/(zmax - zmin + 1e-12)

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

def ou_update(val, mu, tau, sigma, dt):
    return val + (-(val-mu)/tau)*dt + sigma*np.sqrt(dt)*np.random.randn()

def thermal_barrier_jump(Tk=1.0, attempt_prob=0.25, dG_range=(5.0, 15.0)):
    if np.random.rand() > attempt_prob: return 0.0
    dG = np.random.uniform(*dG_range)
    if np.random.rand() < np.exp(-dG/max(Tk,1e-6)):
        mag = (0.5 + 0.5*np.random.rand()) * (1.0 + 0.1*(dG-10.0))
        return mag if np.random.rand() < 0.5 else -mag
    return 0.0

# Odd (nonreciprocal) coupling between (d, theta, eps):
#   [d, theta, eps]^T has an extra drift term creating circulation (breaks detailed balance).
#   Here: thetȧ += odd * eps,   epṡ -= odd * d.
def odd_coupled_update(d, theta, eps, mu_d, mu_th, mu_eps, tau_d, tau_th, tau_eps, sig_d, sig_th, sig_eps, dt, odd):
    dd  = (-(d     - mu_d )/tau_d )*dt + sig_d * np.sqrt(dt)*np.random.randn()
    dth = (-(theta - mu_th)/tau_th)*dt + sig_th* np.sqrt(dt)*np.random.randn()
    de  = (-(eps   - mu_eps)/tau_eps)*dt + sig_eps*np.sqrt(dt)*np.random.randn()
    d      += dd
    theta  += dth + odd * eps * dt
    eps    += de  - odd * d   * dt
    return d, theta, eps

def run_sim(T, N, ex, ey, rays, steps, noise_sigma, Tk, odd, seed):
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
        # stochastic + odd coupling
        d, theta, eps = odd_coupled_update(d, theta, eps,
                                           mu_d, mu_th, mu_eps,
                                           tau_d, tau_th, tau_eps,
                                           sig_d, sig_th, sig_eps,
                                           1.0, odd)
        # thermal jumps
        d   += 0.05 * thermal_barrier_jump(Tk=Tk, attempt_prob=0.22)
        eps += 0.03 * thermal_barrier_jump(Tk=Tk, attempt_prob=0.14)

        # apply eps to modes (as base amplitude; you can modulate per-mode if desired)
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

# ---- Metrics ----
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
    # KL( N(mu,Sigma) || N(-mu,Sigma) ) = 2 * mu^T Sigma^{-1} mu
    mu = dX.mean(axis=0); Xc = dX - mu
    Sig = (Xc.T @ Xc) / (len(dX)-1 + 1e-9)
    try:
        inv = np.linalg.inv(Sig + 1e-9*np.eye(Sig.shape[0]))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sig + 1e-9*np.eye(Sig.shape[0]))
    kl = float(2.0 * mu.T @ inv @ mu)
    return kl

def lag_by_xcorr(a, b, maxlag=10):
    a = (a - a.mean())/(a.std()+1e-9); b = (b - b.mean())/(b.std()+1e-9)
    best_lag = 0; best_val = -1e9
    for L in range(-maxlag, maxlag+1):
        if L<0: v = np.dot(a[-L:], b[:len(b)+L]) / (len(b)+L)
        elif L>0: v = np.dot(a[:len(a)-L], b[L:]) / (len(b)-L)
        else: v = np.dot(a, b) / len(a)
        if v > best_val: best_val, best_lag = v, L
    return best_lag, best_val

def m3_skew(x, tau=1):
    dx = x[tau:] - x[:-tau]
    return float(np.mean(dx**3))

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ex = args.extent; ey = 0.62*args.extent

    # Forward simulation with odd coupling
    phi, foc, snaps = run_sim(args.T, args.grid, ex, ey, args.rays, args.steps, args.noise, args.Tk, args.odd, args.seed)
    labels = snaps.astype(int)

    # Reversed-time series (null for TRS)
    phi_rev = phi[::-1].copy(); foc_rev = foc[::-1].copy(); labels_rev = labels[::-1].copy()

    # ROC/PR forward & reversed
    grid, FPR, TPR, PREC, RECALL, auc, auprc = roc_from_thresholds(phi, labels)
    _,    FPRr,TPRr,PRECr,RECALLr, aucr, auprcr = roc_from_thresholds(phi_rev, labels_rev)

    # Entropy production proxy via increments
    dphi = np.diff(phi); dfoc = np.diff(foc)
    kl = path_KL_gaussian(np.vstack([dphi, dfoc]).T)

    # Lag & time-irreversibility skew
    lag, xc = lag_by_xcorr(-phi, foc, maxlag=8)  # -phi so minima lead positive peaks
    m3_phi = m3_skew(phi, tau=1); m3_foc = m3_skew(foc, tau=1)
    corr = pearson_r(phi, foc)

    # Plots
    plt.figure(figsize=(6.2,4.6))
    plt.plot(FPR, TPR, marker='o', linestyle='none', label="forward")
    plt.plot(FPRr,TPRr,marker='x', linestyle='none', label="reversed")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC_fwd={auc:.2f}, AUC_rev={aucr:.2f})")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"roc_forward_vs_reversed.png"), dpi=200); plt.close()

    plt.figure(figsize=(6.2,4.6))
    plt.plot(RECALL, PREC, marker='o', linestyle='none', label="forward")
    plt.plot(RECALLr,PRECr,marker='x', linestyle='none', label="reversed")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUPRC_fwd={auprc:.2f}, rev={auprcr:.2f})")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"pr_forward_vs_reversed.png"), dpi=200); plt.close()

    t = np.arange(len(phi))
    plt.figure(figsize=(8.4,3.0))
    plt.plot(t, phi, label="phi_med"); plt.plot(t, foc, label="focus")
    plt.xlabel("t"); plt.ylabel("value"); plt.title(f"lag(phi→focus)={lag} (xc={xc:.2f}), corr={corr:.2f}")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"timeseries_with_lag.png"), dpi=200); plt.close()

    # Save metrics
    with open(os.path.join(args.outdir, "trs_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AUC_fwd","AUC_rev","AUPRC_fwd","AUPRC_rev","KL_path","lag_phi_to_focus","xcorr","m3_phi","m3_focus","corr_phi_focus"])
        w.writerow([f"{auc:.6f}", f"{aucr:.6f}", f"{auprc:.6f}", f"{auprcr:.6f}",
                    f"{kl:.6f}", f"{lag:.3f}", f"{xc:.6f}", f"{m3_phi:.6e}", f"{m3_foc:.6e}", f"{corr:.6f}"])

    print(f"AUC_fwd={auc:.3f}, AUC_rev={aucr:.3f}, AUPRC_fwd={auprc:.3f}, AUPRC_rev={auprcr:.3f}")
    print(f"KL_path={kl:.3f}, lag(phi→focus)={lag}, xcorr={xc:.2f}, m3_phi={m3_phi:.2e}, m3_focus={m3_foc:.2e}, corr={corr:.2f}")
    print(f"Saved plots & trs_metrics.csv to {args.outdir}")

if __name__ == "__main__":
    main()
