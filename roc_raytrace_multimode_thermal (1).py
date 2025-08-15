
"""
Full ROC with ray-traced multiplicity (ground truth) + multi-mode ringdown + thermal barriers.

Usage (defaults are tuned for strong caustics; adjust for speed):
    python roc_raytrace_multimode_thermal.py --T 160 --grid 161 --rays 31 --steps 600 --noise 0.06 --Tk 1.0 --outdir outputs/roc_run1

Outputs:
    - roc_phi_snap.png, pr_phi_snap.png, threshold_sweep.png, timeseries_phi_focus.png
    - metrics.csv  (AUC, AUPRC, corr, threshold grid with precision/recall/TPR/FPR)
"""

import os, argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=160)
    ap.add_argument("--grid", type=int, default=161)
    ap.add_argument("--extent", type=float, default=7.2)
    ap.add_argument("--rays", type=int, default=31)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--noise", type=float, default=0.06)
    ap.add_argument("--Tk", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="outputs/roc_run1")
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
    return np.zeros_like(Z) if zmax<=zmin else (Z - zmin)/(zmax - zmin)

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

def trace_rays(sx, sy, x, y, ex, ey, y_launch, h=0.016, steps=600):
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

def run_sim(T, N, ex, ey, rays, steps, noise_sigma, Tk, seed):
    np.random.seed(seed)
    x,y,dx,dy,X,Y,R,TH = make_grid(N, ex, ey)
    M1, M2 = 1.05, 0.95
    d, theta = 2.45, 0.0
    OU_d     = (2.45, 28.0, 0.06*noise_sigma)
    OU_theta = (0.0,  16.0, 0.08*noise_sigma)
    modes0 = [
        (2, 1.18, 13.0, 0.16, 2.9, 0.8, 0.0, 0.02),
        (2, 1.34,  8.0, 0.12, 3.0, 0.9, 0.6, 0.00),
        (3, 1.40,  9.5, 0.08, 3.1, 1.0, 0.9, 0.00),
    ]
    y_launch = np.linspace(-0.7*ey, 0.7*ey, rays)
    phi_med, focus, snaps = [], [], []
    prev_mult = None
    for t in range(T):
        d     = ou_update(d,     *OU_d,     1.0)
        theta = ou_update(theta, *OU_theta, 1.0)
        d    += 0.05 * thermal_barrier_jump(Tk=Tk, attempt_prob=0.22)
        eps_k = 0.03 * thermal_barrier_jump(Tk=Tk, attempt_prob=0.14)
        modes = [(m,om,tau,eps+eps_k,rpk,sig,ph,dr) for (m,om,tau,eps,rpk,sig,ph,dr) in modes0]
        Phi = Phi_total(X,Y,R,TH,M1, M2, d, theta, modes, t)
        sigma, sx, sy = sigma_grad_from_Phi(Phi, dx, dy)
        phi    = phi_a_from_Phi(Phi, dx, dy)
        crosses = trace_rays(sx, sy, x, y, ex, ey, y_launch, steps=steps)
        mult = len(crosses); foc = focus_from_crossings(crosses)
        phi_med.append(float(np.median(phi[R<2.0]))); focus.append(foc)
        snap = (prev_mult is not None) and (abs(mult - prev_mult) >= 2)
        snaps.append(snap); prev_mult = mult
    return np.array(phi_med), np.array(focus), np.array(snaps, bool)

def roc_pr_from_threshold_series(phi_med, snaps, phi_min=1.30, phi_max=1.60, K=61):
    labels = snaps.astype(int)
    grid = np.linspace(phi_min, phi_max, K)
    TPR=[]; FPR=[]; PREC=[]; RECALL=[]
    for th in grid:
        pred = (phi_med < th).astype(int)
        TP = int(np.sum((pred==1) & (labels==1)))
        FP = int(np.sum((pred==1) & (labels==0)))
        FN = int(np.sum((pred==0) & (labels==1)))
        TN = int(np.sum((pred==0) & (labels==0)))
        tpr = TP/(TP+FN+1e-9); fpr = FP/(FP+TN+1e-9)
        prec= TP/(TP+FP+1e-9); rec = tpr
        TPR.append(tpr); FPR.append(fpr); PREC.append(prec); RECALL.append(rec)
    # AUC / AUPRC
    order = np.argsort(FPR); auc = float(np.trapz(np.array(TPR)[order], np.array(FPR)[order]))
    order_pr = np.argsort(RECALL); auprc = float(np.trapz(np.array(PREC)[order_pr], np.array(RECALL)[order_pr]))
    return grid, np.array(FPR), np.array(TPR), np.array(PREC), np.array(RECALL), auc, auprc

def pearson_r(a,b):
    a=(a-a.mean())/(a.std()+1e-9); b=(b-b.mean())/(b.std()+1e-9)
    return float(np.mean(a*b))

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ex = args.extent; ey = 0.62*args.extent

    # Run sim
    phi_med, focus, snaps = run_sim(args.T, args.grid, ex, ey, args.rays, args.steps, args.noise, args.Tk, args.seed)

    # ROC/PR
    grid, FPR, TPR, PREC, RECALL, auc, auprc = roc_pr_from_threshold_series(phi_med, snaps)
    corr = pearson_r(phi_med, focus)

    # Plots
    plt.figure(figsize=(6.4,4.8))
    plt.plot(FPR, TPR, marker='o', linestyle='none')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.2f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "roc_phi_snap.png"), dpi=200); plt.close()

    plt.figure(figsize=(6.4,4.8))
    plt.plot(RECALL, PREC, marker='o', linestyle='none')
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision–Recall (AUPRC={auprc:.2f})")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "pr_phi_snap.png"), dpi=200); plt.close()

    plt.figure(figsize=(6.4,4.8))
    plt.plot(grid, PREC, marker='o', label="precision")
    plt.plot(grid, RECALL, marker='o', label="recall")
    plt.xlabel("phi_a threshold"); plt.ylabel("score"); plt.title("Threshold sweep")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "threshold_sweep.png"), dpi=200); plt.close()

    t = np.arange(len(phi_med))
    plt.figure(figsize=(8.2,3.2))
    plt.plot(t, phi_med, label="phi_med"); plt.plot(t, focus, label="focus")
    plt.xlabel("t"); plt.ylabel("value"); plt.title(f"Time series: corr(phi_med, focus) = {corr:.2f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "timeseries_phi_focus.png"), dpi=200); plt.close()

    # Save metrics
    with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["AUC", "AUPRC", "corr_phi_focus"]); w.writerow([f"{auc:.6f}", f"{auprc:.6f}", f"{corr:.6f}"])
        w.writerow([]); w.writerow(["phi_thresh","TPR","FPR","precision","recall"])
        for th, tpr, fpr, pr, rc in zip(grid, TPR, FPR, PREC, RECALL):
            w.writerow([f"{th:.4f}", f"{tpr:.6f}", f"{fpr:.6f}", f"{pr:.6f}", f"{rc:.6f}"])

    print(f"Done. AUC={auc:.3f}, AUPRC={auprc:.3f}, corr(phi_med, focus)={corr:.2f}")
    print(f"Saved to: {args.outdir}")

if __name__ == "__main__":
    main()
