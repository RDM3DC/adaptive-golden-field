#!/usr/bin/env python3
"""Stochastic Hybrid phi_a / Caustic Prediction Demo (Enhanced)

Adds Ornstein–Uhlenbeck (OU) noise to separation d(t), orientation theta(t), and a ringdown
mode amplitude eps(t). Traces rays in an optical metric, detects "snap" events when
image multiplicity (ray crossings at x=0) jumps by >= snap_delta (default 2), and evaluates
predictors:

Predictors:
    1. phi_med threshold: phi_med < THRESH
    2. focus high-quantile: focus >= Q-quantile (e.g. top 80%)

Metrics: precision, recall, F1 for each; confusion matrix PNG for phi threshold rule; optional CSV.
Also computes correlation corr(phi_med, focus).

Auto-calibration (optional): increase OU sigma for separation until at least target snaps observed.

Outputs:
    - hybrid_timeseries.png       (phi_med, focus, multiplicity, snaps marked)
    - phi_focus_corr.png          (scatter + linear fit + correlation)
    - confusion_matrix.png        (bars for phi threshold TP/FP/FN/TN)
    - metrics.csv (optional via --csv-out)
    - hybrid_noise_clip.gif (if --gif)

Examples:
    python stochastic_hybrid_phi_a.py --frames 200 --ou-d-sigma 0.12 --ou-theta-sigma 0.12 --ou-eps-sigma 0.05 --rays 17 --progress
    python stochastic_hybrid_phi_a.py --auto-calibrate-snaps 5 --progress
    python stochastic_hybrid_phi_a.py --phi-threshold 1.45 --focus-quantile 0.85 --csv-out metrics.csv --gif
"""
from __future__ import annotations
import os, argparse, csv
import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio

# ---------- Grid ----------
N, ex, ey = 181, 7.5, 5.0
x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y)
R = np.sqrt(X*X + Y*Y); TH = np.arctan2(Y, X)

# ---------- Helpers ----------
def norm01(Z: np.ndarray) -> np.ndarray:
    zmin, zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if zmax <= zmin else (Z - zmin)/(zmax - zmin)

# ---------- Optics & phi_a ----------
def sigma_grad_from_Phi(Phi: np.ndarray):
    Aopt = (1 - 2*Phi) / np.clip(1 + 2*Phi, 1e-6, None)
    Aopt = np.clip(Aopt, 1e-6, 1e6)
    sigma = 0.5*np.log(Aopt)
    sx = np.gradient(sigma, dx, axis=1)
    sy = np.gradient(sigma, dy, axis=0)
    return sx, sy

def phi_a_from_Phi(Phi: np.ndarray) -> np.ndarray:
    Px = np.gradient(Phi, dx, axis=1); Py = np.gradient(Phi, dy, axis=0)
    Pxx = np.gradient(Px, dx, axis=1); Pyy = np.gradient(Py, dy, axis=0)
    Pxy = np.gradient(Px, dy, axis=0)
    Lap = Pxx + Pyy
    T2  = Pxx**2 + 2*Pxy**2 + Pyy**2
    Rn, Tn = norm01(-Lap), norm01(T2)
    r0, aR, aT = 0.45, 0.40, 0.25
    r = r0 + aR*Rn + aT*Tn
    return 0.5*(1 + np.sqrt(1 + 4*r))

# ---------- Potentials ----------
def Phi_binary_oriented(M1: float, M2: float, d: float, ang: float, soft: float = 0.24) -> np.ndarray:
    x1, y1 = +0.5*d*np.cos(ang), +0.5*d*np.sin(ang)
    x2, y2 = -x1, -y1
    r1 = np.sqrt((X-x1)**2 + (Y-y1)**2 + soft**2)
    r2 = np.sqrt((X-x2)**2 + (Y-y2)**2 + soft**2)
    return -M1/r1 - M2/r2

def Phi_ringdown_mode(eps: float, omega: float, tau: float, t: float,
                       r_peak: float = 3.0, sig: float = 0.9, soft: float = 0.24) -> np.ndarray:
    base = -1.0/np.sqrt(R*R + soft*soft)
    W = np.exp(- (R - r_peak)**2 / (2*sig*sig))
    return eps * W * base * np.exp(-t/tau) * np.cos(omega*t + 2*TH)

def Phi_total(M1: float, M2: float, d: float, theta: float,
              eps: float, omega: float, tau: float, t: float) -> np.ndarray:
    return Phi_binary_oriented(M1, M2, d, theta) + Phi_ringdown_mode(eps, omega, tau, t)

# ---------- OU Noise ----------
def ou_update(val: float, mu: float, tau: float, sigma: float, dt: float) -> float:
    return val + (-(val - mu)/tau)*dt + sigma*np.sqrt(dt)*np.random.randn()

# ---------- Interp + rays ----------
def bilinear(grid: np.ndarray, xv: float, yv: float) -> float:
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(np.floor(i), 0, N-2)); j0 = int(np.clip(np.floor(j), 0, N-2))
    di, dj = float(i - i0), float(j - j0)
    G00=grid[j0,i0]; G10=grid[j0,i0+1]; G01=grid[j0+1,i0]; G11=grid[j0+1,i0+1]
    return (1-di)*(1-dj)*G00 + di*(1-dj)*G10 + (1-di)*dj*G01 + di*dj*G11

def trace_rays(sx: np.ndarray, sy: np.ndarray, y_launch: np.ndarray, h: float = 0.015, steps: int = 700):
    start_x = -ex + 0.2
    rays, crossings = [], []
    for y0 in y_launch:
        xs = np.empty(steps); ys = np.empty(steps)
        xs[0], ys[0] = start_x, y0
        vx, vy = 1.0, 0.0
        hit = None
        for n in range(1, steps):
            xc, yc = xs[n-1], ys[n-1]
            if not (x[0] <= xc <= x[-1] and y[0] <= yc <= y[-1]):
                xs, ys = xs[:n], ys[:n]
                break
            sxi = bilinear(sx, xc, yc); syi = bilinear(sy, xc, yc)
            ax  = - (sxi*(vx*vx - vy*vy) + 2*syi*vx*vy)
            ay  = - (syi*(vy*vy - vx*vx) + 2*sxi*vx*vy)
            vxm = vx + 0.5*h*ax; vym = vy + 0.5*h*ay
            xs[n] = xc + h*vxm; ys[n] = yc + h*vym
            if hit is None and xs[n-1] <= 0.0 <= xs[n]:
                tloc = (0.0 - xs[n-1])/(xs[n] - xs[n-1] + 1e-12)
                crossings.append(ys[n-1] + tloc*(ys[n] - ys[n-1]))
                hit = True
            sxi2 = bilinear(sx, xs[n], ys[n]); syi2 = bilinear(sy, xs[n], ys[n])
            ax2  = - (sxi2*(vxm*vxm - vym*vym) + 2*syi2*vxm*vym)
            ay2  = - (syi2*(vym*vym - vxm*vxm) + 2*sxi2*vxm*vym)
            vx   = vxm + 0.5*h*ax2; vy = vym + 0.5*h*ay2
        rays.append((xs, ys))
    return rays, crossings

# ---------- CLI & Simulation + Evaluation ----------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Stochastic hybrid phi_a snap prediction with OU noise")
    p.add_argument('--frames', type=int, default=160, help='Number of timesteps')
    p.add_argument('--seed', type=int, default=7, help='RNG seed')
    p.add_argument('--phi-threshold', type=float, default=1.4, help='phi_med threshold')
    p.add_argument('--focus-quantile', type=float, default=0.8, help='Quantile for focus alert rule (0-1)')
    p.add_argument('--lag', type=int, default=1, help='+/- lag for matching alert to snap')
    p.add_argument('--snap-delta', type=int, default=2, help='Multiplicity jump to count as snap (abs diff >= delta)')
    # OU params (mu,tau,sigma)
    p.add_argument('--ou-d-mu', type=float, default=2.6)
    p.add_argument('--ou-d-tau', type=float, default=30.0)
    p.add_argument('--ou-d-sigma', type=float, default=0.05)
    p.add_argument('--ou-theta-mu', type=float, default=0.0)
    p.add_argument('--ou-theta-tau', type=float, default=18.0)
    p.add_argument('--ou-theta-sigma', type=float, default=0.08)
    p.add_argument('--ou-eps-mu', type=float, default=0.10)
    p.add_argument('--ou-eps-tau', type=float, default=10.0)
    p.add_argument('--ou-eps-sigma', type=float, default=0.03)
    p.add_argument('--omega', type=float, default=1.35, help='Ringdown omega')
    p.add_argument('--tau-rd', type=float, default=12.0, help='Ringdown damping time')
    p.add_argument('--d-init', type=float, default=2.6)
    p.add_argument('--theta-init', type=float, default=0.0)
    p.add_argument('--eps-init', type=float, default=0.10)
    p.add_argument('--rays', type=int, default=11, help='Number of rays to launch')
    p.add_argument('--ray-span', type=float, default=0.7, help='Fraction of vertical extent spanned by ray launch band')
    p.add_argument('--gif', action='store_true', help='Emit GIF (hybrid_noise_clip.gif)')
    p.add_argument('--gif-stride', type=int, default=2, help='Frame stride for GIF (every n-th frame)')
    p.add_argument('--csv-out', type=str, default=None, help='Write time series metrics to CSV')
    p.add_argument('--auto-calibrate-snaps', type=int, default=0, help='Target minimum snap events (increase d sigma until reached)')
    p.add_argument('--calib-max-iters', type=int, default=8, help='Max calibration iterations')
    p.add_argument('--calib-factor', type=float, default=1.25, help='Sigma multiply per calibration attempt')
    p.add_argument('--progress', action='store_true', help='Print per-frame metrics')
    return p

def simulate(args):
    np.random.seed(args.seed)
    y_launch = np.linspace(-args.ray_span*ey, args.ray_span*ey, args.rays)

    def run_once(sigma_d):
        d = args.d_init; theta = args.theta_init; eps = args.eps_init
        phi_med_series = []; focus_series=[]; mult_series=[]; snap_marks=[]
        frames=[]; prev_mult=None
        for t in range(args.frames):
            # OU update (override sigma for d if calibrating)
            d     = ou_update(d,     args.ou_d_mu,     args.ou_d_tau,     sigma_d,      1.0)
            theta = ou_update(theta, args.ou_theta_mu, args.ou_theta_tau, args.ou_theta_sigma, 1.0)
            eps   = ou_update(eps,   args.ou_eps_mu,   args.ou_eps_tau,   args.ou_eps_sigma,   1.0)
            Phi = Phi_total(1.05, 0.95, d, theta, eps, args.omega, args.tau_rd, t)
            sx, sy = sigma_grad_from_Phi(Phi)
            phi = phi_a_from_Phi(Phi)
            rays, crosses = trace_rays(sx, sy, y_launch)
            mult = len(crosses)
            focus = 0.0 if mult < 3 else 1.0/(np.std(crosses)+1e-3)
            phi_med = float(np.median(phi[R<2.0]))
            phi_med_series.append(phi_med); focus_series.append(focus); mult_series.append(mult)
            snap = (prev_mult is not None) and (abs(mult - prev_mult) >= args.snap_delta)
            snap_marks.append(snap); prev_mult = mult
            if args.progress:
                print(f"t={t:03d} d={d:.3f} phi_med={phi_med:.3f} mult={mult} snap={snap}")
            if args.gif and (t % args.gif_stride == 0):
                fig, ax = plt.subplots(figsize=(7.5,4.6))
                ax.imshow(phi, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6)
                for xs, ys in rays: ax.plot(xs, ys, lw=0.9)
                ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
                ax.text(0.01,0.99,f"t={t}\nd={d:.2f}\nphi_med={phi_med:.3f}\nmult={mult}",
                        transform=ax.transAxes, ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.25))
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                frames.append(frame)
        return (np.array(phi_med_series), np.array(focus_series), np.array(mult_series), np.array(snap_marks, dtype=bool), frames)

    sigma_d = args.ou_d_sigma
    if args.auto_calibrate_snaps > 0:
        for attempt in range(args.calib_max_iters):
            phi_s, foc_s, mult_s, snaps_s, _ = run_once(sigma_d)
            snap_count = int(snaps_s.sum())
            if snap_count >= args.auto_calibrate_snaps:
                if args.progress:
                    print(f"Calibration reached {snap_count} snaps with sigma_d={sigma_d:.4f} on attempt {attempt+1}")
                # Re-run preserving GIF frames this time
                return run_once(sigma_d)
            sigma_d *= args.calib_factor
            if args.progress:
                print(f"Calibration attempt {attempt+1}: snaps={snap_count} < target; increasing sigma_d -> {sigma_d:.4f}")
        if args.progress:
            print(f"Calibration ended without reaching target; using last sigma_d={sigma_d:.4f}")
        return run_once(sigma_d)
    else:
        return run_once(sigma_d)

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.mean(a * b))

def evaluate(phi_med, focus, mult, snaps, args):
    tlen = len(phi_med)
    # phi threshold rule
    phi_alerts = phi_med < args.phi_threshold
    # focus alert rule (top quantile)
    qcut = np.quantile(focus, args.focus_quantile)
    focus_alerts = focus >= qcut
    def near_snap(idx):
        lo, hi = max(0, idx-args.lag), min(tlen-1, idx+args.lag)
        return snaps[lo:hi+1].any()
    def confusion(alerts):
        TP = sum(near_snap(i) for i,a in enumerate(alerts) if a)
        FP = sum((not near_snap(i)) for i,a in enumerate(alerts) if a)
        FN = int(snaps.sum()) - TP
        TN = tlen - TP - FP - FN
        prec = TP / (TP + FP + 1e-9)
        rec  = TP / (TP + FN + 1e-9)
        f1   = 2*prec*rec/(prec+rec+1e-9)
        return dict(TP=TP,FP=FP,FN=FN,TN=TN,precision=prec,recall=rec,f1=f1,alerts=alerts)
    phi_stats = confusion(phi_alerts)
    focus_stats = confusion(focus_alerts)
    corr = pearson_r(phi_med, focus)
    return phi_stats, focus_stats, corr, qcut

def write_csv(path, phi_med, focus, mult, snaps, phi_alerts, focus_alerts):
    with open(path,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(["t","phi_med","focus","multiplicity","snap","phi_alert","focus_alert"])
        for t,(pm,fc,m,s,pa,fa) in enumerate(zip(phi_med,focus,mult,snaps,phi_alerts,focus_alerts)):
            w.writerow([t,f"{pm:.6f}",f"{fc:.6f}",m,int(s),int(pa),int(fa)])

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    phi_med, focus, mult, snaps, frames = simulate(args)
    phi_stats, focus_stats, corr, qcut = evaluate(phi_med, focus, mult, snaps, args)
    print(f"[phi_thresh] phi<{args.phi_threshold}: TP={phi_stats['TP']} FP={phi_stats['FP']} FN={phi_stats['FN']} TN={phi_stats['TN']} precision={phi_stats['precision']:.2f} recall={phi_stats['recall']:.2f} F1={phi_stats['f1']:.2f}")
    print(f"[focus_q] focus>=Q{args.focus_quantile:.2f}(={qcut:.4f}): TP={focus_stats['TP']} FP={focus_stats['FP']} FN={focus_stats['FN']} TN={focus_stats['TN']} precision={focus_stats['precision']:.2f} recall={focus_stats['recall']:.2f} F1={focus_stats['f1']:.2f}")
    print(f"corr(phi_med, focus) = {corr:.2f}")
    tgrid = np.arange(len(phi_med))
    # Time series plot
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(tgrid, phi_med, label='phi_med', lw=1.3)
    ax1.axhline(args.phi_threshold, ls='--', lw=1, label=f'phi_thresh={args.phi_threshold}')
    ax1.set_xlabel('t'); ax1.set_ylabel('phi_med')
    ax2 = ax1.twinx()
    ax2.plot(tgrid, focus, label='focus', lw=1.1, color='tab:orange')
    ax2.axhline(qcut, ls=':', color='tab:orange', lw=1, label=f'focus_q={qcut:.3f}')
    ax2.set_ylabel('focus (1/std crossings)')
    for tt in tgrid[snaps]:
        ax1.axvline(tt, color='k', alpha=0.12)
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout(); plt.savefig('hybrid_timeseries.png', dpi=200); plt.close()
    # Scatter plot
    fig = plt.figure(figsize=(6,5))
    plt.scatter(phi_med, focus, s=20)
    if len(phi_med) >= 2:
        z = np.polyfit(phi_med, focus, 1); zz = np.poly1d(z)
        xx = np.linspace(phi_med.min(), phi_med.max(), 200)
        plt.plot(xx, zz(xx), lw=1.4)
    plt.xlabel('median phi_a (central)'); plt.ylabel('focus = 1/std(y@x=0)')
    plt.title(f'Hybrid noise: corr(phi_med, focus) = {corr:.2f}')
    plt.tight_layout(); plt.savefig('phi_focus_corr.png', dpi=200); plt.close()
    # Confusion matrix for phi threshold
    fig = plt.figure(figsize=(5.8,3.6))
    vals = [phi_stats['TP'],phi_stats['FP'],phi_stats['FN'],phi_stats['TN']]; labels=['TP','FP','FN','TN']
    plt.bar(labels, vals, color=['tab:green','tab:red','tab:purple','tab:gray'])
    plt.title(f"phi<{args.phi_threshold}: P={phi_stats['precision']:.2f} R={phi_stats['recall']:.2f} F1={phi_stats['f1']:.2f}")
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=200); plt.close()
    if args.csv_out:
        write_csv(args.csv_out, phi_med, focus, mult, snaps, phi_stats['alerts'], focus_stats['alerts'])
        print(f"Saved CSV: {args.csv_out}")
    if args.gif and frames:
        imageio.mimsave('hybrid_noise_clip.gif', frames, fps=6)
        print('Saved GIF: hybrid_noise_clip.gif')
    print('Saved: hybrid_timeseries.png, phi_focus_corr.png, confusion_matrix.png')

if __name__ == '__main__':
    main()
