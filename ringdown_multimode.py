#!/usr/bin/env python3
"""Multi-mode ringdown lensing visualization with ray tracing and correlation metric.

Extends the single-mode ringdown by superposing several damped oscillatory azimuthal modes:
  Phi_total = Phi0 + sum_i eps_i * W_r(R; r_peak_i, sigma_i) * Phi0 * exp(-t/tau_i) * cos(omega_i * t + m_i * TH + phase_i)

Features:
  - Configurable list of modes via --modes string (semicolon-separated tuples)
  - Ray tracing with crossing-based focus metric (focus = 1/std(y crossings at x=0))
  - Median phi_a in central patch vs focus correlation scatter + linear fit
  - GIF + optional MP4 output (with fallback) + optional CSV metrics

Modes string format:
  m,omega,tau,eps,r_peak,sigma_r,phase; m,omega,tau,eps,r_peak,sigma_r,phase; ...
Example:
  --modes "2,1.20,10,0.14,3.0,0.9,0.0;3,1.90,7,0.10,3.2,1.0,1.0"

Quick run:
  python ringdown_multimode.py --frames 30 --progress

Custom modes:
  python ringdown_multimode.py --modes "2,1.2,10,0.16,3.0,0.9,0.0;2,1.55,6,0.07,2.8,0.8,0.7;3,1.9,7,0.1,3.2,1.0,1.0" --frames 36

Dependencies: numpy, matplotlib, imageio, pillow (no SciPy).
"""
from __future__ import annotations
import argparse, math, csv
import numpy as np, imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------------- CLI -----------------
parser = argparse.ArgumentParser(description="Multi-mode ringdown lensing animation + correlation")
parser.add_argument('--frames', type=int, default=30, help='Number of frames (T)')
parser.add_argument('--fps', type=int, default=6, help='Frames per second for animations')
parser.add_argument('--grid-n', type=int, default=201, help='Grid resolution N (creates N x N)')
parser.add_argument('--extent', type=float, default=6.5, help='Half-extent for x and y (square domain)')
parser.add_argument('--mass', type=float, default=1.0, help='Final mass M_f')
parser.add_argument('--soft', type=float, default=0.20, help='Core softening length')
parser.add_argument('--modes', default="2,1.20,10,0.14,3.0,0.9,0.0;3,1.90,7,0.10,3.2,1.0,1.0", help='Semicolon-separated mode tuples: m,omega,tau,eps,r_peak,sigma_r,phase')
parser.add_argument('--rays', type=int, default=11, help='Number of launch rays')
parser.add_argument('--ray-span', type=float, default=0.7, help='Fraction of y extent covered by launch points')
parser.add_argument('--step', type=float, default=0.014, help='Ray integration step size')
parser.add_argument('--steps', type=int, default=850, help='Maximum integration steps per ray')
parser.add_argument('--center-radius', type=float, default=2.0, help='Radius of central patch for median phi_a metric')
parser.add_argument('--gif', dest='gif', action='store_true', help='Enable GIF output (default on)')
parser.add_argument('--no-gif', dest='gif', action='store_false', help='Disable GIF output')
parser.add_argument('--mp4', dest='mp4', action='store_true', help='Enable MP4 output (default on)')
parser.add_argument('--no-mp4', dest='mp4', action='store_false', help='Disable MP4 output')
parser.add_argument('--cmap', default='viridis', help='Colormap for phi_a')
parser.add_argument('--bg', default='white', help='Background face color')
parser.add_argument('--ray-color', default='black', help='Ray path color')
parser.add_argument('--ray-lw', type=float, default=1.05, help='Ray linewidth')
parser.add_argument('--output-stem', default='ringdown_multimode', help='Filename stem for outputs')
parser.add_argument('--save-data', action='store_true', help='Write CSV with per-frame metrics')
parser.add_argument('--progress', action='store_true', help='Print per-frame progress (legacy flag; core metrics always print)')
parser.add_argument('--no-corr', action='store_true', help='Disable correlation plot output')
parser.set_defaults(gif=True, mp4=True)
args = parser.parse_args()

# ---------------- Parse modes -----------------
def parse_modes(spec: str):
    modes = []
    for part in spec.split(';'):
        part = part.strip()
        if not part:
            continue
        vals = [p.strip() for p in part.split(',')]
        if len(vals) != 7:
            raise ValueError(f"Mode '{part}' must have 7 comma-separated values (got {len(vals)}).")
        m, omg, tau, eps, rpk, sig, ph = vals
        modes.append((int(m), float(omg), float(tau), float(eps), float(rpk), float(sig), float(ph)))
    return modes

modes = parse_modes(args.modes)

# Global slow precession drift added to every mode phase (OMEGA * t)
# Adjust OMEGA for faster/slower collective precession; default chosen small.
OMEGA = 0.05

# ---------------- Grid -----------------
N = args.grid_n
ex = ey = args.extent
x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y)
R = np.sqrt(X*X + Y*Y); TH = np.arctan2(Y, X)

# ---------------- Helper functions -----------------
def norm01(Z: np.ndarray) -> np.ndarray:
    zmin, zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if zmax <= zmin else (Z - zmin)/(zmax - zmin)

M_f, soft = args.mass, args.soft

def Phi0(R: np.ndarray) -> np.ndarray:
    return -M_f / np.sqrt(R*R + soft*soft)

def W_r(R: np.ndarray, r_peak: float, sig: float) -> np.ndarray:
    return np.exp(- (R - r_peak)**2 / (2*sig*sig))

def Phi_total(t: int) -> np.ndarray:
    base = Phi0(R)
    dsum = 0.0
    # Added collective precession term: + OMEGA * t
    for (m, omg, tau, eps, rpk, sig, ph) in modes:
        phase = omg*t + m*TH + ph + OMEGA*t
        dsum += eps * W_r(R, rpk, sig) * base * np.exp(-t/tau) * np.cos(phase)
    return base + dsum

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

def bilinear(grid: np.ndarray, xv: float, yv: float) -> float:
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(math.floor(i), 0, N-2)); j0 = int(np.clip(math.floor(j), 0, N-2))
    di, dj = float(i-i0), float(j-j0)
    G00=grid[j0,i0]; G10=grid[j0,i0+1]; G01=grid[j0+1,i0]; G11=grid[j0+1,i0+1]
    return (1-di)*(1-dj)*G00 + di*(1-dj)*G10 + (1-di)*dj*G01 + di*dj*G11

def trace_rays(sx: np.ndarray, sy: np.ndarray, y_launch: np.ndarray, h: float, steps: int):
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
            xs[n] = xc + h*vxm;  ys[n] = yc + h*vym
            if hit is None and xs[n-1] <= 0.0 <= xs[n]:
                tloc = (0.0 - xs[n-1]) / (xs[n] - xs[n-1] + 1e-12)
                y_cross = ys[n-1] + tloc*(ys[n] - ys[n-1])
                hit = y_cross
            sxi2 = bilinear(sx, xs[n], ys[n]); syi2 = bilinear(sy, xs[n], ys[n])
            ax2  = - (sxi2*(vxm*vxm - vym*vym) + 2*syi2*vxm*vym)
            ay2  = - (syi2*(vym*vym - vxm*vxm) + 2*sxi2*vxm*vym)
            vx   = vxm + 0.5*h*ax2; vy = vym + 0.5*h*ay2
        rays.append((xs, ys))
        if hit is not None:
            crossings.append(hit)
    return rays, crossings

# ---------------- Evolution loop -----------------
frames = []
phi_med, focus = [], []
center_mask = (R < args.center_radius)
y_launch = np.linspace(-args.ray_span*ey, args.ray_span*ey, args.rays)

for t in range(args.frames):
    Phi = Phi_total(t)
    sx, sy = sigma_grad_from_Phi(Phi)
    phi = phi_a_from_Phi(Phi)
    rays, crosses = trace_rays(sx, sy, y_launch, h=args.step, steps=args.steps)

    med_val = float(np.median(phi[center_mask]))
    phi_med.append(med_val)
    if len(crosses) < 3:
        focus.append(0.0)
    else:
        focus.append(1.0 / (np.std(crosses) + 1e-3))

    # Per-frame bifurcation flag (central phi_a below threshold)
    bifurcation_flag = (med_val < 1.4)
    # Always print the requested metric line
    print(f"t={t:02d} | phi_med={med_val:.3f} | bifurcation_flag={bifurcation_flag}")
    if args.progress:
        print(f"  (progress) Frame {t+1}/{args.frames} crossings={len(crosses)} focus={focus[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7.0,7.0))
    ax.imshow(phi, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap=args.cmap)
    for xs, ys in rays:
        ax.plot(xs, ys, lw=args.ray_lw, color=args.ray_color)
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title(f"Multi-mode ringdown — t={t:02d},  phi_med≈{med_val:.3f}")
    fig.patch.set_facecolor(args.bg)
    ax.set_facecolor(args.bg)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)

# ---------------- Write animation outputs -----------------
if args.gif:
    gif_name = f"{args.output_stem}.gif"
    duration_ms = max(1, int(round(1000.0 / max(1, args.fps))))
    try:
        imageio.mimsave(gif_name, frames, duration=duration_ms)
        print(f"Saved GIF: {gif_name}")
    except TypeError:
        imageio.mimsave(gif_name, frames, fps=args.fps)
        print(f"Saved GIF (legacy fps path): {gif_name}")

if args.mp4:
    mp4_name = f"{args.output_stem}.mp4"
    wrote = False
    try:
        with imageio.get_writer(mp4_name, fps=args.fps, codec='libx264', quality=8) as w:
            for f in frames:
                w.append_data(f)
        wrote = True
        print(f"Saved MP4: {mp4_name} (imageio-ffmpeg)")
    except Exception as e:
        print(f"Primary MP4 writer failed ({e}); attempting matplotlib fallback...")
    if not wrote:
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=args.fps)
            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.imshow(frames[0], origin='lower')
            ax.axis('off')
            with writer.saving(fig, mp4_name, dpi=100):
                for f in frames:
                    im.set_data(f)
                    writer.grab_frame()
            plt.close(fig)
            wrote = True
            print(f"Saved MP4: {mp4_name} (matplotlib FFMpegWriter)")
        except Exception as e2:
            print(f"ERROR: Could not write MP4: {e2}")
            print("Install imageio-ffmpeg or system ffmpeg.")

# ---------------- Correlation plot + metrics -----------------
phi_arr = np.array(phi_med, dtype=float)
focus_arr = np.array(focus, dtype=float)
if not args.no_corr:
    # Correlation snippet (z-scored manual Pearson) as requested
    if len(phi_arr) >= 2 and np.std(phi_arr) > 0 and np.std(focus_arr) > 0:
        pm = phi_arr; fc = focus_arr
        pmz = (pm - pm.mean()) / (pm.std() + 1e-9)
        fcz = (fc - fc.mean()) / (fc.std() + 1e-9)
        r = float(np.mean(pmz * fcz))
    else:
        r = float('nan')
    print(f"corr(phi_med, focus) = {r:.2f}")
    plt.figure(figsize=(6.2,5.0))
    plt.scatter(phi_arr, focus_arr, s=32)
    if len(phi_arr) >= 2 and np.std(phi_arr) > 0:
        z = np.polyfit(phi_arr, focus_arr, 1)
        zz = np.poly1d(z)
        xx = np.linspace(phi_arr.min(), phi_arr.max(), 200)
        plt.plot(xx, zz(xx), lw=1.5)
    plt.xlabel("median phi_a (central patch)")
    plt.ylabel("focus score = 1/std(y@x=0)")
    plt.title(f"Multi-mode ringdown: phi_a vs focus  (r = {r:.2f})")
    plt.tight_layout()
    corr_name = f"phi_focus_corr_{args.output_stem}.png"
    plt.savefig(corr_name, dpi=200)
    plt.close()
    print(f"Saved: {corr_name}")

if args.save_data:
    csv_name = f"{args.output_stem}_metrics.csv"
    with open(csv_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["frame", "phi_med", "focus"])
        for i, (pm, fs) in enumerate(zip(phi_med, focus)):
            w.writerow([i, f"{pm:.6f}", f"{fs:.6f}"])
    print(f"Saved: {csv_name}")

print("Done.")
