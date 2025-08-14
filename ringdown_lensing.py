#!/usr/bin/env python3
"""Ringdown lensing visualization with ray tracing and phi_a vs focus correlation.

Dependency minimal: numpy, matplotlib, imageio, pillow (no SciPy required).

Phenomenological model:
  Base potential Phi0(R) = - M_f / sqrt(R^2 + soft^2)
  Perturbation dPhi ~ eps * Phi0 * exp(-(R-r_peak)^2/(2 sig_r^2)) * exp(-t/tau) * cos(omega*t + m*Theta + phi0)
  Here m=2 azimuthal structure.

Focus metric: For each frame we trace rays launched from the left; record the y-positions
where they cross x=0; define focus score = 1 / (std(y_cross) + 1e-3) if >=3 crossings else 0.

Correlation: Pearson r between median phi_a in central patch (R < 2) and focus score.

Outputs (selected by flags):
  ringdown_lensing.gif
  ringdown_lensing.mp4 (if enabled and ffmpeg available)
  phi_focus_corr_ringdown.png (scatter + best-fit line)
  ringdown_metrics.csv (optional with --save-data)

Examples:
  python ringdown_lensing.py --frames 30 --progress
  python ringdown_lensing.py --frames 40 --omega 1.4 --eps 0.22 --tau 14 --gif --no-mp4
  python ringdown_lensing.py --save-data --output-stem rd_run1 --rays 15

"""
from __future__ import annotations
import argparse, math, csv, os
import numpy as np, imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------------- CLI -----------------
parser = argparse.ArgumentParser(description="Ringdown lensing animation + correlation plot (no SciPy)")
parser.add_argument('--frames', type=int, default=28, help='Number of frames (T)')
parser.add_argument('--fps', type=int, default=6, help='Frames per second for animations')
parser.add_argument('--grid-n', type=int, default=201, help='Grid resolution N (creates N x N)')
parser.add_argument('--extent', type=float, default=6.5, help='Half-extent for both x and y (square domain)')
parser.add_argument('--m-f', type=float, default=1.0, help='Final mass M_f')
parser.add_argument('--soft', type=float, default=0.20, help='Core softening length')
parser.add_argument('--eps', type=float, default=0.18, help='Perturbation amplitude epsilon')
parser.add_argument('--tau', type=float, default=10.0, help='Damping timescale')
parser.add_argument('--omega', type=float, default=1.20, help='Angular (oscillation) frequency per frame')
parser.add_argument('--phi0', type=float, default=0.0, help='Initial phase offset')
parser.add_argument('--r-peak', type=float, default=3.0, help='Radial location of perturbation peak')
parser.add_argument('--sig-r', type=float, default=0.9, help='Radial Gaussian width of perturbation')
parser.add_argument('--m-mode', type=int, default=2, help='Azimuthal mode number m (cos(m*Theta + ...))')
parser.add_argument('--rays', type=int, default=11, help='Number of launch rays')
parser.add_argument('--ray-span', type=float, default=0.7, help='Fraction of y-extent covered by launch points (0-1)')
parser.add_argument('--step', type=float, default=0.014, help='Ray integration step size h')
parser.add_argument('--steps', type=int, default=850, help='Maximum ray integration steps')
parser.add_argument('--gif', dest='gif', action='store_true', help='Enable GIF output (default on)')
parser.add_argument('--no-gif', dest='gif', action='store_false', help='Disable GIF output')
parser.add_argument('--mp4', dest='mp4', action='store_true', help='Enable MP4 output (default on)')
parser.add_argument('--no-mp4', dest='mp4', action='store_false', help='Disable MP4 output')
parser.add_argument('--cmap', default='viridis', help='Colormap for phi_a')
parser.add_argument('--bg', default='white', help='Background figure face color')
parser.add_argument('--ray-color', default='black', help='Ray path color')
parser.add_argument('--ray-lw', type=float, default=1.05, help='Ray linewidth')
parser.add_argument('--output-stem', default='ringdown_lensing', help='Filename stem for outputs')
parser.add_argument('--center-radius', type=float, default=2.0, help='Radius of central patch for median phi_a metric')
parser.add_argument('--save-data', action='store_true', help='Also write CSV with frame metrics')
parser.add_argument('--progress', action='store_true', help='Print per-frame progress')
parser.set_defaults(gif=True, mp4=True)
args = parser.parse_args()

# ---------------- Grid -----------------
N = args.grid_n
ex = ey = args.extent
x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y)
R = np.sqrt(X*X + Y*Y)
Theta = np.arctan2(Y, X)

# ---------------- Helpers -----------------
def norm01(Z: np.ndarray) -> np.ndarray:
    zmin, zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if zmax <= zmin else (Z - zmin) / (zmax - zmin)

# Potentials
M_f = args.m_f
soft = args.soft

def Phi0(R: np.ndarray) -> np.ndarray:
    return - M_f / np.sqrt(R*R + soft*soft)

def dPhi(R: np.ndarray, Theta: np.ndarray, t: int) -> np.ndarray:
    W = np.exp(- (R - args.r_peak)**2 / (2*args.sig_r*args.sig_r))
    return (args.eps * W * Phi0(R) * np.exp(-t/args.tau) *
            np.cos(args.omega * t + args.m_mode * Theta + args.phi0))

def Phi_total(t: int) -> np.ndarray:
    return Phi0(R) + dPhi(R, Theta, t)

# Optical metric & phi_a

def sigma_grad_from_Phi(Phi: np.ndarray):
    Aopt = (1 - 2*Phi) / np.clip(1 + 2*Phi, 1e-6, None)
    Aopt = np.clip(Aopt, 1e-6, 1e+6)
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

# Bilinear interpolation & ray tracing

def bilinear(grid: np.ndarray, xv: float, yv: float) -> float:
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(math.floor(i), 0, N-2)); j0 = int(np.clip(math.floor(j), 0, N-2))
    di, dj = float(i - i0), float(j - j0)
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

# --------------- Evolution loop ---------------
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

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.imshow(phi, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap=args.cmap)
    for xs, ys in rays:
        ax.plot(xs, ys, lw=args.ray_lw, color=args.ray_color)
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title(f"Ringdown lensing — t={t:02d},  phi_med≈{med_val:.3f}")
    fig.patch.set_facecolor(args.bg)
    ax.set_facecolor(args.bg)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)
    if args.progress:
        print(f"Frame {t+1}/{args.frames} phi_med={med_val:.4f} crossings={len(crosses)}")

# --------------- Write animation outputs ---------------
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

# --------------- Correlation plot ---------------
phi_arr = np.array(phi_med, dtype=float)
focus_arr = np.array(focus, dtype=float)
if len(phi_arr) >= 2 and np.std(phi_arr) > 0 and np.std(focus_arr) > 0:
    r = float(np.cov(phi_arr, focus_arr, ddof=1)[0,1] / (np.std(phi_arr, ddof=1) * np.std(focus_arr, ddof=1)))
else:
    r = float('nan')

plt.figure(figsize=(6.2,5.0))
plt.scatter(phi_arr, focus_arr, s=32)
if len(phi_arr) >= 2:
    z = np.polyfit(phi_arr, focus_arr, 1)
    zz = np.poly1d(z)
    xx = np.linspace(phi_arr.min(), phi_arr.max(), 200)
    plt.plot(xx, zz(xx), lw=1.5)
plt.xlabel("median phi_a (central patch)")
plt.ylabel("focus score = 1/std(y@x=0)")
plt.title(f"Ringdown: phi_a vs focus  (r = {r:.2f})")
plt.tight_layout()
scatter_name = f"phi_focus_corr_{args.output_stem}.png"
plt.savefig(scatter_name, dpi=200)
plt.close()
print(f"Saved: {scatter_name}")

# --------------- Optional CSV ---------------
if args.save_data:
    csv_name = f"{args.output_stem}_metrics.csv"
    with open(csv_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["frame", "phi_med", "focus"])
        for i, (pm, fs) in enumerate(zip(phi_med, focus)):
            w.writerow([i, f"{pm:.6f}", f"{fs:.6f}"])
    print(f"Saved: {csv_name}")

print("Done.")
