#!/usr/bin/env python3
"""Precessing inspiral visualization over adaptive phi_a field.

This extends the base inspiral animation by allowing the binary separation axis to rotate
(precession + orbital motion). A simple phenomenological model is used:

  d'      = -K / d^3 (inward inspiral)
  theta'  = omega_orb + omega_prec
  omega_orb  ~ c_orb * d^(-3/2)
  omega_prec ~ k_s * d^(-2) + k_k * J_eff * d^(-3)

Outputs (by default):
  - inspiral_precessing_phi_a.gif
  - inspiral_precessing_phi_a.mp4 (if ffmpeg backend available and not disabled)

Usage examples:
  python inspiral_precessing_phi_a.py --frames 24 --progress
  python inspiral_precessing_phi_a.py --no-mp4 --gif --fps 10 --frames 40
  python inspiral_precessing_phi_a.py --c-orb 0.25 --k-s 0.3 --k-k 0.15 --spin 0.4

Dependencies:
  pip install numpy matplotlib imageio pillow imageio-ffmpeg
"""
from __future__ import annotations
import os, math, argparse
import numpy as np, imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Precessing inspiral phi_a animation")
parser.add_argument('--frames', type=int, default=24, help='Number of frames')
parser.add_argument('--fps', type=int, default=6, help='Animation frame rate')
parser.add_argument('--width', type=float, default=8.0, help='Figure width (in)')
parser.add_argument('--height', type=float, default=4.0, help='Figure height (in)')
parser.add_argument('--gif', dest='gif', action='store_true', help='Enable GIF output (default on)')
parser.add_argument('--no-gif', dest='gif', action='store_false', help='Disable GIF output')
parser.add_argument('--mp4', dest='mp4', action='store_true', help='Enable MP4 output (default on)')
parser.add_argument('--no-mp4', dest='mp4', action='store_false', help='Disable MP4 output')
parser.add_argument('-o', '--output-stem', default='inspiral_precessing_phi_a', help='Output filename stem')
parser.add_argument('--progress', action='store_true', help='Print per-frame progress')
# Model parameters
parser.add_argument('--K', type=float, default=0.75, help='Inspiral shrink coefficient K (d\' = -K/d^3)')
parser.add_argument('--c-orb', type=float, default=0.30, help='Orbital rate scale coefficient (c_orb)')
parser.add_argument('--k-s', type=float, default=0.28, help='Schwarzschild-like precession coefficient (k_s)')
parser.add_argument('--k-k', type=float, default=0.20, help='Kerr-like spin precession coefficient (k_k)')
parser.add_argument('--spin', type=float, default=0.5, help='Effective dimensionless spin J_eff')
parser.add_argument('--d0', type=float, default=6.0, help='Initial separation')
parser.add_argument('--d-min', type=float, default=2.4, help='Minimum separation clamp to avoid singularity')
parser.add_argument('--m1', type=float, default=1.05, help='Mass 1')
parser.add_argument('--m2', type=float, default=0.95, help='Mass 2')
parser.add_argument('--grid-n', type=int, default=201, help='Grid resolution N (N x N)')
parser.add_argument('--ext-x', type=float, default=8.0, help='Half-extent in x')
parser.add_argument('--ext-y', type=float, default=5.0, help='Half-extent in y')
parser.add_argument('--rays', type=int, default=9, help='Number of launch rays')
parser.add_argument('--ray-span', type=float, default=0.75, help='Fraction of y-extent spanned by rays (0-1)')
parser.add_argument('--step', type=float, default=0.016, help='Ray integration step size')
parser.add_argument('--steps', type=int, default=900, help='Max integration steps per ray')
parser.add_argument('--cmap', default='viridis', help='Colormap for phi_a')
parser.add_argument('--bg', default='black', help='Background face color')
parser.add_argument('--ray-color', default='white', help='Ray line color')
parser.add_argument('--ray-lw', type=float, default=1.1, help='Ray line width')
parser.add_argument('--soft', type=float, default=0.22, help='Softening length in potential')
parser.set_defaults(gif=True, mp4=True)
args = parser.parse_args()

# ---------------- Grid ----------------
N, ex, ey = args.grid_n, args.ext_x, args.ext_y
x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y)

# ---------------- Helpers ----------------
def norm01(Z: np.ndarray) -> np.ndarray:
    Zmin, Zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if Zmax <= Zmin else (Z - Zmin) / (Zmax - Zmin)

def Phi_binary_oriented(M1: float, M2: float, d: float, ang: float, soft: float) -> np.ndarray:
    x1, y1 = +0.5*d*math.cos(ang), +0.5*d*math.sin(ang)
    x2, y2 = -x1, -y1
    r1 = np.sqrt((X-x1)**2 + (Y-y1)**2 + soft**2)
    r2 = np.sqrt((X-x2)**2 + (Y-y2)**2 + soft**2)
    return -M1/r1 - M2/r2

def phi_a_from_Phi(Phi: np.ndarray) -> np.ndarray:
    Phi_x = np.gradient(Phi, dx, axis=1); Phi_y = np.gradient(Phi, dy, axis=0)
    Phi_xx = np.gradient(Phi_x, dx, axis=1); Phi_yy = np.gradient(Phi_y, dy, axis=0)
    Phi_xy = np.gradient(Phi_x, dy, axis=0)
    Lap = Phi_xx + Phi_yy
    T2  = Phi_xx**2 + 2*Phi_xy**2 + Phi_yy**2
    Rn, Tn = norm01(-Lap), norm01(T2)
    r0, aR, aT = 0.45, 0.40, 0.25
    r = r0 + aR*Rn + aT*Tn
    return 0.5*(1.0 + np.sqrt(1.0 + 4.0*r))

def sigma_grad_from_Phi(Phi: np.ndarray):
    Aopt = (1 - 2*Phi) / np.clip(1 + 2*Phi, 1e-6, None)
    Aopt = np.clip(Aopt, 1e-6, 1e6)
    sigma = 0.5*np.log(Aopt)
    sx = np.gradient(sigma, dx, axis=1)
    sy = np.gradient(sigma, dy, axis=0)
    return sx, sy

def bilinear(grid: np.ndarray, xv: float, yv: float) -> float:
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(math.floor(i), 0, N-2)); j0 = int(np.clip(math.floor(j), 0, N-2))
    di, dj = float(i - i0), float(j - j0)
    G00=grid[j0,i0]; G10=grid[j0,i0+1]; G01=grid[j0+1,i0]; G11=grid[j0+1,i0+1]
    return (1-di)*(1-dj)*G00 + di*(1-dj)*G10 + (1-di)*dj*G01 + di*dj*G11

def integrate_rays(sx: np.ndarray, sy: np.ndarray, y_launch: np.ndarray, h: float, steps: int):
    start_x = -ex + 0.2
    rays = []
    for y0 in y_launch:
        xs = np.empty(steps); ys = np.empty(steps)
        xs[0], ys[0] = start_x, y0
        vx, vy = 1.0, 0.0
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
            sxi2 = bilinear(sx, xs[n], ys[n]); syi2 = bilinear(sy, xs[n], ys[n])
            ax2  = - (sxi2*(vxm*vxm - vym*vym) + 2*syi2*vxm*vym)
            ay2  = - (syi2*(vym*vym - vxm*vxm) + 2*sxi2*vxm*vym)
            vx   = vxm + 0.5*h*ax2; vy = vym + 0.5*h*ay2
        rays.append((xs, ys))
    return rays

# ---------------- Evolution ----------------
K = args.K
M1, M2 = args.m1, args.m2
sep = args.d0
theta = 0.0

frames = []
y_launch = np.linspace(-args.ray_span*ey, args.ray_span*ey, args.rays)
for frame_idx in range(args.frames):
    sep = max(args.d_min, sep - K/(sep**3))
    omega_orb  = args.c_orb / (sep**1.5)
    omega_prec = (args.k_s/(sep**2)) + (args.k_k*args.spin/(sep**3))
    theta += (omega_orb + omega_prec)

    Phi   = Phi_binary_oriented(M1, M2, sep, theta, args.soft)
    phi_a = phi_a_from_Phi(Phi)
    sx, sy= sigma_grad_from_Phi(Phi)
    rays  = integrate_rays(sx, sy, y_launch, h=args.step, steps=args.steps)

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    im = ax.imshow(phi_a, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap=args.cmap)
    for xs, ys in rays:
        ax.plot(xs, ys, lw=args.ray_lw, color=args.ray_color)
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title(f"Precessing inspiral — sep={sep:.2f}, θ={(theta%(2*np.pi)):.2f} rad")
    ax.set_facecolor(args.bg)
    fig.patch.set_facecolor(args.bg)
    ax.tick_params(colors='white' if args.bg=='black' else 'black')
    for spine in ax.spines.values():
        spine.set_edgecolor('white' if args.bg=='black' else 'black')

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)
    if args.progress:
        print(f"Frame {frame_idx+1}/{args.frames} sep={sep:.3f} theta={(theta%(2*np.pi)):.3f}")

# ---------------- Write outputs ----------------
if args.gif:
    gif_name = f"{args.output_stem}.gif"
    duration_ms = max(1, int(round(1000.0 / max(1, args.fps))))
    try:
        imageio.mimsave(gif_name, frames, duration=duration_ms)
        print(f"Saved GIF: {gif_name} (~{args.fps} fps)")
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
        print(f"MP4 primary writer failed ({e}); trying matplotlib FFMpegWriter fallback...")
    if not wrote:
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=args.fps)
            fig, ax = plt.subplots(figsize=(args.width, args.height))
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
            print("Install imageio-ffmpeg or ffmpeg binary (e.g., brew install ffmpeg).")

if not (args.gif or args.mp4):
    print("No outputs requested (enable with --gif and/or --mp4).")

print("Done.")
