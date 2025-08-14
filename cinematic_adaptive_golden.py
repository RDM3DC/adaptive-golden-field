#!/usr/bin/env python3
"""Adaptive Golden Field — Cinematic Life-Cycle

Phases (single animation):
  Phase A — Inspiral (GW-style shrink + mild periapsis advance)
  Phase B — Precession spotlight (faster Kerr-like twist)
  Phase C — Ringdown (multi-mode damped QNM beats with rotating petals)

Underlying field: phi_a = (1 + sqrt(1 + 4 r)) / 2 with r mixing normalized (-Laplace Phi, ||Hess Phi||^2).
Rays are traced as optical geodesic-like paths in gradient-derived 'sigma' conformal factor.

Default output: AdaptiveGolden_Cinematic.gif
Optional MP4:   AdaptiveGolden_Cinematic.mp4 (if ffmpeg available)  --mp4 / --no-mp4

Quick start:
  pip install numpy matplotlib imageio pillow imageio-ffmpeg
  python cinematic_adaptive_golden.py

Adjust resolution / story length:
  python cinematic_adaptive_golden.py --N 241 --steps 900 --phase-a-frames 24 --phase-b-frames 16 --phase-c-frames 28

Tune appearance:
  python cinematic_adaptive_golden.py --cmap plasma --ray-color white --bg black

"""
from __future__ import annotations
import argparse, math
import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio

# ---------------- CLI -----------------
parser = argparse.ArgumentParser(description="Adaptive Golden Field cinematic lifecycle animation")
parser.add_argument('--N', type=int, default=181, help='Grid resolution (N x N). Use 241 for higher quality.')
parser.add_argument('--extent-x', type=float, default=7.5, help='Half-extent in x')
parser.add_argument('--extent-y', type=float, default=5.0, help='Half-extent in y')
parser.add_argument('--steps', type=int, default=700, help='Max integration steps per ray (e.g., 900 for high quality)')
parser.add_argument('--ray-step', type=float, default=0.016, help='Ray integrator step size h')
parser.add_argument('--rays', type=int, default=9, help='Number of launch rays')
parser.add_argument('--ray-span', type=float, default=0.75, help='Fraction of vertical extent used for ray launch band (0-1)')
parser.add_argument('--phase-a-frames', type=int, default=16, help='Frames in Phase A (Inspiral)')
parser.add_argument('--phase-b-frames', type=int, default=10, help='Frames in Phase B (Precession spotlight)')
parser.add_argument('--phase-c-frames', type=int, default=18, help='Frames in Phase C (Ringdown multi-mode)')
parser.add_argument('--fps', type=int, default=8, help='FPS for animation outputs')
parser.add_argument('--gif', dest='gif', action='store_true', help='Enable GIF output (default on)')
parser.add_argument('--no-gif', dest='gif', action='store_false', help='Disable GIF output')
parser.add_argument('--mp4', dest='mp4', action='store_true', help='Enable MP4 output (default off)')
parser.add_argument('--no-mp4', dest='mp4', action='store_false', help='Disable MP4 output')
parser.add_argument('--output-stem', default='AdaptiveGolden_Cinematic', help='Output name stem')
parser.add_argument('--cmap', default='viridis', help='Colormap for phi_a')
parser.add_argument('--bg', default='white', help='Background color for figure area')
parser.add_argument('--ray-color', default=None, help='Ray color (default auto white if dark bg else black)')
parser.add_argument('--ray-lw', type=float, default=1.05, help='Ray line width')
parser.add_argument('--progress', action='store_true', help='Print per-frame progress information')
# Inspiral / precession parameters
parser.add_argument('--M1', type=float, default=1.05, help='Mass 1')
parser.add_argument('--M2', type=float, default=0.95, help='Mass 2')
parser.add_argument('--d0', type=float, default=6.0, help='Initial separation for inspiral')
parser.add_argument('--d-min', type=float, default=2.6, help='Separation clamp to avoid singularity')
parser.add_argument('--K', type=float, default=0.75, help='Inspiral shrink coefficient (d'"'"' = -K/d^3)')
parser.add_argument('--c-orb', type=float, default=0.28, help='Orbital rate scale ~ d^(-3/2)')
parser.add_argument('--k-s', type=float, default=0.22, help='Schwarzschild-like precession coefficient ~ d^(-2)')
parser.add_argument('--k-k', type=float, default=0.18, help='Kerr-like spin precession coefficient ~ J/d^3')
parser.add_argument('--spin', type=float, default=0.5, help='Effective dimensionless spin J_eff')
# Precession spotlight rotation increment override
parser.add_argument('--phase-b-dtheta', type=float, default=0.18, help='Rotation added per frame in Phase B')
# Ringdown multi-mode specification (hard-coded list now, could expose in future)
parser.add_argument('--mode-scale', type=float, default=1.0, help='Global scaling multiplier applied to all mode epsilons')
parser.add_argument('--gif-duration-cap', type=int, default=None, help='Optional cap on total frames used in GIF (debug)')
parser.set_defaults(gif=True, mp4=False)
args = parser.parse_args()

# ---------------- Grid -----------------
N = args.N
ex = args.extent_x
ey = args.extent_y
x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y)
R = np.sqrt(X*X + Y*Y)
TH = np.arctan2(Y, X)

# ---------------- Helpers -----------------
def norm01(Z: np.ndarray) -> np.ndarray:
    zmin, zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if zmax <= zmin else (Z - zmin)/(zmax - zmin)

def bilinear(grid: np.ndarray, xv: float, yv: float) -> float:
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(math.floor(i), 0, N-2)); j0 = int(np.clip(math.floor(j), 0, N-2))
    di, dj = float(i-i0), float(j-j0)
    G00=grid[j0,i0]; G10=grid[j0,i0+1]; G01=grid[j0+1,i0]; G11=grid[j0+1,i0+1]
    return (1-di)*(1-dj)*G00 + di*(1-dj)*G10 + (1-di)*dj*G01 + di*dj*G11

# Fields

def Phi_binary_oriented(M1: float, M2: float, d: float, ang: float, soft: float = 0.24) -> np.ndarray:
    x1, y1 = +0.5*d*math.cos(ang), +0.5*d*math.sin(ang)
    x2, y2 = -x1, -y1
    r1 = np.sqrt((X-x1)**2 + (Y-y1)**2 + soft**2)
    r2 = np.sqrt((X-x2)**2 + (Y-y2)**2 + soft**2)
    return -M1/r1 - M2/r2

def Phi0(R: np.ndarray, M: float = 1.0, soft: float = 0.24) -> np.ndarray:
    return -M/np.sqrt(R*R + soft*soft)

def Phi_ringdown_multi(t: int, modes, M: float = 1.0) -> np.ndarray:
    base = Phi0(R, M=M)
    dsum = 0.0
    for (m, omg, tau, eps, rpk, sig, ph, drift) in modes:
        W = np.exp(- (R - rpk)**2 / (2*sig*sig))
        dsum += eps * args.mode_scale * W * base * np.exp(-t/tau) * np.cos(omg*t + m*TH + ph + drift*t)
    return base + dsum

def sigma_grad_from_Phi(Phi: np.ndarray):
    Aopt = (1 - 2*Phi) / np.clip(1 + 2*Phi, 1e-6, None)
    Aopt = np.clip(Aopt, 1e-6, 1e6)
    sigma = 0.5*np.log(Aopt)
    sx = np.gradient(sigma, dx, axis=1)
    sy = np.gradient(sigma, dy, axis=0)
    return sx, sy, sigma

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

# Rays

def trace_rays(sx: np.ndarray, sy: np.ndarray, y_launch: np.ndarray, h: float, steps: int):
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
            xs[n] = xc + h*vxm; ys[n] = yc + h*vym
            sxi2 = bilinear(sx, xs[n], ys[n]); syi2 = bilinear(sy, xs[n], ys[n])
            ax2  = - (sxi2*(vxm*vxm - vym*vym) + 2*syi2*vxm*vym)
            ay2  = - (syi2*(vym*vym - vxm*vxm) + 2*sxi2*vxm*vym)
            vx   = vxm + 0.5*h*ax2; vy = vym + 0.5*h*ay2
        rays.append((xs, ys))
    return rays

# ---------------- Build Frames -----------------
frames = []
y_launch = np.linspace(-args.ray_span*ey, args.ray_span*ey, args.rays)

# Phase A — Inspiral
M1, M2 = args.M1, args.M2
d = args.d0
theta = 0.0
for i in range(args.phase_a_frames):
    d = max(args.d_min, d - args.K/(d**3))
    omega_orb  = args.c_orb / (d**1.5)
    omega_prec = (args.k_s/(d**2)) + (args.k_k*args.spin/(d**3))
    theta += (omega_orb + omega_prec)
    Phi = Phi_binary_oriented(M1, M2, d, theta)
    phi = phi_a_from_Phi(Phi)
    sx, sy, _ = sigma_grad_from_Phi(Phi)
    rays = trace_rays(sx, sy, y_launch, h=args.ray_step, steps=args.steps)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.imshow(phi, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap=args.cmap)
    rc = args.ray_color or ('white' if args.bg.lower() in ('black','k') else 'black')
    for xs, ys in rays: ax.plot(xs, ys, lw=args.ray_lw, color=rc)
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title(f"Phase A — Inspiral  •  sep={d:.2f}")
    fig.patch.set_facecolor(args.bg); ax.set_facecolor(args.bg)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)
    if args.progress:
        print(f"Phase A frame {i+1}/{args.phase_a_frames} d={d:.3f}")

# Phase B — Precession spotlight (fix d, accelerate rotation)
for i in range(args.phase_b_frames):
    theta += args.phase_b_dtheta
    Phi = Phi_binary_oriented(M1, M2, d, theta)
    phi = phi_a_from_Phi(Phi)
    sx, sy, _ = sigma_grad_from_Phi(Phi)
    rays = trace_rays(sx, sy, y_launch, h=args.ray_step, steps=args.steps)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.imshow(phi, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap=args.cmap)
    rc = args.ray_color or ('white' if args.bg.lower() in ('black','k') else 'black')
    for xs, ys in rays: ax.plot(xs, ys, lw=args.ray_lw, color=rc)
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title("Phase B — Precession (Kerr-like twist)")
    fig.patch.set_facecolor(args.bg); ax.set_facecolor(args.bg)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)
    if args.progress:
        print(f"Phase B frame {i+1}/{args.phase_b_frames} theta={theta:.3f}")

# Phase C — Ringdown (multi-mode beats)
# (m, omega, tau, eps, r_peak, sigma_r, phase, driftΩ)
ringdown_modes = [
    (2, 1.22, 11.0, 0.12, 3.0, 0.9, 0.0, 0.02),
    (3, 1.38,  9.0, 0.10, 3.2, 1.0, 0.7, 0.00),
    (2, 1.46,  7.0, 0.08, 2.8, 0.8, 1.1, 0.03),
]
for t in range(args.phase_c_frames):
    Phi = Phi_ringdown_multi(t, ringdown_modes, M=1.0)
    phi = phi_a_from_Phi(Phi)
    sx, sy, _ = sigma_grad_from_Phi(Phi)
    rays = trace_rays(sx, sy, y_launch, h=args.ray_step, steps=args.steps)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.imshow(phi, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap=args.cmap)
    rc = args.ray_color or ('white' if args.bg.lower() in ('black','k') else 'black')
    for xs, ys in rays: ax.plot(xs, ys, lw=max(0.9, args.ray_lw-0.1), color=rc)
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title("Phase C — Ringdown (multi-mode beats)")
    fig.patch.set_facecolor(args.bg); ax.set_facecolor(args.bg)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)
    if args.progress:
        print(f"Phase C frame {t+1}/{args.phase_c_frames}")

# Optionally cap total frames (debug/testing)
if args.gif_duration_cap is not None and args.gif_duration_cap < len(frames):
    frames = frames[:args.gif_duration_cap]

# ---------------- Write outputs -----------------
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
            fig, ax = plt.subplots(figsize=(9,5))
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

print("Done.")
