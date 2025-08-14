#!/usr/bin/env python3
"""Generate an inspiral visualization over adaptive phi_a field.

Outputs (by default):
  - inspiral_gwstyle_phi_a.gif
  - inspiral_gwstyle_phi_a.mp4 (if imageio-ffmpeg is available or matplotlib FFMpegWriter works)

You can disable either via CLI flags.

Dependencies:
  pip install numpy matplotlib imageio pillow imageio-ffmpeg

Optional:
  Set environment variable FRAMES to change number of frames (default 22)

CLI:
  python inspiral_gwstyle_phi_a.py --no-gif --mp4  # only mp4
  python inspiral_gwstyle_phi_a.py --gif --no-mp4  # only gif

Notes:
  - MP4 writing prefers imageio-ffmpeg backend. Falls back to matplotlib animation writer if necessary.
  - The physical model here is illustrative; not a rigorous GR simulation.
"""
from __future__ import annotations
import os, sys, math, argparse
import numpy as np, imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------------- Argument parsing -----------------
parser = argparse.ArgumentParser(description="Inspiral GIF/MP4 generator (GW-style rays over adaptive phi_a)")
parser.add_argument('--gif', dest='gif', action='store_true', help='Write GIF output (default True unless --no-gif)')
parser.add_argument('--no-gif', dest='gif', action='store_false', help='Disable GIF output')
parser.add_argument('--mp4', dest='mp4', action='store_true', help='Write MP4 output (default True unless --no-mp4)')
parser.add_argument('--no-mp4', dest='mp4', action='store_false', help='Disable MP4 output')
parser.add_argument('-o', '--output-stem', default='inspiral_gwstyle_phi_a', help='Output filename stem (default: %(default)s)')
parser.add_argument('--fps', type=int, default=6, help='Frames per second for animation (default: 6)')
parser.add_argument('--width', type=float, default=8.0, help='Figure width in inches (default: 8.0)')
parser.add_argument('--height', type=float, default=4.0, help='Figure height in inches (default: 4.0)')
parser.add_argument('--frames', type=int, default=None, help='Override number of frames (default 22 or env FRAMES)')
parser.add_argument('--progress', action='store_true', help='Print per-frame progress messages')
parser.set_defaults(gif=True, mp4=True)
args = parser.parse_args()

# Allow environment variable to override frames if CLI not given
if args.frames is None:
    env_frames = os.getenv('FRAMES')
    if env_frames and env_frames.isdigit():
        args.frames = int(env_frames)
if args.frames is None:
    args.frames = 22

# ---------------- Grid setup -----------------
N, ex, ey = 201, 8.0, 5.0
x = np.linspace(-ex, ex, N); y = np.linspace(-ey, ey, N)
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y)

# ---------------- Utility helpers -----------------
def norm01(Z: np.ndarray) -> np.ndarray:
    Zmin, Zmax = np.min(Z), np.max(Z)
    return np.zeros_like(Z) if Zmax <= Zmin else (Z - Zmin) / (Zmax - Zmin)

# ---------------- Physics helpers -----------------
def Phi_binary(M1: float, M2: float, d: float, soft: float = 0.22) -> np.ndarray:
    return (-M1 / np.sqrt((X + d/2)**2 + Y**2 + soft**2)
            -M2 / np.sqrt((X - d/2)**2 + Y**2 + soft**2))

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

# ---------------- Interpolation & ray tracing -----------------
def bilinear(grid: np.ndarray, xv: float, yv: float) -> float:
    i = np.interp(xv, x, np.arange(N)); j = np.interp(yv, y, np.arange(N))
    i0 = int(np.clip(math.floor(i), 0, N-2)); j0 = int(np.clip(math.floor(j), 0, N-2))
    di, dj = float(i - i0), float(j - j0)
    G00=grid[j0,i0]; G10=grid[j0,i0+1]; G01=grid[j0+1,i0]; G11=grid[j0+1,i0+1]
    return (1-di)*(1-dj)*G00 + di*(1-dj)*G10 + (1-di)*dj*G01 + di*dj*G11

def integrate_rays(sx: np.ndarray, sy: np.ndarray, y_launch: np.ndarray, h: float = 0.016, steps: int = 900):
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

# ---------------- Animation loop -----------------
K, M1, M2, d = 0.75, 1.05, 0.95, 6.0
frames = []
for frame_idx in range(args.frames):
    d = max(2.4, d - K/(d**3))
    Phi   = Phi_binary(M1, M2, d)
    phi_a = phi_a_from_Phi(Phi)
    sx, sy= sigma_grad_from_Phi(Phi)
    rays  = integrate_rays(sx, sy, np.linspace(-0.75*ey, 0.75*ey, 9))
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    ax.imshow(phi_a, origin='lower', extent=[x.min(),x.max(),y.min(),y.max()], alpha=0.6, cmap='viridis')
    for xs, ys in rays:
        ax.plot(xs, ys, lw=1.1, color='white')
    ax.set_aspect('equal'); ax.set_xlim(x.min(),x.max()); ax.set_ylim(y.min(),y.max())
    ax.set_title(f"Inspiral (GW-style) — sep={d:.2f}")
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    fig.patch.set_facecolor('black')
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    frames.append(frame)
    if args.progress:
        print(f"Frame {frame_idx+1}/{args.frames} (sep={d:.3f})")

# ---------------- Write outputs -----------------
if args.gif:
    gif_name = f"{args.output_stem}.gif"
    # imageio's pillow writer deprecated fps; use duration (ms per frame).
    duration_ms = max(1, int(round(1000.0 / max(1, args.fps))))
    try:
        imageio.mimsave(gif_name, frames, duration=duration_ms)
        print(f"Saved GIF: {gif_name} (duration per frame {duration_ms} ms ~ {args.fps} fps)")
    except TypeError:
        # Fallback in case older imageio still wants fps
        imageio.mimsave(gif_name, frames, fps=args.fps)
        print(f"Saved GIF (legacy fps path): {gif_name}")

if args.mp4:
    mp4_name = f"{args.output_stem}.mp4"
    wrote = False
    # Try imageio ffmpeg writer
    try:
        with imageio.get_writer(mp4_name, fps=args.fps, codec='libx264', quality=8) as w:
            for f in frames:
                w.append_data(f)
        wrote = True
        print(f"Saved MP4 (imageio-ffmpeg): {mp4_name}")
    except Exception as e:
        print(f"(imageio-ffmpeg path failed: {e}) Trying matplotlib FFMpegWriter...")
    if not wrote:
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=args.fps)
            # Re-render using writer (since we already have frames, we'll just dump them through a simple figure)
            fig, ax = plt.subplots(figsize=(args.width, args.height))
            im = ax.imshow(frames[0], origin='lower')
            ax.axis('off')
            with writer.saving(fig, mp4_name, dpi=100):
                for f in frames:
                    im.set_data(f)
                    writer.grab_frame()
            plt.close(fig)
            wrote = True
            print(f"Saved MP4 (matplotlib FFMpegWriter): {mp4_name}")
        except Exception as e2:
            print(f"ERROR: Unable to write MP4: {e2}")
            print("Install imageio-ffmpeg or a system ffmpeg binary.")

if not (args.gif or args.mp4):
    print("No outputs requested. Use --gif and/or --mp4.")

print("Done.")
