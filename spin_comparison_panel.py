#!/usr/bin/env python3
"""Generate a simple 2-panel PNG comparing no-spin vs spin-modified sigma field.

Output:
  spin_panel.png

Dependencies:
  pip install numpy matplotlib

CLI:
  python spin_comparison_panel.py --j1 0.3 --j2 -0.3 --outfile spin_panel.png
"""
from __future__ import annotations
import argparse, numpy as np, matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Spin vs No-Spin sigma panel')
parser.add_argument('--n', type=int, default=200, help='Grid size (default 200)')
parser.add_argument('--ext', type=float, default=6.0, help='Domain half-extent (default 6.0)')
parser.add_argument('--mass1', type=float, default=1.0, help='Mass 1 (default 1)')
parser.add_argument('--mass2', type=float, default=1.0, help='Mass 2 (default 1)')
parser.add_argument('--d', type=float, default=3.0, help='Separation (default 3.0)')
parser.add_argument('--j1', type=float, default=0.3, help='Spin-like parameter J1 (default 0.3)')
parser.add_argument('--j2', type=float, default=-0.3, help='Spin-like parameter J2 (default -0.3)')
parser.add_argument('--spin-scale', type=float, default=0.2, help='Scale factor applied to spin field (default 0.2)')
parser.add_argument('--outfile', default='spin_panel.png', help='Output PNG filename (default spin_panel.png)')
parser.add_argument('--cmap', default='magma', help='Matplotlib colormap (default magma)')
parser.add_argument('--dpi', type=int, default=200, help='Figure DPI (default 200)')
args = parser.parse_args()

N, ext = args.n, args.ext
x = np.linspace(-ext, ext, N); y = np.linspace(-ext, ext, N)
X, Y = np.meshgrid(x, y)

M1, M2, d = args.mass1, args.mass2, args.d
r1 = np.sqrt((X + d/2)**2 + Y**2)
r2 = np.sqrt((X - d/2)**2 + Y**2)
Phi = -M1/r1 - M2/r2

Aopt = (1 - 2*Phi)/np.clip(1 + 2*Phi, 1e-6, None)
Aopt = np.clip(Aopt, 1e-6, 1e+6)
sigma0 = 0.5*np.log(Aopt)

def spin_field(J1: float, J2: float) -> np.ndarray:
    r1_mag = np.sqrt((X + d/2)**2 + Y**2)
    r2_mag = np.sqrt((X - d/2)**2 + Y**2)
    return J1/np.clip(r1_mag**3,1e-6,None) + J2/np.clip(r2_mag**3,1e-6,None)

sigma_spin = sigma0 + args.spin_scale*spin_field(args.j1, args.j2)

fig, axs = plt.subplots(1,2, figsize=(10,5))
for ax, data, title in zip(axs, [sigma0, sigma_spin], ["No Spin", "With Spin"]):
    im = ax.imshow(data, origin='lower', extent=[-ext,ext,-ext,ext], cmap=args.cmap)
    ax.set_title(title)
    ax.set_xlim(-ext,ext); ax.set_ylim(-ext,ext)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout(); plt.savefig(args.outfile, dpi=args.dpi); plt.close()
print(f"Saved: {args.outfile}")
