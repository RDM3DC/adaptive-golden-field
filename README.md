# φ_a Visualization Scripts

Two small, dependency-light scripts to generate visual artifacts for adaptive field / inspiral style demonstrations. Designed to run locally without any external downloads beyond PyPI basics.

## Quick Install

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install numpy matplotlib imageio pillow imageio-ffmpeg
```

(If you do not need MP4 output you can omit `imageio-ffmpeg`.)

## Scripts

### 1. `inspiral_gwstyle_phi_a.py`
Generates an inspiral animation of ray trajectories over an adaptive `phi_a` scalar field derived from a softened binary potential and curvature-like invariants.

Outputs (by default):
- `inspiral_gwstyle_phi_a.gif`
- `inspiral_gwstyle_phi_a.mp4` (if the ffmpeg backend is available)

Run with defaults:
```bash
python inspiral_gwstyle_phi_a.py
```

Progress logging & only GIF:
```bash
python inspiral_gwstyle_phi_a.py --progress --gif --no-mp4
```

Only MP4 with custom fps & frames:
```bash
python inspiral_gwstyle_phi_a.py --no-gif --mp4 --fps 10 --frames 40
```

Change output stem:
```bash
python inspiral_gwstyle_phi_a.py -o my_inspiral
```

Environment variable alternative for frame count:
```bash
FRAMES=30 python inspiral_gwstyle_phi_a.py
```

Key adjustable parameters (CLI flags):
- `--fps` Frame rate.
- `--frames` Number of frames (or use `FRAMES` env var).
- `--width / --height` Figure size in inches.
- `--gif / --no-gif`, `--mp4 / --no-mp4` format toggles.
- `-o / --output-stem` Base name for files.
- `--progress` Per-frame logging.

### 2. `spin_comparison_panel.py`
Creates a side-by-side PNG showing base `sigma` field (no spin) vs a spin-augmented field using a simple additive pseudo-spin term ~ J/r^3.

Run:
```bash
python spin_comparison_panel.py
```

Custom spins / output file:
```bash
python spin_comparison_panel.py --j1 0.4 --j2 -0.25 --outfile panel.png
```

Adjust grid size & extent:
```bash
python spin_comparison_panel.py --n 300 --ext 8.0
```

Colormap & DPI:
```bash
python spin_comparison_panel.py --cmap plasma --dpi 300
```

### Notes on the Physics Simplifications
- Potentials are Newtonian-softened; no GR solver invoked.
- `phi_a` heuristic depends on Laplacian & Hessian norm (trace / invariants) normalized to [0,1].
- Ray integration uses a custom geodesic-like update with gradients of an optical conformal factor derived from the potential.
- Spin field is a toy model (dipole-like falloff) added linearly in log-conformal factor space for visual differentiation.

### Troubleshooting
- If MP4 writing fails: ensure `ffmpeg` is on PATH or install `imageio-ffmpeg` (`pip install imageio-ffmpeg`).
- If you see Import errors in an IDE: verify the virtual environment is active.
- Large grids or many frames can increase memory usage; reduce `--frames` or grid size (`N` constant in script) if needed.

### Possible Extensions
- Add color mapping to ray curvature or affine parameter.
- Adaptive step size integration for rays.
- More realistic inspiral dynamics (e.g., PN evolution of separation & spins).
- Export frame set to a numbered directory for further post-processing.

---
Enjoy exploring the fields! Feel free to adapt / extend.
