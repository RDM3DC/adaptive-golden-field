# Adaptive Golden Field (ϕ_a) — Lensing, Inspiral, Ringdown

**What:** Field-aware Fibonacci where the golden ratio becomes a dynamic field ϕ_a(x,t),
coupled to gravity-like potentials to generate adaptive spirals, caustics, and CAD-ready geometry.

**Highlights**
- Inspiral → precession → multi-mode ringdown (cinematic clip)
- Optical geodesics over ϕ_a; focus metric (1/std of ray crossings)
- Stochastic “protein–GW” hybrid (OU noise) + correlation tests
- Plain-text equations for X + equation-card PNGs

**Quickstart**
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy matplotlib imageio pillow
python cinematic_adaptive_golden.py
```

**Scripts**

* `inspiral_gwstyle_phi_a.py` — GW-like inspiral
* `inspiral_precessing_phi_a.py` — periapsis precession (Kerr-like)
* `ringdown_multimode.py` — multi-mode beats + correlation
* `ringdown_lensing.py` — single-mode lensing + focus correlation
* `stochastic_hybrid_phi_a.py` — noisy hybrid (protein–GW analog)
* `cinematic_adaptive_golden.py` — full life-cycle animation
* `spin_comparison_panel.py` — base vs spin-augmented field comparison

**License**: MIT
