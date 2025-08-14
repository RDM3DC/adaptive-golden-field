# Adaptive Golden Field (ϕ_a) — Lensing, Inspiral, Ringdown

Make the golden ratio **dynamic**:  
\( F_n = F_{n-1} + r(x,t)F_{n-2} \) ⇒ \( \phi_a(x,t) = \frac{1+\sqrt{1+4r(x,t)}}{2} \).  
Couple \(r\) to curvature proxies of a gravity-like potential \(\Phi\) and trace rays in the optical metric to reveal **caustics** that surge, precess, and ring.

**Quickstart**
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python cinematic_adaptive_golden.py                   # outputs: AdaptiveGolden_Cinematic.gif
```

**Scripts**

* `cinematic_adaptive_golden.py` — full life-cycle (inspiral → precession → multi-mode ringdown)
* `inspiral_gwstyle_phi_a.py`, `inspiral_precessing_phi_a.py` — core phases
* `ringdown_multimode.py`, `ringdown_lensing.py` — ringdown + correlation
* `stochastic_hybrid_phi_a.py` — hybrid “protein–GW” noise analog

> Focus metric: `1 / std(y_cross at x=0)` from a fan of rays.
> Heuristic: **ϕ_a < 1.4** often flags bifurcations (caustic snaps).
