# Theory and conventions

## Formalism

The solver implements the generalized 4 × 4 transfer-matrix method of Passler & Paarmann, *J. Opt. Soc. Am. B* **34**, 2128 (2017). For each layer it builds the 4 × 4 Δ matrix from the rotated permittivity tensor, finds the four eigenmodes (two forward, two backward), assembles interface and propagation matrices, and obtains the reflection coefficients. The quantity reported is the p-to-p coefficient `rpp`, plotted as `Im(rpp)`, which highlights polariton branches as bright bands.

Numerics (`src/multilayer_atm/solver_fast.py`): the batched path evaluates a full row across kx. An interior layer's transfer matrix is `exp(−iαΔ)` (α = 2πfd/c), so only the semi-infinite boundaries need sorted forward/backward modes. Isotropic layers use the diagonalised closed form with `q = ±√(εμ − ζ²)`; layers with ε_xz = ε_yz = 0 reduce Δ to `[[0, A], [B, 0]]` with 2 × 2 blocks and are propagated in closed form from a 2 × 2 characteristic polynomial; only tilted anisotropic layers need a batched `np.linalg.eig`. Because `rpp` depends only on the two transmitted substrate modes and three rows of the superstrate inverse, a 4 × 2 block is propagated instead of the full 4 × 4 product. For isofrequency sweeps, `rpp(φ + 180°) = rpp(φ)` always and `rpp(−φ) = rpp(φ)` when every layer has β = 0 and α + γ is a multiple of 90°, so only the independent azimuths are computed. Like every transfer-matrix method, the result loses accuracy when the accumulated layer growth `Σ α·|Im q|` exceeds roughly 20 (thick, strongly evanescent layers at large kx); this limit is inherent to the formalism and unchanged.

## Units

| Quantity | Unit |
|---|---|
| Frequency ω | cm⁻¹ (spectroscopic wavenumber) |
| In-plane wavevector kx | cm⁻¹, or normalized kx/k0 in the UI |
| Thickness | m |
| Euler angles | degrees (API/UI), radians inside the engine |

## Euler angles

Layer orientation uses the z–x–z convention (α, β, γ) of the Passler paper. `solver.passler_euler_radians` only relabels the angles into the (θ, φ, ψ) order the engine expects. The isofrequency sweep rotates the in-plane azimuth by offsetting the stack angles.

## Permittivity model

Each layer has a diagonal tensor ε = diag(εxx, εyy, εzz) in its principal frame (magnetic permeability μ = 1), rotated by the Euler matrix into the lab frame. Drude doping adds `−ωp² / (ω² + iγpω)` to every axis.

## Mode trace

The optical mode is the pole of `rpp`. Per frequency the `|rpp|` maximum on the real kx grid seeds a damped Newton iteration on `1/rpp = 0` in the complex plane (`solver_fast.find_rpp_pole`; with the fast solver all frequencies are iterated together by `find_rpp_poles_batch`, and the `|rpp|` grid of a dispersion map computed just before on the same stack and grid is reused as the seed). Re(kx) is the mode wavevector; Im(kx) its loss. Non-converged frequencies are flagged.

## Vertical-dipole surface-wave map

For a z-oriented dipole at height h above the stack, the reflected Weyl spectrum is proportional to `rpp(q, φ) K∥² / Kz · exp(iKz(h+z))`. Only evanescent components (q > √ε_superstrate · ω) inside the sampled band are kept; a polar-to-Cartesian interpolation and a 2D inverse FFT give the real-space field. The result is a band-limited **proxy** for the surface-wave field including non-resonant evanescent background. It is not total E, not energy flux, and excludes the direct source field. Convergence must be checked against momentum band, angular sampling and FFT size.
