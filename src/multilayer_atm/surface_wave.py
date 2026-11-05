"""Band-limited evanescent field of a vertical dipole above a planar stack."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def compute_surface_wave(
    phi_rad: np.ndarray,
    k_cm1: np.ndarray,
    rpp: np.ndarray,
    *,
    w0_cm1: float,
    source_height_nm: float = 25.0,
    observation_height_nm: float = 0.0,
    epsilon_superstrate: complex = 1.0,
    fft_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x=y in um, normalized complex Ez[y,x], normalized |Ez|²).

    For a z dipole at (0,0,h), the reflected Weyl spectrum is proportional to
    rpp(q,phi) K_parallel² / Kz * exp(i Kz (h+z)). Only evanescent components
    (q > sqrt(epsilon_superstrate)*w0) in the sampled momentum band are included.
    Thus this is a surface-wave field proxy, including nonresonant evanescent
    background; it is neither total E nor energy flux, and it excludes the direct
    source field. No mode extraction or pole search is performed.

    The solver's cm⁻¹ momenta are spectroscopic wavenumbers (K/2pi). Its gamma
    sweep samples the lab propagation azimuth +phi. A periodic polar interpolant
    supplies the Cartesian spectrum, and ifft2 uses exp(+i K_parallel·rho).
    Zero padding to four times q_max gives eight pixels per shortest wavelength.
    The Fourier domain has period fft_size/(8*q_max) in um. Enlarge fft_size and
    the polar sampling to check convergence; a finite band can cause ringing.
    """
    phi = np.asarray(phi_rad, dtype=float)
    k = np.asarray(k_cm1, dtype=float)
    rp = np.asarray(rpp)
    if phi.ndim != 1 or k.ndim != 1 or len(phi) < 4 or len(k) < 2:
        raise ValueError("The wave map needs at least four angles and two momentum samples.")
    if rp.shape != (len(phi), len(k)) or not np.iscomplexobj(rp):
        raise ValueError("The wave map needs complex rpp with shape (Nphi, Nk); recompute isofrequency.")
    if not all(np.all(np.isfinite(v)) for v in (phi, k, rp)):
        raise ValueError("The isofrequency samples must be finite.")
    dphi = np.diff(phi)
    if (np.any(dphi <= 0) or not np.allclose(dphi, dphi[0])
            or not np.isclose(phi[-1] - phi[0] + dphi[0], 2 * np.pi)):
        raise ValueError("The wave map requires a full 360-degree isofrequency sweep (no repeated endpoint).")
    if np.any(np.diff(k) <= 0) or k[0] < 0 or k[-1] <= 0:
        raise ValueError("Momentum samples must be increasing and nonnegative.")
    if not np.isfinite(w0_cm1) or w0_cm1 <= 0:
        raise ValueError("Frequency must be finite and positive.")
    heights = np.array([source_height_nm, observation_height_nm], dtype=float)
    if not np.all(np.isfinite(heights)) or np.any(heights < 0):
        raise ValueError("Source and observation heights must be finite and nonnegative.")
    eps = complex(epsilon_superstrate)
    if not np.isfinite(eps) or eps.real <= 0 or abs(eps.imag) > 1e-8 * max(1.0, abs(eps.real)):
        raise ValueError("The dipole wave map requires a lossless dielectric superstrate (e.g. air).")
    if int(fft_size) != fft_size or not 128 <= fft_size <= 2048:
        raise ValueError("FFT size must be an integer between 128 and 2048.")

    q_max = k[-1] * 1e-4  # cycles / um
    dx = 1.0 / (8.0 * q_max)
    n = int(fft_size)
    q = np.fft.fftfreq(n, d=dx)
    qx, qy = np.meshgrid(q, q)
    qr = np.hypot(qx, qy)
    angle = phi[0] + np.mod(np.arctan2(qy, qx) - phi[0], 2 * np.pi)
    interp = RegularGridInterpolator(
        (np.append(phi, phi[0] + 2 * np.pi), k * 1e-4),
        np.concatenate((rp, rp[:1]), axis=0), bounds_error=False, fill_value=0.0,
    )
    # ponytail: linear interpolation of sampled rpp; refine the polar grid for narrow resonances.
    spectrum = interp((angle, qr))
    evanescent = (qr**2 > eps.real * (w0_cm1 * 1e-4)**2) & (qr >= k[0] * 1e-4) & (qr <= q_max)
    kappa = 2 * np.pi * np.sqrt(np.maximum(qr**2 - eps.real * (w0_cm1 * 1e-4)**2, 0.0))
    weight = np.zeros_like(qr)
    weight[evanescent] = (2 * np.pi * qr[evanescent])**2 / kappa[evanescent]
    spectrum *= -1j * weight * np.exp(-kappa * heights.sum() * 1e-3)
    field = np.fft.fftshift(np.fft.ifft2(spectrum))
    peak = float(np.max(np.abs(field)))
    if peak > 0:
        field /= peak
    xy_um = (np.arange(n) - n // 2) * dx
    return xy_um, field, np.abs(field)**2


def compute_pole_wave(
    phi_rad: np.ndarray,
    k_pole_cm1: np.ndarray,
    dinv_dk: np.ndarray,
    ok: np.ndarray,
    *,
    w0_cm1: float,
    source_height_nm: float = 25.0,
    observation_height_nm: float = 0.0,
    epsilon_superstrate: complex = 1.0,
    half_width_um: float = 5.0,
    n_pix: int = 301,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mode-only field of the same vertical dipole: the pole (residue) part of the Weyl integral.

    Near its pole rpp ~ R/(q - q_p), R = 1/(d(1/rpp)/dq). For a ray at azimuth phi the radial
    integral, closed in the first quadrant, gives 2 pi i q_p S0(q_p) R exp(i 2 pi q_p rho
    cos(phi - theta)) for cos(phi - theta) > 0 and nothing otherwise (the cut is smoothed over cos < 0.2, away from the stationary azimuth), with the same Weyl
    weight S0 = -i (2 pi q)^2 / kappa exp(-kappa (h_s + h_o)) as :func:`compute_surface_wave`
    (kappa is now complex). The azimuth integral is a plain sum, so each mode branch keeps its
    own wavelength, decay and anisotropy. The non-resonant background and the branch cut are
    dropped, and the result is normalised to the peak outside the dipole core
    (rho >= max(0.1 um, 5% of the half-width)). Azimuths without a pole (``ok`` False) contribute 0.
    The azimuth sampling is refined by linear interpolation so rings stay unaliased out to the view edge.
    """
    phi = np.asarray(phi_rad, dtype=float)
    kp = np.asarray(k_pole_cm1, dtype=complex)
    dk = np.asarray(dinv_dk, dtype=complex)
    ok = np.asarray(ok, dtype=bool)
    if phi.ndim != 1 or not (phi.shape == kp.shape == dk.shape == ok.shape) or len(phi) < 4:
        raise ValueError("Pole arrays must be 1-D with one entry per azimuth (at least four).")
    if not ok.any():
        raise ValueError("No converged pole found; adjust the frequency or the momentum range.")
    eps = complex(epsilon_superstrate)
    if eps.real <= 0 or abs(eps.imag) > 1e-8 * max(1.0, abs(eps.real)):
        raise ValueError("The dipole wave map requires a lossless dielectric superstrate (e.g. air).")
    hs = float(source_height_nm) + float(observation_height_nm)
    if hs < 0 or half_width_um <= 0 or int(n_pix) < 11:
        raise ValueError("Heights must be nonnegative, the half-width positive and n_pix at least 11.")

    q = kp * 1e-4  # cycles / um
    kappa = 2 * np.pi * np.sqrt(q**2 - eps.real * (w0_cm1 * 1e-4) ** 2 + 0j)
    kappa = np.where(kappa.real < 0, -kappa, kappa)
    amp = 2j * np.pi * q * (-1j) * (2 * np.pi * q) ** 2 / kappa * np.exp(-kappa * hs * 1e-3) / (dk * 1e4)
    amp = np.where(ok, amp, 0.0)

    # refine the azimuth grid: enough rays that neighbouring ones stay within ~1/4 period at the view edge
    reach = half_width_um * np.sqrt(2.0)
    n_f = int(np.clip(np.ceil(8 * np.pi * reach * np.max(q.real[ok])), len(phi), 8192))
    phi_f = np.linspace(phi[0], phi[0] + 2 * np.pi, n_f, endpoint=False)
    period = 2 * np.pi
    q_f = np.interp(phi_f, phi[ok], q.real[ok], period=period) + 1j * np.interp(phi_f, phi[ok], q.imag[ok], period=period)
    a_f = np.interp(phi_f, phi[ok], amp.real[ok], period=period) + 1j * np.interp(phi_f, phi[ok], amp.imag[ok], period=period)
    near = np.argmin(np.abs(np.angle(np.exp(1j * (phi_f[:, None] - phi[None, :])))), axis=1)
    a_f = np.where(ok[near], a_f, 0.0)

    xy = np.linspace(-half_width_um, half_width_um, int(n_pix))
    X, Y = np.meshgrid(xy, xy)
    rho, th = np.hypot(X, Y), np.arctan2(Y, X)
    field = np.zeros(X.shape, dtype=complex)
    for lo in range(0, n_f, 16):
        sl = slice(lo, lo + 16)
        cosd = np.cos(phi_f[sl, None, None] - th[None])
        term = a_f[sl, None, None] * np.exp(2j * np.pi * q_f[sl, None, None] * rho[None] * cosd)
        field += (np.clip(cosd / 0.2, 0.0, 1.0) * term).sum(axis=0)  # smooth cos>0 cut: no ray-edge ripple
    field *= 2 * np.pi / n_f
    core = rho >= max(0.1, 0.05 * half_width_um)
    peak = float(np.max(np.abs(field[core]))) if core.any() else 0.0
    if peak > 0:
        field /= peak
    return xy, field, np.abs(field) ** 2


def compute_dispersion_wave_ft(
    w_cm1: np.ndarray,
    k_cm1: np.ndarray,
    rpp: np.ndarray,
    *,
    source_height_nm: float = 25.0,
    observation_height_nm: float = 0.0,
    epsilon_superstrate: complex = 1.0,
    half_width_um: float = 2.0,
    n_pix: int = 301,
) -> tuple[np.ndarray, np.ndarray]:
    """Line cut (y = 0) of the evanescent dipole field versus frequency, from the complex rpp(w, kx) map.

    Per frequency: E(x) = sum_q rpp(q) (2 pi q)^2 / kappa exp(-kappa (h_s + h_o)) exp(i 2 pi q x) dq over the
    evanescent samples q > sqrt(eps) w present in the map (kx > 0 only: the field of the +x
    propagating waves). Each row is normalised to its maximum outside the
    source core (|x| >= max(0.1 um, 5% of the half-width)). Returns (x in um, 0 <= x <= half_width, field[nw, nx]).
    """
    w, k, rp = np.asarray(w_cm1, float), np.asarray(k_cm1, float), np.asarray(rpp)
    if rp.shape != (len(w), len(k)) or not np.iscomplexobj(rp) or len(k) < 2:
        raise ValueError("The real-space map needs complex rpp with shape (Nw, Nk); recompute the dispersion map.")
    eps = complex(epsilon_superstrate)
    if eps.real <= 0 or abs(eps.imag) > 1e-8 * max(1.0, abs(eps.real)):
        raise ValueError("The dipole wave map needs a lossless dielectric superstrate (e.g. air).")
    if min(source_height_nm, observation_height_nm) < 0 or half_width_um <= 0:
        raise ValueError("Heights must be nonnegative and the half-width positive.")
    x = np.linspace(0.0, half_width_um, int(n_pix))
    q = k * 1e-4  # cycles / um
    dq = np.gradient(q)
    hs = (source_height_nm + observation_height_nm) * 1e-3
    field = np.zeros((len(w), len(x)), dtype=complex)
    phase = np.exp(2j * np.pi * q[:, None] * x[None, :])  # (nq, nx)
    for j, wj in enumerate(w):
        qw2 = eps.real * (wj * 1e-4) ** 2
        ev = q**2 > qw2
        if not ev.any():
            continue
        kappa = 2 * np.pi * np.sqrt(q[ev] ** 2 - qw2)
        weight = rp[j, ev] * (2 * np.pi * q[ev]) ** 2 / kappa * np.exp(-kappa * hs) * dq[ev]
        field[j] = weight @ phase[ev]
    core = x >= max(0.1, 0.05 * half_width_um)
    peak = np.max(np.abs(field[:, core]), axis=1, keepdims=True) if core.any() else np.ones((len(w), 1))
    field = np.divide(field, peak, out=np.zeros_like(field), where=peak > 0)
    return x, field


def compute_dispersion_wave_poles(
    kx_mode_cm1: np.ndarray,
    ok: np.ndarray,
    *,
    half_width_um: float = 2.0,
    n_pix: int = 301,
) -> tuple[np.ndarray, np.ndarray]:
    """Plane wave of each mode: exp(i 2 pi q x), x >= 0, with q the complex pole kx (cycles/um), unit amplitude at x = 0.

    Frequencies without a pole (``ok`` False) are zero. Re(q) sets the wavelength, Im(q) the decay.
    Returns (x in um, field[nw, nx]).
    """
    kx = np.asarray(kx_mode_cm1, complex)
    ok = np.asarray(ok, bool) & np.isfinite(kx)
    if half_width_um <= 0 or not ok.any():
        raise ValueError("No converged pole inside the map range; adjust the frequency or momentum range.")
    q = np.where(ok, kx.real + 1j * np.abs(kx.imag), 0.0) * 1e-4
    x = np.linspace(0.0, half_width_um, int(n_pix))
    field = np.exp(2j * np.pi * q[:, None] * x[None, :])
    field[~ok] = 0.0
    return x, field
