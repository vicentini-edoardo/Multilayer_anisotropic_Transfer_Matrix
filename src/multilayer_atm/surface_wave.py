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
