"""Self-contained data containers and constants for the transfer-matrix engine.

These replace the ``Layer`` / ``System`` objects that used to come from
``pyGTM`` (``GTM.GTMcore``). They are pure data holders plus the frequency- and
orientation-dependent permittivity tensor construction; all of the numerical
transfer-matrix work lives in :mod:`multilayer_atm.solver_fast`.

The Euler-rotation matrix and the lab-frame permittivity construction are ported
verbatim from the Passler & Paarmann (2017) formalism (matching the original
pyGTM implementation), so results are numerically identical to the pyGTM backend.
pyGTM is no longer imported at runtime by the engine; it is kept only as an
optional validation reference for the test suite.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

# Physical constant and mode-sorting thresholds (mirror the values previously
# taken from GTM.GTMcore so branch decisions are identical).
C_CONST = 299792458.0  # speed of light, m/s
QSD_THR = 1e-10  # threshold for wavevector (birefringence) comparison
ZERO_THR = 1e-10  # threshold for eigenvalue clean-up

EpsFunc = Callable[[float], complex]


def vacuum_eps(_f: float) -> complex:
    """Vacuum permittivity (1.0) for any frequency."""
    return 1.0 + 0.0j


class Layer:
    """A single anisotropic layer.

    Holds the layer thickness, the three principal-axis permittivity functions
    ``epsilon_i(f)`` and the crystal orientation (Euler angles, in radians). After
    :py:meth:`calculate_epsilon` the rotated lab-frame tensor is available as
    ``self.epsilon`` (3x3). ``self.mu`` and ``self.thick`` complete the data the
    numerical engine reads.
    """

    def __init__(
        self,
        thickness: float = 1.0e-6,
        epsilon1: Optional[EpsFunc] = None,
        epsilon2: Optional[EpsFunc] = None,
        epsilon3: Optional[EpsFunc] = None,
        theta: float = 0.0,
        phi: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        self.thick = float(thickness)
        self.mu = 1.0 + 0.0j
        self.epsilon = np.identity(3, dtype=np.complex128)

        self._eps1: EpsFunc = epsilon1 if epsilon1 is not None else vacuum_eps
        self._eps2: EpsFunc = epsilon2 if epsilon2 is not None else self._eps1
        self._eps3: EpsFunc = epsilon3 if epsilon3 is not None else self._eps1

        self.set_euler(theta, phi, psi)

    def set_euler(self, theta: float, phi: float, psi: float) -> None:
        """Build the crystal-frame rotation matrix from the Euler angles (rad)."""
        self.theta, self.phi, self.psi = theta, phi, psi
        euler = np.identity(3, dtype=np.complex128)
        euler[0, 0] = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
        euler[0, 1] = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
        euler[0, 2] = np.sin(theta) * np.sin(phi)
        euler[1, 0] = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
        euler[1, 1] = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
        euler[1, 2] = -np.sin(theta) * np.cos(phi)
        euler[2, 0] = np.sin(theta) * np.sin(psi)
        euler[2, 1] = np.sin(theta) * np.cos(psi)
        euler[2, 2] = np.cos(theta)
        self.euler = euler
        # The lab->crystal rotation is the inverse; precompute it once since the
        # orientation is frequency-independent.
        self.euler_inv = np.linalg.inv(euler)

    def calculate_epsilon(self, f: float) -> np.ndarray:
        """Set and return the rotated lab-frame permittivity tensor at frequency f (Hz)."""
        eps_xtal = np.diag(
            np.array([self._eps1(f), self._eps2(f), self._eps3(f)], dtype=np.complex128)
        )
        self.epsilon = self.euler_inv @ (eps_xtal @ self.euler)
        return self.epsilon


class System:
    """An optical system: superstrate, substrate and the interior layers."""

    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.superstrate: Layer = Layer()
        self.substrate: Layer = Layer()

    def set_superstrate(self, layer: Layer) -> None:
        self.superstrate = layer

    def set_substrate(self, layer: Layer) -> None:
        self.substrate = layer

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def initialize_sys(self, f: float) -> None:
        """Evaluate every layer's permittivity tensor at frequency f (Hz)."""
        self.superstrate.calculate_epsilon(f)
        self.substrate.calculate_epsilon(f)
        for layer in self.layers:
            layer.calculate_epsilon(f)
