"""Layer and system containers for the transfer-matrix engine.

These construct frequency- and orientation-dependent permittivity tensors.
The numerical transfer-matrix work lives in :mod:`multilayer_atm.solver_fast`.
Euler rotation follows the Passler and Paarmann (2017) formalism.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

# Physical constant and mode-sorting thresholds shared with solver_fast.
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
        # Whether the (rotated) tensor is a scalar multiple of the identity. Set by
        # calculate_epsilon; lets the engine use closed-form isotropic modes and
        # skip the 4x4 eigen-decomposition. Defaults True (vacuum identity tensor).
        self.is_isotropic = True
        # Whether the rotated tensor has no coupling between the in-plane (x, y) and
        # out-of-plane (z) axes (eps_xz = eps_yz = 0, e.g. any in-plane rotation of a
        # crystal whose principal axis 3 is along z). Enables the closed-form 2x2
        # propagation in solver_fast. Defaults True (vacuum identity tensor).
        self.is_block_diagonal = True

        self._eps1: EpsFunc = epsilon1 if epsilon1 is not None else vacuum_eps
        self._eps2: EpsFunc = epsilon2 if epsilon2 is not None else self._eps1
        self._eps3: EpsFunc = epsilon3 if epsilon3 is not None else self._eps1
        # Principal-axis permittivities pre-evaluated for a whole frequency grid
        # (see prepare_frequencies): {f_hz: (eps1, eps2, eps3)}.
        self._prepared: dict = {}

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

    def prepare_frequencies(self, f_values) -> None:
        """Pre-evaluate the principal-axis permittivities on a frequency grid (Hz).

        The axis callables accept arrays, so one vectorised call replaces one scalar
        call per row of a dispersion map (the material models are interpolation /
        oscillator sums whose cost is dominated by call overhead). A later
        :py:meth:`calculate_epsilon` at exactly one of these frequencies reuses the
        stored values; any other frequency is evaluated on the fly.
        """
        f = np.asarray(f_values, dtype=float).ravel()
        e = [np.asarray(fn(f), dtype=np.complex128).reshape(f.shape) for fn in (self._eps1, self._eps2, self._eps3)]
        self._prepared = {float(fi): (e[0][i], e[1][i], e[2][i]) for i, fi in enumerate(f)}

    def batch_at(self, f_values: np.ndarray) -> LayerBatch:
        """Rotated tensors for every frequency in ``f_values`` (Hz), as a :class:`LayerBatch`."""
        f = np.asarray(f_values, dtype=float).ravel()
        diag = np.stack(
            [np.broadcast_to(np.asarray(fn(f), dtype=np.complex128), f.shape) for fn in (self._eps1, self._eps2, self._eps3)]
        )  # (3, N)
        eps = np.einsum("ik,kn,kj->ijn", self.euler_inv, diag, self.euler, optimize=True)
        scale = np.abs(eps[0, 0]) + np.abs(eps[1, 1]) + np.abs(eps[2, 2])
        off = sum(np.abs(eps[i, j]) for i in range(3) for j in range(3) if i != j)
        spread = np.abs(eps[0, 0] - eps[1, 1]) + np.abs(eps[0, 0] - eps[2, 2])
        zc = np.abs(eps[0, 2]) + np.abs(eps[1, 2]) + np.abs(eps[2, 0]) + np.abs(eps[2, 1])
        return LayerBatch(
            self.thick, self.mu, eps,
            bool(np.all(off + spread <= 1e-12 * scale)),
            bool(np.all(zc <= 1e-12 * scale)),
        )

    def calculate_epsilon(self, f: float) -> np.ndarray:
        """Set and return the rotated lab-frame permittivity tensor at frequency f (Hz)."""
        axes = self._prepared.get(f)
        if axes is None:
            axes = (self._eps1(f), self._eps2(f), self._eps3(f))
        eps_xtal = np.diag(np.array(axes, dtype=np.complex128))
        self.epsilon = self.euler_inv @ (eps_xtal @ self.euler)

        # Classify isotropy from the resulting tensor: a rotated isotropic medium
        # stays scalar (R^-1 (eps I) R = eps I), so off-diagonals and the spread of
        # the diagonal vanish to round-off. This also catches doped/custom isotropic
        # materials whose axis functions are not the same object.
        e = self.epsilon
        scale = abs(e[0, 0]) + abs(e[1, 1]) + abs(e[2, 2])
        off = (
            abs(e[0, 1]) + abs(e[0, 2]) + abs(e[1, 0])
            + abs(e[1, 2]) + abs(e[2, 0]) + abs(e[2, 1])
        )
        spread = abs(e[0, 0] - e[1, 1]) + abs(e[0, 0] - e[2, 2])
        self.is_isotropic = (off + spread) <= 1e-12 * scale
        zcoupling = abs(e[0, 2]) + abs(e[1, 2]) + abs(e[2, 0]) + abs(e[2, 1])
        self.is_block_diagonal = zcoupling <= 1e-12 * scale
        return self.epsilon


class LayerBatch:
    """Layer data for many (frequency, sample) pairs at once.

    ``epsilon`` has shape (3, 3, N): the rotated lab-frame tensor of every sample, so
    ``epsilon[i, j]`` is an (N,) array and the batched solver code, which only ever
    reads entries ``epsilon[i, j]``, works unchanged for per-sample frequencies.
    ``is_isotropic`` / ``is_block_diagonal`` hold only if they hold for *every* sample
    (the general paths are valid for any tensor, so this is always safe).
    """

    def __init__(self, thick: float, mu: complex, epsilon: np.ndarray, is_isotropic: bool, is_block_diagonal: bool) -> None:
        self.thick = thick
        self.mu = mu
        self.epsilon = epsilon
        self.is_isotropic = is_isotropic
        self.is_block_diagonal = is_block_diagonal

    def take(self, index: np.ndarray) -> "LayerBatch":
        return LayerBatch(self.thick, self.mu, self.epsilon[:, :, index], self.is_isotropic, self.is_block_diagonal)


class SystemBatch:
    """A :class:`System` evaluated at an array of frequencies (one tensor per sample)."""

    def __init__(self, superstrate: LayerBatch, substrate: LayerBatch, layers: List[LayerBatch]) -> None:
        self.superstrate = superstrate
        self.substrate = substrate
        self.layers = layers

    def take(self, index: np.ndarray) -> "SystemBatch":
        return SystemBatch(self.superstrate.take(index), self.substrate.take(index), [l.take(index) for l in self.layers])


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

    def batch_at(self, f_values: np.ndarray) -> SystemBatch:
        """Evaluate every layer at each frequency in ``f_values`` (Hz) as a :class:`SystemBatch`."""
        return SystemBatch(
            self.superstrate.batch_at(f_values),
            self.substrate.batch_at(f_values),
            [layer.batch_at(f_values) for layer in self.layers],
        )

    def prepare_frequencies(self, f_values) -> None:
        """Vectorised pre-evaluation of every layer's principal axes on a frequency grid."""
        for layer in (self.superstrate, self.substrate, *self.layers):
            layer.prepare_frequencies(f_values)

    def initialize_sys(self, f: float) -> None:
        """Evaluate every layer's permittivity tensor at frequency f (Hz)."""
        self.superstrate.calculate_epsilon(f)
        self.substrate.calculate_epsilon(f)
        for layer in self.layers:
            layer.calculate_epsilon(f)
