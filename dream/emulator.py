"""JAX reconstruction and batched Galerkin solves."""

import jax
import jax.numpy as jnp
import numpy as np

from .database import BlockData, ComponentData, Compression, EmulatorData
from .interp import multilinear


class Emulator:
    """A differentiable map from physical parameters to user observables.

    Padding is internal: padded rows have identity on the diagonal, zero
    sources, and zero functionals. ``regularization`` adds an absolute diagonal
    shift to active reduced rows only, changes the approximation, and defaults
    to zero. Geometry extrapolation clamps; use ``in_grid`` on the host to check.
    """

    def __init__(self, data: EmulatorData, observable, *, regularization=0.0):
        if not data.blocks:
            raise ValueError('data must contain at least one block')
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError('regularization must be finite and nonnegative')
        self.data, self.observable = data, observable
        self.regularization = regularization
        self._rank = [len(block.s0) for block in data.blocks]
        width = max(self._rank)
        nfunc = max(block.LX.shape[0] for block in data.blocks)
        matrices, sources, functionals = [], [], []
        for k, block in zip(self._rank, data.blocks):
            m = np.eye(width, dtype=complex)
            m[:k, :k] = block.M0 + regularization * np.eye(k)
            matrices.append(m)
            sources.append(np.pad(block.s0, (0, width - k)))
            functionals.append(np.pad(block.LX, ((0, nfunc - len(block.LX)), (0, width - k))))
        self._M0 = jnp.asarray(np.stack(matrices), dtype=jnp.complex128)
        self._s0 = jnp.asarray(np.stack(sources), dtype=jnp.complex128)
        self._LX = jnp.asarray(np.stack(functionals), dtype=jnp.complex128)
        self._terms = []
        for f, definition in enumerate(data.components):
            axes = tuple(jnp.linspace(lo, hi, n) for lo, hi, n in definition.grid)
            pairs = []
            for field in ('matrix', 'source'):
                items = [getattr(block.components[f], field) for block in data.blocks]
                nmodes = max(len(item.modes) for item in items)
                means, modes, coefficients = [], [], []
                for k, item in zip(self._rank, items):
                    pad = ((0, width - k),) * item.mean.ndim
                    means.append(np.pad(item.mean, pad))
                    modes.append(np.pad(item.modes, ((0, nmodes - len(item.modes)),) + pad))
                    coefficients.append(np.pad(item.coeff_grid, ((0, 0),) * len(axes) + ((0, nmodes - len(item.modes)),)))
                pairs.append(tuple(jnp.asarray(np.stack(values)) for values in (means, modes, coefficients)))
            self._terms.append((definition, axes, pairs))
        self._solve_jit = jax.jit(self._solve)
        self._functionals_jit = jax.jit(self._functionals)
        self._predict_jit = jax.jit(self._predict)

    @property
    def rank(self) -> list[int]:
        return list(self._rank)

    def _omega(self, omega):
        omega = jnp.asarray(omega, dtype=jnp.float64)
        if omega.shape != (len(self.data.param_names),):
            raise ValueError(f'omega must have shape ({len(self.data.param_names)},)')
        return omega

    def _system(self, omega):
        omega = self._omega(omega)
        matrix, source = self._M0, self._s0
        for definition, axes, pairs in self._terms:
            geometry = omega[jnp.asarray(definition.geometry, dtype=int)]
            weight = 1.0 if definition.strength is None else omega[definition.strength] / definition.reference
            updates = []
            for mean, modes, coeff in pairs:
                values = jax.vmap(lambda grid: multilinear(grid, axes, geometry))(coeff)
                update = jax.vmap(lambda c, v: jnp.tensordot(c, v, axes=1))(values, modes)
                updates.append(weight * (mean + update))
            matrix, source = matrix + updates[0], source + updates[1]
        return matrix, source

    def _solve(self, omega):
        matrix, source = self._system(omega)
        return jax.vmap(jnp.linalg.solve)(matrix, source)

    def reduced_solution(self, omega) -> list[jax.Array]:
        """Return each block's active coefficients, with no visible padding."""
        solution = self._solve_jit(omega)
        return [solution[b, :k] for b, k in enumerate(self._rank)]

    def _functionals(self, omega):
        return jnp.einsum('bfk,bk->bf', self._LX, self._solve(omega))

    def functionals(self, omega):
        """Return (n_blocks, max_n_functionals), with zero right padding."""
        return self._functionals_jit(omega)

    def _predict(self, omega):
        omega = self._omega(omega)
        return jnp.asarray(self.observable(self._functionals(omega), omega), dtype=jnp.float64)

    def predict(self, omega):
        """Evaluate the jitted, float64 parameter-to-observable map."""
        return self._predict_jit(omega)

    def in_grid(self, omega) -> bool:
        """Host-side finite-input and geometry-domain check, including shared axes."""
        omega = np.asarray(omega)
        if omega.shape != (len(self.data.param_names),) or not np.all(np.isfinite(omega)):
            return False
        return all(lo <= omega[index] <= hi for component in self.data.components
                   for index, (lo, hi, _) in zip(component.geometry, component.grid))

    def truncate(self, k: int):
        """Slice every reduced tensor to its leading rank-k subspace.

        No new compression or interpolation is performed. Equality to a fresh
        build is exact before floating-point and independent SVD-mode truncation
        errors. Requested rank is capped separately at each block's rank.
        """
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        blocks = []
        for block in self.data.blocks:
            terms = []
            for component in block.components:
                compressed = []
                for item in (component.matrix, component.source):
                    slices = (slice(None, k),) * item.mean.ndim
                    compressed.append(Compression(item.mean[slices], item.modes[(slice(None),) + slices],
                                                  item.coeff_grid, item.singular_values))
                terms.append(ComponentData(*compressed))
            blocks.append(BlockData(block.M0[:k, :k], block.s0[:k], block.LX[:, :k], terms))
        data = EmulatorData(list(self.data.param_names), list(self.data.components), blocks, self.data.projection)
        return Emulator(data, self.observable, regularization=self.regularization)
