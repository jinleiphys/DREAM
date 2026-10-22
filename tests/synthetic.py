"""Deterministic, small linear systems shared by emulator tests."""

import jax.numpy as jnp
import numpy as np

from dream.problem import Component


class SyntheticProblem:
    """Four distinct additive terms with nonzero source increments.

    ``bounds`` has shape (n_params, 2); ``truth`` is its midpoint. Tests can
    replace ``_components`` to vary metadata, including interpolation grids.
    ``assemble`` is an independent full-order reference for later tests.
    """

    def __init__(self, n_blocks, sizes, symmetric, seed, components, grid_counts, n_functionals):
        if n_blocks < 1 or len(sizes) < n_blocks or any(n < 1 for n in sizes[:n_blocks]):
            raise ValueError("sizes must provide a positive size for each block")
        self.n_blocks = n_blocks
        self.sizes = tuple(sizes[:n_blocks])
        self.param_names = ["a_strength", "b_strength", "b_shape", "c_strength", "radius", "diffuseness"]
        self.bounds = np.array([(0.7, 1.5), (0.5, 1.2), (-0.4, 0.6), (0.6, 1.4), (0.35, 0.7), (0.07, 0.16)])
        self.truth = self.bounds.mean(axis=1)
        counts = (grid_counts, grid_counts) if isinstance(grid_counts, int) else tuple(grid_counts)
        if len(counts) != 2:
            raise ValueError("grid_counts must be an integer or a pair")
        definitions = {
            "a": Component("a", 0, (), reference=1.3),
            "b": Component("b", 1, (2,), reference=0.8, grid=((-0.4, 0.6, counts[0]),)),
            "c": Component("c", 3, (4, 5), reference=1.1, grid=((0.35, 0.7, counts[0]), (0.07, 0.16, counts[1]))),
            "d": Component("d", None, (4,), grid=((0.35, 0.7, counts[0]),)),
        }
        self._components = [definitions[name] for name in components]
        n_functions = (n_functionals,) * n_blocks if isinstance(n_functionals, int) else tuple(n_functionals)
        if len(n_functions) != n_blocks or any(n < 1 for n in n_functions):
            raise ValueError("n_functionals must give a positive count for each block")
        rng = np.random.default_rng(seed)

        def random_complex(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        self._fixed, self._terms, self._functionals = [], [], []
        for block, size in enumerate(self.sizes):
            def operator(scale):
                matrix = random_complex((size, size)) * scale / np.sqrt(size)
                return (matrix + matrix.T) / 2 if symmetric else matrix

            self._fixed.append(((3.0 + 0.02 * block) * np.eye(size) + operator(0.08), random_complex(size)))
            self._terms.append({name: (operator(0.05), random_complex(size) * 0.04) for name in definitions})
            self._functionals.append(random_complex((n_functions[block], size)) / np.sqrt(size))

    def components(self):
        return list(self._components)

    def fixed(self, block):
        return self._fixed[block]

    def component(self, f, block, geometry):
        f = self._components[f]
        matrix, source = self._terms[block][f.name]
        if f.name == "a":
            return matrix, source
        if f.name == "b":
            factor = np.exp(0.3 * geometry[0])
            return factor * matrix, (1 + geometry[0] ** 2) * source
        radial = np.linspace(0, 1, self.sizes[block])
        if f.name == "c":
            radius, diffuseness = geometry
            profile = 1 / (1 + np.exp((radial - radius) / diffuseness))
            return np.diag((0.12 + 0.03j) * profile), profile * source
        radius, = geometry
        profile = np.exp(-((radial - radius) / 0.25) ** 2)
        return radius * matrix, profile * source

    def assemble(self, omega, block):
        matrix, source = (array.copy() for array in self.fixed(block))
        for index, f in enumerate(self.components()):
            dmatrix, dsource = self.component(index, block, np.asarray(omega)[list(f.geometry)])
            weight = 1.0 if f.strength is None else omega[f.strength] / f.reference
            matrix += weight * dmatrix
            source += weight * dsource
        return matrix, source

    def solve(self, omega, block):
        matrix, source = self.assemble(omega, block)
        return np.linalg.solve(matrix, source)

    def functionals(self, block):
        return self._functionals[block]

    def observable(self, z, omega):
        return (jnp.abs(z) ** 2).ravel()


def make_synthetic(n_blocks=3, sizes=(40, 55, 30), symmetric=False, seed=0, *,
                   components=("a", "b", "c", "d"), grid_counts=5, n_functionals=2):
    """Build a reproducible problem, optionally selecting terms/grid sizes."""
    return SyntheticProblem(n_blocks, sizes, symmetric, seed, components, grid_counts, n_functionals)
