"""Adapter contract and numerical validation for parameterized linear systems."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Component:
    """An additive operator/source pair evaluated at geometry coordinates.

    A component with a strength index is multiplied by
    ``omega[strength] / reference``. A component without one has unit weight.
    Geometry contains parameter indices; grid contains matching (lo, hi, n)
    interpolation specifications, with zero to two geometry coordinates.
    """

    name: str
    strength: int | None
    geometry: tuple[int, ...]
    reference: float = 1.0
    grid: tuple[tuple[float, float, int], ...] = ()

    def __post_init__(self):
        """Keep numeric metadata JSON-native without repairing invalid shapes."""
        def scalar(value):
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            return value

        for name in ('strength', 'reference'):
            object.__setattr__(self, name, scalar(getattr(self, name)))
        if isinstance(self.geometry, tuple):
            object.__setattr__(self, 'geometry', tuple(scalar(v) for v in self.geometry))
        if isinstance(self.grid, tuple):
            object.__setattr__(self, 'grid', tuple(
                tuple(scalar(v) for v in axis) if isinstance(axis, tuple) else axis
                for axis in self.grid
            ))


class Problem(Protocol):
    """Full-order adapter for ``M(omega) c = s(omega)``.

    Operators and sources may be real or complex floating arrays. ``fixed``
    returns shapes (N, N), (N,); ``component`` returns increments of the same
    shapes and receives the component's integer index and geometry *values*,
    ordered as in the metadata.
    ``functionals`` returns a matrix of shape (n_func, N), applied as ``L @ c``.
    ``observable`` must accept JAX arrays and return a real floating vector.
    Its input z has shape (n_blocks, max(n_func)), with zero padding on the
    right for blocks with fewer functionals. Block sizes may differ.
    """

    param_names: list[str]
    n_blocks: int

    def components(self) -> list[Component]: ...

    def fixed(self, block: int) -> tuple[ArrayLike, ArrayLike]: ...

    def component(
        self, f: int, block: int, geometry: np.ndarray
    ) -> tuple[ArrayLike, ArrayLike]: ...

    def solve(self, omega: ArrayLike, block: int) -> ArrayLike: ...

    def functionals(self, block: int) -> ArrayLike: ...

    def observable(self, z: jax.Array, omega: jax.Array) -> jax.Array: ...


def _array(value, label, *, shape=None, ndim=None, real=False):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a numeric array") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{label}: expected shape {shape}, got {array.shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label}: expected {ndim}D shape, got {array.shape}")
    if array.dtype.kind not in ("f" if real else "fc"):
        kind = "real floating" if real else "floating or complex floating"
        raise ValueError(f"{label}: expected {kind} dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: all entries must be finite")
    return array


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _finite_real(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _pair(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label}: expected an (operator, source) pair")
    return value


def validate_problem(problem: Problem, omega: ArrayLike) -> None:
    """Check metadata, arrays, JAX observables, and solves at one parameter point.

    Each adapter solve must agree with independently assembled NumPy solutions
    to relative Euclidean error at most 1e-8. This is a local contract check,
    not a validation of the model's physics or its full parameter domain.
    Adapter arrays are never modified.
    """
    names = problem.param_names
    if (
        not isinstance(names, list) or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("param_names: expected a nonempty list of unique nonempty strings")
    n_params = len(names)
    omega = _array(omega, "omega", shape=(n_params,), real=True)
    if not _integer(problem.n_blocks) or problem.n_blocks < 1:
        raise ValueError("n_blocks: expected a positive integer")
    components = problem.components()
    if not isinstance(components, list):
        raise ValueError("components(): expected a list of Component objects")
    seen = set()
    for i, f in enumerate(components):
        if not isinstance(f, Component):
            raise ValueError(f"components()[{i}]: expected a Component")
        label = f"component {f.name!r}"
        if not isinstance(f.name, str) or not f.name.strip() or f.name in seen:
            raise ValueError(f"{label}: name must be a unique nonempty string")
        seen.add(f.name)
        if f.strength is not None and (
            not _integer(f.strength) or not 0 <= f.strength < n_params
        ):
            raise ValueError(f"{label}: strength must be None or an index in [0, {n_params})")
        if (
            not isinstance(f.geometry, tuple) or len(f.geometry) > 2
            or any(not _integer(index) or not 0 <= index < n_params for index in f.geometry)
            or len(set(f.geometry)) != len(f.geometry)
        ):
            raise ValueError(f"{label}: geometry must contain zero to two unique parameter indices")
        if not _finite_real(f.reference) or f.reference == 0:
            raise ValueError(f"{label}: reference must be finite, real, and nonzero")
        if not isinstance(f.grid, tuple) or len(f.grid) != len(f.geometry):
            raise ValueError(f"{label}: grid must have one (lo, hi, n) tuple per geometry index")
        for axis, grid in enumerate(f.grid):
            if not isinstance(grid, tuple) or len(grid) != 3:
                raise ValueError(f"{label}: grid[{axis}] must be a (lo, hi, n) tuple")
            lo, hi, n = grid
            if not _finite_real(lo) or not _finite_real(hi) or lo >= hi:
                raise ValueError(f"{label}: grid[{axis}] requires finite lo < hi")
            if not _integer(n) or n < 2:
                raise ValueError(f"{label}: grid[{axis}] requires an integer n >= 2")

    projected = []
    for block in range(problem.n_blocks):
        label = f"fixed block {block}"
        matrix, source = _pair(problem.fixed(block), label)
        matrix = _array(matrix, f"{label} M", ndim=2)
        if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{label} M: expected nonempty square shape, got {matrix.shape}")
        size = matrix.shape[0]
        source = _array(source, f"{label} s", shape=(size,))
        matrix = np.array(matrix, dtype=np.complex128, copy=True)
        source = np.array(source, dtype=np.complex128, copy=True)
        for index, f in enumerate(components):
            label = f"component {f.name!r} block {block}"
            geometry = omega[list(f.geometry)].copy()
            dmatrix, dsource = _pair(problem.component(index, block, geometry), label)
            dmatrix = _array(dmatrix, f"{label} dM", shape=(size, size))
            dsource = _array(dsource, f"{label} ds", shape=(size,))
            weight = 1.0 if f.strength is None else omega[f.strength] / f.reference
            matrix += weight * dmatrix
            source += weight * dsource
        _array(matrix, f"assembled block {block} M")
        _array(source, f"assembled block {block} s")
        try:
            expected = np.linalg.solve(matrix, source)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"assembled block {block} M: singular linear system") from exc
        _array(expected, f"reference solve block {block}")
        actual = _array(problem.solve(omega.copy(), block), f"solve block {block}", shape=(size,))
        error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), np.finfo(float).tiny)
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError(f"solve block {block}: relative error {error:.6g} exceeds 1e-8")
        functionals = _array(problem.functionals(block), f"functionals block {block} L", ndim=2)
        if functionals.shape[1] != size or functionals.shape[0] < 1:
            raise ValueError(f"functionals block {block} L: expected shape (n_func >= 1, {size}), got {functionals.shape}")
        projected.append(functionals @ actual)

    z = np.zeros((problem.n_blocks, max(len(row) for row in projected)), dtype=np.complex128)
    for block, row in enumerate(projected):
        z[block, :len(row)] = row
    z, parameters = jnp.asarray(z), jnp.asarray(omega)
    try:
        observable = problem.observable(z, parameters)
    except Exception as exc:
        raise ValueError("observable: evaluation failed for padded z and omega") from exc
    _array(observable, "observable", ndim=1, real=True)
    try:
        jax.eval_shape(problem.observable, z, parameters)
    except Exception as exc:
        raise ValueError("observable: must support JAX tracing with padded z and omega") from exc
