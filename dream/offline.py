"""Sampling, ordered solution bases, and compression of reduced operators."""

from dataclasses import dataclass

import numpy as np

from ._logging import logger
from .database import BlockData, ComponentData, Compression, EmulatorData
from .problem import validate_problem


@dataclass
class Basis:
    """Per-block orthonormal columns ordered by snapshot singular value."""

    X: list[np.ndarray]
    singular_values: list[np.ndarray]
    projection: str = 'hermitian'


def latin_hypercube(lo, hi, n: int, seed: int) -> np.ndarray:
    """One randomly placed sample per stratum along each box coordinate."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if (lo.ndim != 1 or hi.shape != lo.shape or not np.all(np.isfinite([lo, hi]))
            or np.any(hi <= lo) or not isinstance(n, (int, np.integer)) or n < 1):
        raise ValueError('require finite matching 1D bounds lo < hi and positive integer n')
    rng = np.random.default_rng(seed)
    unit = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in lo])
    return lo + (hi - lo) * unit


def build_basis(problem, samples, rank=None, rtol=None, projection='hermitian') -> Basis:
    """SVD of solution snapshots; keep a count or singular values > rtol*s[0].

    With neither selection argument all available columns are retained. Explicit
    rank is capped at each block's available dimension. Transpose projection
    uses X.T rather than X.conj().T; the basis itself is unchanged.
    """
    if projection not in ('hermitian', 'transpose'):
        raise ValueError('projection must be hermitian or transpose')
    if rank is not None and (not isinstance(rank, (int, np.integer)) or rank < 1):
        raise ValueError('rank must be a positive integer')
    if rtol is not None and (not np.isfinite(rtol) or not 0 <= rtol < 1):
        raise ValueError('rtol must be in [0, 1)')
    if rank is not None and rtol is not None:
        raise ValueError('choose rank or rtol, not both')
    samples = np.asarray(samples, dtype=float)
    if (samples.ndim != 2 or samples.shape[1] != len(problem.param_names)
            or len(samples) == 0 or not np.all(np.isfinite(samples))):
        raise ValueError('samples must be a finite nonempty (n_samples, n_params) array')
    validate_problem(problem, samples[0])
    columns, spectra = [], []
    for block in range(problem.n_blocks):
        snapshots = np.column_stack([problem.solve(omega, block) for omega in samples])
        if not np.all(np.isfinite(snapshots)):
            raise ValueError(f'nonfinite snapshots in block {block}')
        u, s, _ = np.linalg.svd(snapshots, full_matrices=False)
        k = min(rank, len(s)) if rank is not None else len(s)
        if rtol is not None:
            k = max(1, int(np.count_nonzero(s > rtol * s[0])))
        columns.append(u[:, :k])
        spectra.append(s)
        logger.info('Built block %d basis, rank %d of %d', block, k, len(s))
    return Basis(columns, spectra, projection)


def _compress(values, grid_shape, rtol):
    """Complex SVD; trailing tensor dimensions stay explicit for rank slicing."""
    mean = values.mean(axis=0)
    flat = (values - mean).reshape(len(values), -1)
    if not np.any(flat):
        return Compression(mean, np.empty((0,) + mean.shape, dtype=values.dtype),
                           np.empty(grid_shape + (0,), dtype=values.dtype), np.empty(0))
    u, s, vh = np.linalg.svd(flat, full_matrices=False)
    k = int(np.count_nonzero(s > rtol * s[0]))
    return Compression(mean, vh[:k].reshape((k,) + mean.shape),
                       (u[:, :k] * s[:k]).reshape(grid_shape + (k,)), s)


def build_database(problem, basis: Basis, mode_rtol=1e-10) -> EmulatorData:
    """Project fixed terms and sample/compress each component's geometry grid.

    Matrix and source are compressed separately so their relative scales do not
    affect mode selection. Grid interpolation is performed only online.
    """
    if not np.isfinite(mode_rtol) or not 0 <= mode_rtol < 1:
        raise ValueError('mode_rtol must be in [0, 1)')
    if basis.projection not in ('hermitian', 'transpose'):
        raise ValueError('projection must be hermitian or transpose')
    if len(basis.X) != problem.n_blocks:
        raise ValueError('basis must contain one array per block')
    components = problem.components()
    blocks = []
    for block, x in enumerate(basis.X):
        matrix, source = problem.fixed(block)
        if x.ndim != 2 or x.shape[0] != len(source) or x.shape[1] < 1:
            raise ValueError(f'incompatible basis shape for block {block}')
        left = x.conj().T if basis.projection == 'hermitian' else x.T
        terms = []
        for f, definition in enumerate(components):
            axes = [np.linspace(lo, hi, n) for lo, hi, n in definition.grid]
            shape = tuple(len(axis) for axis in axes)
            matrices, sources = [], []
            for index in np.ndindex(shape):
                geometry = np.array([axis[i] for axis, i in zip(axes, index)])
                dm, ds = problem.component(f, block, geometry)
                matrices.append(left @ dm @ x)
                sources.append(left @ ds)
            terms.append(ComponentData(_compress(np.asarray(matrices), shape, mode_rtol),
                                       _compress(np.asarray(sources), shape, mode_rtol)))
        blocks.append(BlockData(left @ matrix @ x, left @ source,
                                np.asarray(problem.functionals(block)) @ x, terms))
    return EmulatorData(list(problem.param_names), components, blocks, basis.projection)
