"""Direct-solver checks and synchronized accuracy-versus-cost measurements."""

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class GateResult:
    chi2_emu: np.ndarray
    max_resid: np.ndarray
    passed: bool
    failed_indices: np.ndarray


def direct_observable(problem, omega):
    """Apply the adapter's same observable to independently solved functionals."""
    rows = [np.asarray(problem.functionals(b)) @ np.asarray(problem.solve(omega, b))
            for b in range(problem.n_blocks)]
    z = np.zeros((len(rows), max(map(len, rows))), dtype=complex)
    for block, row in enumerate(rows):
        z[block, :len(row)] = row
    return np.asarray(problem.observable(jnp.asarray(z), jnp.asarray(omega, dtype=jnp.float64)))


def _inputs(emulator, points, sigma):
    points, sigma = np.asarray(points, dtype=float), np.asarray(sigma, dtype=float)
    if (points.ndim != 2 or len(points) == 0
            or points.shape[1] != len(emulator.data.param_names) or not np.all(np.isfinite(points))):
        raise ValueError('points must be a nonempty finite (n_points, n_params) array')
    if sigma.ndim != 1 or not len(sigma) or np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError('sigma must be a finite positive vector')
    return points, sigma


def _compare(emulator, points, sigma, reference, chi2_max, resid_max):
    predictions = np.asarray([emulator.predict(point) for point in points])
    if predictions.shape != reference.shape or reference.shape[1:] != sigma.shape:
        raise ValueError('sigma and observable vector shapes must match')
    residuals = (predictions - reference) / sigma
    chi2 = np.sum(residuals**2, axis=1)
    maximum = np.max(np.abs(residuals), axis=1)
    valid = np.isfinite(chi2) & np.isfinite(maximum) & (chi2 < chi2_max) & (maximum < resid_max)
    return GateResult(chi2, maximum, bool(np.all(valid)), np.flatnonzero(~valid))


def gate(emulator, problem, points, sigma, chi2_max=1.0, resid_max=0.5) -> GateResult:
    """Compare at supplied points; chi2_emu is a sum, not a per-datum average.

    A point passes only if both strict inequalities hold. Nonfinite predictions
    fail. Use posterior points to test the region actually used by inference.
    """
    points, sigma = _inputs(emulator, points, sigma)
    reference = np.asarray([direct_observable(problem, point) for point in points])
    return _compare(emulator, points, sigma, reference, chi2_max, resid_max)


def _wall(function, omega, repetitions):
    for _ in range(2):
        jax.block_until_ready(function(omega))
    elapsed = []
    for _ in range(repetitions):
        start = perf_counter()
        jax.block_until_ready(function(omega))
        elapsed.append(perf_counter() - start)
    return float(np.median(elapsed))


def rank_sweep(emulator, problem, points, sigma, ranks, n_rep=20) -> dict:
    """Gate each truncated basis and time forward and Gaussian value-and-grad.

    Times are median seconds at the first point, after two warmups. The gradient
    timing uses a diagonal Gaussian likelihood centered on the first direct
    observable. Reference solves are reused across ranks and excluded from time.
    """
    points, sigma = _inputs(emulator, points, sigma)
    if not isinstance(n_rep, (int, np.integer)) or n_rep < 1 or not len(ranks):
        raise ValueError('require positive integer n_rep and nonempty ranks')
    reference = np.asarray([direct_observable(problem, point) for point in points])
    result = {key: [] for key in ('rank', 'chi2_max', 'chi2_median', 'resid_max', 't_forward', 't_grad')}
    omega = jnp.asarray(points[0])
    for rank in ranks:
        reduced = emulator.truncate(rank)
        check = _compare(reduced, points, sigma, reference, 1.0, 0.5)
        gradient = jax.jit(jax.value_and_grad(
            lambda w: -0.5 * jnp.sum(((reduced.predict(w) - reference[0]) / sigma)**2)))
        values = (rank, np.max(check.chi2_emu), np.median(check.chi2_emu), np.max(check.max_resid),
                  _wall(reduced.predict, omega, n_rep), _wall(gradient, omega, n_rep))
        for key, value in zip(result, values):
            result[key].append(value)
    return result
