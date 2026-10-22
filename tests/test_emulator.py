import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dream.emulator import Emulator
from dream.interp import multilinear
from dream.offline import build_basis, build_database, latin_hypercube
from tests.synthetic import make_synthetic


def make_emulator(problem, *, rank=None, projection='hermitian', mode_rtol=0):
    samples = latin_hypercube(*problem.bounds.T, max(problem.sizes) + 5, seed=9)
    basis = build_basis(problem, samples, rank=rank, projection=projection)
    return Emulator(build_database(problem, basis, mode_rtol=mode_rtol), problem.observable), samples


def direct(problem, omega):
    rows = [problem.functionals(b) @ problem.solve(omega, b) for b in range(problem.n_blocks)]
    z = np.zeros((len(rows), max(map(len, rows))), dtype=complex)
    for b, row in enumerate(rows):
        z[b, :len(row)] = row
    return np.asarray(problem.observable(jnp.asarray(z), jnp.asarray(omega)))


def test_strength_only_exact():
    problem = make_synthetic(3, (12, 9, 7), components=('a',))
    emulator, _ = make_emulator(problem)
    for w in latin_hypercube(*problem.bounds.T, 3, seed=2):
        np.testing.assert_allclose(emulator.predict(w), direct(problem, w), rtol=1e-10, atol=1e-13)


def test_geometry_converges():
    errors = []
    for n in (6, 12, 24):
        problem = make_synthetic(1, (8,), grid_counts=n)
        emulator, _ = make_emulator(problem)
        points = latin_hypercube(*problem.bounds.T, 5, seed=7)
        errors.append(np.linalg.norm([emulator.predict(w) - direct(problem, w) for w in points]))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < errors[0] / 5


def test_ad_matches_fd():
    problem = make_synthetic(2, (10, 8), grid_counts=7)
    emulator, _ = make_emulator(problem)
    w = problem.bounds[:, 0] + 0.437 * np.diff(problem.bounds).ravel()
    step = 1e-5 * (problem.bounds[:, 1] - problem.bounds[:, 0])
    gradient = jax.grad(lambda p: emulator.predict(p).sum())(jnp.asarray(w))
    fd = []
    for delta in np.diag(step):
        fd.append((emulator.predict(w + delta).sum() - emulator.predict(w - delta).sum()) / (2 * np.max(delta)))
    np.testing.assert_allclose(gradient, fd, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('projection', ['hermitian', 'transpose'])
def test_truncate_equals_rebuild(projection):
    problem = make_synthetic(2, (12, 9), symmetric=True)
    emulator, samples = make_emulator(problem, projection=projection)
    k = 5
    rebuilt = Emulator(build_database(problem, build_basis(problem, samples, rank=k, projection=projection), mode_rtol=0), problem.observable)
    for w in samples[:3]:
        np.testing.assert_allclose(emulator.truncate(k).predict(w), rebuilt.predict(w), rtol=1e-10, atol=1e-13)


def test_unequal_blocks():
    problem = make_synthetic(3, (9, 6, 4), n_functionals=(1, 3, 2), components=('a',))
    emulator, _ = make_emulator(problem)
    assert emulator.rank == [9, 6, 4]
    assert [len(c) for c in emulator.reduced_solution(problem.truth)] == [9, 6, 4]
    assert emulator.functionals(problem.truth).shape == (3, 3)
    np.testing.assert_array_equal(np.asarray(emulator.functionals(problem.truth))[0, 1:], 0)
    np.testing.assert_allclose(emulator.predict(problem.truth), direct(problem, problem.truth), rtol=1e-10, atol=1e-13)


def test_none_strength_and_0d():
    problem = make_synthetic(1, (6,), components=('a', 'd'))
    emulator, _ = make_emulator(problem)
    w = problem.truth.copy()
    w[4] = problem.components()[1].grid[0][0]
    np.testing.assert_allclose(emulator.predict(w), direct(problem, w), rtol=1e-10)


def test_symmetric_transpose_projection(record_property):
    problem = make_synthetic(1, (12,), symmetric=True, components=('a',))
    for projection in ('hermitian', 'transpose'):
        emulator, _ = make_emulator(problem, rank=4, projection=projection)
        relative = np.linalg.norm(emulator.predict(problem.truth) - direct(problem, problem.truth)) / np.linalg.norm(direct(problem, problem.truth))
        record_property(projection + '_relative_error', relative)
        assert relative < 1e-8


def test_float32_omega_and_in_grid():
    problem = make_synthetic(1, (6,))
    emulator, _ = make_emulator(problem)
    assert emulator.predict(problem.truth.astype(np.float32)).dtype == jnp.float64
    assert emulator.in_grid(problem.truth)
    w = problem.truth.copy()
    w[4] = 0.1
    assert not emulator.in_grid(w)
    edge = w.copy()
    edge[4] = problem.bounds[4, 0]
    np.testing.assert_allclose(emulator.predict(w), emulator.predict(edge), rtol=1e-13)
    assert not emulator.in_grid(np.full(6, np.nan))


def test_float32_observable_returns_float64():
    problem = make_synthetic(1, (6,), components=('a',))
    base, _ = make_emulator(problem)
    emulator = Emulator(base.data, lambda z, w: problem.observable(z, w).astype(jnp.float32))
    assert emulator.predict(problem.truth).dtype == jnp.float64


@pytest.mark.parametrize('projection', ['hermitian', 'transpose'])
def test_truncate_caps_each_block(projection):
    problem = make_synthetic(3, (9, 6, 4), symmetric=True)
    emulator, samples = make_emulator(problem, projection=projection)
    truncated = emulator.truncate(5)
    rebuilt, _ = make_emulator(problem, rank=5, projection=projection)
    assert truncated.rank == [5, 5, 4]
    for w in samples[:3]:
        np.testing.assert_allclose(truncated.predict(w), rebuilt.predict(w), rtol=1e-10, atol=1e-13)


def test_multilinear():
    axes = (jnp.array([0., 1.]), jnp.array([0., 2.]))
    values = jnp.array([[0., 6.], [2., 8.]])
    assert float(multilinear(values, axes, jnp.array([0.25, 0.5]))) == 2.
    assert float(multilinear(values, axes, jnp.array([-1., 3.]))) == 6.
    assert float(multilinear(jnp.array(3.), (), jnp.array([]))) == 3.
    assert float(multilinear(jnp.array([1., 3.]), axes[:1], jnp.array([0.25]))) == 1.5


def test_shared_strength():
    from dataclasses import replace

    problem = make_synthetic(1, (6,), components=('a', 'b'))
    problem._components[1] = replace(problem._components[1], strength=0)
    emulator, _ = make_emulator(problem)
    for value in (0.8, 1.2):
        omega = problem.truth.copy()
        omega[0] = value
        np.testing.assert_allclose(emulator.predict(omega), direct(problem, omega), rtol=1e-10)
