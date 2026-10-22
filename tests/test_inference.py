import jax
import jax.numpy as jnp
import numpy as np
import pytest

numpyro = pytest.importorskip('numpyro')
from numpyro import distributions as dist

from dream.emulator import Emulator
from dream.inference import check_prior_support, log_likelihood, run_nuts
from dream.offline import build_basis, build_database, latin_hypercube
from dream.problem import Component


class InferenceProblem:
    param_names = ['amplitude', 'radius']
    n_blocks = 1

    def components(self):
        return [Component('amplitude', 0, ()), Component('shape', None, (1,), grid=((0.2, 1., 8),))]

    def fixed(self, block):
        return np.eye(2, dtype=complex), np.ones(2, dtype=complex)

    def component(self, f, block, geometry):
        source = np.array([1., 0.]) if f == 0 else np.array([0., geometry[0]])
        return np.zeros((2, 2), dtype=complex), source

    def solve(self, omega, block):
        return 1 + np.asarray(omega, dtype=complex)

    def functionals(self, block):
        return np.eye(2, dtype=complex)

    def observable(self, z, omega):
        return (jnp.abs(z)**2).ravel()


@pytest.fixture
def emulator():
    problem = InferenceProblem()
    samples = latin_hypercube(np.array([0.2, 0.2]), np.array([1.5, 1.]), 6, seed=1)
    return Emulator(build_database(problem, build_basis(problem, samples, rank=2)), problem.observable)


@pytest.mark.parametrize('prior', [dist.Uniform(0.1, 1.), dist.Normal(0.6, 0.1), dist.HalfNormal(1.)])
def test_prior_outside_grid_raises(emulator, prior):
    with pytest.raises(ValueError, match=r'radius.*grid'):
        check_prior_support(emulator, {'amplitude': dist.Normal(1., 1.), 'radius': prior})


def test_allow_extrapolation_skips_check(emulator):
    check_prior_support(emulator, {'radius': dist.Normal(0., 1.)}, allow_extrapolation=True)


def test_nuts_recovers_truth(emulator):
    truth = np.array([0.8, 0.6])
    y0 = np.asarray(emulator.predict(truth))
    sigma = 0.02 * y0
    y = y0 + np.random.default_rng(1).normal(size=2) * sigma
    result = run_nuts(emulator, {'radius': dist.Uniform(0.2, 1.), 'amplitude': dist.Uniform(0.2, 1.5)}, y, sigma,
                      num_warmup=120, num_samples=300, seed=4)
    for name, value in zip(emulator.data.param_names, truth):
        low, high = np.quantile(result[name], [0.025, 0.975])
        assert low < value < high
    assert result['_edge_fraction'] == 0.


def test_prior_named_likelihood():
    problem = InferenceProblem()
    problem.param_names = ['likelihood', 'radius']
    samples = latin_hypercube(np.array([0.2, 0.2]), np.array([1.5, 1.]), 6, seed=1)
    emulator = Emulator(build_database(problem, build_basis(problem, samples, rank=2)), problem.observable)
    y = np.asarray(emulator.predict(np.array([0.8, 0.6])))
    result = run_nuts(emulator, {'likelihood': dist.Uniform(0.2, 1.5), 'radius': dist.Uniform(0.2, 1.)},
                      y, 0.02 * y, num_warmup=10, num_samples=10, seed=4)
    assert result['likelihood'].shape == (10,)
    assert np.all(np.isfinite(result['likelihood']))


def test_transform_used(emulator):
    transform = lambda theta: jnp.array([theta[0], 2 * theta[0]])
    y = emulator.predict(jnp.array([0.3, 0.6]))
    sigma = jnp.full(2, 0.02)
    ll = log_likelihood(emulator, y, sigma, transform=transform)
    plain = log_likelihood(emulator, y, sigma)
    theta = jnp.array([0.31])
    np.testing.assert_allclose(ll(theta), plain(transform(theta)))
    np.testing.assert_allclose(jax.grad(ll)(theta)[0], jax.grad(plain)(transform(theta)) @ jnp.array([1., 2.]))
    result = run_nuts(emulator, {'coordinate': dist.Uniform(0.25, 0.35)}, y, sigma,
                      transform=transform, allow_extrapolation=True, num_warmup=50, num_samples=80, seed=3)
    assert abs(np.mean(result['coordinate']) - 0.3) < 0.02
    assert result['_edge_fraction'] == 0.


def test_geometry_transform_requires_explicit_override(emulator):
    with pytest.raises(ValueError, match='transform'):
        run_nuts(emulator, {'radius': dist.Uniform(0.2, 1.), 'amplitude': dist.Uniform(0.2, 1.)},
                 np.ones(2), np.ones(2), transform=lambda x: x * 2, num_samples=1, num_warmup=1)


def test_bad_sigma(emulator):
    with pytest.raises(ValueError, match='sigma'):
        log_likelihood(emulator, np.ones(2), np.zeros(2))


def test_edge_warning(emulator, caplog):
    with caplog.at_level('WARNING', logger='dream'):
        result = run_nuts(emulator, {'amplitude': dist.Uniform(0.5, 0.6), 'radius': dist.Uniform(0.2, 0.21)},
                          np.array([2.4, 1.45]), np.ones(2), num_warmup=30, num_samples=30, seed=2)
    assert result['_edge_fraction'] == 1.
    assert 'grid edge' in caplog.text
