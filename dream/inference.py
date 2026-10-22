"""Optional NumPyro inference with explicit geometry-domain checks."""

import jax
import jax.numpy as jnp
import numpy as np

from ._logging import logger


def log_likelihood(emulator, y, sigma, transform=None):
    """Return a jitted diagonal-Gaussian log likelihood on sampled coordinates.

    ``jax.value_and_grad`` may be applied to the returned function. ``transform``
    maps the ordered sampled-coordinate vector to the full emulator input.
    """
    y, sigma = np.asarray(y, dtype=float), np.asarray(sigma, dtype=float)
    if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError('y must be a nonempty finite vector')
    if sigma.shape != y.shape or np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError('sigma must match y and be finite and positive')
    y, sigma = jnp.asarray(y), jnp.asarray(sigma)

    @jax.jit
    def likelihood(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        omega = theta if transform is None else transform(theta)
        prediction = emulator.predict(omega)
        if prediction.shape != y.shape:
            raise ValueError('observable shape must match y and sigma')
        return -0.5 * jnp.sum(((prediction - y) / sigma)**2 + jnp.log(2 * jnp.pi * sigma**2))

    return likelihood


def check_prior_support(emulator, priors, allow_extrapolation=False):
    """Require scalar physical-coordinate geometry priors inside every grid.

    Unbounded or unknown support counts as exceeding the grid. Shared geometry
    must lie inside the intersection of all grids using that coordinate.
    """
    if allow_extrapolation:
        return
    for component in emulator.data.components:
        for index, (low, high, _) in zip(component.geometry, component.grid):
            name = emulator.data.param_names[index]
            if name not in priors:
                raise ValueError(f'{name}: missing prior support for grid [{low}, {high}]')
            support = priors[name].support
            lower = np.asarray(getattr(support, 'lower_bound', -np.inf))
            upper = np.asarray(getattr(support, 'upper_bound', np.inf))
            if (lower.ndim or upper.ndim or not np.isfinite(lower) or not np.isfinite(upper)
                    or lower < low or upper > high):
                raise ValueError(f'{name}: prior support [{lower}, {upper}] exceeds grid [{low}, {high}]')


def run_nuts(emulator, priors, y, sigma, transform=None, num_warmup=500,
             num_samples=1000, num_chains=1, seed=0, allow_extrapolation=False):
    """Sample scalar named priors with a dense mass matrix on one CPU or device.

    Without a transform, priors must exactly name the emulator parameters, and
    their dictionary order is ignored. With a transform, theta follows prior
    insertion order. An arbitrary transform's image cannot be inferred from
    marginal supports: if geometry is present it requires the explicit
    ``allow_extrapolation=True`` override. The caller then owns the bound proof.

    Returns flattened chain samples and ``_edge_fraction``, the fraction of
    transformed draws within one cell of any geometry edge (including outside).
    This diagnostic is not a convergence or posterior-calibration guarantee.
    """
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_median

    if not priors or any(name.startswith('_') for name in priors):
        raise ValueError('priors must be nonempty and names must not begin with an underscore')
    for name, prior in priors.items():
        if prior.batch_shape or prior.event_shape:
            raise ValueError(f'{name}: each prior must be scalar')
    if transform is None:
        if set(priors) != set(emulator.data.param_names):
            raise ValueError('priors must exactly name all emulator parameters without a transform')
        names = list(emulator.data.param_names)
        check_prior_support(emulator, priors, allow_extrapolation)
    else:
        names = list(priors)
        if any(c.geometry for c in emulator.data.components) and not allow_extrapolation:
            raise ValueError('cannot certify geometry bounds of an arbitrary transform; prove its bounds '
                             'in user code and set allow_extrapolation=True explicitly')
    likelihood = log_likelihood(emulator, y, sigma, transform)

    def model():
        theta = jnp.stack([numpyro.sample(name, priors[name]) for name in names])
        numpyro.factor('_likelihood', likelihood(theta))

    sampler = MCMC(NUTS(model, dense_mass=True, init_strategy=init_to_median()),
                   num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                   chain_method='sequential', progress_bar=False)
    logger.info('NUTS: %d chains, %d warmup, %d retained per chain', num_chains, num_warmup, num_samples)
    sampler.run(jax.random.PRNGKey(seed))
    result = {name: np.asarray(value) for name, value in sampler.get_samples().items()}
    theta = np.column_stack([result[name] for name in names])
    omega = theta if transform is None else np.asarray(jax.vmap(transform)(jnp.asarray(theta)))
    if omega.shape != (len(theta), len(emulator.data.param_names)):
        raise ValueError('transform must return the full 1D emulator input vector')
    edges = np.zeros(len(theta), dtype=bool)
    for component in emulator.data.components:
        for index, (low, high, n) in zip(component.geometry, component.grid):
            cell = (high - low) / (n - 1)
            edges |= (omega[:, index] <= low + cell) | (omega[:, index] >= high - cell)
    fraction = float(edges.mean())
    result['_edge_fraction'] = fraction
    if fraction > 0.01:
        logger.warning('%.2f%% of posterior samples lie within one cell of a grid edge', 100 * fraction)
    return result
