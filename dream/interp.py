"""Differentiable multilinear interpolation with clamped extrapolation."""

import itertools

import jax.numpy as jnp


def multilinear(grid_values, axes: tuple, x):
    """Interpolate over zero, one, or two leading grid axes.

    Outside an axis range the coordinate is clamped to its nearest edge. The
    derivative there is zero. Interior knots use the right cell's derivative;
    outer endpoints use half the adjacent cell's derivative, following JAX's
    clip convention. Derivatives need not be continuous. All trailing value
    dimensions are retained.
    """
    if not axes:
        return jnp.asarray(grid_values)
    if len(axes) > 2:
        raise ValueError('multilinear supports at most two axes')
    indices, fractions = [], []
    for dim, axis in enumerate(axes):
        value = jnp.clip(x[dim], axis[0], axis[-1])
        index = jnp.clip(jnp.searchsorted(axis, value, side='right') - 1, 0, len(axis) - 2)
        indices.append(index)
        fractions.append((value - axis[index]) / (axis[index + 1] - axis[index]))
    result = jnp.zeros(jnp.shape(grid_values)[len(axes):], dtype=jnp.asarray(grid_values).dtype)
    for corner in itertools.product((0, 1), repeat=len(axes)):
        weight = 1.0
        for high, fraction in zip(corner, fractions):
            weight = weight * (fraction if high else 1 - fraction)
        result = result + weight * grid_values[tuple(i + high for i, high in zip(indices, corner))]
    return result
