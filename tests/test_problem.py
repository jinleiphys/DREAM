from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dream.problem import Component, validate_problem
from tests.synthetic import make_synthetic


@pytest.mark.parametrize("symmetric", [False, True])
def test_validate_synthetic(symmetric):
    problem = make_synthetic(symmetric=symmetric)
    validate_problem(problem, problem.truth)
    if symmetric:
        for block in range(problem.n_blocks):
            matrices = [problem.fixed(block)[0]]
            matrices += [
                problem.component(index, block, problem.truth[list(f.geometry)])[0]
                for index, f in enumerate(problem.components())
            ]
            for matrix in matrices:
                np.testing.assert_allclose(matrix, matrix.T)


def test_component_is_frozen():
    component = Component("a", 0, ())
    assert component.reference == 1.0 and component.grid == ()
    with pytest.raises(FrozenInstanceError):
        component.name = "changed"


def test_bad_component_shape_names_component():
    problem = make_synthetic(n_blocks=1, sizes=(8,))
    original = problem.component

    def malformed(f, block, geometry):
        matrix, source = original(f, block, geometry)
        return (matrix[:-1], source) if problem.components()[f].name == "b" else (matrix, source)

    problem.component = malformed
    with pytest.raises(ValueError, match=r"component 'b'.*block 0.*dM.*shape"):
        validate_problem(problem, problem.truth)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"name": ""}, "name"),
        ({"strength": 6}, "strength"),
        ({"strength": True}, "strength"),
        ({"geometry": (2, 4, 5)}, "geometry"),
        ({"geometry": (-1,)}, "geometry"),
        ({"geometry": (2, 2)}, "geometry"),
        ({"reference": 0.0}, "reference"),
        ({"reference": np.inf}, "reference"),
        ({"grid": ()}, "grid"),
        ({"grid": ((0.0, 1.0, 1),)}, "grid"),
        ({"grid": ((1.0, 0.0, 3),)}, "grid"),
        ({"grid": ((0.0, np.inf, 3),)}, "grid"),
    ],
)
def test_bad_metadata(changes, message):
    problem = make_synthetic(n_blocks=1, sizes=(8,))
    original = problem.components()
    original[1] = replace(original[1], **changes)
    problem.components = lambda: original
    with pytest.raises(ValueError, match=message):
        validate_problem(problem, problem.truth)


def test_bad_solve_rejected():
    problem = make_synthetic(n_blocks=1, sizes=(8,))
    original = problem.solve
    problem.solve = lambda omega, block: original(omega, block) * 1.01
    with pytest.raises(ValueError, match=r"solve.*block 0.*relative"):
        validate_problem(problem, problem.truth)


def test_validation_does_not_mutate_adapter_arrays():
    problem = make_synthetic(n_blocks=1, sizes=(8,))
    matrix, source = problem.fixed(0)
    expected_matrix, expected_source = matrix.copy(), source.copy()
    matrix.flags.writeable = source.flags.writeable = False
    validate_problem(problem, problem.truth)
    np.testing.assert_array_equal(matrix, expected_matrix)
    np.testing.assert_array_equal(source, expected_source)


@pytest.mark.parametrize("omega", [np.zeros(5), np.full(6, np.nan), np.zeros(6, dtype=int)])
def test_bad_omega(omega):
    with pytest.raises(ValueError, match="omega"):
        validate_problem(make_synthetic(n_blocks=1, sizes=(8,)), omega)


def test_bad_operator_dtype_and_nonfinite():
    problem = make_synthetic(n_blocks=1, sizes=(8,))
    matrix, source = problem.fixed(0)
    problem.fixed = lambda block: (matrix.astype(bool), source)
    with pytest.raises(ValueError, match=r"fixed.*M.*dtype"):
        validate_problem(problem, problem.truth)
    matrix = matrix.copy()
    matrix[0, 0] = np.nan
    problem.fixed = lambda block: (matrix, source)
    with pytest.raises(ValueError, match=r"fixed.*M.*finite"):
        validate_problem(problem, problem.truth)


@pytest.mark.parametrize(
    "observable, message",
    [
        (lambda z, omega: z.ravel(), "real"),
        (lambda z, omega: jnp.abs(z) ** 2, "shape"),
        (lambda z, omega: jnp.full((2,), jnp.nan), "finite"),
        (lambda z, omega: np.asarray(z).real.ravel(), "JAX"),
    ],
)
def test_bad_observable(observable, message):
    problem = make_synthetic(n_blocks=1, sizes=(8,))
    problem.observable = observable
    with pytest.raises(ValueError, match=message):
        validate_problem(problem, problem.truth)


def test_fixture_configuration_and_padding():
    problem = make_synthetic(
        n_blocks=2, sizes=(8, 10), components=("b", "c", "d"),
        grid_counts=(4, 3), n_functionals=(1, 3),
    )
    validate_problem(problem, problem.truth)
    assert [f.name for f in problem.components()] == ["b", "c", "d"]
    assert [g[2] for g in problem.components()[1].grid] == [4, 3]
    z = jnp.zeros((2, 3), dtype=jnp.complex128)
    result = jax.jit(problem.observable)(z, jnp.asarray(problem.truth))
    assert result.shape == (6,) and result.dtype == jnp.float64


def test_fixture_deterministic_and_nonzero_component_sources():
    first = make_synthetic(n_blocks=1, sizes=(8,), seed=12)
    second = make_synthetic(n_blocks=1, sizes=(8,), seed=12)
    np.testing.assert_array_equal(first.fixed(0)[0], second.fixed(0)[0])
    for index, f in enumerate(first.components()):
        _, source = first.component(index, 0, first.truth[list(f.geometry)])
        assert np.linalg.norm(source) > 0
