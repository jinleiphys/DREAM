from dataclasses import fields, is_dataclass, replace

import h5py
import numpy as np
import pytest

from dream.database import load, save
from dream.offline import build_basis, build_database, latin_hypercube
from dream.problem import validate_problem
from tests.synthetic import make_synthetic


def assert_equal(a, b):
    if is_dataclass(a):
        for field in fields(a):
            assert_equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.fixture
def offline_data():
    problem = make_synthetic(n_blocks=2, sizes=(9, 7), grid_counts=4)
    samples = latin_hypercube(*problem.bounds.T, 15, seed=2)
    basis = build_basis(problem, samples, rank=7)
    return problem, basis, build_database(problem, basis)


def test_latin_hypercube():
    lo, hi = np.array([-2., 1.]), np.array([3., 2.])
    samples = latin_hypercube(lo, hi, 10, seed=3)
    np.testing.assert_array_equal(samples, latin_hypercube(lo, hi, 10, seed=3))
    for column in ((samples - lo) / (hi - lo)).T:
        np.testing.assert_array_equal(np.sort((column * 10).astype(int)), np.arange(10))


def test_basis_orthonormal_and_ordered(offline_data):
    _, basis, _ = offline_data
    for x, s in zip(basis.X, basis.singular_values):
        np.testing.assert_allclose(x.conj().T @ x, np.eye(x.shape[1]), atol=1e-12)
        assert np.all(np.diff(s) <= 0)


def test_compressed_grid_reconstructs(offline_data):
    problem, basis, data = offline_data
    for block, reduced in enumerate(data.blocks):
        x = basis.X[block]
        for f, entry in enumerate(reduced.components):
            axes = [np.linspace(*axis) for axis in data.components[f].grid]
            shape = tuple(len(axis) for axis in axes)
            for index in np.ndindex(shape):
                geometry = np.array([axis[i] for axis, i in zip(axes, index)])
                matrix, source = problem.component(f, block, geometry)
                for compressed, expected in ((entry.matrix, x.conj().T @ matrix @ x),
                                             (entry.source, x.conj().T @ source)):
                    actual = compressed.mean + np.tensordot(compressed.coeff_grid[index], compressed.modes, axes=1)
                    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_h5_roundtrip(offline_data, tmp_path):
    data = offline_data[2]
    path = tmp_path / 'data.h5'
    save(data, path)
    assert_equal(data, load(path))
    with h5py.File(path) as file:
        assert file.attrs['schema'] == 1
        assert file.attrs['dream_version'] == '0.1.0'
        assert len(file.attrs['components_hash']) == 64


@pytest.mark.parametrize('all_fields', [False, True])
def test_numpy_metadata_roundtrip(offline_data, tmp_path, all_fields):
    problem, _, data = offline_data
    definition = data.components[1]
    changes = {'reference': np.float64(definition.reference)}
    if all_fields:
        changes.update(
            strength=np.int64(definition.strength),
            geometry=tuple(np.int64(index) for index in definition.geometry),
            grid=tuple((np.float64(lo), np.float64(hi), np.int64(n))
                       for lo, hi, n in definition.grid),
        )
    definitions = list(data.components)
    definitions[1] = replace(definition, **changes)
    problem._components = definitions
    validate_problem(problem, problem.truth)
    path = tmp_path / 'numpy_metadata.h5'
    save(replace(data, components=definitions), path)
    assert_equal(data, load(path, components=definitions))


@pytest.mark.parametrize('changes, message', [
    ({'strength': np.bool_(True)}, 'strength'),
    ({'reference': np.bool_(True)}, 'reference'),
    ({'geometry': [np.int64(2)]}, 'geometry'),
    ({'grid': [(0., 1., 3)]}, 'grid'),
    ({'grid': ([0., 1., 3],)}, 'grid'),
])
def test_metadata_normalization_preserves_invalid_types(changes, message):
    problem = make_synthetic(n_blocks=1, sizes=(6,))
    problem._components[1] = replace(problem._components[1], **changes)
    with pytest.raises(ValueError, match=message):
        validate_problem(problem, problem.truth)


def test_components_hash_mismatch_raises(offline_data, tmp_path):
    path = tmp_path / 'data.h5'
    save(offline_data[2], path)
    with h5py.File(path, 'r+') as file:
        file.attrs['components_hash'] = 'incorrect'
    with pytest.raises(ValueError, match='components_hash'):
        load(path)


def test_expected_components_and_schema(offline_data, tmp_path):
    data = offline_data[2]
    path = tmp_path / 'data.h5'
    save(data, path)
    changed = [replace(data.components[0], reference=9.)] + data.components[1:]
    with pytest.raises(ValueError, match='components_hash'):
        load(path, components=changed)
    with h5py.File(path, 'r+') as file:
        file.attrs['schema'] = 999
    with pytest.raises(ValueError, match='schema'):
        load(path)


def test_rank_tolerance_and_invalid_inputs():
    problem = make_synthetic(n_blocks=1, sizes=(6,))
    samples = latin_hypercube(*problem.bounds.T, 10, seed=0)
    basis = build_basis(problem, samples, rtol=1e-4, projection='transpose')
    assert basis.projection == 'transpose'
    assert 1 <= basis.X[0].shape[1] <= 6
    for kwargs in ({'rank': 0}, {'rtol': -1}, {'rank': 3, 'rtol': 0.1}, {'projection': 'other'}):
        with pytest.raises(ValueError):
            build_basis(problem, samples, **kwargs)
