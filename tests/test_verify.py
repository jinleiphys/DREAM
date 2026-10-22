import numpy as np
import pytest

from dream.plotting import plot_rank_sweep
from dream.verify import direct_observable, gate, rank_sweep
from tests.synthetic import make_synthetic
from tests.test_emulator import make_emulator


@pytest.fixture
def case():
    problem = make_synthetic(2, (12, 9), grid_counts=5)
    emulator, _ = make_emulator(problem)
    points = np.stack([problem.truth, problem.bounds[:, 0], problem.bounds[:, 1]])
    sigma = np.full(4, 1e-5)
    return emulator, problem, points, sigma


def test_gate_passes_full_rank(case):
    result = gate(*case)
    assert result.passed
    assert result.failed_indices.size == 0
    assert result.chi2_emu.shape == (3,)
    assert np.max(result.chi2_emu) < 1e-15


def test_direct_observable_accepts_list_arrays(monkeypatch):
    problem = make_synthetic(2, (12, 9), grid_counts=5)
    expected = direct_observable(problem, problem.truth)
    functionals, solve = problem.functionals, problem.solve
    monkeypatch.setattr(problem, 'functionals', lambda block: functionals(block).tolist())
    monkeypatch.setattr(problem, 'solve', lambda omega, block: solve(omega, block).tolist())
    np.testing.assert_allclose(direct_observable(problem, problem.truth), expected)


def test_gate_fails_rank2(case):
    emulator, *args = case
    result = gate(emulator.truncate(2), *args)
    assert not result.passed
    assert result.failed_indices.size > 0


def test_rank_sweep_monotone_tail(case):
    result = rank_sweep(*case, ranks=[2, 6, 12], n_rep=3)
    assert set(result) == {'rank', 'chi2_max', 'chi2_median', 'resid_max', 't_forward', 't_grad'}
    assert result['chi2_max'][1] < result['chi2_max'][0]
    assert result['chi2_max'][2] < result['chi2_max'][0]
    assert np.all(np.asarray(result['t_forward']) > 0)
    assert np.all(np.asarray(result['t_grad']) > 0)


def test_plot_writes_file(case, tmp_path):
    result = rank_sweep(*case, ranks=[2, 6, 12], n_rep=1)
    path = tmp_path / 'rank.pdf'
    plot_rank_sweep(result, path)
    assert path.stat().st_size > 1000


@pytest.mark.parametrize('sigma', [np.zeros(4), np.full(4, -1.), np.full(4, np.nan), np.ones(3)])
def test_gate_rejects_bad_errors(case, sigma):
    with pytest.raises(ValueError, match='sigma'):
        gate(*case[:3], sigma)


def test_gate_rejects_empty_points(case):
    with pytest.raises(ValueError, match='points'):
        gate(case[0], case[1], np.empty((0, 6)), case[3])
