import numpy as np
import pytest

from examples import jitr_optical

pytest.importorskip('jitr')
pytestmark = pytest.mark.example

from dream.problem import validate_problem
from examples.jitr_optical.adapter import JitrOpticalProblem
from examples.jitr_optical import run_example


def test_adapter_matches_jitr_smatrix():
    problem = JitrOpticalProblem(lmax=4, nbasis=24, grid_size=8)
    validate_problem(problem, problem.truth)
    z = np.array([problem.functionals(b) @ problem.solve(problem.truth, b) for b in range(problem.n_blocks)])
    np.testing.assert_allclose(problem.smatrix(z), problem.direct_smatrix(problem.truth), rtol=1e-11, atol=1e-12)
    real = problem.truth.copy()
    real[3] = 0.
    np.testing.assert_allclose(np.abs(problem.direct_smatrix(real)), 1., atol=1e-12)
    free = real.copy()
    free[0] = 0.
    np.testing.assert_allclose(problem.direct_smatrix(free), 1., atol=1e-8)


def test_quick_example(tmp_path):
    result = run_example.main(['--quick', '--output', str(tmp_path)])
    assert result['gate']['passed']
    assert result['intervals']['V'][0] < result['truth'][0] < result['intervals']['V'][1]
    assert result['gate']['n_points'] >= 9
    assert (tmp_path / 'rank_sweep.pdf').stat().st_size > 1000


def test_file_entrypoint_without_pythonpath(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / 'examples' / 'jitr_optical' / 'run_example.py'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--quick' in result.stdout
