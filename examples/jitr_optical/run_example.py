"""Build, check, calibrate and time a small public-solver emulator on a CPU."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.jitr_optical.adapter import JitrOpticalProblem
import jax
import numpy as np
from numpyro import distributions as dist

from dream.emulator import Emulator
from dream.inference import run_nuts
from dream.offline import Basis, build_basis, build_database, latin_hypercube
from dream.plotting import plot_rank_sweep
from dream.problem import validate_problem
from dream.verify import direct_observable, gate, rank_sweep

logger = logging.getLogger('dream.example')


def _gate_summary(result):
    return {'passed': result.passed, 'n_points': len(result.chi2_emu),
            'chi2_max': float(np.max(result.chi2_emu)), 'resid_max': float(np.max(result.max_resid)),
            'chi2_emu': result.chi2_emu.tolist(), 'max_resid': result.max_resid.tolist(),
            'failed_indices': result.failed_indices.tolist()}


def main(argv=None):
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/jitr_optical'))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    problem = JitrOpticalProblem(grid_size=8 if args.quick else 16)
    validate_problem(problem, problem.truth)
    logger.info('Adapter validated; building %d blocks on %s', problem.n_blocks, jax.default_backend())
    samples = latin_hypercube(*problem.bounds.T, 20 if args.quick else 60, seed=0)
    basis = build_basis(problem, samples, rtol=1e-10)
    y0 = direct_observable(problem, problem.truth)
    sigma = 0.05 * y0
    y = y0 + np.random.default_rng(1).normal(size=len(y0)) * sigma
    comparisons, candidates = {}, {}
    points = latin_hypercube(*problem.bounds.T, 6, seed=22)
    for definition in problem.components():
        for index, (low, high, n) in zip(definition.geometry, definition.grid):
            axes = np.linspace(low, high, n)
            points[:, index] = axes[np.argmin(abs(points[:, index, None] - axes), axis=1)]
    for projection in ('hermitian', 'transpose'):
        projected = Basis(basis.X, basis.singular_values, projection)
        emulator = Emulator(build_database(problem, projected, mode_rtol=1e-12), problem.observable)
        candidates[projection] = emulator
        diagnostic = emulator.truncate(8)
        check = gate(diagnostic, problem, points, sigma)
        matrices = [np.asarray(diagnostic._system(w)[0]) for w in points]
        condition = max(np.linalg.cond(m[:k, :k]) for batch in matrices for m, k in zip(batch, diagnostic.rank))
        comparisons[projection] = {'rank8': _gate_summary(check), 'condition_max': float(condition),
                                   'full': _gate_summary(gate(emulator, problem, points, sigma))}
        logger.info('%s projection: rank8 chi2 %.4g, max residual %.4g, max condition %.4g',
                    projection, comparisons[projection]['rank8']['chi2_max'],
                    comparisons[projection]['rank8']['resid_max'], condition)
    h_error = comparisons['hermitian']['rank8']['chi2_max']
    t_error = comparisons['transpose']['rank8']['chi2_max']
    projection = 'transpose' if t_error < 0.99 * h_error else 'hermitian'
    emulator = candidates[projection]
    logger.info('Selected %s; active ranks %s', projection, emulator.rank)
    priors = {name: dist.TruncatedNormal(float(value), float(value * scale), low=float(low), high=float(high))
              for name, value, scale, (low, high) in zip(problem.param_names, problem.truth,
                                                        (0.1, 0.05, 0.05, 0.1, 0.05, 0.05), problem.bounds)}
    posterior = run_nuts(emulator, priors, y, sigma, num_warmup=100 if args.quick else 500,
                         num_samples=200 if args.quick else 1000, seed=1)
    draws = np.column_stack([posterior[name] for name in problem.param_names])
    indices = np.random.default_rng(10).choice(len(draws), size=8, replace=False)
    posterior_points = np.vstack([draws.mean(axis=0), draws[indices]])
    check = gate(emulator, problem, posterior_points, sigma)
    errors = []
    for w in posterior_points:
        reference = problem.direct_smatrix(w)
        predicted = np.asarray(problem.smatrix(emulator.functionals(w)))
        errors.append(abs(predicted - reference) / np.maximum(abs(reference), np.finfo(float).tiny))
    per_block = np.max(errors, axis=0)
    logger.info('Posterior gate: %s, chi2 max %.6g, max residual %.6g', check.passed,
                np.max(check.chi2_emu), np.max(check.max_resid))
    logger.info('Maximum relative S error per block: %s', np.array2string(per_block, precision=5))
    intervals = {name: np.quantile(posterior[name], [0.025, 0.975]).tolist() for name in problem.param_names}
    for name, value in zip(problem.param_names, problem.truth):
        logger.info('%s truth %.6g, posterior 95%% interval %s', name, value, intervals[name])
    full = max(emulator.rank)
    ranks = sorted(set(min(rank, full) for rank in (4, 8, 12, 16, 24, full)))
    sweep = rank_sweep(emulator, problem, posterior_points, sigma, ranks, n_rep=20)
    plot_rank_sweep(sweep, args.output / 'rank_sweep.pdf')
    plot_rank_sweep(sweep, args.output / 'rank_sweep.png')
    result = {'quick': args.quick, 'projection': projection, 'projection_comparison': comparisons,
              'rank': emulator.rank, 'truth': problem.truth.tolist(), 'intervals': intervals,
              'gate': _gate_summary(check), 's_relative_error_per_block': per_block.tolist(),
              'edge_fraction': posterior['_edge_fraction'], 'sweep': sweep,
              'wall_seconds': perf_counter() - started}
    np.savez(args.output / 'samples.npz', **posterior, gate_points=posterior_points, y=y, sigma=sigma)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    logger.info('Finished in %.2f seconds; results in %s', result['wall_seconds'], args.output)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    logging.getLogger('dream').setLevel(logging.INFO)
    result = main()
    if not result['gate']['passed']:
        raise SystemExit('posterior verification gate failed')
