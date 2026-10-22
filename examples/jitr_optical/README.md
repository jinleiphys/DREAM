# Optical-potential example

This example connects DREAM to [jitr](https://github.com/beykyle/jitr), a public
BSD-3-Clause solver by Kyle Beyer. It demonstrates six-parameter inference for
spin-independent neutron scattering from a mass-40, charge-20 target at
14.1 MeV laboratory energy. It is an illustrative model, with spin-orbit terms
omitted and target mass approximated by 40 atomic mass units.

From the repository root, with Python 3.12 or newer:

```bash
python -m pip install -e '.[dev,example]'
python examples/jitr_optical/run_example.py --quick
python examples/jitr_optical/run_example.py
```

For an existing environment without an editable installation, prepend
`PYTHONPATH=$PWD` to the example command. Outputs go to `outputs/jitr_optical/`;
use `--output DIRECTORY` to keep separate runs. The example directs numba's JIT
cache to a temporary directory unless `NUMBA_CACHE_DIR` is already set.

The potential is

\[
U(r)=-V f(r;r_v A^{1/3},a_v)-4iW_d f(r;r_d A^{1/3},a_d)[1-f(r;r_d A^{1/3},a_d)],
\quad f(r;R,a)=[1+\exp((r-R)/a)]^{-1}.
\]

Depths are in MeV, lengths in fm, and the observable is in mb/sr. The input
truth is `(48, 1.25, 0.65, 6, 1.30, 0.55)`. These illustrative values are not a
fitted global parametrization. Mock data use a direct jitr solve at 40 angles
between 10 and 170 degrees plus Gaussian noise with seed 1. Each error bar is
5% of the noiseless cross section. The truncated-normal priors are centered on
this known mock truth, with depth widths 10% and geometry widths 5%; supports
are the training box (depths +/-20%, geometry +/-10%).

The full run uses 13 partial waves, 40 mesh functions, a 15 fm channel radius,
60 Latin-hypercube snapshots, 16 by 16 component grids, and 500 warmup plus
1000 retained NUTS samples. Quick mode uses 20 snapshots, 8 by 8 grids, and
100 warmup plus 200 retained samples. Solution bases use a relative singular
value cutoff of 1e-10; operator/source compression uses 1e-12. No diagonal
regularization is added.

jitr 3.0.1 consumes potential values on its radial mesh. Its returned
wavefunction coefficients include the external boundary derivative. The
adapter divides by that scalar to obtain the fixed-source response `c=A^-1 b`.
For jitr's dimensionless boundary `a=k*r_channel`, `R=b.T@c/a**2`, followed by
`S=(Hminus-a*R*Hminus_prime)/(Hplus-a*R*Hplus_prime)`. Tests compare this map to
jitr's own S matrix, and test the real-potential and zero-potential limits.

Projection selection compares both projections at rank 8 on six held-out
points snapped to geometry grid nodes, eliminating interpolation error from
that comparison. Transpose is selected if its maximum chi2 error is at least
1% smaller; otherwise Hermitian is retained. Both condition numbers and full
basis errors are recorded. The posterior gate then compares direct solves
with the selected emulator at the posterior mean and eight seeded posterior
draws. Both `chi2_emu < 1` and `max_abs_error/sigma < 0.5` must hold everywhere.
Per-partial-wave S errors are the maximum relative errors over these nine
points, independent of the noise scale.

`summary.json` records the gate, projection comparison, intervals, S errors,
rank-sweep data, and wall time. `samples.npz` retains the posterior samples,
mock data/errors, and gate points. `rank_sweep.pdf` and `rank_sweep.png` show
maximum chi2 error against synchronized forward and gradient timing; labels
are the requested ranks, capped independently per block. The curve can plateau
because geometry-grid interpolation error remains after solution-rank
convergence. A credible interval containing this one truth is a demonstration,
not a general calibration test or a claim about experimental data.

## Example results

Results of one full run (`python examples/jitr_optical/run_example.py`, one NUTS
chain, CPU). The full pipeline, including offline construction, compilation,
sampling, checks, and plotting, takes well under a minute on a laptop.

The nine-point posterior gate passed: maximum chi2 error **0.00549224**,
maximum pointwise error **0.02822378 sigma**. All six truths lay
inside their marginal 95% intervals; for V the interval was
[47.435737, 48.404078] MeV around a 48 MeV truth.
The grid-edge fraction was 0.000.

| Projection, rank 8 | Max chi2 error | Max pointwise error / sigma | Max condition number |
|---|---:|---:|---:|
| Hermitian | 4.8323749e-05 | 0.0033893011 | 6454.44 |
| Transpose | 1.028e-05 | 0.0015985873 | 435.679 |

Transpose was selected. The full basis ranks for l=0 through 12 were
`25, 24, 23, 22, 21, 20, 18, 17, 15, 13, 12, 11, 10`.

| l | Maximum relative S error over the nine posterior points |
|---:|---:|
| 0 | 0.00027246173 |
| 1 | 0.00016686674 |
| 2 | 0.00030900943 |
| 3 | 0.00037644668 |
| 4 | 0.00028343377 |
| 5 | 0.0004249772 |
| 6 | 7.3038308e-05 |
| 7 | 1.5869492e-05 |
| 8 | 3.6556081e-06 |
| 9 | 8.5815924e-07 |
| 10 | 2.0258015e-07 |
| 11 | 4.5899653e-08 |
| 12 | 9.7460585e-09 |

The numbers are written to `summary.json` in the output directory, for example
`python examples/jitr_optical/run_example.py --output outputs/jitr_full`.
The `--quick` setting is a reduced smoke test. Timings vary by machine.
