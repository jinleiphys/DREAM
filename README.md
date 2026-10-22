# DREAM

**DREAM** (Differentiable Reduced-basis Emulator with Automatic gradients for
Markov-chain inference) turns an expensive, parameterized linear solver into a
fast emulator whose outputs and exact gradients can be used directly by
gradient-based Bayesian samplers such as Hamiltonian Monte Carlo (HMC) and its
adaptive form, the No-U-Turn Sampler (NUTS).

The original solver is never rewritten or differentiated. It is used only
offline, to export the parameter-dependent operators of its discretized problem
and a set of solutions. DREAM compresses these objects and rebuilds a small
reduced problem from differentiable operations, so every evaluation of the
emulator returns the observables together with their exact derivatives with
respect to all parameters.

## What it is for

DREAM is meant for calibration problems in which

- each forward calculation is a large linear solve, for example a
  coupled-channels, R-matrix, bound-state, or other discretized scattering or
  structure calculation, and a posterior needs of order 10^5 such evaluations;
- the parameters enter the operator through a few additive terms, each with a
  linear strength and a nonlinear "geometry" of at most two parameters, as in
  phenomenological potentials built from Woods-Saxon form factors
  (depth, radius, diffuseness);
- gradients are wanted, for HMC/NUTS, posterior-mode searches, or sensitivity
  studies, but the production solver does not provide them.

It provides the tools needed to check the approximation against the original
solver at the points that matter (typically the posterior the emulator itself
produces), to see how accuracy trades against cost as the basis size changes,
and to sample a posterior with NUTS. It does not assume any particular physics:
everything problem specific lives in one small adapter class written by the user.

## How it works

The adapter describes independent blocks $b$ (partial waves, total angular
momenta, energies, ...), each a linear system

$$
M_b(\boldsymbol\omega)\,\mathbf c_b=\mathbf s_b(\boldsymbol\omega),\qquad
M_b(\boldsymbol\omega)=M_{0b}+\sum_f w_f(\boldsymbol\omega)\,\Delta M_{fb}(\mathbf g_f),
\qquad
\mathbf s_b(\boldsymbol\omega)=\mathbf s_{0b}+\sum_f w_f(\boldsymbol\omega)\,\Delta\mathbf s_{fb}(\mathbf g_f),
$$

where component $f$ carries a linear strength $w_f=\omega_{\alpha_f}/\omega^{\rm ref}_{\alpha_f}$
and a geometry $\mathbf g_f$ of zero, one, or two parameters. The observable is
any function, written in JAX, of a few linear functionals of the solution,
$\mathbf z_b=L_b\mathbf c_b$, for instance the boundary values from which an
S-matrix and a cross section follow.

**Offline.** Full solutions at sampled parameter points are collected per block
and compressed by a singular value decomposition, $C_b=U_b\Sigma_bV_b^\dagger$.
The leading $n_b$ columns of $U_b$, ordered by singular value, form the solution
basis $X_b$. Every operator and source is projected onto it,

$$
\widehat M_{0b}=X_b^\dagger M_{0b}X_b,\qquad
\widehat{\mathbf s}_{0b}=X_b^\dagger\mathbf s_{0b},\qquad
\widehat L_b=L_bX_b ,
$$

and each component is evaluated on a regular grid in its geometry and compressed
once more,

$$
X_b^\dagger\,\Delta M_{fb}(\mathbf g)\,X_b\simeq\overline M_{fb}
+\sum_{m=1}^{K_f}\gamma_{fbm}(\mathbf g)\,B_{fbm},
$$

so that only the scalar coefficients $\gamma_{fbm}$ have to be interpolated.

**Online.** For a parameter vector the coefficients are interpolated
(multilinearly), the reduced operator is assembled, and the small Galerkin
system is solved in JAX:

$$
\widehat M_b(\boldsymbol\omega)\,\mathbf c^{\rm red}_b=\widehat{\mathbf s}_b(\boldsymbol\omega),
\qquad \mathbf z_b=\widehat L_b\,\mathbf c^{\rm red}_b,
\qquad \mathbf y=\mathcal O(\mathbf z,\boldsymbol\omega).
$$

Automatic differentiation then gives the exact gradient of the emulated
observable or likelihood. Differentiating the reduced solve,

$$
\frac{\partial\mathbf c^{\rm red}_b}{\partial\theta}
=\widehat M_b^{-1}\left[\frac{\partial\widehat{\mathbf s}_b}{\partial\theta}
-\frac{\partial\widehat M_b}{\partial\theta}\,\mathbf c^{\rm red}_b\right],
$$

reuses the factorization of the forward solve, so the cost of a gradient does not
grow with the number of parameters beyond the assembly of the parameter derivatives.

**Verification.** The emulator error is measured against the original solver in
the units of the data errors $\sigma_i$,

$$
\chi^2_{\rm emu}=\sum_i\left(\frac{y^{\rm emu}_i-y^{\rm direct}_i}{\sigma_i}\right)^2,
\qquad
r_{\max}=\max_i\frac{|y^{\rm emu}_i-y^{\rm direct}_i|}{\sigma_i},
$$

and a point passes when $\chi^2_{\rm emu}<1$ and $r_{\max}<0.5$. Because the basis
columns are ordered by singular value, the emulator with fewer modes is obtained
exactly by keeping the leading rows and columns of every reduced object, which
makes accuracy-versus-cost (rank) sweeps free.

Details, including the derivative conventions at grid knots and what each error
measure does and does not guarantee, are in [docs/method.md](docs/method.md).

## Install and test

Python 3.10 or newer is required for the core. From a source checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not example"
```

For the worked example, which uses the public R-matrix solver
[jitr](https://github.com/beykyle/jitr) and needs Python 3.12 or newer:

```bash
python -m pip install -e ".[dev,example]"
python -m pytest -m example
python examples/jitr_optical/run_example.py --quick
python examples/jitr_optical/run_example.py
```

Core dependencies are `jax>=0.4.30`, `numpy>=1.24`, and `h5py>=3.8`.
Extras are `inference` (NumPyro), `plotting` (matplotlib), `example` (jitr), and
`dev` (pytest, NumPyro, matplotlib). The example adapter is written for jitr 3.0.1.
Continuous integration runs the core tests on Python 3.10 and 3.12 and the quick
example on Python 3.12.

**Importing `dream` enables JAX float64 globally.** Emulator inputs and outputs
are float64 and operators complex128. The library logs through the `dream`
logger and does not configure application logging.

## Quick start

A complete adapter for the one-dimensional system `(2 + strength) * c = 1`.
Replace its methods with calls to your own solver. No inheritance is required.

```python
import jax.numpy as jnp
import numpy as np
from dream.problem import Component, validate_problem
from dream.offline import build_basis, build_database
from dream.emulator import Emulator
from dream.verify import gate

class ScalarProblem:
    param_names = ['strength']
    n_blocks = 1

    def components(self):
        return [Component('shift', strength=0, geometry=())]

    def fixed(self, block):
        return np.array([[2.0]]), np.array([1.0])

    def component(self, f, block, geometry):
        return np.array([[1.0]]), np.array([0.0])

    def solve(self, omega, block):
        return np.array([1 / (2 + omega[0])])

    def functionals(self, block):
        return np.array([[1.0]])

    def observable(self, z, omega):
        return (jnp.abs(z)**2).ravel()

problem = ScalarProblem()
validate_problem(problem, np.array([0.5]))
samples = np.array([[0.2], [0.5], [0.8]])
data = build_database(problem, build_basis(problem, samples, rank=1))
emulator = Emulator(data, problem.observable)
np.testing.assert_allclose(emulator.predict(np.array([0.4])), [1 / 2.4**2])
check = gate(emulator, problem, np.array([[0.4], [0.7]]), sigma=np.array([0.01]))
assert check.passed
```

## The adapter

A problem is any object with these attributes and methods:

| Member | Returns |
|---|---|
| `param_names`, `n_blocks` | parameter names; number of independent blocks |
| `components()` | list of `Component(name, strength, geometry, reference, grid)` |
| `fixed(block)` | parameter-independent operator `(N, N)` and source `(N,)` |
| `component(f, block, geometry)` | operator and source of component `f` at its reference strength |
| `solve(omega, block)` | full solution `(N,)`, used for snapshots and verification |
| `functionals(block)` | matrix `L` `(n_func, N)` of linear functionals of the solution |
| `observable(z, omega)` | JAX function from functionals `z` `(n_blocks, n_func)` to the observable vector |

Components may share strength or geometry parameters. A map from sampled
coordinates to the emulator inputs (for example correlated or transformed
priors) is passed to the inference helper as a JAX `transform`.
See [docs/interface.md](docs/interface.md).

## Main functions

| Module | Function | Purpose |
|---|---|---|
| `dream.problem` | `validate_problem` | check an adapter's shapes and its solver against a dense solve |
| `dream.offline` | `latin_hypercube`, `build_basis`, `build_database` | snapshots, ordered solution basis, compressed reduced operators |
| `dream.database` | `save`, `load` | HDF5 storage with a schema version and a hash of the components |
| `dream.emulator` | `Emulator.predict`, `.truncate(k)`, `.in_grid` | jitted differentiable map; rank-k restriction; grid check |
| `dream.verify` | `gate`, `rank_sweep` | comparison with the original solver; accuracy versus cost |
| `dream.inference` | `log_likelihood`, `run_nuts` | Gaussian likelihood with gradient; NumPyro NUTS with grid-box checks |
| `dream.plotting` | `plot_rank_sweep` | accuracy-versus-cost figure |

## Worked example

[examples/jitr_optical](examples/jitr_optical/README.md) emulates neutron elastic
scattering on a medium-mass target with a six-parameter optical potential (real
volume and imaginary surface Woods-Saxon terms), using jitr as the original
solver. It builds the emulator, samples the posterior of seeded mock data with
NUTS, checks the emulator against jitr at the posterior mean and posterior draws,
reports S-matrix errors, and produces a rank-sweep figure. A full run takes well
under a minute on a laptop CPU.

## Limits

- The emulator is only as good as the snapshot set; check it with `gate` at the
  posterior it produces, and enrich the snapshots where it fails.
- Geometry interpolation is constant outside its grid. `run_nuts` refuses priors
  on geometry parameters that extend beyond the grids unless
  `allow_extrapolation=True` is passed, and reports the fraction of samples near
  a grid edge.
- The gradient is the exact gradient of the emulated quantity. Its agreement
  with derivatives of the original solver is a separate question that the user
  can test by finite differences of the original solver.
- Parameters that enter one coupled nonlinear function, rather than a few
  additive low-dimensional components, need a different coefficient model.

## Citing DREAM

If you use DREAM, please cite

```bibtex
@article{Lei:2026szy,
  author        = {Lei, Jin},
  title         = {High-Dimensional Bayesian Calibration of Expensive Nuclear Models with Differentiable Emulation},
  year          = {2026},
  eprint        = {2605.30980},
  archivePrefix = {arXiv},
  primaryClass  = {nucl-th}
}
```

The reduced-basis construction for coupled-channels scattering on which it builds is

```bibtex
@article{Lei:2025rb,
  author  = {Lei, Jin},
  title   = {Reduced basis emulator for elastic scattering in continuum-discretized coupled-channels calculations},
  journal = {Phys. Rev. C},
  volume  = {113},
  pages   = {044610},
  year    = {2026},
  doi     = {10.1103/n24x-d9gm},
  eprint  = {2512.17687},
  archivePrefix = {arXiv}
}
```

If you use the worked example, please also cite jitr.

## License and contact

BSD-3-Clause, see [LICENSE](LICENSE). Release notes: [CHANGELOG](CHANGELOG.md).
Questions and bug reports: [SUPPORT](SUPPORT.md), Jin Lei, Tongji University,
jinl@tongji.edu.cn.
