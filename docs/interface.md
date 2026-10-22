# Adapter interface

The only adapter contract is `dream.problem.Problem`, a structural Python
Protocol. An adapter need not inherit it. The complete scalar synthetic example
in the [README](../README.md) is the smallest working implementation; the
larger deterministic fixture in `tests/synthetic.py` exercises complex systems,
unequal block sizes, sources, and shared geometry.

## Methods and shapes

| Member | Meaning |
|---|---|
| `param_names: list[str]` | Unique names in the physical input-vector order |
| `n_blocks: int` | Number of independent systems, at least one |
| `components() -> list[Component]` | Ordered additive operator/source definitions |
| `fixed(block) -> (M0, s0)` | Parameter-independent arrays `(N, N)` and `(N,)` |
| `component(f, block, geometry) -> (dM, ds)` | Component **integer index** `f`, at its reference strength and supplied geometry values |
| `solve(omega, block) -> c` | Full solver, shape `(N,)`, solving the exported system |
| `functionals(block) -> L` | Linear observation matrix `(n_func, N)`, applied as `L @ c` |
| `observable(z, omega) -> y` | JAX code returning one real vector |

Each block may have a different `N` and number of functionals. The observable
receives complex `z` of shape `(n_blocks, max_n_func)`, padded with zeros on the
right. No conjugation is applied to `L`. Arrays may be real or complex floating
values; integer, object, and nonfinite arrays are rejected by validation.
The observable must remain JAX-traceable and differentiable where derivatives
are needed. User physics belongs in this callback and the adapter.

`validate_problem(problem, omega)` checks component metadata, array shapes and
types, and compares each adapter solution against a separately assembled
`numpy.linalg.solve`, with relative solution error at most 1e-8. It also traces
the observable. This checks one point, not the whole parameter domain or the
model's physical validity. Call it before generating many snapshots.

## Components

`Component` is a frozen dataclass with fields:

```python
Component(name, strength, geometry, reference=1.0, grid=())
```

`strength` is an input index or `None`; `geometry` contains zero, one, or two
input indices. The component function receives the geometry **values** in that
order. A component with a strength has multiplier `omega[strength]/reference`;
otherwise its multiplier is one. `reference` must be nonzero. A grid contains
one `(low, high, count)` tuple per geometry coordinate, with `low < high` and
`count >= 2`. A zero-dimensional component has `geometry=(), grid=()` and is
evaluated once. Components can share strength or geometry indices. Shared
geometry priors must fit the intersection of all associated grid domains.

For example, for `param_names=['depth', 'radius', 'width']`, a two-coordinate
component can be defined as
`Component('profile', 0, (1, 2), 50.0, ((1.0, 1.5, 16), (0.4, 0.8, 16)))`.
Its exported increment includes the reference depth 50; the online multiplier
is `depth/50`. A second component can use the same radius index 1 without
introducing another sampled coordinate.

## Offline construction and storage

`latin_hypercube(lo, hi, n, seed)` returns a reproducible sample matrix with one
sample in each stratum of each parameter. Users may supply any finite sample
matrix instead. `build_basis(problem, samples, rank=k)` selects up to `k`
columns in each block. Alternatively, `rtol=t` keeps singular values strictly
larger than `t` times the leading value, retaining at least one column. With
neither option, every available column is retained. Singular spectra and
ordered basis arrays are available as `basis.singular_values` and `basis.X`.

The default projection is `projection='hermitian'`; `projection='transpose'`
is available for testing complex symmetric problems. Compare both against the
full solver in the configuration being used. `build_database(problem, basis,
mode_rtol=1e-10)` constructs compressed reduced components. Matrix and source
compression use separate SVDs so their scales do not determine each other's
mode counts. The tolerance is a relative singular-value cutoff, not a bound
on observable error.

`dream.database.save(data, path)` writes HDF5 schema 1. `load(path)` validates
the schema and component-definition hash; `load(path, components=problem.components())`
also checks against the current adapter definitions. The file stores the DREAM
version, parameter names, projection, grids, means, modes, and coefficients.
The hash is SHA-256 of the ordered component definitions' representation.
It does not authenticate adapter code or observables. Supply the matching
observable when constructing `Emulator(load(path), problem.observable)`;
executable code is never serialized.

## Online use, verification and inference

`emulator.predict(omega)` uses JIT compilation and returns float64 observables.
`functionals(omega)` returns padded functionals; `reduced_solution(omega)`
returns a list of unpadded coefficient vectors. `rank` returns each active
block rank. `truncate(k)` slices the stored tensors, capping at each block's
rank. It does not revisit the full solver or recompress the components.
An optional nonnegative `regularization` adds an absolute diagonal shift to
active reduced rows. It defaults to zero and changes the approximation.

`gate(emulator, problem, points, sigma)` compares both predictions using the
same observable, returning `chi2_emu`, `max_resid`, `passed`, and
`failed_indices`. A sum-of-squares error below 1 and every pointwise error
below 0.5 data-error units are the defaults. Supply posterior points, including
the posterior mean, to check the inference region. The gate fails on nonfinite
results and rejects invalid error bars. It measures emulator error relative
to the configured full solver, not full-solver convergence or model accuracy.

`rank_sweep(..., ranks=[4, 8, 16], n_rep=20)` returns error metrics and median
seconds for forward and likelihood value-and-gradient evaluations. Timings
synchronize JAX and exclude two warmups and all full-solver work.
`dream.plotting.plot_rank_sweep(result, path)` writes a closed-frame plot
without requiring a graphical display.

`dream.inference.log_likelihood(emulator, y, sigma, transform=None)` returns a
jitted diagonal-Gaussian function; use `jax.value_and_grad` for its value and
gradient. `run_nuts` accepts scalar NumPyro priors, uses a dense mass matrix,
and returns the flattened samples. The entry `_edge_fraction` gives the share of
draws whose geometry lies within one grid spacing of a grid boundary (or beyond
it); a warning is logged when this share exceeds one percent.

Without a transform, prior keys must exactly match `param_names`, and dictionary
order does not matter. With `transform(theta) -> omega`, theta follows prior
insertion order. The map must use JAX. Generic transforms have no inferable
image bounds: for a model with geometry, they require the explicit
`allow_extrapolation=True` override after the caller proves the mapped domain
in user code. The edge check still uses the transformed physical inputs.
This limitation is preferable to claiming a support guarantee from a finite
sample of an arbitrary nonlinear map.

Example direct-coordinate priors can use
`dist.TruncatedNormal(center, scale, low=grid_low, high=grid_high)` for geometry.
An unbounded normal geometry prior is refused. Array-valued priors and names
beginning with `_` are unsupported. Multiple chains run sequentially, suitable
for a single CPU device. The thin helper does not certify convergence; inspect
chains and use independent convergence and calibration diagnostics for research.
