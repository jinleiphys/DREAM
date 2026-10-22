# Reduced operators and differentiable solves

For independent blocks b, the adapter exports

$$
M_b(\omega)=M_{0b}+\sum_f w_f(\omega)\,\Delta M_{fb}(g_f),\qquad
s_b(\omega)=s_{0b}+\sum_f w_f(\omega)\,\Delta s_{fb}(g_f).
$$

Here `w_f = omega[strength]/reference`, or 1 without a strength parameter.
The geometry `g_f` contains zero, one, or two physical input coordinates.
Strength and geometry indices can be shared by different components.

## Offline stage

Full solutions at chosen parameter points form the columns of a snapshot
matrix `C_b`. Its complex SVD is `C_b = U_b diag(s_b) V_b^H`. The first `k_b`
columns of `U_b` define an orthonormal solution basis `X_b`, ordered by singular
value. Choosing a snapshot tolerance controls this basis approximation; it does
not by itself bound an observable error.

Let `P_b = X_b^H` for the default Hermitian projection, or `P_b = X_b^T` for the
optional transpose projection. Reduced arrays are

$$
\widehat M_{0b}=P_b M_{0b}X_b,\quad
\widehat s_{0b}=P_b s_{0b},\quad \widehat L_b=L_bX_b.
$$

Each component is projected on its regular geometry grid. The matrix and source
are compressed separately. Flatten a grid tensor into rows `D_i`, subtract its
mean, and compute `D - mean = U diag(s) V^H`. Retained rows of `V^H` are the modes
and `U diag(s)` contains their grid coefficients. Thus

$$
\widehat{\Delta M}_{fb}(g)\simeq \overline M_{fb}
 +\sum_m \gamma_{fbm}(g) B_{fbm}.
$$

The source uses the same construction with its own mean, modes and coefficients.
Both modes and coefficients may be complex. The retained singular values exceed
`mode_rtol` times the leading singular value. Constant arrays require zero
modes, only their mean. For no geometry, there is one component evaluation.

## Online stage and derivatives

Multilinear interpolation supplies the coefficients at the requested geometry.
The small reduced systems are assembled and solved in JAX, then

$$
\widehat M_b(\omega)c_b^{\rm red}=\widehat s_b(\omega),\qquad
z_b=\widehat L_b c_b^{\rm red},\qquad y=\mathcal O(z,\omega).
$$

Only the adapter's observable defines the physical meaning of `z`. It may also
depend explicitly on `omega`. Unequal block ranks are padded internally with
identity matrix rows and zero sources; the active coefficients are unchanged.
The linear solves are vmapped over blocks. Optional diagonal regularization is
zero by default; a nonzero shift changes the solved problem.

JAX differentiates this *emulated* map, including its solves and observable.
Within a grid cell the interpolation is differentiable. Derivatives usually
jump at interior grid knots; the implementation selects the cell on the right.
At a clamping endpoint, JAX's clip convention gives half the interior derivative.
Outside the grid the interpolation is constant and its geometry derivative is
zero. These derivatives are not a proof of agreement with derivatives of the
full solver. Finite-difference tests compare AD to the emulator's own derivative.

## Truncation, errors and timing

Because each object is a projection onto ordered columns, the leading rank-k
system is obtained by slicing its matrix rows/columns, sources, functionals and
modes. This is exact for the stored approximation. A separate rebuild can differ
by floating-point effects and its independently chosen component-SVD cutoff;
with untruncated component modes, the tests verify equality near machine
precision. This distinction matters if an aggressive compression tolerance is
used. Increasing solution rank cannot remove geometry-grid interpolation error.

At validation point p, the two reported errors are

$$
\chi^2_{\rm emu,p}=\sum_i\left[\frac{y_{\rm emu,i}-y_{\rm direct,i}}
 {\sigma_i}\right]^2,\qquad
r_{\max,p}=\max_i\frac{|y_{\rm emu,i}-y_{\rm direct,i}|}{\sigma_i}.
$$

The first is a sum, not a reduced chi-square. Default acceptance requires both
`chi2_emu < 1` and `r_max < 0.5` at every supplied point. The reference and emulator
use identical observable post-processing. The example gates the posterior mean
and eight draws from its own posterior and also reports S-matrix errors that do
not depend on data uncertainties. Agreement refers to that solver configuration.

Rank sweeps reuse the same reference solutions. Forward and Gaussian
value-and-gradient times are synchronized median wall times after two warmup
calls, excluding compilation. They are not total offline or inference cost.
The example also records full pipeline wall time, including compilation and
plot export but excluding Python startup and module imports.

## References

1. Jin Lei, *High-Dimensional Bayesian Calibration of Expensive Nuclear Models
   with Differentiable Emulation*, [arXiv:2605.30980](https://arxiv.org/abs/2605.30980)
   (2026).
2. Jin Lei, *Reduced basis emulator for elastic scattering in
   continuum-discretized coupled-channels calculations*, Physical Review C
   **113**, 044610 (2026), [doi:10.1103/n24x-d9gm](https://doi.org/10.1103/n24x-d9gm),
   [arXiv:2512.17687](https://arxiv.org/abs/2512.17687).

