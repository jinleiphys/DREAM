"""Six-parameter, spin-independent neutron elastic-scattering adapter."""

import jax.numpy as jnp
import numpy as np
from jitr.reactions.system import ProjectileTargetSystem
from jitr.rmatrix import Solver
from jitr.utils.constants import AMU, MASS_N
from jitr.utils.kinematics import classical_kinematics

from dream.problem import Component


class JitrOpticalProblem:
    """Illustrative real volume and absorptive surface potentials, in MeV and fm.

    The unknown is the boundary response c = A^-1 b, not the physically
    normalized scattering wave. jitr returns that wave multiplied by its
    external derivative; dividing by the derivative gives the fixed-source
    system. With dimensionless a = k*r_channel, R = (b.T @ c)/a**2.
    """

    param_names = ['V', 'r_v', 'a_v', 'W_d', 'r_d', 'a_d']

    def __init__(self, target=(40, 20), projectile='neutron', E_lab=14.1,
                 lmax=12, nbasis=40, channel_radius=15., grid_size=16):
        if projectile != 'neutron':
            raise ValueError('this example implements a neutral projectile only')
        self.target = target
        self.n_blocks = lmax + 1
        self.nbasis = nbasis
        self.truth = np.array([48., 1.25, 0.65, 6., 1.30, 0.55])
        spans = np.array([0.2, 0.1, 0.1, 0.2, 0.1, 0.1])
        self.bounds = self.truth[:, None] * np.stack([1 - spans, 1 + spans], axis=1)
        self.grid_size = grid_size
        self.kinematics = classical_kinematics(target[0] * AMU, MASS_N, E_lab)
        self.k = self.kinematics.k
        self.a = self.k * channel_radius
        system = ProjectileTargetSystem(self.a, lmax, mass_target=target[0] * AMU,
                                        mass_projectile=MASS_N, Ztarget=target[1], Zproj=0)
        self.channels, self.asymptotics = system.get_partial_wave_channels(*self.kinematics)
        self.solver = Solver(nbasis)
        self.boundary = self.solver.precompute_boundaries(self.a)
        self.radial = self.solver.radial_grid(self.a, self.k)
        self._free = [self.solver.free_matrix(c.a, c.l, c.E, c.mu) for c in self.channels]
        self._asym = {name: jnp.asarray([getattr(a, name)[0] for a in self.asymptotics])
                      for name in ('Hp', 'Hm', 'Hpp', 'Hmp')}
        self.angles = np.linspace(10., 170., 40)
        self._legendre = jnp.asarray(np.polynomial.legendre.legvander(np.cos(np.deg2rad(self.angles)), lmax).T)
        self._weights = jnp.arange(1, 2 * self.n_blocks, 2)

    def components(self):
        return [Component(name, strength, geometry, self.truth[strength],
                          tuple((*self.bounds[index], self.grid_size) for index in geometry))
                for name, strength, geometry in [('volume', 0, (1, 2)), ('surface', 3, (4, 5))]]

    def fixed(self, block):
        return self._free[block], self.boundary

    def _potential(self, f, geometry):
        radius, diffuseness = geometry
        form = 1 / (1 + np.exp((self.radial - radius * self.target[0]**(1 / 3)) / diffuseness))
        if f == 0:
            return -self.truth[0] * form
        return -4j * self.truth[3] * form * (1 - form)

    def component(self, f, block, geometry):
        matrix = self.solver.interaction_matrix(self.k, self.channels[block].E[0], self.a, 1,
                                                local_potential=self._potential(f, geometry))
        return matrix, np.zeros(self.nbasis, dtype=complex)

    def _jitr_solve(self, omega, block):
        potential = (omega[0] / self.truth[0] * self._potential(0, omega[1:3])
                     + omega[3] / self.truth[3] * self._potential(1, omega[4:6]))
        return self.solver.solve(self.channels[block], self.asymptotics[block], local_potential=potential,
                                 free_matrix=self._free[block], basis_boundary=self.boundary, wavefunction=True)

    def solve(self, omega, block):
        _, _, coefficients, derivative = self._jitr_solve(np.asarray(omega), block)
        if abs(derivative[0]) < np.finfo(float).tiny:
            raise ValueError('cannot remove a zero boundary-derivative normalization')
        return coefficients[0] / derivative[0]

    def functionals(self, block):
        return self.boundary[None, :]

    def smatrix(self, z):
        rmatrix = jnp.asarray(z)[:, 0] / self.a**2
        values = self._asym
        return ((values['Hm'] - self.a * rmatrix * values['Hmp'])
                / (values['Hp'] - self.a * rmatrix * values['Hpp']))

    def observable(self, z, omega):
        amplitude = (self._weights * (self.smatrix(z) - 1)) @ self._legendre / (2j * self.k)
        return 10 * jnp.abs(amplitude)**2  # mb/sr; 1 fm^2 = 10 mb

    def direct_smatrix(self, omega):
        """Independent public-solver S matrix for checking the observable map."""
        return np.asarray([self._jitr_solve(omega, block)[1][0, 0] for block in range(self.n_blocks)])
