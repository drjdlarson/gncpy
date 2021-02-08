import numpy as np
import numpy.linalg as la
from warnings import warn
from copy import deepcopy

import gncpy.math as gmath


class SigmaPoints():
    """ Helper class that defines sigma points.

    Args:
        state0 (n x 1 numpy array): Initial state.
        alpha (float): Tunig parameter, influences the spread of sigma
            points about the mean. In range (0, 1].
        kappa (float): Tunig parameter, influences the spread of sigma
            points about the mean. In range [0, inf].
        beta (float, optional): Tunig parameter for distribution type.
            In range [0, Inf]. Defaults to 2 for gaussians.
    """

    def __init__(self, **kwargs):
        self.weights_mean = kwargs.get('weights_mean', [])
        self.weights_cov = kwargs.get('weights_cov', [])
        self.alpha = kwargs.get('alpha', 1)
        self.kappa = kwargs.get('kappa', 0)
        self.beta = kwargs.get('beta', 2)
        self.n = kwargs.get('n', 0)
        self.points = kwargs.get('points', [])

    @property
    def lam(self):
        return self.alpha**2 * (self.n + self.kappa) - self.n

    @property
    def mean(self):
        return gmath.weighted_sum_vec(self.weights_mean, self.points)

    @property
    def cov(self):
        x_bar = self.mean
        cov_lst = [(x - x_bar) @ (x - x_bar).T for x in self.points]
        return gmath.weighted_sum_mat(self.weights_cov, cov_lst)

    def init_weights(self):
        lam = self.lam
        self.weights_mean = [lam / (self.n + lam)]
        self.weights_cov = [lam / (self.n + lam)
                            + 1 - self.alpha**2 + self.beta]
        w = 1 / (2 * (self.n + lam))
        for ii in range(1, 2 * self.n + 1):
            self.weights_mean.append(w)
            self.weights_cov.append(w)

    def update_points(self, x, cov):
        S = la.cholesky((self.n + self.lam) * cov)

        self.points = [x]

        for ii in range(0, self.n):
            self.points.append(x + S[:, [ii]])

        for ii in range(self.n, 2 * self.n):
            self.points.append(x - S[:, [ii - self.n]])


class Particle:
    def __init__(self, **kwargs):
        self.point = kwargs.get('point', np.array([[]]))
        self.uncertainty = kwargs.get('uncertainty', np.array([[]]))

        self.sigmaPoints = kwargs.get('sigmaPoints', None)

    @property
    def mean(self):
        if self.sigmaPoints is not None:
            ref = deepcopy(self.sigmaPoints)
            ref.update_points(self.point, self.uncertainty)
            return ref.mean
        else:
            return self.point

    @mean.setter
    def mean(self, x):
        warn('Particle mean is read only')


class ParticleDistIter:
    def __init__(self, partDist):
        self._partDist = partDist
        self._index = 0

    def __next__(self):
        if self._index < self._partDist.num_particles:
            p = self._partDist._particles[self._index]
            w = self._partDist.weights[self._index]
            res = (p, w)
            self._index += 1
            return res
        raise StopIteration


class ParticleDistribution:
    """ Particle distribution object

    Helper class for managing arbitrary distributions of particles.

    Attributes:
        particles (list): List of 2d numpy arrays, one per particle
        weights (list): List of weights, one per particle
    """

    def __init__(self, **kwargs):
        self._particles = []
        self._weights = []

    @property
    def particles(self):
        if self.num_particles > 0:
            return [x.point for x in self._particles]
        else:
            return self._particles

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, lst):
        warn('Use function to add weight. SKIPPING')

    @particles.setter
    def particles(self, lst):
        warn('Use function to add particle. SKIPPING')

    @property
    def uncertainties(self):
        if self.num_particles > 0 and isinstance(self._particles[0], Particle):
            return [x.uncertainty for x in self._particles]
        else:
            return []

    def add_particle(self, p, w):
        self._particles.append(p)
        self._weights.append(w)

    def clear_particles(self):
        self._particles = []
        self._weights = []

    def update_weights(self, w_lst):
        if len(w_lst) != self.num_particles:
            warn('Different number of weights than particles')
        else:
            self._weights = w_lst

    @property
    def mean(self):
        """ Mean of the particles
        """
        return gmath.weighted_sum_vec(self.weights, self.particles)

    @property
    def covariance(self):
        """Covariance of the particles

        """
        if self.num_particles == 0:
            cov = np.array([[]])
        else:
            x_dim = self.particles[0].size
            cov = np.cov(np.hstack(self.particles)).reshape((x_dim, x_dim))
        return cov

    @property
    def num_particles(self):
        """ Number of particles
        """
        return len(self._particles)

    def __iter__(self):
        return ParticleDistIter(self)
