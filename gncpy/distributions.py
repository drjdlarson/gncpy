"""Standard distributions for use with the package classes."""
import numpy as np
import numpy.linalg as la
from warnings import warn

import gncpy.math as gmath


class SigmaPoints():
    """Helper class that defines sigma points.

    Attributes
    ----------
    weights_mean : list
        List of integer weights for calculating the mean of the points.
    weights_cov : list
        List of integer weights for calculating the covariance of the points.
    alpha : float, optional
        Tunig parameter, influences the spread of sigma points about the mean.
        In range (0, 1]. The default is 1.
    kappa : float, optional
        Tunig parameter, influences the spread of sigma points about the mean.
        In range [0, inf]. The default is 0.
    beta : float, optional
        Tunig parameter for distribution type. In range [0, Inf]. Defaults
        to 2 which is ideal for Gaussians.
    n : int
        Length of a single point vector.
    points : list
        List of N x 1 numpy arrays representing the sigma points.
    """

    def __init__(self, alpha=1, kappa=0, beta=2, n=0):
        self.weights_mean = np.array([])
        self.weights_cov = np.array([])
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.n = n
        self.points = []

    @property
    def lam(self):
        """Read only derived parameter of the sigma points."""
        return self.alpha**2 * (self.n + self.kappa) - self.n

    @property
    def mean(self):
        """Mean of the points, accounting for the weights."""
        return gmath.weighted_sum_vec(self.weights_mean,
                                      self.points).reshape((self.points.shape[1], 1))

    @property
    def cov(self):
        """Covariance of the points, accounting for the weights."""
        x_bar = self.mean
        diff = (self.points - x_bar.ravel()).reshape(self.points.shape[0], x_bar.size, 1)
        return gmath.weighted_sum_mat(self.weights_cov,
                                      diff @ diff.reshape(self.points.shape[0],
                                                          1, x_bar.size))

    def init_weights(self):
        """Initializes the weights based on other parameters.

        This should be called to setup the weight vectors after setting
        `alpha`, `kappa`, `beta`, and `n`.
        """
        lam = self.lam
        self.weights_mean = np.nan * np.ones(2 * self.n + 1)
        self.weights_cov = np.nan * np.ones(2 * self.n + 1)
        self.weights_mean[0] = lam / (self.n + lam)
        self.weights_cov[0] = lam / (self.n + lam) + 1 - self.alpha**2 \
            + self.beta

        w = 1 / (2 * (self.n + lam))
        self.weights_mean[1:] = w
        self.weights_cov[1:] = w

    def update_points(self, x, cov):
        """Updates the sigma points given some initial point and covariance.

        Parameters
        ----------
        x : N x 1 numpy array
            Point to represent by sigma points.
        cov : N x N numpy array
            Covariance matrix of the given point.

        Returns
        -------
        None.
        """
        loc_cov = cov
        S = la.cholesky((self.n + self.lam) * loc_cov)

        # self.points = [None] * (2 * self.n + 1)
        self.points = np.nan * np.ones((2 * self.n + 1, x.size))
        self.points[0, :] = x.flatten()
        self.points[1:self.n + 1, :] = x.ravel() + S.T
        self.points[self.n + 1:, :] = x.ravel() - S.T
        # self.points[0] = x

        # for ii in range(0, self.n):
        #     self.points[ii + 1] = x + S[:, [ii]]

        # for ii in range(self.n, 2 * self.n):
        #     self.points[ii + 1] = x - S[:, [ii - self.n]]

        # brk = 1


class Particle:
    """Helper class for defining single particles in a particle distribution.

    Attributes
    ----------
    point : N x 1 numpy array
        The location of the particle.
    uncertainty : N x N numpy array, optional
        The uncertainty of the point, this does not always need to be specified.
        Check with the filter if this is needed.
    sigmaPoints : :class:`.distributions.SigmaPoints`, optional
        Sigma points used to represent the particle. This is not always needed,
        check with the filter to determine if this is necessary.
    """

    def __init__(self, point=np.array([[]]), uncertainty=np.array([[]]),
                 sigmaPoints=None):
        self.point = point
        self.uncertainty = uncertainty

        self.sigmaPoints = sigmaPoints

    @property
    def mean(self):
        """Read only mean value of the particle.

        If no sigma points are used then it is the same as `point`. Otherwise
        it is the mean of the sigma points.

        Returns
        -------
        N x 1 numpy array
            The mean value.
        """
        if self.sigmaPoints is not None:
            # self.sigmaPoints.update_points(self.point, self.uncertainty)
            return self.sigmaPoints.mean
        else:
            return self.point

    @mean.setter
    def mean(self, x):
        warn('Particle mean is read only')


class _ParticleDistIter:
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
    """Particle distribution object.

    Helper class for managing arbitrary distributions of particles.
    """

    def __init__(self, **kwargs):
        self._particles = []
        self._weights = []

    @property
    def particles(self):
        """Particles in the distribution.

        Must be set by the :meth:`.distributions.ParticleDistribution.add_particle`
        method.

        Returns
        -------
        list
            Each element is a :class:`.distributions.Particle` object.
        """
        if self.num_particles > 0:
            return [x.mean for x in self._particles]
        else:
            return self._particles

    @property
    def weights(self):
        """Weights of the partilces.

        Must be set by the :meth:`.distributions.ParticleDistribution.add_particle`
        or :meth:`.distributions.ParticleDistribution.update_weights` methods.

        Returns
        -------
        list
            Each element is a float representing the weight of the particle.
        """
        return self._weights

    @weights.setter
    def weights(self, lst):
        raise RuntimeError('Use function to add weight.')

    @particles.setter
    def particles(self, lst):
        raise RuntimeError('Use function to add particle.')

    @property
    def uncertainties(self):
        """Read only uncertainty of each particle.

        Returns
        -------
        list
            Each element is a N x N numpy array
        """
        return [x.uncertainty for x in self._particles]

    def add_particle(self, p, w):
        """Adds a particle and weight to the distribution.

        Parameters
        ----------
        p : :class:`.distributions.Particle` or list
            Particle to add or list of particles.
        w : float or list
            Weight of the particle or list of weights.

        Returns
        -------
        None.
        """
        if isinstance(p, list):
            self._particles.extend(p)
        else:
            self._particles.append(p)

        if isinstance(w, list):
            self._weights.extend(w)
        else:
            self._weights.append(w)

    def clear_particles(self):
        """Clears the particle and weight lists."""
        self._particles = []
        self._weights = []

    def update_weights(self, w_lst):
        """Updates the weights to match the given list.

        Checks that the lenght of the weights matches the number of particles.
        """
        if len(w_lst) != self.num_particles:
            warn('Different number of weights than particles')
        else:
            self._weights = w_lst

    @property
    def mean(self):
        """Mean of the particles."""
        if any(np.abs(self.weights) == np.inf):
            if self.num_particles > 0:
                mean = np.zeros(self.particles[0].shape)
            else:
                mean = np.array([[]])
        else:
            mean = gmath.weighted_sum_vec(self.weights, self.particles)
        return mean

    @property
    def covariance(self):
        """Covariance of the particles."""
        if self.num_particles == 0:
            cov = np.array([[]])
        else:
            x_dim = self.particles[0].size
            cov = np.cov(np.hstack(self.particles)).reshape((x_dim, x_dim))
            cov = (cov + cov.T) * 0.5
        return cov

    @property
    def num_particles(self):
        """Number of particles."""
        return len(self._particles)

    def __iter__(self):
        """Custom iterator for looping over the object.

        Returns
        -------
        :class:`.distributions.Particle`
            Current particle in distribution.
        float
            Weight of the current particle.
        """
        return _ParticleDistIter(self)
