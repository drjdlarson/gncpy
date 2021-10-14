"""Standard distributions for use with the package classes."""
import numpy as np
import numpy.linalg as la
from warnings import warn

import gncpy.math as gmath


class _QuadPointIter:
    def __init__(self, quadPoints):
        self._quadPoints = quadPoints
        self.__index = 0

    def __next__(self):
        try:
            point = self._quadPoints.points[self.__index, :]
            result = (point.reshape((self._quadPoints.num_axes, 1)),
                      self._quadPoints.weights[self.__index])
        except IndexError:
            raise StopIteration
        self.__index += 1
        return result


class QuadraturePoints:
    """Helper class that defines quadrature points.

    Attributes
    ----------
    points_per_axis : int
        Number of points to use per axis
    num_axes : int
        Number of axis in each point. This can be set manually, but will be updated
        when :meth:`self.update_points` is called to match the supplied mean.
    weights : numpy array
        Weight of each quadrature point
    points : M x N numpy array
        Each row corresponds to a quadrature point and the total number of rows
        is the total number of points.
    """

    def __init__(self, points_per_axis=None, num_axes=None):
        self.points_per_axis = points_per_axis
        self.num_axes = num_axes
        self.points = np.array([[]])
        self.weights = np.array([])

    @property
    def num_points(self):
        """Expected number of points."""
        return int(self.points_per_axis**self.num_axes)

    @property
    def mean(self):
        """Mean of the points, accounting for the weights."""
        return gmath.weighted_sum_vec(self.weights,
                                      self.points).reshape((self.points.shape[1], 1))

    @property
    def cov(self):
        """Covariance of the points, accounting for the weights."""
        x_bar = self.mean
        diff = (self.points - x_bar.ravel()).reshape(self.points.shape[0], x_bar.size, 1)
        return gmath.weighted_sum_mat(self.weights,
                                      diff @ diff.reshape(self.points.shape[0],
                                                          1, x_bar.size))

    def _factor_scale_matrix(self, scale, have_sqrt):
        if have_sqrt:
            return scale
        else:
            return la.cholesky(scale)

    def update_points(self, mean, scale, have_sqrt=False):
        """Updates the sigma points given some initial point and covariance.

        Parameters
        ----------
        mean : N x 1 numpy array
            Point to represent by quadrature points.
        scale : N x N numpy array
            Covariance or square root of the covariance matrix of the given point.
        have_sqrt : bool
            Optional flag indicating if the square root of the matrix was
            supplied. The default is False.

        Returns
        -------
        None.
        """
        self._num_axes = mean.size

        sqrt_cov = self._factor_scale_matrix(scale, have_sqrt)

        self.points = np.nan * np.ones((self.num_points, self.num_axes))

        # TODO: figure out what these should be
        self.weights = np.nan * np.ones(self.num_points)

        # TODO: figure out what these should be
        quad_points = []

        for row, direction in enumerate(quad_points):
            self.points[row, :] = (mean + sqrt_cov @ direction.reshape((self._num_axes, 1))).ravel()

    def __iter__(self):
        """Custom iterator for looping over the object.

        Returns
        -------
        N x 1 numpy array
            Current point in set.
        float
            Weight of the current point.
        """
        return _QuadPointIter(self)


class SigmaPoints(QuadraturePoints):
    """Helper class that defines sigma points.

    Notes
    -----
    This can be interpretted as a speacial case of the Quadrature points. See
    :cite:`Sarkka2015_OntheRelationbetweenGaussianProcessQuadraturesandSigmaPointMethods`
    for details.

    Attributes
    ----------
    alpha : float, optional
        Tunig parameter, influences the spread of sigma points about the mean.
        In range (0, 1]. The default is 1.
    kappa : float, optional
        Tunig parameter, influences the spread of sigma points about the mean.
        In range [0, inf]. The default is 0.
    beta : float, optional
        Tunig parameter for distribution type. In range [0, Inf]. Defaults
        to 2 which is ideal for Gaussians.
    """

    def __init__(self, alpha=1, kappa=0, beta=2, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta

    @property
    def lam(self):
        """Read only derived parameter of the sigma points."""
        return self.alpha**2 * (self.num_axes + self.kappa) - self.num_axes

    @property
    def num_points(self):
        """Read only expected number of points."""
        return int(2 * self.num_axes + 1)

    @property
    def weights_mean(self):
        """Wights for calculating the mean."""
        return self.weights[0:self.num_points]

    @weights_mean.setter
    def weights_mean(self, val):
        if self.weights.size != 2 * self.num_points:
            self.weights = np.nan * np.ones(2 * self.num_points)
        self.weights[0:self.num_points] = val

    @property
    def weights_cov(self):
        """Wights for calculating the covariance."""
        return self.weights[self.num_points:]

    @weights_cov.setter
    def weights_cov(self, val):
        if self.weights.size != 2 * self.num_points:
            self.weights = np.nan * np.ones(2 * self.num_points)
        self.weights[self.num_points:] = val

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
        self.weights_mean = np.nan * np.ones(self.num_points)
        self.weights_cov = np.nan * np.ones(self.num_points)
        self.weights_mean[0] = lam / (self.num_axes + lam)
        self.weights_cov[0] = lam / (self.num_axes + lam) + 1 - self.alpha**2 \
            + self.beta

        w = 1 / (2 * (self.num_axes + lam))
        self.weights_mean[1:] = w
        self.weights_cov[1:] = w

    def update_points(self, x, scale, have_sqrt=False):
        """Updates the sigma points given some initial point and covariance.

        Parameters
        ----------
        x : N x 1 numpy array
            Point to represent by sigma points.
        scale : N x N numpy array
            Covariance or square root of the covariance matrix of the given point.
        have_sqrt : bool
            Optional flag indicating if the square root of the matrix was
            supplied. The default is False.

        Returns
        -------
        None.
        """
        self.num_axes = x.size
        if have_sqrt:
            factor = np.sqrt(self.num_axes + self.lam)
        else:
            factor = self.num_axes + self.lam
        S = self._factor_scale_matrix(factor * scale, have_sqrt)

        self.points = np.nan * np.ones((2 * self.num_axes + 1, self.num_axes))
        self.points[0, :] = x.flatten()
        self.points[1:self.num_axes + 1, :] = x.ravel() + S.T
        self.points[self.num_axes + 1:, :] = x.ravel() - S.T


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
        self.__index = 0

    def __next__(self):
        try:
            result = (self._partDist._particles[self.__index],
                      self._partDist.weights[self.__index])
        except IndexError:
            raise StopIteration
        self.__index += 1
        return result


class ParticleDistribution:
    """Particle distribution object.

    Helper class for managing arbitrary distributions of particles.
    """

    def __init__(self, **kwargs):
        self._particles = []
        self._weights = []

        self.__need_mean_lst_update = True
        self.__need_uncert_lst_update = True
        self.__means = []
        self.__uncertianties = []

        self.__index = 0

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
            if self.__need_mean_lst_update:
                self.__need_mean_lst_update = False
                self.__means = [x.mean for x in self._particles if x]
            return self.__means
        else:
            return []

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
        if self.num_particles > 0:
            if self.__need_uncert_lst_update:
                self.__need_uncert_lst_update = False
                self.__uncertianties = [x.uncertainty for x in self._particles
                                        if x]
            return self.__uncertianties
        else:
            return []

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
        self.__need_mean_lst_update = True
        self.__need_uncert_lst_update = True
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
        self.__need_mean_lst_update = True
        self.__need_uncert_lst_update = True
        self._particles = []
        self._weights = []

    def update_weights(self, w_lst):
        """Updates the weights to match the given list.

        Checks that the length of the weights matches the number of particles.
        """
        self.__need_mean_lst_update = True
        self.__need_uncert_lst_update = True

        if len(w_lst) != self.num_particles:
            warn('Different number of weights than particles')
        else:
            self._weights = w_lst

    @property
    def mean(self):
        """Mean of the particles."""
        if self.num_particles == 0 or any(np.abs(self.weights) == np.inf):
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
