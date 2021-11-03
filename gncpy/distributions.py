"""Standard distributions for use with the package classes."""
import numpy as np
import numpy.linalg as la
import numpy.polynomial.hermite_e as herm_e
import numpy.random as rnd
from warnings import warn
import itertools
import matplotlib.pyplot as plt
import enum

import gncpy.math as gmath
import gncpy.plotting as pltUtil


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
    r"""Helper class that defines quadrature points.

    Notes
    -----
    This implements the Probabilist's version of the Gauss-Hermite quadrature
    points. This consists of the Hermite polynomial

    .. math::
        H_{e_n}(x) = (-1)^n \exp{\frac{x^2}{2}} \frac{\partial^n}{\partial x^n} \exp{-\frac{x^2}{2}}

    and its associated weights. For details see
    :cite:`Press1992_NumericalRecipesinCtheArtofScientificComputing`,
    :cite:`Golub1969_CalculationofGaussQuadratureRules` and for a multi-variate
    extension :cite:`Jackel2005_ANoteonMultivariateGaussHermiteQuadrature`.

    Attributes
    ----------
    points_per_axis : int
        Number of points to use per axis
    num_axes : int
        Number of axis in each point. This can be set manually, but will be updated
        when :meth:`.update_points` is called to match the supplied mean.
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
        """Read only expected number of points."""
        return int(self.points_per_axis**self.num_axes)

    @property
    def mean(self):
        """Mean of the points, accounting for the weights."""
        return gmath.weighted_sum_vec(self.weights,
                                      self.points).reshape((self.num_axes, 1))

    @property
    def cov(self):
        """Covariance of the points, accounting for the weights."""
        x_bar = self.mean
        diff = (self.points - x_bar.ravel()).reshape(self.points.shape[0], x_bar.size, 1)
        return gmath.weighted_sum_mat(self.weights,
                                      diff @ diff.reshape(self.points.shape[0],
                                                          1, self.num_axes))

    def _factor_scale_matrix(self, scale, have_sqrt):
        if have_sqrt:
            return scale
        else:
            return la.cholesky(scale)

    def update_points(self, mean, scale, have_sqrt=False):
        """Updates the quadrature points given some initial point and scale.

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
        def create_combos(points_per_ax, num_ax, tot_points):
            combos = np.meshgrid(*itertools.repeat(range(points_per_ax), num_ax))

            return np.array(combos).reshape((num_ax, tot_points)).T

        self.num_axes = mean.size

        sqrt_cov = self._factor_scale_matrix(scale, have_sqrt)

        self.points = np.nan * np.ones((self.num_points, self.num_axes))
        self.weights = np.nan * np.ones(self.num_points)

        # get standard values for 1 axis case, use "Probabilist's" Hermite polynomials
        quad_points, weights = herm_e.hermegauss(self.points_per_axis)

        ind_combos = create_combos(self.points_per_axis, self.num_axes,
                                   self.num_points)

        for ii, inds in enumerate(ind_combos):
            point = quad_points[inds].reshape((inds.size, 1))
            self.points[ii, :] = (mean + sqrt_cov @ point).ravel()
            self.weights[ii] = np.prod(weights[inds])

        self.weights = self.weights / np.sum(self.weights)

        # note: Sum[w * p @ p.T] should equal the identity matrix here

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

    def plot_points(self, inds, x_lbl='X Position', y_lbl='Y Position',
                    ttl='Weighted Positions', size_factor=100**2, **kwargs):
        """Plots the weighted points.

        Keywrod arguments are processed with
        :meth:`gncpy.plotting.init_plotting_opts`. This function
        implements

            - f_hndl
            - Any title/axis/text options

        Parameters
        ----------
        inds : list or int
            Indices of the point vector to plot. Can be a list of at most 2
            elements. If only 1 is given a bar chart is created.
        x_lbl : string, optional
            Label for the x-axis. The default is 'X Position'.
        y_lbl : string, optional
            Label for the y-axis. The default is 'Y Position'.
        ttl : string, optional
            Title of the plot. The default is 'Weighted positions'.
        size_factor : int, optional
            Factor to multiply the weight by when determining the marker size.
            Only used if plotting 2 indices. The default is 100**2.
        **kwargs : dict
            Additional standard plotting options.

        Returns
        -------
        fig : matplotlib figure handle
            Handle to the figure used.
        """
        opts = pltUtil.init_plotting_opts(**kwargs)
        fig = opts['f_hndl']

        if fig is None:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

        if isinstance(inds, list):
            if len(inds) >= 2:
                fig.axes[0].scatter(self.points[:, inds[0]],
                                    self.points[:, inds[1]],
                                    s=size_factor * self.weights, color='k')
                fig.axes[0].grid(True)

            elif len(inds) == 1:
                fig.axes[0].bar(self.points[:, inds[0]],
                                self.weights)
        else:
            fig.axes[0].bar(self.points[:, inds[0]],
                            self.weights)

        pltUtil.set_title_label(fig, 0, opts, ttl=ttl, x_lbl=x_lbl,
                                y_lbl=y_lbl)

        fig.tight_layout()

        return fig


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


class GSMTypes(enum.Enum):
    STUDENTS_T = enum.auto()
    SYMMETRIC_A_STABLE = enum.auto()

    def __str__(self):
        """Return the enum name for strings."""
        return self.name


class GaussianScaleMixture:
    def __init__(self):
        self.type = None

    def sample(self, rng=None):
        """Draw a sample from the specified GSM type.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        float
            randomly sampled value from the GSM.
        """
        if rng is None:
            rng = rnd.default_rng()

        if self.type is GSMTypes.STUDENTS_T:
            return self._sample_student_t(rng)

        elif self.type is GSMTypes.SYMMETRIC_A_STABLE:
            return self._sample_SaS(rng)

        else:
            raise RuntimeError('GSM type: {} is not supported'.format(self.type))

    def _sample_student_t(self, rng):
        pass

    def _sample_SaS(self, rng):
        pass
