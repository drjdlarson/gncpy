import numpy as np

import gncpy.math as gmath


class ParticleDistribution:
    """ Particle distribution object

    Helper class for managing arbitrary distributions of particles.

    Attributes:
        particles (list): List of 2d numpy arrays, one per particle
        weights (list): List of weights, one per particle
    """

    def __init__(self, **kwargs):
        self.particles = kwargs.get('particles', [])
        self.weights = kwargs.get('weights', [])

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
        return len(self.particles)