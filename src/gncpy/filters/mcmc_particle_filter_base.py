import abc

from gncpy.filters.particle_filter import ParticleFilter


class MCMCParticleFilterBase(ParticleFilter):
    """Generic base class for Particle filters with an optional Markov Chain Monte Carlo move step.

    Attributes
    ----------
    use_MCMC : bool, optional
        Flag indicating if the move step is run. The default is False.
    """

    require_copy_can_dist = True

    def __init__(self, use_MCMC=False, **kwargs):
        self.use_MCMC = use_MCMC

        super().__init__(**kwargs)

    @abc.abstractmethod
    def move_particles(self, timestep, meas, old_weights, **kwargs):
        """Generic interface for the movement function.

        This must be overridden in the child class. It is recommended to keep
        the same function signature to allow for standardized wrappers.
        """
        raise RuntimeError("Must implement thid function in derived class")
