"""Defines custom exceptions and errors for algorithm specific problems."""


class ParticleDepletionError(Exception):
    """Thrown when a PF has depleted all its particles."""

    pass


class ExtremeMeasurementNoiseError(Exception):
    """Thrown when the estimated measurement noise covariance is ill formed.

    This may happen when bad measurements are used for estimation, as may be
    the case when a filter is used as the inner filter for a Generalized
    Labeled Multi-Bernoulli (GLMB) filter.
    """

    pass
