"""Defines custom exceptions and errors for algorithm specific problems."""


class ParticleDepletionError(Exception):
    """Thrown when a PF has depleted all its particles."""

    pass
