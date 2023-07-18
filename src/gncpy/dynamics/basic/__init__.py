"""Basic models and classes that can be extended.

These provide an easy to use interface for some common dynamic objects and
and their associated models. They have been designed to integrate well with the
filters in :mod:`gncpy.filters`.
"""

from .clohessy_wiltshire_orbit2d import ClohessyWiltshireOrbit2d
from .clohessy_wiltshire_orbit import ClohessyWiltshireOrbit
from .coordinated_turn_known import CoordinatedTurnKnown
from .coordinated_turn_unknown import CoordinatedTurnUnknown
from .curvilinear_motion import CurvilinearMotion
from .double_integrator import DoubleIntegrator
from .dynamics_base import DynamicsBase
from .irobot_create import IRobotCreate
from .karlgaard_orbit import KarlgaardOrbit
from .linear_dynamics_base import LinearDynamicsBase
from .nonlinear_dynamics_base import NonlinearDynamicsBase
from .tschauner_hempel_orbit import TschaunerHempelOrbit
