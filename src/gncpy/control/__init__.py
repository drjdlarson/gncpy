"""Implements control algorithms and models.

Some algorithms may also be used for path planning.
"""
from gncpy.control._control import StateControl, StateControlParams
from .lqr import LQR
from .elqr import ELQR
from .pronav import Pronav
