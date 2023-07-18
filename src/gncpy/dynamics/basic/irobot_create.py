import numpy as np
from .nonlinear_dynamics_base import NonlinearDynamicsBase


class IRobotCreate(NonlinearDynamicsBase):
    """A differential drive robot based on the iRobot Create.

    This has a control model predefined because the dynamics themselves do not
    change the state.

    Notes
    -----
    This is taken from :cite:`Berg2016_ExtendedLQRLocallyOptimalFeedbackControlforSystemswithNonLinearDynamicsandNonQuadraticCost`
    It represents a 2 wheel robot with some distance between its wheels.
    """

    state_names = ("pos_x", "pos_v", "turn_angle")

    def __init__(self, wheel_separation=0.258, radius=0.335 / 2, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        wheel_separation : float, optional
            Distance between the two wheels in meters.
        radius : float
            Radius of the bounding box for the robot in meters.
        **kwargs : dict
            Additional arguments for the parent class.
        """
        super().__init__(**kwargs)
        self._wheel_separation = wheel_separation
        self.radius = radius

        def g0(t, x, u, *args):
            return 0.5 * ((u[0] + u[1]) * np.cos(x[2])).item()

        def g1(t, x, u, *args):
            return 0.5 * ((u[0] + u[1]) * np.sin(x[2])).item()

        def g2(t, x, u, *args):
            return (u[1] - u[0]) / self.wheel_separation

        self._control_model = [g0, g1, g2]

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, model):
        self._control_model = model

    @property
    def wheel_separation(self):
        """Read only wheel separation distance."""
        return self._wheel_separation

    @property
    def cont_fnc_lst(self):
        """Implements the contiuous time dynamics."""

        def f0(t, x, *args):
            return 0

        def f1(t, x, *args):
            return 0

        def f2(t, x, *args):
            return 0

        return [f0, f1, f2]
