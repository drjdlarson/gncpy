import numpy as np
from .nonlinear_dynamics_base import NonlinearDynamicsBase


class KarlgaardOrbit(NonlinearDynamicsBase):
    """Implements the non-linear Karlgaar elliptical orbit model.

    Notes
    -----
    It uses the numerical integration of the second order approximation in
    dimensionless sphereical coordinates. See
    :cite:`Karlgaard2003_SecondOrderRelativeMotionEquations` for details.
    """

    state_names = (
        "non-dim radius",
        "non-dim az angle",
        "non-dim elv angle",
        "non-dim radius ROC",
        "non-dim az angle ROC",
        "non-dim elv angle ROC",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, model):
        self._control_model = model

    @property
    def cont_fnc_lst(self):
        """Continuous time dynamics.

        Returns
        -------
        list
            functions of the form :code:`(t, x, *args)`.
        """

        # returns non-dim radius ROC
        def f0(t, x, *args):
            return x[3]

        # returns non-dim az angle ROC
        def f1(t, x, *args):
            return x[4]

        # returns non-dim elv angle ROC
        def f2(t, x, *args):
            return x[5]

        # returns non-dim radius ROC ROC
        def f3(t, x, *args):
            r = x[0]
            phi = x[2]
            theta_d = x[4]
            phi_d = x[5]
            return (
                (-3 * r**2 + 2 * r * theta_d - phi**2 + theta_d**2 + phi_d**2)
                + 3 * r
                + 2 * theta_d
            )

        # returns non-dim az angle ROC ROC
        def f4(t, x, *args):
            r = x[0]
            theta = x[1]
            r_d = x[3]
            theta_d = x[4]
            return (2 * r * r_d + 2 * theta * theta_d - 2 * theta_d * r_d) - 2 * r_d

        # returns non-dim elv angle ROC ROC
        def f5(t, x, *args):
            phi = x[2]
            r_d = x[3]
            theta_d = x[4]
            phi_d = x[5]
            return (-2 * theta_d * phi - 2 * phi_d * r_d) - phi

        return [f0, f1, f2, f3, f4, f5]
