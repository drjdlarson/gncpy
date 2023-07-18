import numpy as np
from .nonlinear_dynamics_base import NonlinearDynamicsBase


class TschaunerHempelOrbit(NonlinearDynamicsBase):
    """Implements the non-linear Tschauner-Hempel elliptical orbit model.

    Notes
    -----
    It is the general elliptical orbit of an object around another target
    object as defined in
    :cite:`Tschauner1965_RendezvousZuEineminElliptischerBahnUmlaufendenZiel`.
    The states are defined as positions in a
    Local-Vertical-Local-Horizontal (LVLH) frame. Note, the true anomaly is
    that of the target object. For more details see
    :cite:`Okasha2013_GuidanceNavigationandControlforSatelliteProximityOperationsUsingTschaunerHempelEquations`
    and :cite:`Lange1965_FloquetTheoryOrbitalPerturbations`

    Attributes
    ----------
    mu : float, optional
        gravitational parameter in :math:`m^3 s^{-2}`. The default is 3.986004418 * 10**14.
    semi_major : float
        semi-major axis in meters. The default is None.
    eccentricity : float, optional
        eccentricity. The default is 1.
    """

    state_names = (
        "x pos",
        "y pos",
        "z pos",
        "x vel",
        "y vel",
        "z vel",
        "targets true anomaly",
    )

    def __init__(
        self, mu=3.986004418 * 10**14, semi_major=None, eccentricity=1, **kwargs
    ):
        self.mu = mu
        self.semi_major = semi_major
        self.eccentricity = eccentricity

        super().__init__(**kwargs)

    @property
    def cont_fnc_lst(self):
        """Continuous time dynamics.

        Returns
        -------
        list
            functions of the form :code:`(t, x, *args)`.
        """

        # returns x velocity
        def f0(t, x, *args):
            return x[3]

        # returns y velocity
        def f1(t, x, *args):
            return x[4]

        # returns z velocity
        def f2(t, x, *args):
            return x[5]

        # returns x acceleration
        def f3(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6]))) ** 3
            n = np.sqrt(mu / a**3)

            C1 = mu / R3
            wz = n * (1 + e * np.cos(x[6])) ** 2 / (1 - e2) ** (3.0 / 2.0)
            wz_dot = -2 * mu * e * np.sin(x[6]) / R3

            return (wz**2 + 2 * C1) * x[0] + wz_dot * x[1] + 2 * wz * x[4]

        # returns y acceleration
        def f4(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6]))) ** 3
            n = np.sqrt(mu / a**3)

            C1 = mu / R3
            wz = n * (1 + e * np.cos(x[6])) ** 2 / (1 - e2) ** (3.0 / 2.0)
            wz_dot = -2 * mu * e * np.sin(x[6]) / R3

            return (wz**2 - C1) * x[1] - wz_dot * x[0] - 2 * wz * x[3]

        # returns z acceleration
        def f5(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            mu = self.mu

            e2 = e**2
            R3 = ((a * (1 - e2)) / (1 + e * np.cos(x[6]))) ** 3

            C1 = mu / R3

            return -C1 * x[2]

        # returns true anomaly ROC
        def f6(t, x, *args):
            e = self.eccentricity
            a = self.semi_major
            p = a * (1 - e**2)

            H = np.sqrt(self.mu * p)
            R = p / (1 + e * np.cos(x[6]))
            return H / R**2

        return [f0, f1, f2, f3, f4, f5, f6]
