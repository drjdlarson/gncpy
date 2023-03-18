import numpy as np
import scipy.stats as stats
from copy import deepcopy
from warnings import warn

import gncpy.distributions as gdistrib
from gncpy.filters.mcmc_particle_filter_base import MCMCParticleFilterBase
from gncpy.filters.unscented_kalman_filter import UnscentedKalmanFilter

class UnscentedParticleFilter(MCMCParticleFilterBase):
    """Implements an unscented particle filter.

    Notes
    -----
    For details on the filter see
    :cite:`VanDerMerwe2000_TheUnscentedParticleFilter` and
    :cite:`VanDerMerwe2001_TheUnscentedParticleFilter`.
    """

    require_copy_prop_parts = False

    def __init__(self, **kwargs):
        self.candDist = None

        self._filt = UnscentedKalmanFilter()

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["candDist"] = deepcopy(self.candDist)
        filt_state["_filt"] = self._filt.save_filter_state()

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.candDist = filt_state["candDist"]
        self._filt.load_filter_state(filt_state["_filt"])

    @property
    def meas_likelihood_fnc(self):
        r"""A function that returns the likelihood of the measurement.

        This has the signature :code:`f(y, y_hat, *args)` where `y` is
        the measurement as an Nm x 1 numpy array, and `y_hat` is the estimated
        measurement.

        Notes
        -----
        This represents :math:`p(y_t \vert x_t)` in the importance
        weight

        .. math::

            w_t \propto \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        and has the assumed form :math:`\mathcal{N}(y_t, R)` for measurement
        noise covariance :math:`R`.

        Returns
        -------
        callable
            function to return the measurement likelihood.
        """
        return lambda y, y_hat: stats.multivariate_normal.pdf(
            y.ravel(), y_hat.ravel(), self._filt.meas_noise
        )

    @meas_likelihood_fnc.setter
    def meas_likelihood_fnc(self, val):
        warn("Measuremnet likelihood has an assumed form.")

    @property
    def proposal_fnc(self):
        r"""A function that returns the probability for the proposal distribution.

        This has the signature :code:`f(x_hat, x, y, *args)` where
        `x_hat` is a :class:`gncpy.distributions.Particle` of the estimated
        state, `x` is the particle it is conditioned on, and `y` is the
        measurement.

        Notes
        -----
        This represents :math:`q(x_t \vert x_{t-1}, y_t)` in the importance
        weight

        .. math::

            w_t \propto \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        and has the assumed form :math:`\mathcal{N}(\bar{x}_{t}, \hat{P}_t)`

        Returns
        -------
        callable
            function to return the proposal probability.
        """
        return lambda x_hat, part: stats.multivariate_normal.pdf(
            x_hat.ravel(), part.mean.ravel(), part.uncertainty
        )

    @proposal_fnc.setter
    def proposal_fnc(self, val):
        warn("Proposal distribution has an assumed form.")

    @property
    def proposal_sampling_fnc(self):
        r"""A function that returns a random sample from the proposal distribtion.

        This should be consistent with the PDF specified in the
        :meth:`gncpy.filters.ParticleFilter.proposal_fnc`.

        Notes
        -----
        This assumes :math:`x` is drawn from :math:`\mathcal{N}(\bar{x}_{t}, \hat{P}_t)`
        Returns
        -------
        callable
            function to return a random sample.
        """
        return lambda part: self.rng.multivariate_normal(
            part.mean.ravel(), part.uncertainty
        ).reshape(part.mean.shape)

    @proposal_sampling_fnc.setter
    def proposal_sampling_fnc(self, val):
        warn("Proposal sampling has an assumed form.")

    @property
    def transition_prob_fnc(self):
        r"""A function that returns the transition probability for the state.

        This has the signature :code:`f(x_hat, x, cov)` where
        `x_hat` is an N x 1 numpy array representing the propagated state, and
        `x` is the state it is conditioned on.

        Notes
        -----
        This represents :math:`p(x_t \vert x_{t-1})` in the importance
        weight

        .. math::

            w_t \propto \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        and has the assumed form :math:`\mathcal{N}(f(x_{t-1}), P_t)` for the
        covariance :math:`P_t`.

        Returns
        -------
        callable
            function to return the transition probability.
        """
        return lambda x_hat, x, cov: stats.multivariate_normal.pdf(
            x_hat.ravel(), x.ravel(), cov
        )

    @transition_prob_fnc.setter
    def transition_prob_fnc(self, val):
        warn("Transistion distribution has an assumed form.")

    # @property
    # def cov(self):
    #     """Read only covariance of the particles.

    #     This is a weighted sum of each particles uncertainty.

    #     Returns
    #     -------
    #     N x N numpy array
    #         covariance matrix.

    #     """
    #     return gmath.weighted_sum_mat(self._particleDist.weights,
    #                                   self._particleDist.uncertainties)

    @property
    def meas_noise(self):
        """Measurement noise matrix.

        This is a wrapper to keep the UPF measurement noise and the internal UKF
        measurement noise synced.

        Returns
        -------
        Nm x Nm numpy array
            measurement noise.
        """
        return self._filt.meas_noise

    @meas_noise.setter
    def meas_noise(self, meas_noise):
        self._filt.meas_noise = meas_noise

    @property
    def proc_noise(self):
        """Process noise matrix.

        This is a wrapper to keep the UPF process noise and the internal UKF
        process noise synced.

        Returns
        -------
        N x N numpy array
            process noise.
        """
        return self._filt.proc_noise

    @proc_noise.setter
    def proc_noise(self, proc_noise):
        self._filt.proc_noise = proc_noise

    def set_state_model(self, **kwargs):
        """Sets the state model for the filter.

        This calls the UKF's set state function
        (:meth:`gncpy.filters.UnscentedKalmanFilter.set_state_model`).
        """
        self._filt.set_state_model(**kwargs)

    def set_measurement_model(self, **kwargs):
        r"""Sets the measurement model for the filter.

        This is a wrapper for the inner UKF's set_measurement model function.
        It is assumed that the measurement model is the same as that of the UKF.
        See :meth:`gncpy.filters.UnscentedKalmanFilter.set_measurement_model`
        for details.
        """
        self._filt.set_measurement_model(**kwargs)

    def _predict_loop(self, timestep, ukf_kwargs, dist):
        newDist = gdistrib.ParticleDistribution()
        num_parts = dist.num_particles
        new_parts = [None] * num_parts
        new_weights = [None] * num_parts
        for ii, (origPart, w) in enumerate(dist):
            part = gdistrib.Particle()
            self._filt.cov = origPart.uncertainty.copy()
            self._filt._stateSigmaPoints = deepcopy(origPart.sigmaPoints)
            part.point = self._filt.predict(timestep, origPart.point, **ukf_kwargs)
            part.uncertainty = self._filt.cov
            part.sigmaPoints = self._filt._stateSigmaPoints
            new_parts[ii] = part
            new_weights[ii] = w
        newDist.add_particle(new_parts, new_weights)
        return newDist

    def predict(self, timestep, ukf_kwargs={}):
        """Prediction step of the UPF.

        This calls the UKF prediction function on every particle.

        Parameters
        ----------
        timestep : float
            Current timestep.
        ukf_kwargs : dict, optional
            Additional arguments to pass to the UKF prediction function. The
            default is {}.

        Returns
        -------
        state : N x 1 numpy array
            The predicted state.
        """
        self._particleDist = self._predict_loop(
            timestep, ukf_kwargs, self._particleDist
        )
        if self.use_MCMC:
            if self.candDist is None:  # first timestep
                self.candDist = deepcopy(self._particleDist)
            else:
                self.candDist = self._predict_loop(timestep, ukf_kwargs, self.candDist)
        return self._calc_state()

    def _inner_correct(self, timestep, meas, state, filt_kwargs):
        """Wrapper so child class can override."""
        return self._filt.correct(timestep, meas, state, **filt_kwargs)

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        return self._filt._est_meas(timestep, cur_state, n_meas, meas_fun_args)[0]

    def _correct_loop(self, timestep, meas, ukf_kwargs, dist):
        unnorm_weights = np.nan * np.ones(dist.num_particles)
        new_parts = [None] * dist.num_particles
        rel_likeli = [None] * dist.num_particles
        for ii, (p, w) in enumerate(dist):
            self._filt.cov = p.uncertainty.copy()
            self._filt._stateSigmaPoints = deepcopy(p.sigmaPoints)

            # create a new particle and correct the predicted point with the measurement
            part = gdistrib.Particle()
            part.point, rel_likeli[ii] = self._inner_correct(
                timestep, meas, p.point, ukf_kwargs
            )
            part.uncertainty = self._filt.cov

            # draw a sample from the proposal distribution using the corrected point
            samp = self.proposal_sampling_fnc(part)

            # transition probability of the sample given the predicted point
            trans_prob = self.transition_prob_fnc(samp, p.point, p.uncertainty)

            # probability of the sampled value given the corrected value
            proposal_prob = self.proposal_fnc(samp, part)

            # get new weight
            if proposal_prob < np.finfo(float).eps:
                proposal_prob = np.finfo(float).eps
            unnorm_weights[ii] = rel_likeli[ii] * trans_prob / proposal_prob

            # update the new particle with the sampled point
            part.point = samp
            part.sigmaPoints = self._filt._stateSigmaPoints
            part.sigmaPoints.update_points(part.point, part.uncertainty)
            new_parts[ii] = part
        # update info for next UKF
        newDist = gdistrib.ParticleDistribution()
        tot = np.sum(unnorm_weights)
        # protect against divide by 0
        if tot < np.finfo(float).eps:
            tot = np.finfo(float).eps
        newDist.add_particle(new_parts, (unnorm_weights / tot).tolist())

        return newDist, rel_likeli, unnorm_weights

    def correct(self, timestep, meas, ukf_kwargs={}, move_kwargs={}):
        """Correction step of the UPF.

        This optionally can perform a MCMC move step as well.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            measurement.
        ukf_kwargs : dict, optional
            Additional arguments to pass to the UKF correction function. The
            default is {}.
        move_kwargs : dict, optional
            Additional arguments to pass to the movement function.
            The default is {}.

        Returns
        -------
        state : N x 1 numpy array
            corrected state.
        rel_likeli : list
            each element is a float representing the relative likelihood of the
            particles (unnormalized).
        inds_removed : list
            each element is an int representing the index of any particles
            that were removed during the selection process.
        """
        # if first timestep and have not called predict yet
        if self.use_MCMC and self.candDist is None:
            self.candDist = deepcopy(self._particleDist)
        # call UKF correction on each particle
        (self._particleDist, rel_likeli, unnorm_weights) = self._correct_loop(
            timestep, meas, ukf_kwargs, self._particleDist
        )
        if self.use_MCMC:
            (self.candDist, can_rel_likeli, can_unnorm_weights) = self._correct_loop(
                timestep, meas, ukf_kwargs, self.candDist
            )
        # perform selection/resampling
        (inds_removed, old_weights, rel_likeli) = self._selection(
            unnorm_weights, rel_likeli_in=rel_likeli
        )

        # optionally move particles
        if self.use_MCMC:
            rel_likeli = self.move_particles(
                timestep,
                meas,
                old_weights,
                rel_likeli,
                can_unnorm_weights,
                can_rel_likeli,
                **move_kwargs
            )
        return (self._calc_state(), rel_likeli, inds_removed)

    def move_particles(
        self, timestep, meas, old_weights, old_likeli, can_weight, can_likeli
    ):
        r"""Movement function for the MCMC move step.

        This modifies the internal particle distribution but does not return a
        value.

        Notes
        -----
        This implements a metropolis-hastings algorithm.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            measurement.
        old_weights : :class:`gncpy.distributions.ParticleDistribution`
            Distribution before the measurement correction.

        Returns
        -------
        None.
        """
        accept_prob = self.rng.random()
        num_parts = self._particleDist.num_particles
        new_parts = [None] * num_parts
        new_likeli = [None] * num_parts
        for ii, (can, exp, ex_weight, can_weight, ex_like, can_like) in enumerate(
            zip(
                self.candDist,
                self._particleDist,
                old_weights,
                can_weight,
                old_likeli,
                can_likeli,
            )
        ):

            if accept_prob <= np.min([1, can_weight / ex_weight]):
                # accept move
                new_parts[ii] = deepcopy(can[0])
                new_likeli[ii] = can_like
            else:
                # reject move
                new_parts[ii] = exp[0]
                new_likeli[ii] = ex_like
        self._particleDist.clear_particles()
        self._particleDist.add_particle(new_parts, [1 / num_parts] * num_parts)

        return new_likeli
