import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy
from warnings import warn

import gncpy.distributions as gdistrib
import gncpy.errors as gerr
import gncpy.plotting as pltUtil
from gncpy.filters.bayes_filter import BayesFilter


class ParticleFilter(BayesFilter):
    """Implements a basic Particle Filter.

    Notes
    -----
    The implementation is based on
    :cite:`Simon2006_OptimalStateEstimationKalmanHInfinityandNonlinearApproaches`
    and uses Sampling-Importance Resampling (SIR) sampling. Other resampling
    methods can be added in derived classes.

    Attributes
    ----------
    require_copy_prop_parts : bool
        Flag indicating if the propagated particles need to be copied if this
        filter is being manipulated externally. This is a constant value that
        should not be modified outside of the class, but can be overridden by
        inherited classes.
    require_copy_can_dist : bool
        Flag indicating if a candidate distribution needs to be copied if this
        filter is being manipulated externally. This is a constant value that
        should not be modified outside of the class, but can be overridden by
        inherited classes.
    """

    require_copy_prop_parts = True
    require_copy_can_dist = False

    def __init__(
        self,
        dyn_obj=None,
        dyn_fun=None,
        part_dist=None,
        transition_prob_fnc=None,
        rng=None,
        **kwargs
    ):

        self.__meas_likelihood_fnc = None
        self.__proposal_sampling_fnc = None
        self.__proposal_fnc = None
        self.__transition_prob_fnc = None

        if rng is None:
            rng = rnd.default_rng(1)
        self.rng = rng

        self._dyn_fnc = None
        self._dyn_obj = None

        self._meas_mat = None
        self._meas_fnc = None

        if dyn_obj is not None or dyn_fun is not None:
            self.set_state_model(dyn_obj=dyn_obj, dyn_fun=dyn_fun)
        self._particleDist = gdistrib.ParticleDistribution()
        if part_dist is not None:
            self.init_from_dist(part_dist)
        self.prop_parts = []

        super().__init__(**kwargs)

    def save_filter_state(self):
        """Saves filter variables so they can be restored later."""
        filt_state = super().save_filter_state()

        filt_state["__meas_likelihood_fnc"] = self.__meas_likelihood_fnc
        filt_state["__proposal_sampling_fnc"] = self.__proposal_sampling_fnc
        filt_state["__proposal_fnc"] = self.__proposal_fnc
        filt_state["__transition_prob_fnc"] = self.__transition_prob_fnc

        filt_state["rng"] = self.rng

        filt_state["_dyn_fnc"] = self._dyn_fnc
        filt_state["_dyn_obj"] = self._dyn_obj

        filt_state["_meas_mat"] = self._meas_mat
        filt_state["_meas_fnc"] = self._meas_fnc

        filt_state["_particleDist"] = deepcopy(self._particleDist)
        filt_state["prop_parts"] = deepcopy(self.prop_parts)

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.__meas_likelihood_fnc = filt_state["__meas_likelihood_fnc"]
        self.__proposal_sampling_fnc = filt_state["__proposal_sampling_fnc"]
        self.__proposal_fnc = filt_state["__proposal_fnc"]
        self.__transition_prob_fnc = filt_state["__transition_prob_fnc"]

        self.rng = filt_state["rng"]

        self._dyn_fnc = filt_state["_dyn_fnc"]
        self._dyn_obj = filt_state["_dyn_obj"]

        self._meas_mat = filt_state["_meas_mat"]
        self._meas_fnc = filt_state["_meas_fnc"]

        self._particleDist = filt_state["_particleDist"]
        self.prop_parts = filt_state["prop_parts"]

    @property
    def meas_likelihood_fnc(self):
        r"""A function that returns the likelihood of the measurement.

        This must have the signature :code:`f(y, y_hat, *args)` where `y` is
        the measurement as an Nm x 1 numpy array, and `y_hat` is the estimated
        measurement.

        Notes
        -----
        This represents :math:`p(y_t \vert x_t)` in the importance
        weight

        .. math::

            w_t = w_{t-1} \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        Returns
        -------
        callable
            function to return the measurement likelihood.
        """
        return self.__meas_likelihood_fnc

    @meas_likelihood_fnc.setter
    def meas_likelihood_fnc(self, val):
        self.__meas_likelihood_fnc = val

    @property
    def proposal_fnc(self):
        r"""A function that returns the probability for the proposal distribution.

        This must have the signature :code:`f(x_hat, x, y, *args)` where
        `x_hat` is a :class:`gncpy.distributions.Particle` of the estimated
        state, `x` is the particle it is conditioned on, and `y` is the
        measurement.

        Notes
        -----
        This represents :math:`q(x_t \vert x_{t-1}, y_t)` in the importance
        weight

        .. math::

            w_t = w_{t-1} \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        Returns
        -------
        callable
            function to return the proposal probability.
        """
        return self.__proposal_fnc

    @proposal_fnc.setter
    def proposal_fnc(self, val):
        self.__proposal_fnc = val

    @property
    def proposal_sampling_fnc(self):
        """A function that returns a random sample from the proposal distribtion.

        This should be consistent with the PDF specified in the
        :meth:`gncpy.filters.ParticleFilter.proposal_fnc`.

        Returns
        -------
        callable
            function to return a random sample.
        """
        return self.__proposal_sampling_fnc

    @proposal_sampling_fnc.setter
    def proposal_sampling_fnc(self, val):
        self.__proposal_sampling_fnc = val

    @property
    def transition_prob_fnc(self):
        r"""A function that returns the transition probability for the state.

        This must have the signature :code:`f(x_hat, x, *args)` where
        `x_hat` is an N x 1 numpy array representing the propagated state, and
        `x` is the state it is conditioned on.

        Notes
        -----
        This represents :math:`p(x_t \vert x_{t-1})` in the importance
        weight

        .. math::

            w_t = w_{t-1} \frac{p(y_t \vert x_t) p(x_t \vert x_{t-1})}{q(x_t \vert x_{t-1}, y_t)}

        Returns
        -------
        callable
            function to return the transition probability.
        """
        return self.__transition_prob_fnc

    @transition_prob_fnc.setter
    def transition_prob_fnc(self, val):
        self.__transition_prob_fnc = val

    def set_state_model(self, dyn_obj=None, dyn_fun=None):
        """Sets the state model.

        Parameters
        ----------
        dyn_obj : :class:gncpy.dynamics.DynamicsBase`, optional
            Dynamic object to use. The default is None.
        dyn_fun : callable, optional
            function that returns the next state. It must have the signature
            `f(t, x, *args)` and return a N x 1 numpy array. The default is None.

        Raises
        ------
        RuntimeError
            If no model is specified.

        Returns
        -------
        None.
        """
        if dyn_obj is not None:
            self._dyn_obj = deepcopy(dyn_obj)
        elif dyn_fun is not None:
            self._dyn_fnc = dyn_fun
        else:
            msg = "Invalid state model specified. Check arguments"
            raise RuntimeError(msg)

    def set_measurement_model(self, meas_mat=None, meas_fun=None):
        r"""Sets the measurement model for the filter.

        This can either set the constant measurement matrix, or a set of
        non-linear functions (potentially time varying) to map states to
        measurements.

        Notes
        -----
        The constant matrix assumes a measurement model of the form

        .. math::
            \tilde{y}_{k+1} = H x_{k+1}^-

        and the non-linear case assumes

        .. math::
            \tilde{y}_{k+1} = h(t, x_{k+1}^-)

        Parameters
        ----------
        meas_mat : Nm x N numpy array, optional
            Measurement matrix that transforms the state to estimated
            measurements. The default is None.
        meas_fun_lst : list, optional
            Non-linear functions that return the expected measurement for the
            given state. Each function must have the signature `h(t, x, *args)`.
            The default is None.

        Raises
        ------
        RuntimeError
            Rasied if no arguments are specified.

        Returns
        -------
        None.
        """
        if meas_mat is not None:
            self._meas_mat = meas_mat
        elif meas_fun is not None:
            self._meas_fnc = meas_fun
        else:
            raise RuntimeError("Invalid combination of inputs")

    @property
    def cov(self):
        """Read only covariance of the particles.

        Returns
        -------
        N x N numpy array
            covariance matrix.

        """
        return self._particleDist.covariance

    @cov.setter
    def cov(self, x):
        raise RuntimeError("Covariance is read only")

    @property
    def num_particles(self):
        """Read only number of particles used by the filter.

        Returns
        -------
        int
            Number of particles.
        """
        return self._particleDist.num_particles

    def init_from_dist(self, dist, make_copy=True):
        """Initialize the distribution from a distribution object.

        Parameters
        ----------
        dist : :class:`gncpy.distributions.ParticleDistribution`
            Distribution object to use.
        make_copy : bool, optional
            Flag indicating if a deepcopy of the input distribution should be
            performed. The default is True.

        Returns
        -------
        None.
        """
        if make_copy:
            self._particleDist = deepcopy(dist)
        else:
            self._particleDist = dist

    def extract_dist(self, make_copy=True):
        """Extracts the particle distribution used by the filter.

        Parameters
        ----------
        make_copy : bool, optional
            Flag indicating if a deepcopy of the distribution should be
            performed. The default is True.

        Returns
        -------
        :class:`gncpy.distributions.ParticleDistribution`
            Particle distribution object used by the filter
        """
        if make_copy:
            return deepcopy(self._particleDist)
        else:
            return self._particleDist

    def init_particles(self, particle_lst):
        """Initializes the particle distribution with the given list of points.

        Parameters
        ----------
        particle_lst : list
            List of numpy arrays, one for each particle.
        """
        num_parts = len(particle_lst)
        if num_parts <= 0:
            warn("No particles to initialize. SKIPPING")
            return
        self._particleDist.clear_particles()
        self._particleDist.add_particle(particle_lst, [1.0 / num_parts] * num_parts)

    def _calc_state(self):
        return self._particleDist.mean

    def predict(
        self, timestep, dyn_fun_params=(), sampling_args=(), transition_args=()
    ):
        """Predicts the next state.

        Parameters
        ----------
        timestep : float
            Current timestep.
        dyn_fun_params : tuple, optional
            Extra arguments to be passed to the dynamics function. The default
            is ().
        sampling_args : tuple, optional
            Extra arguments to be passed to the proposal sampling function.
            The default is ().

        Raises
        ------
        RuntimeError
            If no state model is set.

        Returns
        -------
        N x 1 numpy array
            predicted state.

        """
        if self._dyn_obj is not None:
            self.prop_parts = [
                self._dyn_obj.propagate_state(timestep, x, state_args=dyn_fun_params)
                for x in self._particleDist.particles
            ]
            mean = self._dyn_obj.propagate_state(
                timestep, self._particleDist.mean, state_args=dyn_fun_params
            )
        elif self._dyn_fnc is not None:
            self.prop_parts = [
                self._dyn_fnc(timestep, x, *dyn_fun_params)
                for x in self._particleDist.particles
            ]
            mean = self._dyn_fnc(timestep, self._particleDist.mean, *dyn_fun_params)
        else:
            raise RuntimeError("No state model set")
        new_weights = [
            w * self.transition_prob_fnc(x, mean, *transition_args)
            if self.transition_prob_fnc is not None
            else w
            for x, w in zip(self.prop_parts, self._particleDist.weights)
        ]

        new_parts = [
            self.proposal_sampling_fnc(p, self.rng, *sampling_args)
            for p in self.prop_parts
        ]

        self._particleDist.clear_particles()
        for p, w in zip(new_parts, new_weights):
            part = gdistrib.Particle(point=p)
            self._particleDist.add_particle(part, w)
        return self._calc_state()

    def _est_meas(self, timestep, cur_state, n_meas, meas_fun_args):
        if self._meas_fnc is not None:
            est_meas = self._meas_fnc(timestep, cur_state, *meas_fun_args)
        elif self._meas_mat is not None:
            est_meas = self._meas_mat @ cur_state
        else:
            raise RuntimeError("No measurement model set")
        return est_meas

    def _selection(self, unnorm_weights, rel_likeli_in=None):
        new_parts = [None] * self.num_particles
        old_weights = [None] * self.num_particles
        rel_likeli_out = [None] * self.num_particles
        inds_kept = []
        probs = self.rng.random(self.num_particles)
        cumulative_weight = np.cumsum(self._particleDist.weights)
        failed = False
        for ii, r in enumerate(probs):
            inds = np.where(cumulative_weight >= r)[0]
            if inds.size > 0:
                new_parts[ii] = deepcopy(self._particleDist._particles[inds[0]])
                old_weights[ii] = unnorm_weights[inds[0]]
                if rel_likeli_in is not None:
                    rel_likeli_out[ii] = rel_likeli_in[inds[0]]
                if inds[0] not in inds_kept:
                    inds_kept.append(inds[0])
            else:
                failed = True
        if failed:
            tot = np.sum(self._particleDist.weights)
            self._particleDist.clear_particles()
            msg = (
                "Failed to select enough particles, "
                + "check weights (sum = {})".format(tot)
            )
            raise gerr.ParticleDepletionError(msg)
        inds_removed = [
            ii for ii in range(0, self.num_particles) if ii not in inds_kept
        ]

        self._particleDist.clear_particles()
        w = 1 / len(new_parts)
        self._particleDist.add_particle(new_parts, [w] * len(new_parts))

        return inds_removed, old_weights, rel_likeli_out

    def correct(
        self,
        timestep,
        meas,
        meas_fun_args=(),
        meas_likely_args=(),
        proposal_args=(),
        selection=True,
    ):
        """Corrects the state estimate.

        Parameters
        ----------
        timestep : float
            Current timestep.
        meas : Nm x 1 numpy array
            Current measurement.
        meas_fun_args : tuple, optional
            Arguments for the measurement matrix function if one has
            been specified. The default is ().
        meas_likely_args : tuple, optional
            additional agruments for the measurement likelihood function.
            The default is ().
        proposal_args : tuple, optional
            Additional arguments for the proposal distribution function. The
            default is ().
        selection : bool, optional
            Flag indicating if the selection step should be performed. The
            default is True.

        Raises
        ------
        RuntimeError
            If no measurement model is set

        Returns
        -------
        state : N x 1 numpy array
            corrected state.
        rel_likeli : numpy array
            The unnormalized measurement likelihood of each particle.
        inds_removed : list
            each element is an int representing the index of any particles
            that were removed during the selection process.

        """
        # calculate weights
        est_meas = [
            self._est_meas(timestep, p, meas.size, meas_fun_args)
            for p in self._particleDist.particles
        ]
        if self.meas_likelihood_fnc is None:
            rel_likeli = np.ones(len(est_meas))
        else:
            rel_likeli = np.array(
                [self.meas_likelihood_fnc(meas, y, *meas_likely_args) for y in est_meas]
            ).ravel()
        if self.proposal_fnc is None or len(self.prop_parts) == 0:
            prop_fit = np.ones(len(self._particleDist.particles))
        else:
            prop_fit = np.array(
                [
                    self.proposal_fnc(x_hat, cond, meas, *proposal_args)
                    for x_hat, cond in zip(
                        self._particleDist.particles, self.prop_parts
                    )
                ]
            ).ravel()
        inds = np.where(prop_fit < np.finfo(float).eps)[0]
        if inds.size > 0:
            prop_fit[inds] = np.finfo(float).eps
        unnorm_weights = rel_likeli / prop_fit * np.array(self._particleDist.weights)

        tot = np.sum(unnorm_weights)
        if tot > 0 and tot != np.inf:
            weights = unnorm_weights / tot
        else:
            weights = np.inf * np.ones(unnorm_weights.size)
        self._particleDist.update_weights(weights)

        # resample
        if selection:
            inds_removed, rel_likeli = self._selection(
                unnorm_weights, rel_likeli_in=rel_likeli.tolist()
            )[0:3:2]
        else:
            inds_removed = []
        return (self._calc_state(), rel_likeli, inds_removed)

    def plot_particles(
        self,
        inds,
        title="Particle Distribution",
        x_lbl="State",
        y_lbl="Probability",
        **kwargs
    ):
        """Plots the particle distribution.

        This will either plot a histogram for a single index, or plot a 2-d
        heatmap/histogram if a list of 2 indices are given. The 1-d case will
        have the counts normalized to represent the probability.

        Parameters
        ----------
        inds : int or list
            Index of the particle vector to plot.
        title : string, optional
            Title of the plot. The default is 'Particle Distribution'.
        x_lbl : string, optional
            X-axis label. The default is 'State'.
        y_lbl : string, optional
            Y-axis label. The default is 'Probability'.
        **kwargs : dict
            Additional plotting options for :meth:`gncpy.plotting.init_plotting_opts`
            function. Values implemented here are `f_hndl`, `lgnd_loc`, and
            any values relating to title/axis text formatting.

        Returns
        -------
        f_hndl : matplotlib figure
            Figure object the data was plotted on.
        """
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts["f_hndl"]
        lgnd_loc = opts["lgnd_loc"]

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)
        h_opts = {"histtype": "stepfilled", "bins": "auto", "density": True}
        if (not isinstance(inds, list)) or len(inds) == 1:
            if isinstance(inds, list):
                ii = inds[0]
            else:
                ii = inds
            x = [p[ii, 0] for p in self._particleDist.particles]
            f_hndl.axes[0].hist(x, **h_opts)
        else:
            x = [p[inds[0], 0] for p in self._particleDist.particles]
            y = [p[inds[1], 0] for p in self._particleDist.particles]
            f_hndl.axes[0].hist2d(x, y)
        pltUtil.set_title_label(f_hndl, 0, opts, ttl=title, x_lbl=x_lbl, y_lbl=y_lbl)
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl

    def plot_weighted_particles(
        self,
        inds,
        x_lbl="State",
        y_lbl="Weight",
        title="Weighted Particle Distribution",
        **kwargs
    ):
        """Plots the weight vs state distribution of the particles.

        This generates a bar chart and only works for single indices.

        Parameters
        ----------
        inds : int
            Index of the particle vector to plot.
        x_lbl : string, optional
            X-axis label. The default is 'State'.
        y_lbl : string, optional
            Y-axis label. The default is 'Weight'.
        title : string, optional
            Title of the plot. The default is 'Weighted Particle Distribution'.
        **kwargs : dict
            Additional plotting options for :meth:`gncpy.plotting.init_plotting_opts`
            function. Values implemented here are `f_hndl`, `lgnd_loc`, and
            any values relating to title/axis text formatting.

        Returns
        -------
        f_hndl : matplotlib figure
            Figure object the data was plotted on.
        """
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts["f_hndl"]
        lgnd_loc = opts["lgnd_loc"]

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)
        if (not isinstance(inds, list)) or len(inds) == 1:
            if isinstance(inds, list):
                ii = inds[0]
            else:
                ii = inds
            x = [p[ii, 0] for p in self._particleDist.particles]
            y = [w for p, w in self._particleDist]
            f_hndl.axes[0].bar(x, y)
        else:
            warn("Only 1 element supported for weighted particle distribution")
        pltUtil.set_title_label(f_hndl, 0, opts, ttl=title, x_lbl=x_lbl, y_lbl=y_lbl)
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl
