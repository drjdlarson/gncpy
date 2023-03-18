import numpy as np
import numpy.random as rnd
import scipy.stats as stats
from queue import deque
from warnings import warn

import gncpy.distributions as gdistrib
import gncpy.errors as gerr
from serums.enums import GSMTypes
from gncpy.filters.bayes_filter import BayesFilter
from gncpy.filters.bootstrap_filter import BootstrapFilter


class _GSMProcNoiseEstimator:
    """Helper class for estimating proc noise in the GSM filters."""

    def __init__(self):
        self.q_fifo = deque([], None)
        self.b_fifo = deque([], None)

        self.startup_delay = 1

        self._last_q_hat = None
        self._call_count = 0

    @property
    def maxlen(self):
        return self.q_fifo.maxlen

    @maxlen.setter
    def maxlen(self, val):
        self.q_fifo = deque([], val)
        self.b_fifo = deque([], val)

    @property
    def win_len(self):
        return len(self.q_fifo)

    def estimate_next(self, cur_est, pred_state, pred_cov, cor_state, cor_cov):
        self._call_count += 1

        # update bk
        bk = pred_cov - cur_est - cor_cov
        if self.b_fifo.maxlen is None or len(self.b_fifo) < self.b_fifo.maxlen:
            bk_last = np.zeros(bk.shape)
        else:
            bk_last = self.b_fifo.pop()
        self.b_fifo.appendleft(bk)

        # update qk and qk_hat
        qk = cor_state - pred_state
        if self._last_q_hat is None:
            self._last_q_hat = np.zeros(qk.shape)
        if self.q_fifo.maxlen is None or len(self.q_fifo) < self.q_fifo.maxlen:
            qk_last = np.zeros(qk.shape)
        else:
            qk_last = self.q_fifo.pop()
        self.q_fifo.appendleft(qk)

        qk_klast_diff = qk - qk_last
        win_len = self.win_len
        inv_win_len = 1 / win_len
        win_len_m1 = win_len - 1
        if self._call_count <= np.max([1, self.startup_delay]):
            return cur_est
        self._last_q_hat += inv_win_len * qk_klast_diff

        # estimate cov
        qk_khat_diff = qk - self._last_q_hat
        qklast_khat_diff = qk_last - self._last_q_hat
        bk_diff = bk_last - bk
        next_est = cur_est + 1 / win_len_m1 * (
            qk_khat_diff @ qk_khat_diff.T
            - qklast_khat_diff @ qklast_khat_diff.T
            + inv_win_len * qk_klast_diff @ qk_klast_diff.T
            + win_len_m1 * inv_win_len * bk_diff
        )

        # for numerical reasons
        for ii in range(next_est.shape[0]):
            next_est[ii, ii] = np.abs(next_est[ii, ii])
        return next_est


class GSMFilterBase(BayesFilter):
    """Base implementation of a Gaussian Scale Mixture (GSM) filter.

    This should be inherited from to include specific implementations of the
    core filter by exposing the necessary core filter attributes.

    Notes
    -----
    This is based on a generic version of
    :cite:`VilaValls2012_NonlinearBayesianFilteringintheGaussianScaleMixtureContext`
    which extends :cite:`VilaValls2011_BayesianFilteringforNonlinearStateSpaceModelsinSymmetricStableMeasurementNoise`.
    This class does not implement a specific core filter, that is up to the
    child classes.

    Attributes
    ----------
    enable_proc_noise_estimation : bool
        Flag indicating if the process noise should be estimated.
    """

    def __init__(self, enable_proc_noise_estimation=False, **kwargs):
        """Initializes the class.

        Parameters
        ----------
        enable_proc_noise_estimation : bool, optional
            Flag indicating if the process noise should be estimated. The
            default is False.
        **kwargs : dict
            Additional keyword arguements for the parent class.
        """
        super().__init__(**kwargs)

        self.enable_proc_noise_estimation = enable_proc_noise_estimation

        self._coreFilter = None
        self._meas_noise_filters = []
        self._import_w_factory_lst = None
        self._procNoiseEstimator = _GSMProcNoiseEstimator()

    def save_filter_state(self):
        """Saves filter variables so they can be restored later.

        Note that to pickle the resulting dictionary the :code:`dill` package
        may need to be used due to potential pickling of functions.
        """
        filt_state = super().save_filter_state()

        filt_state["enable_proc_noise_estimation"] = self.enable_proc_noise_estimation

        if self._coreFilter is not None:
            filt_state["_coreFilter"] = (
                type(self._coreFilter),
                self._coreFilter.save_filter_state(),
            )
        else:
            filt_state["_coreFilter"] = (None, self._coreFilter)
        filt_state["_meas_noise_filters"] = [
            (type(f), f.save_filter_state()) for f in self._meas_noise_filters
        ]

        filt_state["_import_w_factory_lst"] = self._import_w_factory_lst
        filt_state["_procNoiseEstimator"] = self._procNoiseEstimator

        return filt_state

    def load_filter_state(self, filt_state):
        """Initializes filter using saved filter state.

        Attributes
        ----------
        filt_state : dict
            Dictionary generated by :meth:`save_filter_state`.
        """
        super().load_filter_state(filt_state)

        self.enable_proc_noise_estimation = filt_state["enable_proc_noise_estimation"]

        cls_type = filt_state["_coreFilter"][0]
        if cls_type is not None:
            self._coreFilter = cls_type()
            self._coreFilter.load_filter_state(filt_state["_coreFilter"][1])
        else:
            self._coreFilter = None
        num_m_filts = len(filt_state["_meas_noise_filters"])
        self._meas_noise_filters = [None] * num_m_filts
        for ii, (cls_type, vals) in enumerate(filt_state["_meas_noise_filters"]):
            if cls_type is not None:
                self._meas_noise_filters[ii] = cls_type()
                self._meas_noise_filters[ii].load_filter_state(vals)
        self._import_w_factory_lst = filt_state["_import_w_factory_lst"]
        self._procNoiseEstimator = filt_state["_procNoiseEstimator"]

    def set_state_model(self, **kwargs):
        """Wrapper for the core filters set state model function."""
        if self._coreFilter is not None:
            self._coreFilter.set_state_model(**kwargs)
        else:
            warn("Core filter is not set, use an inherited class.", RuntimeWarning)

    def set_measurement_model(self, **kwargs):
        """Wrapper for the core filters set measurement model function."""
        if self._coreFilter is not None:
            self._coreFilter.set_measurement_model(**kwargs)
        else:
            warn("Core filter is not set, use an inherited class.", RuntimeWarning)

    def _define_student_t_pf(self, gsm, rng, num_parts):
        def import_w_factory(inov_cov):
            def import_w_fnc(meas, parts):
                stds = np.sqrt(parts[:, 2] * parts[:, 1] ** 2 + inov_cov)
                return np.array(
                    [stats.norm.pdf(meas.item(), scale=scale) for scale in stds]
                )

            return import_w_fnc

        def gsm_import_dist_factory():
            def import_dist_fnc(parts, _rng):
                new_parts = np.nan * np.ones(parts.particles.shape)

                disc = 0.99
                a = (3 * disc - 1) / (2 * disc)
                h = np.sqrt(1 - a ** 2)
                last_means = np.mean(parts.particles, axis=0)
                means = a * parts.particles[:, 0:2] + (1 - a) * last_means[0:2]

                # df, sig
                for ind in range(means.shape[1]):
                    std = np.sqrt(h ** 2 * np.cov(parts.particles[:, ind]))

                    for ii, m in enumerate(means):
                        samp = stats.norm.rvs(loc=m[ind], scale=std, random_state=_rng)
                        new_parts[ii, ind] = samp
                df = np.mean(new_parts[:, 0])
                if df < 0:
                    msg = "Degree of freedom must be > 0 {:.4f}".format(df)
                    raise gerr.ParticleEstimationDomainError(msg)
                new_parts[:, 2] = stats.invgamma.rvs(
                    df / 2,
                    scale=1 / (2 / df),
                    random_state=_rng,
                    size=new_parts.shape[0],
                )
                return new_parts

            return import_dist_fnc

        pf = BootstrapFilter()
        pf.importance_dist_fnc = gsm_import_dist_factory()
        pf.particleDistribution = gdistrib.SimpleParticleDistribution()

        df_scale = gsm.df_range[1] - gsm.df_range[0]
        df_loc = gsm.df_range[0]
        df_particles = stats.uniform.rvs(
            loc=df_loc, scale=df_scale, size=num_parts, random_state=rng
        )

        sig_scale = gsm.scale_range[1] - gsm.scale_range[0]
        sig_loc = gsm.scale_range[0]
        sig_particles = stats.uniform.rvs(
            loc=sig_loc, scale=sig_scale, size=num_parts, random_state=rng
        )

        z_particles = np.nan * np.ones(num_parts)
        for ii, v in enumerate(df_particles):
            z_particles[ii] = stats.invgamma.rvs(
                v / 2, scale=1 / (2 / v), random_state=rng
            )
        pf.particleDistribution.particles = np.stack(
            (df_particles, sig_particles, z_particles), axis=1
        )
        pf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        pf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        pf.rng = rng

        return pf, import_w_factory

    def _define_cauchy_pf(self, gsm, rng, num_parts):
        def import_w_factory(inov_cov):
            def import_w_fnc(meas, parts):
                stds = np.sqrt(parts[:, 1] * parts[:, 0] ** 2 + inov_cov)
                return np.array(
                    [stats.norm.pdf(meas.item(), scale=scale) for scale in stds]
                )

            return import_w_fnc

        def gsm_import_dist_factory():
            def import_dist_fnc(parts, _rng):
                new_parts = np.nan * np.ones(parts.particles.shape)

                disc = 0.99
                a = (3 * disc - 1) / (2 * disc)
                h = np.sqrt(1 - a ** 2)
                last_means = np.mean(parts.particles, axis=0)
                means = a * parts.particles[:, [0]] + (1 - a) * last_means[0]

                # df, sig
                for ind in range(means.shape[1]):
                    std = np.sqrt(h ** 2 * np.cov(parts.particles[:, ind]))

                    for ii, m in enumerate(means):
                        samp = stats.norm.rvs(loc=m[ind], scale=std, random_state=_rng)
                        new_parts[ii, ind] = samp
                new_parts[:, 1] = stats.invgamma.rvs(
                    1 / 2, scale=1 / 2, random_state=_rng, size=new_parts.shape[0]
                )
                return new_parts

            return import_dist_fnc

        pf = BootstrapFilter()
        pf.importance_dist_fnc = gsm_import_dist_factory()
        pf.particleDistribution = gdistrib.SimpleParticleDistribution()

        sig_scale = gsm.scale_range[1] - gsm.scale_range[0]
        sig_loc = gsm.scale_range[0]
        sig_particles = stats.uniform.rvs(
            loc=sig_loc, scale=sig_scale, size=num_parts, random_state=rng
        )

        z_particles = stats.invgamma.rvs(
            1 / 2, scale=1 / 2, random_state=rng, size=num_parts
        )

        pf.particleDistribution.particles = np.stack(
            (sig_particles, z_particles), axis=1
        )
        pf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        pf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        pf.rng = rng

        return pf, import_w_factory

    def set_meas_noise_model(
        self,
        bootstrap_lst=None,
        importance_weight_factory_lst=None,
        gsm_lst=None,
        num_parts=None,
        rng=None,
    ):
        """Initializes the measurement noise estimators.

        The filters and importance
        weight factories can be provided or a list of
        :class:`serums.models.GaussianScaleMixture` objecst, list of particles,
        and a random number generator. If the latter set is given then
        bootstrap filters are constructed automatically. The recommended way
        of specifying the filters is to provide GSM objects.

        Notes
        -----
        This uses independent bootstrap particle filters for each measurement
        based on the provided model information.

        Parameters
        ----------
        bootstrap_lst : list, optional
            List of :class:`.BootstrapFilter` objects that have already been
            initialized. If given then importance weight factory list must also
            be given and of the same length. The default is None.
        importance_weight_factory_lst : list, optional
            List of callables, each takes a 1 x 1 numpy array or float as input
            and returns a callable with the signature `f(z, parts)` where
            `z` is a float that is the difference between the estimated measurement
            and actual measurement (so the distribution is 0 mean) and `parts` is
            a numpy array of all the particles from the bootrap filter `f` must
            return a numpy array of weights for each particle. See the
            :attr:`.BootstrapFilter.importance_weight_fnc` for more details. The
            default is None.
        gsm_lst : list, optional
            List of :class:`serums.models.GaussianScaleMixture` objects, one
            per measurement. Requires `num_parts` also be specified and optionally
            `rng`. The default is None.
        num_parts : list or int, optional
            Number of particles to use in each automatically constructed filter.
            If only one number is supplied all filters will use the same number
            of particles. The default is None.
        rng : numpy random generator, optional
            Random number generator to use in constructed filters. If supplied,
            each filter uses the same instance, otherwise a new generator is
            created for each filter using numpy's default initialization routine
            with no supplied seed. Only used if a `gsm_lst` is supplied.

        Raises
        ------
        RuntimeError
            If an incorrect combination of input arguments is provided.

        Todo
        ----
        Allow for GSM Object to use location parameter when specifying noise models
        """
        if bootstrap_lst is not None:
            self._meas_noise_filters = bootstrap_lst
            if importance_weight_factory_lst is not None:
                if len(importance_weight_factory_lst) != len(bootstrap_lst):
                    msg = (
                        "Importance weight factory list "
                        + "length ({:d}) ".format(len(importance_weight_factory_lst))
                        + "does not match the number of bootstrap filters ({:d})".format(
                            len(bootstrap_lst)
                        )
                    )
                    raise RuntimeError(msg)
                self._import_w_factory_lst = importance_weight_factory_lst
            else:
                msg = (
                    "Must supply an importance weight factory when "
                    + "specifying the bootstrap filters."
                )
                raise RuntimeError(msg)
        elif gsm_lst is not None:
            num_filts = len(gsm_lst)
            if not (isinstance(num_parts, list) or isinstance(num_parts, tuple)):
                if num_parts is None:
                    msg = "Must specify number of particles when giving a list of GSM objects."
                    raise RuntimeError(msg)
                else:
                    num_parts = [num_parts] * num_filts
            self._meas_noise_filters = [None] * num_filts
            self._import_w_factory_lst = [None] * num_filts
            for ii, gsm in enumerate(gsm_lst):
                if rng is None:
                    rng = rnd.default_rng()
                if gsm.type is GSMTypes.STUDENTS_T:
                    (
                        self._meas_noise_filters[ii],
                        self._import_w_factory_lst[ii],
                    ) = self._define_student_t_pf(gsm, rng, num_parts[ii])
                elif GSMTypes.CAUCHY:
                    (
                        self._meas_noise_filters[ii],
                        self._import_w_factory_lst[ii],
                    ) = self._define_cauchy_pf(gsm, rng, num_parts[ii])
                else:
                    msg = (
                        "GSM filter can not automatically setup Bootstrap "
                        + "filter for GSM Type {:s}. ".format(gsm.type)
                        + "Update implementation."
                    )
                    raise RuntimeError(msg)
        else:
            msg = (
                "Incorrect input arguement combination. See documentation for details."
            )
            raise RuntimeError(msg)

    def set_process_noise_model(
        self, initial_est=None, filter_length=None, startup_delay=None
    ):
        """Sets the filter parameters for estimating process noise.

        This assumes the same process noise model as in
        :cite:`VilaValls2011_BayesianFilteringforNonlinearStateSpaceModelsinSymmetricStableMeasurementNoise`.

        Parameters
        ----------
        initial_est : N x N numpy array, optional
            Initial estimate of the covariance, can either be set here or by
            manually setting the process noise variable. The default is None,
            this assumes process noise was manually set.
        filter_length : int, optional
            Number of past samples to use in the filter, must be > 1. The
            default is None, this implies all past samples are used or the
            previously set value is maintained.
        startup_delay : int, optional
            Number of samples to delay before runing the filter, used to fill
            the FIFO buffers. Must be >= 1. The default is None, this means
            either the previous value is maintained or a value of 1 is used.

        Returns
        -------
        None.
        """
        self.enable_proc_noise_estimation = True

        if filter_length is not None:
            self._procNoiseEstimator.maxlen = filter_length
        if initial_est is None and (
            self.proc_noise is None or self.proc_noise.size <= 0
        ):
            msg = (
                "Please manually set the initial process noise "
                + "or specify a value here."
            )
            warn(msg)
        elif initial_est is not None:
            self.proc_noise = initial_est
        if startup_delay is not None:
            self._procNoiseEstimator.startup_delay = startup_delay

    @property
    def cov(self):
        """Covariance of the filter."""
        if self._coreFilter is None:
            return np.array([[]])
        else:
            return self._coreFilter.cov

    @cov.setter
    def cov(self, val):
        self._coreFilter.cov = val

    @property
    def proc_noise(self):
        """Wrapper for the process noise covariance of the core filter."""
        if self._coreFilter is None:
            return np.array([[]])
        else:
            return self._coreFilter.proc_noise

    @proc_noise.setter
    def proc_noise(self, val):
        self._coreFilter.proc_noise = val

    @property
    def meas_noise(self):
        """Measurement noise of the core filter, estimated online and does not need to be set."""
        if self._coreFilter is None:
            return np.array([[]])
        else:
            return self._coreFilter.meas_noise

    @meas_noise.setter
    def meas_noise(self, val):
        warn(
            "Measurement noise is estimated online. NOT SETTING VALUE HERE.",
            RuntimeWarning,
        )

    def predict(self, timestep, cur_state, **kwargs):
        """Prediction step of the GSM filter.

        This optionally estimates the process noise then calls the core filters
        prediction function.
        """
        return self._coreFilter.predict(timestep, cur_state, **kwargs)

    def correct(self, timestep, meas, cur_state, core_filt_kwargs={}):
        """Correction step of the GSM filter.

        This optionally estimates the measurement noise then calls the core
        filters correction function.
        """
        # setup core filter for estimating measurement noise during
        # correction function call
        def est_meas_noise(est_meas, inov_cov):
            m_diag = np.nan * np.ones(est_meas.size)
            f_meas = (meas - est_meas).ravel()
            inov_cov = np.diag(inov_cov)
            for ii, filt in enumerate(self._meas_noise_filters):
                filt.importance_weight_fnc = self._import_w_factory_lst[ii](
                    inov_cov[ii]
                )
                filt.predict(timestep)
                state = filt.correct(timestep, f_meas[ii].reshape((1, 1)))
                if state.size == 3:
                    m_diag[ii] = state[2] * state[1] ** 2  # z * sig^2
                else:
                    m_diag[ii] = state[1] * state[0] ** 2  # z * sig^2
            return np.diag(m_diag)

        self._coreFilter.set_measurement_noise_estimator(est_meas_noise)

        pred_cov = self.cov.copy()
        cor_state, meas_fit_prob = self._coreFilter.correct(
            timestep, meas, cur_state, **core_filt_kwargs
        )

        # update process noise estimate (if applicable)
        if self.enable_proc_noise_estimation:
            self.proc_noise = self._procNoiseEstimator.estimate_next(
                self.proc_noise, cur_state, pred_cov, cor_state, self.cov
            )
        return cor_state, meas_fit_prob

    def plot_particles(self, filt_inds, dist_inds, **kwargs):
        """Plots the particle distribution for every given measurement index.

        Parameters
        ----------
        filt_inds : int or list
            Index of the measurement index/indices to plot the particle distribution
            for.
        dist_inds : int or list
            Index of the particle(s) in the given filter(s) to plot. See the
            :meth:`.BootstrapFilter.plot_particles` for details.
        **kwargs : dict
            Additional keyword arguements. See :meth:`.BootstrapFilter.plot_particles`.

        Returns
        -------
        figs : dict
            Each value in the dictionary is a matplotlib figure handle.
        keys : list
            Each value is a string corresponding to a key in the resulting dictionary
        """
        figs = {}
        key_base = "meas_noise_particles_F{:02d}_D{:02d}"
        keys = []
        if not isinstance(filt_inds, list):
            key = key_base.format(filt_inds, dist_inds)
            figs[key] = self._meas_noise_filters[filt_inds].plot_particles(
                dist_inds, **kwargs
            )
            keys.append(key)
        else:
            for ii in filt_inds:
                key = key_base.format(ii, dist_inds)
                figs[key] = self._meas_noise_filters[ii].plot_particles(
                    dist_inds, **kwargs
                )
                keys.append(key)
        return figs, keys
