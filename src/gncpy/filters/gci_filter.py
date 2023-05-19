from gncpy.filters.bayes_filter import BayesFilter
from warnings import warn
import numpy as np
import gncpy.data_fusion as gdf


class GCIFilter(BayesFilter):
    def __init__(
        self,
        base_filter=None,
        meas_model_list=[],
        meas_noise_list=[],
        weight_list=None,
        optimizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_filter = base_filter
        self.meas_model_list = meas_model_list
        self.meas_noise_list = meas_noise_list
        self.optimizer = optimizer
        if weight_list is not None:
            self.weight_list = weight_list
        else:
            self.weight_list = [
                1 / (len(self.meas_model_list)) for ii in range(0, len(meas_model_list))
            ]

    def save_filter_state(self):
        filt_state = super().save_filter_state()
        filt_state["base_filter_state"] = self.base_filter.save_filter_state()
        filt_state["base_filter"] = self.base_filter
        filt_state["weight_list"] = self.weight_list
        filt_state["optimizer"] = self.optimizer
        filt_state["meas_model_list"] = self.meas_model_list
        filt_state["meas_noise_list"] = self.meas_noise_list

        return filt_state

    def load_filter_state(self, filt_state):
        super().load_filter_state(filt_state)
        self.base_filter = filt_state["base_filter"]
        self.base_filter.load_filter_state(filt_state["base_filter_state"])
        self.weight_list = filt_state["weight_list"]
        self.optimizer = filt_state["optimizer"]
        self.meas_model_list = filt_state["meas_model_list"]
        self.meas_noise_list = filt_state["meas_noise_list"]

    def set_state_model(self, dyn_obj=None):
        self.base_filter.set_state_model(dyn_obj=dyn_obj)

    def set_measurement_model(self, meas_model_list=None):
        warn(
            "Measurement models defined in the constructor,"
            " this function will overwrite initialized measurement model list."
        )
        if meas_model_list is not None:
            self.meas_model_list = meas_model_list

    # def set_measurement_noise_estimator(self, func_list):
    #     self._est_meas_noise_fnc = func_list

    # def _est_meas(self, timestep, cur_state, n_meas, ii, meas_fun_args):
    #     if isinstance(self.meas_model_list[ii] , types.FunctionType):
    #         meas_mat = self.meas_model_list[ii](meas_fun_args)
    #     else:
    #         meas_mat = self.meas_model_list[ii]
    #     est_meas = meas_mat @ cur_state
    #     return est_meas, meas_mat

    @property
    def cov(self):
        return self.base_filter.cov

    @cov.setter
    def cov(self, val):
        self.base_filter.cov = val

    @cov.getter
    def cov(self):
        return self.base_filter.cov

    @property
    def proc_noise(self):
        return self.base_filter.proc_noise

    @proc_noise.setter
    def proc_noise(self, val):
        self.base_filter.proc_noise = val

    @proc_noise.getter
    def proc_noise(self):
        return self.base_filter.proc_noise

    def predict(self, timestep, cur_state, **kwargs):
        return self.base_filter.predict(timestep, cur_state, **kwargs)

    # def predict(self, timestep, cur_state, cur_input=None, state_mat_args=(), input_mat_args=()):
    #     return self.base_filter.predict(timestep, cur_state, cur_input=cur_input, state_mat_args=state_mat_args, input_mat_args=input_mat_args)
    def correct(self, timestep, meas_list, cur_state, meas_fun_args=()):
        # est_list.append(cur_state)
        n_est_list = []
        n_prob_list = []
        n_cov_list = []
        saved_state = self.base_filter.save_filter_state()
        for ii, meas in enumerate(meas_list):
            self.base_filter.load_filter_state(saved_state)
            if isinstance(self.meas_model_list[ii], list):
                self.base_filter.set_measurement_model(
                    meas_fun_lst=self.meas_model_list[ii]
                )
            else:
                self.base_filter.set_measurement_model(
                    meas_mat=self.meas_model_list[ii]
                )
            self.base_filter.meas_noise = self.meas_noise_list[ii]
            n_est, n_prob = self.base_filter.correct(
                timestep, meas, cur_state, meas_fun_args
            )
            n_est_list.append(n_est)
            n_cov_list.append(self.base_filter.cov)
            n_prob_list.append(n_prob)

        eye_list = [np.eye(len(cur_state)) for ii in range(len(self.meas_model_list))]
        new_est, new_cov, new_weight_list = gdf.GeneralizedCovarianceIntersection(
            n_est_list, n_cov_list, self.weight_list, eye_list, optimizer=self.optimizer
        )
        # new_est, new_cov, new_weight_list = gdf.GeneralizedCovarianceIntersection(n_est_list, n_cov_list, self.weight_list, self.meas_model_list, optimizer=self.optimizer)
        meas_fit_prob = 0
        for ii, prob in enumerate(n_prob_list):
            meas_fit_prob += new_weight_list[ii] * prob

        self.base_filter.cov = new_cov
        self.weight_list = new_weight_list

        return new_est, meas_fit_prob
