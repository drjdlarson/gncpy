from copy import deepcopy
from gncpy.filters import InteractingMultipleModel, GCIFilter
from warnings import warn
import numpy as np
import gncpy.data_fusion as gdf


class IMMGCIFilter(InteractingMultipleModel, GCIFilter):
    def __init__(
        self,
        meas_model_list=[],
        meas_noise_list=[],
        weight_list=None,
        optimizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
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
        filt_state = {}
        filt_tup_list = []
        for filt in self.in_filt_list:
            filt_dict = filt.save_filter_state()
            filt_tup_list.append((type(filt), filt_dict))

        filt_state["in_filt_list"] = filt_tup_list.copy()
        filt_state["model_trans_mat"] = self.model_trans_mat.copy()
        filt_state["filt_weights"] = self.filt_weights.copy()
        filt_state["cur_out_state"] = self.cur_out_state.copy()
        filt_state["filt_weight_history"] = self.filt_weight_history.copy()
        filt_state["mean_list"] = deepcopy(self.mean_list)
        filt_state["cov_list"] = deepcopy(self.cov_list)
        filt_state["weight_list"] = self.weight_list
        filt_state["optimizer"] = self.optimizer
        filt_state["meas_model_list"] = self.meas_model_list
        filt_state["meas_noise_list"] = self.meas_noise_list

        return filt_state

    def load_filter_state(self, filt_state):
        self.in_filt_list = []
        filt_tup_list = filt_state["in_filt_list"]
        for tup in filt_tup_list:
            cls_type = tup[0]
            if cls_type is not None:
                filt = cls_type()
                filt.load_filter_state(tup[1])
            else:
                filt = None
            self.in_filt_list.append(filt)
        self.model_trans_mat = filt_state["model_trans_mat"]
        self.filt_weights = filt_state["filt_weights"]
        self.cur_out_state = filt_state["cur_out_state"]
        self.filt_weight_history = filt_state["filt_weight_history"]
        self.mean_list = filt_state["mean_list"]
        self.cov_list = filt_state["cov_list"]
        self.weight_list = filt_state["weight_list"]
        self.optimizer = filt_state["optimizer"]
        self.meas_model_list = filt_state["meas_model_list"]
        self.meas_noise_list = filt_state["meas_noise_list"]

    def set_measurement_model(self, meas_model_list=None):
        warn(
            "Measurement models defined in the constructor,"
            " this function will overwrite initialized measurement model list."
        )
        if meas_model_list is not None:
            self.meas_model_list = meas_model_list

    def predict(self, timestep, **kwargs):
        return super().predict(timestep, **kwargs)

    def correct(self, timestep, meas_list, meas_fun_args=(), **kwargs):
        # initialize lists
        est_list = []
        cov_list = []
        meas_fit_prob_list = []
        new_weight_list = []
        all_weights = []
        eye_list = [
            np.eye(self.cov_list[0].shape[0]) for ii in range(len(self.meas_model_list))
        ]
        # loop over motion models
        for ii, filt in enumerate(self.in_filt_list):
            cur_est_list = []
            cur_cov_list = []
            cur_weight_list = []
            saved_state = filt.save_filter_state()
            # loop over measurements
            for jj, meas in enumerate(meas_list):
                filt.load_filter_state(saved_state)
                if isinstance(self.meas_model_list[jj], list):
                    filt.set_measurement_model(meas_fun_lst=self.meas_model_list[jj])
                else:
                    filt.set_measurement_model(meas_mat=self.meas_model_list[jj])
                filt.meas_noise = self.meas_noise_list[ii]
                new_state, new_prob = filt.correct(
                    timestep, meas, self.mean_list[ii].reshape((-1, 1))
                )
                est_list.append(new_state)
                cov_list.append(filt.cov)
                cur_est_list.append(new_state)
                cur_cov_list.append(filt.cov)
                meas_fit_prob_list.append(new_prob)
                cur_weight_list.append(new_prob * self.filt_weights[ii])
            # fuse measurements for each motion model
            model_est, model_cov, model_weights = gdf.GeneralizedCovarianceIntersection(
                cur_est_list,
                cur_cov_list,
                self.weight_list,
                eye_list,
                optimizer=self.optimizer,
            )
            filt.cov = model_cov.copy()
            self.mean_list[ii] = model_est.copy()
            self.cov_list[ii] = filt.cov.copy()
            all_weights = all_weights + model_weights
            new_weight_list.append(np.sum(cur_weight_list))
        if np.sum(new_weight_list) == 0:
            new_weight_list = new_weight_list * 0
        else:
            new_weight_list = new_weight_list / np.sum(new_weight_list)
        self.filt_weights = new_weight_list

        out_meas_fit_prob = 0
        for ii, prob in enumerate(meas_fit_prob_list):
            out_meas_fit_prob += all_weights[ii] * prob

        # compute output state from combined motion models
        out_state = np.zeros(np.shape(self.mean_list[0]))
        for ii in range(0, len(self.in_filt_list)):
            out_state = out_state + new_weight_list[ii] * self.mean_list[ii]
        out_state = out_state.reshape((np.shape(out_state)[0], 1))
        self.cur_out_state = out_state

        return (out_state, out_meas_fit_prob)
