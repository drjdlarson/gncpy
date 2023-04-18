import types

import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt


def GeneralizedCovarianceIntersection(est_list, cov_list, weight_list, meas_model_list, optimizer=None):

    def obj_func(w_list, cov_list=None, meas_model_list=None):
        if cov_list is None or meas_model_list is None:
            raise ValueError("cov_list cannot be nonetype")
        if len(cov_list) != len(w_list) or len(cov_list) != len(meas_model_list) or len(w_list) != len(meas_model_list):
            raise ValueError("must be same amount of weights and covariances")
        new_cov = np.zeros((meas_model_list[0].T @ cov_list[0] @ meas_model_list[0]).shape)
        for ii, cov in enumerate(cov_list):
            new_cov = new_cov + w_list[ii] * meas_model_list[ii].T @ la.inv(cov_list[ii]) @ meas_model_list[ii]
        return np.trace(la.inv(new_cov))

    obs_mat_list = []
    for ii, model in enumerate(meas_model_list):
        if isinstance(model, types.FunctionType):
            obs_mat_list.append(model(0, est_list[ii]))
        else:
            obs_mat_list.append(model)

    if optimizer is not None:
        opt_out = sopt.minimize(obj_func, weight_list, method=optimizer, args=(cov_list, obs_mat_list,))
    else:
        opt_out = sopt.minimize(obj_func, weight_list, args=(cov_list, obs_mat_list,), constraints=(sopt.LinearConstraint(np.ones(len(est_list)).reshape([1, len(est_list)]), lb=1, ub=1), sopt.LinearConstraint(np.eye(len(est_list)), lb=0, ub=1)))
    new_cov = np.zeros(np.shape(obs_mat_list[0].T @ cov_list[0] @ obs_mat_list[0]))

    new_weight_list = []
    for w in opt_out.x:
        new_weight_list.append(w / np.sum(opt_out.x))

    for ii, cov in enumerate(cov_list):
        new_cov = new_cov + new_weight_list[ii] * obs_mat_list[ii].T @ la.inv(cov_list[ii]) @ obs_mat_list[ii]

    new_cov = la.inv(new_cov)
    gain_list = []
    new_est = np.zeros(np.shape(new_cov @ obs_mat_list[0].T @ la.inv(cov_list[0]) @ est_list[0]))
    for ii, cov in enumerate(cov_list):
        gain_list.append(new_weight_list[ii] * new_cov @ obs_mat_list[ii].T @ la.inv(cov))
        new_est = new_est + gain_list[ii] @ est_list[ii]

    return new_est, new_cov, new_weight_list
