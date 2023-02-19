import types

import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import scipy.optimize as sopt
import scipy.linalg as sla
import scipy.integrate as s_integrate
import scipy.stats as stats
import abc
from collections import deque
from warnings import warn
from copy import deepcopy
import matplotlib.pyplot as plt

import gncpy.math as gmath
import gncpy.plotting as pltUtil
import gncpy.distributions as gdistrib
from serums.enums import GSMTypes
import gncpy.dynamics.basic as gdyn
import gncpy.errors as gerr

def GeneralizedCovarianceIntersection(est_list, cov_list, weight_list, meas_model_list, optimizer=None):

    def obj_func(w_list, cov_list=None, meas_model_list=None):
        if cov_list is None or meas_model_list is None:
            raise ValueError("cov_list cannot be nonetype")
        if len(cov_list) != len(w_list) or len(cov_list) != len(meas_model_list) or len(w_list) != len(meas_model_list):
            raise ValueError("must be same amount of weights and covariances")
        new_cov = np.zeros(np.shape(cov_list[0]))
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
        # opt_out = sopt.minimize(obj_func, weight_list, method=optimizer, args=(cov_list, obs_mat_list,),
        #                         constraints=(sopt.LinearConstraint(np.ones(np.shape(est_list[0].reshape(-1, 1))), lb=0, ub=1)))
        # opt_out = sopt.minimize(obj_func, weight_list, method=optimizer, args=(cov_list, obs_mat_list,),
        #                             constraints=(sopt.NonlinearConstraint(constraint_func, lb=0, ub=1)))
    else:
        opt_out = sopt.minimize(obj_func, weight_list, args=(cov_list, obs_mat_list,), constraints=(sopt.LinearConstraint(np.array([1,1]).reshape(1, 2), lb=1, ub=1), sopt.LinearConstraint(np.eye(2), lb=0, ub=1)))
        # opt_out = sopt.minimize(obj_func, weight_list, args=(cov_list, obs_mat_list,), constraints=(sopt.NonlinearConstraint(constraint_func, lb=0, ub=1)))
    new_cov = np.zeros(np.shape(cov_list[0]))
    # weight_list = opt_out.x
    new_weight_list = []
    for w in opt_out.x:
        new_weight_list.append(w / np.sum(opt_out.x))
    # weight_list =
    for ii, cov in enumerate(cov_list):
        new_cov = new_cov + new_weight_list[ii] * cov_list[ii]

    gain_list = []
    new_est = np.zeros(np.shape(est_list[0]))
    for ii, cov in enumerate(cov_list):
        gain_list.append(new_weight_list[ii] * new_cov @ obs_mat_list[ii] @ la.inv(cov))
        new_est = new_est + gain_list[ii] @ est_list[ii]

    return new_est, new_cov, new_weight_list



# class GeneralizedCovarianceIntersection:
#     def __init__(self, meas_fun_list, weight_list=None filter=None, optimizer=None):
#
#     if weight_list is not None:
#         self.weight_list = gain_list
#     else:
#         self.weight_list = [1/len(meas_fun_list) for ii in range(0, len(meas_fun_list))]
#
#     self.gain_list = []
#     self.filter = filter
#     self.optimizer = optimizer
#
#     def _calculate_weights(self, cov_list):
#         new_cov = np.array([])
#         self.weight_list = []
#
#         self.weight_list = self.optimizer.optimize(function, self.weight_list, constraints)
#
#         for ii, cov in enumerate(cov_list):
#             new_cov = self.weight_list[ii] * la.inv(cov_list)
#
#         #Calculate gains
#         self.gain_list = []
#
#         return la.inv(new_cov)
#
#     def correct(self, mean_list, cov_list, obs_list):
#
#         for meas in meas_list:
#         # call correct for inner filter
#
#         # weight calculation
#         new_cov = self._calculate_weights(cov_list)
#
#         new_est = 0
#         for ii, weight in enumerate(self.weight_list):
#             if self.gain_list is not None:
#                 self.gain_list[ii] = self.weight_list[ii] * new_cov @ la.inv(cov_list[ii])
#             else:
#                 self.gain_list.append(self.weight_list[ii] * new_cov @ la.inv(cov_list[ii]))
#             new_est += self.gain_list[ii] * mean_list[ii]
#
#         return new_est, new_cov
#
#         # convex optimization?
#         # constraint sum(weights) = 1
#         # minimize trace of Pz = omega_1 inv(P1) + ... omega_n inv(Pn)
#         # New mean
        # new cov

