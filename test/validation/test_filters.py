import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import numpy.testing as test
import scipy.stats as stats
from scipy.stats.distributions import chi2

import gncpy.filters as filters


def test_ParticleFilter():
    rng = rnd.default_rng()
    num_parts = 5000
    max_time = 30

    true_state = np.zeros((1, 1))
    noise_state = true_state.copy()
    proc_noise_std = np.array([[0.2]])
    proc_mean = np.array([1])
    meas_noise_std = np.array([[1]])
    F = np.array([[0.75]])
    H = np.array([[2]])

    init_parts = [2 * rng.random(true_state.shape) - 1
                  for ii in range(0, num_parts)]

    # define particle filter
    particleFilter = filters.ParticleFilter()

    def f(x, **kwargs):
        cov = kwargs['proc_cov']
        rng = kwargs['rng']
        return F @ x + rng.multivariate_normal(proc_mean,
                                               cov).reshape(x.shape)

    def meas_likelihood(meas, est, **kwargs):
        m = meas.copy().reshape(meas.size)
        meas_cov = kwargs['meas_cov']
        return stats.multivariate_normal.pdf(m, mean=est,
                                             cov=meas_cov)

    def meas_mod(x, **kwargs):
        return H @ x

    particleFilter.dyn_fnc = f
    particleFilter.meas_likelihood_fnc = meas_likelihood
    particleFilter.set_meas_model(meas_mod)
    particleFilter.init_particles(init_parts)

    for tt in range(1, max_time):
        _ = particleFilter.predict(proc_cov=proc_noise_std**2,
                                   rng=rng)

        # calculate true state and measurement for this timestep
        true_state = F @ true_state + proc_mean
        noise_state = F @ noise_state \
            + rng.multivariate_normal(proc_mean,
                                      proc_noise_std**2).reshape(true_state.shape)
        meas = H @ noise_state + meas_noise_std * rng.normal()

        output = particleFilter.correct(meas, meas_cov=meas_noise_std**2)[0]

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    alpha = 0.05
    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)

    # calc from https://math.bme.hu/~marib/tobbvalt/tv5.pdf
    z_stat = (output - true_state).T @ la.inv(exp_cov) \
        @ (output - true_state)
    z_stat = z_stat.item()
    l_crit_val = chi2.ppf(alpha/2, output.size)
    u_crit_val = chi2.ppf(1 - alpha/2, output.size)

    assert z_stat > l_crit_val, "Test statistic too low"
    assert z_stat < u_crit_val, "Test statistic too high"


def test_StudentsTFilter():
    rng = rnd.default_rng(1)
    max_time = 30
    num_monte = 5000

    true_state = np.zeros((1, 1))
    t_state_0 = true_state.copy()
    F = np.array([[0.75]])
    H = np.array([[2]])

    meas_noise_dof = 3
    meas_noise_scale = 1
    proc_noise_dof = 3
    proc_noise_scale = 1

    F = np.array([[0.75]])
    H = np.array([[2]])

    # setup STF
    stf = filters.StudentsTFilter()

    Q = np.array([[proc_noise_scale]])
    R = np.array([[meas_noise_scale]])

    stf.set_state_mat(mat=F)
    stf.set_input_mat(mat=np.zeros((true_state.shape[0], 1)))
    stf.set_proc_noise(mat=Q)

    stf.set_meas_mat(mat=H)
    stf.meas_noise = R

    stf.meas_noise_dof = meas_noise_dof
    stf.proc_noise_dof = proc_noise_dof
    stf.dof = 3

    stf.scale = 10 * Q

    def f(x, **kwargs):
        rng = kwargs['rng']
        return F @ x + proc_noise_scale \
            * rng.standard_t(proc_noise_dof)

    def meas_likelihood(meas, est, **kwargs):
        m = meas.copy().reshape(meas.size)
        return stats.t.pdf(m, meas_noise_dof, loc=est, scale=meas_noise_scale)

    def meas_mod(x, **kwargs):
        return H @ x

    # run STF in monte carlo
    t_dist = np.zeros(num_monte)
    true_state = t_state_0.copy()
    stf_dist = np.zeros(num_monte)
    for ii in range(0, num_monte):
        # stf_state = np.zeros(true_state.shape)
        stf_state = 2 * rng.random(true_state.shape) - 1
        for tt in range(1, max_time):
            stf_state = stf.predict(cur_state=stf_state, rng=rng)

            # calculate true state and measurement for this timestep
            true_state = f(true_state, rng=rng)
            meas = H @ true_state + meas_noise_scale \
                * rng.standard_t(meas_noise_dof)

            stf_state = stf.correct(meas=meas, cur_state=stf_state)[0]
        stf_dist[ii] = stf_state.item()
        t_dist[ii] = true_state[0].item()

    # chi2 test to check monte carlo matches PF particles
    exp_freq, bin_edges = np.histogram(t_dist, bins=11, density=False)
    obs_freq, _ = np.histogram(stf_dist, bins=bin_edges, density=False)

    test_stat, p_val = stats.chisquare(obs_freq, exp_freq, ddof=3)

    alpha = 0.05
    assert p_val > alpha / 2, "Distribution does not match expected"
