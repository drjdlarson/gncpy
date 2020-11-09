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
