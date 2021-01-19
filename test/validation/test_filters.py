import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import numpy.testing as test
import scipy.stats as stats
from scipy.stats.distributions import chi2

import gncpy.filters as filters
import gncpy.sampling as sampling


def test_ParticleFilter():
    rng = rnd.default_rng()
    num_parts = 2000
    max_time = 30

    true_state = np.zeros((1, 1))
    noise_state = true_state.copy()
    proc_noise_std = np.array([[0.2]])
    proc_mean = np.array([1.0])
    meas_noise_std = np.array([[1.0]])
    F = np.array([[0.75]])
    H = np.array([[2.0]])

    init_parts = [2 * rng.random(true_state.shape) - 1
                  for ii in range(0, num_parts)]

    # define particle filter
    particleFilter = filters.ParticleFilter()

    def f(x, **kwargs):
        return F @ x

    def meas_likelihood(meas, est, **kwargs):
        m = meas.copy().reshape(meas.size)
        meas_cov = kwargs['meas_cov']
        return stats.multivariate_normal.pdf(m, mean=est,
                                             cov=meas_cov)

    def meas_mod(x, **kwargs):
        return H @ x

    def proposal_sampling_fnc(x, **kwargs):
        cov = kwargs['proc_cov']
        rng = kwargs['rng']
        return rng.multivariate_normal(proc_mean, cov).reshape(x.shape)

    def proposal_fnc(x_hat, cond, **kwargs):
        cov = kwargs['proc_cov']
        return stats.multivariate_normal.pdf((x_hat - cond), mean=proc_mean,
                                             cov=cov)

    particleFilter.dyn_fnc = f
    particleFilter.meas_likelihood_fnc = meas_likelihood
    particleFilter.proposal_sampling_fnc = proposal_sampling_fnc
    particleFilter.proposal_fnc = proposal_fnc

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

        output = particleFilter.correct(meas, proc_cov=proc_noise_std**2,
                                        meas_cov=meas_noise_std**2)[0]

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


def test_UnscentedKalmanFilter():
    rng = rnd.default_rng()
    level_sig = 0.05
    max_time = 50

    alpha = 1
    kappa = 0
    sigma_w = 0.1
    sigma_v = 3

    proc_noise = sigma_w**2
    meas_noise = sigma_v**2

    def dyn_fnc(x, **kwargs):
        a = 0.5
        t = kwargs['t']
        return a * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * t)

    def meas_mod(x, **kwargs):
        b = 1 / 20
        return b * x**2

    state0 = np.array([[5]])
    cov0 = np.array([[2]])

    ukf = filters.UnscentedKalmanFilter()
    ukf.cov = cov0.copy()
    ukf.dyn_fnc = dyn_fnc
    ukf.set_meas_model(meas_mod)
    ukf.set_proc_noise(mat=proc_noise)
    ukf.meas_noise = meas_noise
    ukf.init_sigma_points(state0, alpha, kappa)

    state = state0.copy()
    pred_state = state0.copy()
    for t in range(1, max_time):
        pred_state = ukf.predict(cur_state=pred_state, t=t)

        state = dyn_fnc(state, t=t) + sigma_w * rng.normal()
        meas = meas_mod(state) + sigma_v * rng.normal()

        pred_state = ukf.correct(cur_state=pred_state, meas=meas)[0]

    exp_state = np.array([[3.46935396909752]])
    z_stat = (pred_state - exp_state).T @ la.inv(np.sqrt(ukf.cov))
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_MaxCorrEntUKF():
    rng = rnd.default_rng()
    level_sig = 0.05
    max_time = 50

    alpha = 1
    kappa = 0
    sigma_w = 0.1
    sigma_v = 3

    proc_noise = np.array([[sigma_w**2]])
    meas_noise = np.array([[sigma_v**2]])

    def dyn_fnc(x, **kwargs):
        a = 0.5
        t = kwargs['t']
        return a * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * t)

    def meas_mod(x, **kwargs):
        b = 1 / 20
        return b * x**2

    state0 = np.array([[5]])
    cov0 = np.array([[2]])

    ukf = filters.MaxCorrEntUKF()
    ukf.cov = cov0.copy()
    ukf.dyn_fnc = dyn_fnc
    ukf.set_meas_model(meas_mod)
    ukf.set_proc_noise(mat=proc_noise)
    ukf.meas_noise = meas_noise
    ukf.kernel_bandwidth = 10

    ukf._stateSigmaPoints.alpha = alpha
    ukf._stateSigmaPoints.kappa = kappa
    ukf._stateSigmaPoints.beta = 2
    ukf._stateSigmaPoints.n = state0.size
    ukf._stateSigmaPoints.init_weights()

    state = state0.copy()
    pred_state = state0.copy()
    for t in range(1, max_time):
        orig_state = pred_state.copy()
        pred_state = ukf.predict(cur_state=pred_state, t=t)

        state = dyn_fnc(state, t=t) + sigma_w * rng.normal()
        meas = meas_mod(state) + sigma_v * rng.normal()

        pred_state = ukf.correct(past_state=orig_state, cur_state=pred_state,
                                 meas=meas)[0]

    exp_state = np.array([[3.46935396909752]])
    z_stat = (pred_state - exp_state).T @ la.inv(np.sqrt(ukf.cov))
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_UnscentedParticleFilter():
    rng = rnd.default_rng()
    num_parts = 500
    max_time = 30
    level_sig = 0.05

    alpha = 1
    kappa = 0

    true_state = np.zeros((1, 1))
    noise_state = true_state.copy()
    proc_noise_std = np.array([[0.2]])
    proc_cov = proc_noise_std**2
    proc_mean = np.array([1.0])
    meas_mean = np.array([0.0])
    meas_noise_std = np.array([[0.1]])
    meas_cov = meas_noise_std**2
    F = np.array([[0.75]])
    H = np.array([[2.0]])

    init_parts = [2 * rng.random(true_state.shape) - 1
                  for ii in range(0, num_parts)]
    init_covs = [np.array([[5]]).copy() for ii in range(0, num_parts)]

    # define particle filter
    upf = filters.UnscentedParticleFilter()

    def f(x, **kwargs):
        return F @ x

    def meas_likelihood(meas, est, **kwargs):
        m = meas.copy().reshape(meas.size)
        return stats.multivariate_normal.pdf(m, mean=est,
                                             cov=meas_cov)

    def meas_mod(x, **kwargs):
        return H @ x

    def proposal_sampling_fnc(x, **kwargs):
        rng = kwargs['rng']
        cov = kwargs['cov']
        mean = x.flatten()
        return rng.multivariate_normal(mean, cov).reshape(x.shape)

    def proposal_fnc(x_hat, cond, **kwargs):
        cov = kwargs['cov']
        return stats.multivariate_normal.pdf(x_hat, mean=cond,
                                             cov=cov)

    upf.dyn_fnc = f
    upf.meas_likelihood_fnc = meas_likelihood
    upf.proposal_sampling_fnc = proposal_sampling_fnc
    upf.proposal_fnc = proposal_fnc

    upf.set_meas_model(meas_mod)
    upf.set_proc_noise(mat=proc_noise_std**2)
    upf.meas_noise = meas_cov

    upf.init_particles(init_parts, init_covs)
    upf.init_UKF(alpha, kappa, true_state.size)

    for tt in range(1, max_time):
        upf.predict(rng=rng)

        # calculate true state and measurement for this timestep
        true_state = F @ true_state + proc_mean

        noise = rng.multivariate_normal(proc_mean, proc_cov)
        noise_state = F @ noise_state + noise.reshape(true_state.shape)

        m_noise = rng.multivariate_normal(meas_mean, meas_cov)
        meas = H @ noise_state + m_noise

        pred_state = upf.correct(meas)[0]

    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_MaxCorrEntUPF():
    rng = rnd.default_rng()
    num_parts = 500
    max_time = 30
    level_sig = 0.05

    alpha = 1
    kappa = 0

    true_state = np.zeros((1, 1))
    noise_state = true_state.copy()
    proc_noise_std = np.array([[0.2]])
    proc_cov = proc_noise_std**2
    proc_mean = np.array([1.0])
    meas_mean = np.array([0.0])
    meas_noise_std = np.array([[0.1]])
    meas_cov = meas_noise_std**2
    F = np.array([[0.75]])
    H = np.array([[2.0]])

    init_parts = [2 * rng.random(true_state.shape) - 1
                  for ii in range(0, num_parts)]
    init_covs = [np.array([[5]]).copy() for ii in range(0, num_parts)]

    # define particle filter
    upf = filters.MaxCorrEntUPF()
    upf.kernel_bandwidth = 10

    def f(x, **kwargs):
        return F @ x

    def meas_likelihood(meas, est, **kwargs):
        m = meas.copy().reshape(meas.size)
        return stats.multivariate_normal.pdf(m, mean=est,
                                             cov=meas_cov)

    def meas_mod(x, **kwargs):
        return H @ x

    def proposal_sampling_fnc(x, **kwargs):
        rng = kwargs['rng']
        cov = kwargs['cov']
        mean = x.flatten()
        return rng.multivariate_normal(mean, cov).reshape(x.shape)

    def proposal_fnc(x_hat, cond, **kwargs):
        cov = kwargs['cov']
        return stats.multivariate_normal.pdf(x_hat, mean=cond,
                                             cov=cov)

    upf.dyn_fnc = f
    upf.meas_likelihood_fnc = meas_likelihood
    upf.proposal_sampling_fnc = proposal_sampling_fnc
    upf.proposal_fnc = proposal_fnc

    upf.set_meas_model(meas_mod)
    upf.set_proc_noise(mat=proc_noise_std**2)
    upf.meas_noise = meas_cov

    upf.init_particles(init_parts, init_covs)
    upf.init_UKF(alpha, kappa, true_state.size)

    pred_state = np.mean(init_parts, axis=0)
    for tt in range(1, max_time):
        orig_state = pred_state.copy()
        upf.predict(rng=rng)

        # calculate true state and measurement for this timestep
        true_state = F @ true_state + proc_mean

        noise = rng.multivariate_normal(proc_mean, proc_cov)
        noise_state = F @ noise_state + noise.reshape(true_state.shape)

        m_noise = rng.multivariate_normal(meas_mean, meas_cov)
        meas = H @ noise_state + m_noise

        pred_state = upf.correct(meas, orig_state)[0]

    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_UnscentedParticleFilterMCMC():
    rng = rnd.default_rng()
    num_parts = 300
    max_time = 30
    level_sig = 0.05

    alpha = 1
    kappa = 0

    true_state = np.zeros((1, 1))
    noise_state = true_state.copy()
    proc_noise_std = np.array([[0.2]])
    proc_cov = proc_noise_std**2
    proc_mean = np.array([1.0])
    meas_mean = np.array([0.0])
    meas_noise_std = np.array([[0.1]])
    meas_cov = meas_noise_std**2
    F = np.array([[0.75]])
    H = np.array([[2.0]])

    init_parts = [2 * rng.random(true_state.shape) - 1
                  for ii in range(0, num_parts)]
    init_covs = [np.array([[5]]).copy() for ii in range(0, num_parts)]

    # define particle filter
    upf = filters.UnscentedParticleFilter()

    def f(x, **kwargs):
        return F @ x

    def meas_likelihood(meas, est, **kwargs):
        m = meas.copy().reshape(meas.size)
        return stats.multivariate_normal.pdf(m, mean=est.copy(),
                                             cov=meas_cov)

    def meas_mod(x, **kwargs):
        return H @ x

    def proposal_sampling_fnc(x, **kwargs):
        rng = kwargs['rng']
        cov = kwargs['cov']
        mean = x.copy().flatten()
        return rng.multivariate_normal(mean, cov).reshape(x.shape)

    def proposal_fnc(x_hat, cond, **kwargs):
        cov = kwargs['cov']
        return stats.multivariate_normal.pdf(x_hat, mean=cond,
                                             cov=cov)

    upf.use_MCMC = True

    upf.dyn_fnc = f
    upf.meas_likelihood_fnc = meas_likelihood
    upf.proposal_sampling_fnc = proposal_sampling_fnc
    upf.proposal_fnc = proposal_fnc

    upf.set_meas_model(meas_mod)
    upf.set_proc_noise(mat=proc_noise_std**2)
    upf.meas_noise = meas_cov

    upf.init_particles(init_parts, init_covs)
    upf.init_UKF(alpha, kappa, true_state.size)

    for tt in range(1, max_time):
        upf.predict(rng=rng)

        # calculate true state and measurement for this timestep
        true_state = F @ true_state + proc_mean

        noise = rng.multivariate_normal(proc_mean, proc_cov)
        noise_state = F @ noise_state + noise.reshape(true_state.shape)

        m_noise = rng.multivariate_normal(meas_mean, meas_cov)
        meas = H @ noise_state + m_noise

        pred_state = upf.correct(meas)[0]

    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    assert p_val > level_sig, "p-value too low, final state is unexpected"
