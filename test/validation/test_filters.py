import sys
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
import scipy.linalg as la
import matplotlib.pyplot as plt

import gncpy.filters as gfilts
import gncpy.dynamics as gdyn
import gncpy.distributions as gdistrib


global_seed = 69
d2r = np.pi / 180
r2d = 1 / d2r

debug_figs = False


def test_KF_dynObj():  # noqa
    m_noise = 0.02
    p_noise = 0.2

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)

    filt = gfilts.KalmanFilter()
    filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.cov = 0.25 * np.eye(4)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)

    vx0 = 2
    vy0 = 1

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    t_states = states.copy()

    A = gdyn.DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        n_state = m_mat @ (t_states[kk + 1, :] + np.sqrt(np.diag(filt.proc_noise))
                           * rng.standard_normal(1)).reshape((4, 1))
        meas = n_state + m_noise * rng.standard_normal(n_state.size).reshape(n_state.shape)

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = states - t_states

    # plot states
    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, states[:, ii], color='b')
            fig.axes[ii].plot(time, t_states[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('States (obj)')
        fig.tight_layout()

        # plot stds
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, stds[:, ii], color='b')
            fig.axes[ii].plot(time, pre_stds[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

    sig_num = 1
    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    assert all(bounding > (stats.norm.sf(-sig_num) - stats.norm.sf(sig_num))), 'bounding failed'

    # z = (states[-1, :] - t_states[-1, :]) / (stds[-1, :])
    # level_sig = 0.05
    # pval = 2 * stats.norm.sf(np.abs(z))
    # assert all(pval > level_sig), 'p value too low'


def test_KF_mat():  # noqa
    m_noise = 0.02
    p_noise = 0.2

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)

    filt = gfilts.KalmanFilter()
    filt.set_state_model(state_mat=gdyn.DoubleIntegrator().get_state_mat(0, dt))
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.cov = 0.25 * np.eye(4)
    gamma = np.array([0, 0, 1, 1]).reshape((4, 1))
    filt.proc_noise = gamma @ np.array([[p_noise**2]]) @ gamma.T
    filt.meas_noise = m_noise**2 * np.eye(2)

    vx0 = 2
    vy0 = 1

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    t_states = states.copy()

    A = gdyn.DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        n_state = m_mat @ (t_states[kk + 1, :].reshape((4, 1))
                           + gamma * p_noise * rng.standard_normal(1))
        meas = n_state + m_noise * rng.standard_normal(n_state.size).reshape(n_state.shape)

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = states - t_states

    # plot states
    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, states[:, ii], color='b')
            fig.axes[ii].plot(time, t_states[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('States (mat)')
        fig.tight_layout()

        # plot stds
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, stds[:, ii], color='b')
            fig.axes[ii].plot(time, pre_stds[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (mat)')
        fig.tight_layout()

    sig_num = 1
    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    assert all(bounding > (stats.norm.sf(-sig_num) - stats.norm.sf(sig_num))), 'bounding failed'

    # z = (states[-1, :] - t_states[-1, :]) / (stds[-1, :])
    # level_sig = 0.05
    # pval = 2 * stats.norm.sf(np.abs(z))
    # assert all(pval > level_sig), 'p value too low'


def test_EKF_dynObj():  # noqa
    rng = rnd.default_rng(global_seed)

    p_posx_std = 0.2
    p_posy_std = 0.2
    p_turn_std = 0.1 * d2r

    m_posx_std = 0.2
    m_posy_std = 0.2
    m_turn_std = 0.2 * d2r

    dt = 0.01
    t0, t1 = 0, 10 + dt

    coordTurn = gdyn.CoordinatedTurn(dt=dt)
    filt = gfilts.ExtendedKalmanFilter()
    filt.set_state_model(dyn_obj=coordTurn)
    m_mat = np.eye(5)
    filt.set_measurement_model(meas_mat=m_mat)
    filt.proc_noise = coordTurn.get_dis_process_noise_mat(dt, p_posx_std,
                                                          p_posy_std,
                                                          p_turn_std)
    filt.meas_noise = np.diag([m_posx_std, m_posx_std, m_posy_std, m_posy_std, m_turn_std])**2
    filt.cov = 0.03**2 * np.eye(5)
    filt.cov[4, 4] = (0.3 * d2r)**2

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 5))
    stds = np.nan * np.ones(states.shape)
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    states[0, :] = np.array([10, 0, 0, 10, 25 * d2r])
    t_states = states.copy()

    gamma = np.array([[dt**2 / 2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2 / 2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
    Q = np.diag([p_posx_std**2, p_posy_std**2, p_turn_std**2])
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((5, 1))).flatten()

        t_states[kk + 1, :] = coordTurn.propagate_state(t, t_states[kk, :].reshape((5, 1))).flatten()
        meas = m_mat @ (t_states[kk + 1, :].reshape((5, 1)) + gamma @ np.sqrt(np.diag(Q)).reshape((3, 1))
                        * rng.standard_normal(1))
        meas = meas + (np.sqrt(np.diag(filt.meas_noise))
                       * rng.standard_normal(meas.size)).reshape(meas.shape)

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((5, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = states - t_states

    # plot states
    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(coordTurn.state_names):
            fig.add_subplot(5, 1, ii + 1)

            vals = states[:, ii].copy()
            if ii == len(coordTurn.state_names) - 1:
                vals *= r2d
            fig.axes[ii].plot(time, vals, color='b')

            vals = t_states[:, ii].copy()
            if ii == len(coordTurn.state_names) - 1:
                vals *= r2d
            fig.axes[ii].plot(time, vals, color='r')

            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('States (obj)')
        fig.tight_layout()

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        fig.axes[0].plot(states[:, 0], states[:, 2])
        fig.axes[0].grid(True)
        fig.axes[0].set_xlabel('x pos (m)')
        fig.axes[0].set_ylabel('y pos (m)')
        fig.suptitle('Position (obj)')

    # plot stds
        fig = plt.figure()
        for ii, s in enumerate(coordTurn.state_names):
            fig.add_subplot(5, 1, ii + 1)

            vals = stds[:, ii].copy()
            if ii == len(coordTurn.state_names) - 1:
                vals *= r2d
            fig.axes[ii].plot(time, vals, color='b')

            vals = np.abs(errs[:, ii])
            if ii == len(coordTurn.state_names) - 1:
                vals *= r2d
            fig.axes[ii].plot(time, vals, color='r')

            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

    sig_num = 1
    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    print(bounding)
    assert all(bounding > (stats.norm.sf(-sig_num) - stats.norm.sf(sig_num))), 'bounding failed'


def test_STF_dynObj():  # noqa
    m_noise = 0.02
    p_noise = 0.2

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)

    filt = gfilts.StudentsTFilter()
    filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)

    filt.proc_noise_dof = 3
    filt.meas_noise_dof = 3
    filt.dof = 4

    filt.scale = ((filt.dof - 2) / filt.dof) * 0.25**2 * np.eye(4)
    gamma = np.array([0, 0, 1, 1]).reshape((4, 1))
    p_scale = ((filt.proc_noise_dof - 2) / filt.proc_noise_dof) * np.array([[p_noise**2]])
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(dt, p_scale)
    filt.meas_noise = ((filt.meas_noise_dof - 2) / filt.meas_noise_dof) * m_noise**2 * np.eye(2)

    vx0 = 2
    vy0 = 1

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    t_states = states.copy()

    A = gdyn.DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        n_state = m_mat @ (t_states[kk + 1, :].reshape((4, 1)) + gamma * p_noise
                           * rng.standard_t(filt.proc_noise_dof))
        meas = n_state + m_noise * rng.standard_t(filt.meas_noise_dof,
                                                  size=n_state.size).reshape(n_state.shape)

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = states - t_states

    # plot states
    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, states[:, ii], color='b')
            fig.axes[ii].plot(time, t_states[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('States (obj)')
        fig.tight_layout()

        # plot stds
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, stds[:, ii], color='b')
            fig.axes[ii].plot(time, pre_stds[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

    sig_num = 1
    bounding = np.sum(np.abs(errs) <= sig_num * stds, axis=0) / time.size
    print(bounding)
    # check approx bounding
    thresh = stats.t.sf(-sig_num, filt.dof) - stats.t.sf(sig_num, filt.dof)
    assert all(bounding >= thresh - 0.05), 'bounding failed'


def test_UKF_dynObj():  # noqa
    m_noise = 0.02
    p_noise = 0.2

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)

    filt = gfilts.UnscentedKalmanFilter()
    filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.cov = 0.25 * np.eye(4)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)
    alpha = 0.5
    kappa = 1

    vx0 = 2
    vy0 = 1

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    filt.init_sigma_points(states[0, :].reshape((4, 1)), alpha, kappa)

    t_states = states.copy()

    A = gdyn.DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        n_state = m_mat @ (t_states[kk + 1, :] + np.sqrt(np.diag(filt.proc_noise))
                           * rng.standard_normal(1)).reshape((4, 1))
        meas = n_state + m_noise * rng.standard_normal(n_state.size).reshape(n_state.shape)

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = states - t_states

    # plot states
    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, states[:, ii], color='b')
            fig.axes[ii].plot(time, t_states[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('States (obj)')
        fig.tight_layout()

        # plot stds
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, stds[:, ii], color='b')
            fig.axes[ii].plot(time, pre_stds[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

    sig_num = 1
    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    assert all(bounding > (stats.norm.sf(-sig_num) - stats.norm.sf(sig_num))), 'bounding failed'


def test_PF_dyn_fnc():  # noqa
    print('test PF')

    rng = rnd.default_rng(global_seed)
    num_parts = 2000
    t0, t1 = 0, 3
    dt = 0.1
    level_sig = 0.05

    time = np.arange(t0, t1, dt)
    true_state = np.ones((1, 1))
    pred_state = true_state.copy()
    proc_noise_std = np.array([[1.0]])
    proc_mean = np.array([1.0])
    meas_noise_std = np.array([[0.2]])
    F = np.array([[0.75]])
    H = np.array([[2.0]])

    distrib = gdistrib.ParticleDistribution()
    for ii in range(0, num_parts):
        p = gdistrib.Particle()
        p.point = 2 * proc_noise_std * rng.random(true_state.shape) - proc_noise_std + true_state
        distrib.add_particle(p, 1 / num_parts)

    def f(t, x, *args):
        return F @ x

    def transition_prob_fnc(x, mean, *args):
        z = ((x - mean) / proc_noise_std).item()
        return stats.norm.pdf(z)

    def meas_likelihood(meas, est, *args):
        z = ((meas - est) / meas_noise_std).item()
        return stats.norm.pdf(z)

    def proposal_sampling_fnc(x, rng, *args):
        noise = proc_mean + proc_noise_std * rng.standard_normal()
        return x + noise

    def proposal_fnc(x_hat, mean, y, *args):
        # return 1
        z = ((x_hat - mean) / proc_noise_std).item()
        return stats.norm.pdf(z)

    # define particle filter
    pf = gfilts.ParticleFilter()
    pf.rng = rng

    pf.meas_likelihood_fnc = meas_likelihood
    pf.proposal_sampling_fnc = proposal_sampling_fnc
    pf.proposal_fnc = proposal_fnc
    pf.transition_prob_fnc = transition_prob_fnc

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(dyn_fun=f)

    pf.init_from_dist(distrib)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()
        sampling_args = ()
        pred_state = pf.predict(tt, sampling_args=sampling_args)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()

        proposal_args = ()
        pred_state = pf.correct(tt, meas, proposal_args=proposal_args)[0]

    if debug_figs:
        pf.plot_particles(0, title='Final Particle Distribution')

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    print(pred_state)
    print(true_state)
    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_UPF_dyn_fnc():  # noqa
    print('test UPF')

    rng = rnd.default_rng(global_seed)
    num_parts = 100
    t0, t1 = 0, 3
    dt = 0.1
    level_sig = 0.05

    time = np.arange(t0, t1, dt)
    true_state = np.ones((1, 1))
    pred_state = true_state.copy()
    proc_noise_std = np.array([[1.0]])
    proc_mean = np.array([1.0])
    meas_noise_std = np.array([[0.2]])
    F = np.array([[0.75]])
    H = np.array([[2.0]])
    alpha = 0.5
    kappa = 1

    distrib = gdistrib.ParticleDistribution()
    for ii in range(0, num_parts):
        p = gdistrib.Particle()
        p.point = 2 * proc_noise_std * rng.random(true_state.shape) - proc_noise_std + true_state
        p.uncertainty = 0.5**2 * np.eye(1)
        p.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa,
                                             n=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.UnscentedParticleFilter(rng=rng)

    def f(t, *args):
        return F

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        pred_state = pf.predict(tt)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()


        pred_state = pf.correct(tt, meas)[0]

    if debug_figs:
        pf.plot_particles(0, title='Final Particle Distribution')

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    print(pred_state)
    print(true_state)
    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_UPF_dynObj():  # noqa
    print('test UPF dynObj')

    rng = rnd.default_rng(global_seed)
    num_parts = 100
    dt = 0.01
    t0, t1 = 0, 6 + dt
    # level_sig = 0.05

    time = np.arange(t0, t1, dt)

    m_noise_std = 0.02
    p_noise_std = 0.2

    meas_mat = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    dynObj = gdyn.DoubleIntegrator()
    pf = gfilts.UnscentedParticleFilter(use_MCMC=False, rng=rng)
    pf.set_state_model(dyn_obj=dynObj)
    pf.set_measurement_model(meas_mat=meas_mat)

    proc_noise = dynObj.get_dis_process_noise_mat(dt,
                                                  np.array([[p_noise_std**2]]))
    # proc_noise = p_noise_std**2 * np.diag([0, 0, 1, 1])
    pf.proc_noise = proc_noise.copy()
    pf.meas_noise = m_noise_std**2 * np.eye(meas_mat.shape[0])

    true_state = np.array([20, 80, 3, -3]).reshape((4, 1))

    distrib = gdistrib.ParticleDistribution()
    b_cov = np.diag([3**2, 5**2, 2**2, 1])
    alpha = 0.5
    kappa = 1
    spread = 2 * np.sqrt(np.diag(b_cov)).reshape((true_state.shape))
    l_bnd = true_state - spread / 2
    for ii in range(0, num_parts):
        part = gdistrib.Particle()
        part.point = l_bnd + spread * rng.random(true_state.shape)
        part.uncertainty = b_cov.copy()
        part.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa,
                                                n=true_state.size)
        part.sigmaPoints.init_weights()
        part.sigmaPoints.update_points(part.point, part.uncertainty)
        distrib.add_particle(part, 1 / num_parts)

    pf.init_from_dist(distrib)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    xy_pos = np.zeros((time.size, 2))
    xy_pos[0, :] = true_state[0:2, 0]
    true_pos = xy_pos.copy()
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        ukf_kwargs_pred = {'state_mat_args': (dt, )}
        pred_state = pf.predict(tt, ukf_kwargs=ukf_kwargs_pred)

        # calculate true state and measurement for this timestep
        p_noise = rng.multivariate_normal(np.zeros(proc_noise.shape[0]),
                                          proc_noise)
        true_state = dynObj.propagate_state(tt, true_state, state_args=(dt,))
        true_pos[kk + 1, :] = true_state[0:2, 0]
        m_noise = rng.multivariate_normal(np.zeros(meas_mat.shape[0]),
                                          m_noise_std**2 * np.eye(2))
        meas = meas_mat @ (true_state + p_noise.reshape(true_state.shape)) \
            + m_noise.reshape((2, 1))

        pred_state = pf.correct(tt, meas)[0]
        xy_pos[kk + 1, :] = pred_state[0:2, 0]

    if debug_figs:
        pf.plot_particles(0, title='Final Particle Distribution')

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        fig.axes[0].plot(xy_pos[:, 0], xy_pos[:, 1], label='est')
        fig.axes[0].plot(true_pos[:, 0], true_pos[:, 1], label='true', color='k')
        fig.axes[0].legend()
        fig.axes[0].grid(True)
        ttl = 'X/Y Position'
        fig.suptitle(ttl)
        fig.axes[0].set_xlabel('x-position (m)')
        fig.axes[0].set_ylabel('y-position (m)')
        fig.canvas.manager.set_window_title(ttl)

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    # note this calculation fails with the given process noise
    # exp_cov = la.solve_discrete_are(dynObj.get_state_mat(0, dt).T, meas_mat.T,
    #                                 proc_noise,
    #                                 m_noise_std**2 * np.eye(2))

    # sqrt_inv_cov = la.inv(la.cholesky(pf.cov))
    # inv_cov = sqrt_inv_cov.T @ sqrt_inv_cov
    # chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)
    # crit_val = stats.chi2.ppf(1 - level_sig, df=true_state.size)

    print(pred_state.flatten())
    print(true_state.flatten())
    print(pf.cov)
    # print((chi_stat, crit_val))
    # print(exp_cov)
    # Note test method is likely incorrectly implemented for multivariate z
    # assert chi_stat < crit_val, "values are different"


def test_MCMC_UPF_dyn_fnc():  # noqa
    print('test MCMC-UPF')

    rng = rnd.default_rng(global_seed)
    num_parts = 30
    t0, t1 = 0, 3
    dt = 0.1
    level_sig = 0.05

    time = np.arange(t0, t1, dt)
    true_state = np.ones((1, 1))
    pred_state = true_state.copy()
    proc_noise_std = np.array([[1.0]])
    proc_mean = np.array([1.0])
    meas_noise_std = np.array([[0.2]])
    F = np.array([[0.75]])
    H = np.array([[2.0]])
    alpha = 0.5
    kappa = 1

    distrib = gdistrib.ParticleDistribution()
    for ii in range(0, num_parts):
        p = gdistrib.Particle()
        p.point = 2 * proc_noise_std * rng.random(true_state.shape) - proc_noise_std + true_state
        p.uncertainty = 0.5**2 * np.eye(1)
        p.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa,
                                             n=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.UnscentedParticleFilter(use_MCMC=True)

    def f(t, *args):
        return F

    def meas_likelihood(meas, est, *args):
        z = ((meas - est) / meas_noise_std).item()
        return stats.norm.pdf(z)

    def proposal_sampling_fnc(x, rng):
        noise = proc_mean + proc_noise_std * rng.standard_normal()
        return x + noise

    def proposal_fnc(x_hat, cond, cov, *args):
        return 1
        # z = ((x_hat - cond) / proc_noise_std).item()
        # return stats.norm.pdf(z)

    pf.meas_likelihood_fnc = meas_likelihood
    pf.proposal_sampling_fnc = proposal_sampling_fnc
    pf.proposal_fnc = proposal_fnc

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()
        sampling_args = (rng, )
        pred_state = pf.predict(tt)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()

        proposal_args = ()
        move_kwargs = {'rng': rng, 'sampling_args': sampling_args,
                       'proposal_args': proposal_args}
        pred_state = pf.correct(tt, meas, sampling_args=sampling_args,
                                proposal_args=proposal_args, rng=rng,
                                move_kwargs=move_kwargs)[0]

    if debug_figs:
        pf.plot_particles(0, title='Final Particle Distribution')

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    print(pred_state)
    print(true_state)
    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_max_corr_ent_UKF_dynObj():  # noqa
    m_noise = 0.02
    p_noise = 0.2

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)

    filt = gfilts.MaxCorrEntUKF()
    filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.cov = 0.25 * np.eye(4)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)
    filt.kernel_bandwidth = 10
    alpha = 0.5
    kappa = 1

    vx0 = 2
    vy0 = 1

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    filt.init_sigma_points(states[0, :].reshape((4, 1)), alpha, kappa)

    t_states = states.copy()

    A = gdyn.DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        n_state = m_mat @ (t_states[kk + 1, :] + np.sqrt(np.diag(filt.proc_noise))
                           * rng.standard_normal(1)).reshape((4, 1))
        meas = n_state + m_noise * rng.standard_normal(n_state.size).reshape(n_state.shape)

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)),
                                         states[kk, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = states - t_states

    # plot states
    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, states[:, ii], color='b')
            fig.axes[ii].plot(time, t_states[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('States (obj)')
        fig.tight_layout()

        # plot stds
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, stds[:, ii], color='b')
            fig.axes[ii].plot(time, pre_stds[:, ii], color='r')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

    sig_num = 1
    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    assert all(bounding > (stats.norm.sf(-sig_num) - stats.norm.sf(sig_num))), 'bounding failed'


def test_MCUPF_dyn_fnc():  # noqa
    print('test MCUPF')

    rng = rnd.default_rng(global_seed)
    num_parts = 20
    t0, t1 = 0, 3
    dt = 0.1
    level_sig = 0.05

    time = np.arange(t0, t1, dt)
    true_state = np.ones((1, 1))
    pred_state = true_state.copy()
    proc_noise_std = np.array([[1.0]])
    proc_mean = np.array([1.0])
    meas_noise_std = np.array([[0.2]])
    F = np.array([[0.75]])
    H = np.array([[2.0]])
    alpha = 0.5
    kappa = 1

    distrib = gdistrib.ParticleDistribution()
    for ii in range(0, num_parts):
        p = gdistrib.Particle()
        p.point = 2 * proc_noise_std * rng.random(true_state.shape) - proc_noise_std + true_state
        p.uncertainty = 0.5**2 * np.eye(1)
        p.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa,
                                             n=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.MaxCorrEntUPF(use_MCMC=False)
    pf.kernel_bandwidth = 10

    def f(t, *args):
        return F

    def meas_likelihood(meas, est, *args):
        z = ((meas - est) / meas_noise_std).item()
        return stats.norm.pdf(z)

    def proposal_sampling_fnc(x, rng):
        noise = proc_mean + proc_noise_std * rng.standard_normal()
        return x + noise

    def proposal_fnc(x_hat, cond, cov, *args):
        return 1
        # z = ((x_hat - cond) / proc_noise_std).item()
        # return stats.norm.pdf(z)

    pf.meas_likelihood_fnc = meas_likelihood
    pf.proposal_sampling_fnc = proposal_sampling_fnc
    pf.proposal_fnc = proposal_fnc

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()
        sampling_args = (rng, )
        past_state = pred_state.copy()
        pred_state = pf.predict(tt)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()

        proposal_args = ()
        move_kwargs = {'rng': rng, 'sampling_args': sampling_args,
                       'proposal_args': proposal_args}
        pred_state = pf.correct(tt, meas, past_state, sampling_args=sampling_args,
                                proposal_args=proposal_args, rng=rng,
                                move_kwargs=move_kwargs)[0]

    if debug_figs:
        pf.plot_particles(0, title='Final Particle Distribution')

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    print(pred_state)
    print(true_state)
    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_MCMC_MCUPF_dyn_fnc():  # noqa
    print('test MCMC-MCUPF')

    rng = rnd.default_rng(global_seed)
    num_parts = 10
    t0, t1 = 0, 3
    dt = 0.1
    level_sig = 0.05

    time = np.arange(t0, t1, dt)
    true_state = np.ones((1, 1))
    pred_state = true_state.copy()
    proc_noise_std = np.array([[1.0]])
    proc_mean = np.array([1.0])
    meas_noise_std = np.array([[0.2]])
    F = np.array([[0.75]])
    H = np.array([[2.0]])
    alpha = 0.5
    kappa = 1

    distrib = gdistrib.ParticleDistribution()
    for ii in range(0, num_parts):
        p = gdistrib.Particle()
        p.point = 2 * proc_noise_std * rng.random(true_state.shape) - proc_noise_std + true_state
        p.uncertainty = 0.5**2 * np.eye(1)
        p.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa,
                                             n=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.MaxCorrEntUPF(use_MCMC=True)
    pf.kernel_bandwidth = 10

    def f(t, *args):
        return F

    def meas_likelihood(meas, est, *args):
        z = ((meas - est) / meas_noise_std).item()
        return stats.norm.pdf(z)

    def proposal_sampling_fnc(x, rng):
        noise = proc_mean + proc_noise_std * rng.standard_normal()
        return x + noise

    def proposal_fnc(x_hat, cond, cov, *args):
        return 1
        # z = ((x_hat - cond) / proc_noise_std).item()
        # return stats.norm.pdf(z)

    pf.meas_likelihood_fnc = meas_likelihood
    pf.proposal_sampling_fnc = proposal_sampling_fnc
    pf.proposal_fnc = proposal_fnc

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()
        sampling_args = (rng, )
        past_state = pred_state.copy()
        pred_state = pf.predict(tt)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()

        proposal_args = ()
        move_kwargs = {'rng': rng, 'sampling_args': sampling_args,
                       'proposal_args': proposal_args}
        pred_state = pf.correct(tt, meas, past_state, sampling_args=sampling_args,
                                proposal_args=proposal_args, rng=rng,
                                move_kwargs=move_kwargs)[0]

    if debug_figs:
        pf.plot_particles(0, title='Final Particle Distribution')

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    exp_cov = la.solve_discrete_are(F.T, H.T, proc_noise_std**2,
                                    meas_noise_std**2)
    exp_std = np.sqrt(exp_cov)
    z_stat = (pred_state - true_state).T @ la.inv(exp_std)
    z_stat = z_stat.item()
    p_val = stats.norm.sf(abs(z_stat)) * 2

    print(pred_state)
    print(true_state)
    assert p_val > level_sig, "p-value too low, final state is unexpected"


# %% Main
if __name__ == "__main__":
    from timeit import default_timer as timer

    plt.close('all')
    debug_figs = True

    start = timer()

    # test_KF_dynObj()
    # test_KF_mat()

    # test_EKF_dynObj()

    # test_STF_dynObj()

    # test_UKF_dynObj()
    # test_max_corr_ent_UKF_dynObj()

    # test_PF_dyn_fnc()
    # test_UPF_dyn_fnc()
    test_UPF_dynObj()
    # test_MCMC_UPF_dyn_fnc()
    # test_MCUPF_dyn_fnc()
    # test_MCMC_MCUPF_dyn_fnc()

    end = timer()
    print('{:.2f} s'.format(end - start))
    print('Close all plots to exit')
    plt.show()
