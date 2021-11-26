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

    filt_state = filt.save_filter_state()
    filt = gfilts.KalmanFilter()
    filt.load_filter_state(filt_state)

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

    filt_state = filt.save_filter_state()
    filt = gfilts.ExtendedKalmanFilter()
    filt.load_filter_state(filt_state)

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

    filt_state = filt.save_filter_state()
    filt = gfilts.StudentsTFilter()
    filt.load_filter_state(filt_state)

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
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(t))
            sys.stdout.flush()
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

    filt_state = filt.save_filter_state()
    filt = gfilts.UnscentedKalmanFilter()
    filt.load_filter_state(filt_state)

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
    num_parts = 1000
    t0, t1 = 0, 4
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

    filt_state = pf.save_filter_state()
    pf = gfilts.ParticleFilter()
    pf.load_filter_state(filt_state)

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
    t0, t1 = 0, 4
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
                                             num_axes=true_state.size)
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

    filt_state = pf.save_filter_state()
    pf = gfilts.UnscentedParticleFilter()
    pf.load_filter_state(filt_state)

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
    print(exp_cov)
    assert p_val > level_sig, "p-value too low, final state is unexpected"


def test_UPF_dynObj():  # noqa
    print('test UPF dynObj')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)
    num_parts = 250
    dt = 0.01
    t0, t1 = 0, 6 + dt
    level_sig = 0.05

    time = np.arange(t0, t1, dt)

    m_noise_std = 0.02
    p_noise_std = 0.2

    meas_mat = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    dynObj = gdyn.DoubleIntegrator()
    pf = gfilts.UnscentedParticleFilter(use_MCMC=False, rng=filt_rng)
    pf.set_state_model(dyn_obj=dynObj)
    pf.set_measurement_model(meas_mat=meas_mat)

    proc_noise = p_noise_std**2 * np.diag([dt**2, dt**2, 1, 1])
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
                                                num_axes=true_state.size)
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
    crit_val = stats.chi2.ppf(1 - level_sig, df=true_state.size)
    exp_cov = la.solve_discrete_are(dynObj.get_state_mat(0, dt).T, meas_mat.T,
                                    proc_noise,
                                    m_noise_std**2 * np.eye(2))

    inv_cov = la.inv(exp_cov)
    chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)
    # print((chi_stat.item(), crit_val))
    # if chi_stat >= crit_val:
    #     print("values are different (expected cov)")
    # else:
    #     print('pass (expected cov)')

    # inv_cov = la.inv(pf.cov)
    # chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)
    # print((chi_stat.item(), crit_val))
    # if chi_stat >= crit_val:
    #     print("values are different (calculated cov)")
    # else:
    #     print('pass (calculated cov)')

    print(pred_state.flatten())
    print(true_state.flatten())
    print(pf.cov)
    print((chi_stat, crit_val))
    print(exp_cov)
    assert chi_stat < crit_val, "values are different"


def test_MCMC_UPF_dyn_fnc():  # noqa
    print('test MCMC-UPF')

    rng = rnd.default_rng(global_seed)
    num_parts = 50
    t0, t1 = 0, 4
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
                                             num_axes=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.UnscentedParticleFilter(use_MCMC=True, rng=rng)

    def f(t, *args):
        return F

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    filt_state = pf.save_filter_state()
    pf = gfilts.UnscentedParticleFilter()
    pf.load_filter_state(filt_state)

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


def test_MCMC_UPF_dynObj():  # noqa
    print('test MCMC UPF dynObj')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)
    num_parts = 75
    dt = 0.01
    t0, t1 = 0, 6 + dt
    level_sig = 0.05

    time = np.arange(t0, t1, dt)

    m_noise_std = 0.02
    p_noise_std = 0.2

    meas_mat = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    dynObj = gdyn.DoubleIntegrator()
    pf = gfilts.UnscentedParticleFilter(use_MCMC=True, rng=filt_rng)
    pf.set_state_model(dyn_obj=dynObj)
    pf.set_measurement_model(meas_mat=meas_mat)

    proc_noise = p_noise_std**2 * np.diag([dt**2, dt**2, 1, 1])
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
                                                num_axes=true_state.size)
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
    crit_val = stats.chi2.ppf(1 - level_sig, df=true_state.size)
    exp_cov = la.solve_discrete_are(dynObj.get_state_mat(0, dt).T, meas_mat.T,
                                    proc_noise,
                                    m_noise_std**2 * np.eye(2))

    inv_cov = la.inv(exp_cov)
    chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)
    # print((chi_stat.item(), crit_val))
    # if chi_stat >= crit_val:
    #     print("values are different (expected cov)")
    # else:
    #     print('pass (expected cov)')

    # inv_cov = la.inv(pf.cov)
    # chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)
    # print((chi_stat.item(), crit_val))
    # if chi_stat >= crit_val:
    #     print("values are different (calculated cov)")
    # else:
    #     print('pass (calculated cov)')

    print(pred_state.flatten())
    print(true_state.flatten())
    print(pf.cov)
    print((chi_stat, crit_val))
    print(exp_cov)
    assert chi_stat < crit_val, "values are different"


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

    filt_state = filt.save_filter_state()
    filt = gfilts.MaxCorrEntUKF()
    filt.load_filter_state(filt_state)

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
                                             num_axes=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.MaxCorrEntUPF(use_MCMC=False, rng=rng)
    pf.kernel_bandwidth = 10

    def f(t, *args):
        return F

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    filt_state = pf.save_filter_state()
    pf = gfilts.MaxCorrEntUPF()
    pf.load_filter_state(filt_state)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        past_state = pred_state.copy()
        pred_state = pf.predict(tt)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()

        pred_state = pf.correct(tt, meas, past_state)[0]

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
                                             num_axes=true_state.size)
        p.sigmaPoints.init_weights()
        p.sigmaPoints.update_points(p.point, p.uncertainty)
        distrib.add_particle(p, 1 / num_parts)

    # define particle filter
    pf = gfilts.MaxCorrEntUPF(use_MCMC=True, rng=rng)
    pf.kernel_bandwidth = 10

    def f(t, *args):
        return F

    pf.proc_noise = proc_noise_std**2
    pf.meas_noise = meas_noise_std**2

    pf.set_measurement_model(meas_mat=H)
    pf.set_state_model(state_mat_fun=f)

    pf.init_from_dist(distrib)

    filt_state = pf.save_filter_state()
    pf = gfilts.MaxCorrEntUPF()
    pf.load_filter_state(filt_state)

    if debug_figs:
        pf.plot_particles(0, title='Init Particle Distribution')

    print('\tStarting sim')
    for kk, tt in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        past_state = pred_state.copy()
        pred_state = pf.predict(tt)

        # calculate true state and measurement for this timestep
        p_noise = proc_mean + proc_noise_std * rng.normal()
        true_state = F @ true_state + p_noise
        meas = H @ true_state + meas_noise_std * rng.normal()

        pred_state = pf.correct(tt, meas, past_state)[0]

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


def test_QKF_dynObj():  # noqa
    print('Test QKF dynObj')

    m_noise_std = 0.02
    p_noise_std = 0.001
    sig_num = 1
    level_sig = 0.05

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)
    dynObj = gdyn.DoubleIntegrator()

    filt = gfilts.QuadratureKalmanFilter(points_per_axis=5)
    filt.set_state_model(dyn_obj=dynObj)
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    proc_noise = p_noise_std**2 * np.eye(4)
    meas_noise = m_noise_std**2 * np.eye(2)
    filt.cov = 0.1**2 * np.eye(4)
    filt.proc_noise = proc_noise.copy()
    filt.meas_noise = meas_noise.copy()

    vx0 = 2
    vy0 = 1

    filt_state = filt.save_filter_state()
    filt = gfilts.QuadratureKalmanFilter()
    filt.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))

    t_states = states.copy()

    for kk, t in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(t))
            sys.stdout.flush()

        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        p_noise = rng.multivariate_normal(np.zeros(proc_noise.shape[0]),
                                          proc_noise).reshape((4, 1))
        t_states[kk + 1, :] = dynObj.propagate_state(t,
                                                     t_states[kk, :].reshape((4, 1)),
                                                     state_args=(dt,)).ravel()

        n_state = t_states[kk + 1, :].reshape((4, 1)) + p_noise
        m_noise = rng.multivariate_normal(np.zeros(meas_noise.shape[0]),
                                          meas_noise)
        meas = m_mat @ n_state + m_noise.reshape((2, 1))

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = t_states - states

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
            fig.axes[ii].plot(time, sig_num * stds[:, ii], color='r')
            fig.axes[ii].plot(time, np.abs(errs[:, ii]), color='k')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        fig.axes[0].plot(t_states[:, 0], t_states[:, 1], color='r', label='true')
        fig.axes[0].plot(states[:, 0], states[:, 1], color='b', label='est')
        fig.axes[0].grid(True)
        fig.suptitle('Positions')
        fig.axes[0].legend()
        fig.tight_layout()

        filt.plot_quadrature([0, 1])

    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    min_bound = stats.norm.sf(-sig_num) - stats.norm.sf(sig_num)
    print('Bounding is:')
    print(bounding)
    print('must be > {}'.format(min_bound))
    if debug_figs:
        if not all(bounding > min_bound):
            print('!!!!!!!!!!!!!!!bounding failed!!!!!!!!!!!!!!!')
    else:
        assert all(bounding > min_bound), 'bounding failed'

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    pred_state = states[-1, :].reshape((4, 1))
    true_state = t_states[-1, :].reshape((4, 1))

    crit_val = stats.chi2.ppf(1 - level_sig, df=true_state.size)
    exp_cov = la.solve_discrete_are(dynObj.get_state_mat(0, dt).T, m_mat.T,
                                    proc_noise,
                                    meas_noise)

    inv_cov = la.inv(exp_cov)
    chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)

    print(pred_state.flatten())
    print(true_state.flatten())
    print(filt.cov)
    print((chi_stat, crit_val))
    print(exp_cov)
    if debug_figs:
        if chi_stat >= crit_val:
            print('!!!!!!!!!!!!!!!values are different!!!!!!!!!!!!!!!')
    else:
        assert chi_stat < crit_val, "values are different"


def test_SQKF_dynObj():  # noqa
    print('Test SQKF dynObj')

    m_noise_std = 0.02
    p_noise_std = 0.001
    sig_num = 1
    level_sig = 0.05

    dt = 0.01
    t0, t1 = 0, 10 + dt

    rng = rnd.default_rng(global_seed)
    dynObj = gdyn.DoubleIntegrator()

    filt = gfilts.SquareRootQKF(points_per_axis=5)
    filt.set_state_model(dyn_obj=dynObj)
    m_mat = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    proc_noise = p_noise_std**2 * np.eye(4)
    meas_noise = m_noise_std**2 * np.eye(2)
    filt.cov = 0.1**2 * np.eye(4)
    filt.proc_noise = proc_noise.copy()
    filt.meas_noise = meas_noise.copy()

    vx0 = 2
    vy0 = 1

    filt_state = filt.save_filter_state()
    filt = gfilts.SquareRootQKF()
    filt.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    states[0, :] = np.array([0, 0, vx0, vy0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))

    t_states = states.copy()

    for kk, t in enumerate(time[:-1]):
        if np.mod(kk, int(1 / dt)) == 0:
            print('\t\t{:.2f}'.format(t))
            sys.stdout.flush()

        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((4, 1)),
                                         state_mat_args=(dt,)).flatten()
        p_noise = rng.multivariate_normal(np.zeros(proc_noise.shape[0]),
                                          proc_noise).reshape((4, 1))
        t_states[kk + 1, :] = dynObj.propagate_state(t,
                                                     t_states[kk, :].reshape((4, 1)),
                                                     state_args=(dt,)).ravel()

        n_state = t_states[kk + 1, :].reshape((4, 1)) + p_noise
        m_noise = rng.multivariate_normal(np.zeros(meas_noise.shape[0]),
                                          meas_noise)
        meas = m_mat @ n_state + m_noise.reshape((2, 1))

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    errs = t_states - states

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
            fig.axes[ii].plot(time, sig_num * stds[:, ii], color='r')
            fig.axes[ii].plot(time, np.abs(errs[:, ii]), color='k')
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + ' std')

        fig.axes[-1].set_xlabel('time (s)')
        fig.suptitle('Filter standard deviations (obj)')
        fig.tight_layout()

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        fig.axes[0].plot(t_states[:, 0], t_states[:, 1], color='r', label='true')
        fig.axes[0].plot(states[:, 0], states[:, 1], color='b', label='est')
        fig.axes[0].grid(True)
        fig.suptitle('Positions')
        fig.axes[0].legend()
        fig.tight_layout()

        filt.plot_quadrature([0, 1])

    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    min_bound = stats.norm.sf(-sig_num) - stats.norm.sf(sig_num)
    print('Bounding is:')
    print(bounding)
    print('must be > {}'.format(min_bound))
    if debug_figs:
        if not all(bounding > min_bound):
            print('!!!!!!!!!!!!!!!bounding failed!!!!!!!!!!!!!!!')
    else:
        assert all(bounding > min_bound), 'bounding failed'

    # check that particle distribution matches the expected mean assuming
    # that the covariance is the same and its normally distributed
    pred_state = states[-1, :].reshape((4, 1))
    true_state = t_states[-1, :].reshape((4, 1))

    crit_val = stats.chi2.ppf(1 - level_sig, df=true_state.size)
    exp_cov = la.solve_discrete_are(dynObj.get_state_mat(0, dt).T, m_mat.T,
                                    proc_noise,
                                    meas_noise)

    inv_cov = la.inv(exp_cov)
    chi_stat = (pred_state - true_state).T @ inv_cov @ (pred_state - true_state)

    print(pred_state.flatten())
    print(true_state.flatten())
    print(filt.cov)
    print((chi_stat, crit_val))
    print(exp_cov)
    if debug_figs:
        if chi_stat >= crit_val:
            print('!!!!!!!!!!!!!!!values are different!!!!!!!!!!!!!!!')
    else:
        assert chi_stat < crit_val, "values are different"


def test_QKF_GSM_dyn_fnc():
    print('Test QKF-GSM')

    dt = 1
    t0, t1 = 0, 170 + dt
    print_interval = 20

    rng = rnd.default_rng(global_seed)

    # define state and measurement models
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.array([[np.sqrt(x[0, 0]**2 + x[1, 0]**2)],
                         [np.arctan2(x[1, 0], x[0, 0])]])

    m_dfs = (2, 2)
    m_vars = (100, 0.001)

    # define base GSM parameters
    filt = gfilts.QKFGaussianScaleMixtureFilter()
    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun=meas_fun)
    filt.cov = np.diag((5 * 10**4, 5 * 10**4, 8, 8, 0.02, 0.02))

    # define measurement noise filters
    # def range_rvs():
    #     idx = 0
    #     return np.array([[stats.invgamma.rvs(m_dfs[idx] / 2,
    #                                          scale=m_vars[idx] / (2 / m_dfs[idx]),
    #                                          random_state=rng)]])

    # def angle_rvs():
    #     idx = 1
    #     return np.array([[stats.invgamma.rvs(m_dfs[idx] / 2,
    #                                          scale=m_vars[idx] / (2 / m_dfs[idx]),
    #                                          random_state=rng)]])

    # filt.set_meas_noise_model(500, [range_rvs, angle_rvs],
    #                           rng=rng)

    num_parts = 500
    filt._meas_noise_filters = [None] * 2

    mf = gfilts.BootstrapFilter()

    def import_dist_fnc(parts, _rng):
        idx = 0
        n_parts = parts.num_particles
        df = stats.uniform.rvs(loc=1, scale=4,
                               size=n_parts, random_state=_rng)
        z = np.nan * np.ones(n_parts)
        for ii, v in enumerate(df):
            z[ii] = stats.invgamma.rvs(v / 2, scale=1 / (2 / v),
                                       random_state=_rng)

        sig = stats.uniform.rvs(loc=0, scale=5 * np.sqrt(m_vars[idx]),
                                size=n_parts, random_state=_rng)

        return z * sig**2

    def import_w_fnc(meas, parts):
        return np.array([stats.norm.pdf(meas.item(), scale=np.sqrt(var))
                         for var in parts])

    mf.importance_dist_fnc = import_dist_fnc
    mf.importance_weight_fnc = import_w_fnc
    mf.particleDistribution = gdistrib.SimpleParticleDistribution()
    df_distrib = gdistrib.SimpleParticleDistribution()
    df_distrib.particles = stats.uniform.rvs(loc=1, scale=4, size=num_parts,
                                             random_state=rng)
    df_distrib.num_parts_per_ind = np.ones(num_parts)
    sig_distrib = gdistrib.SimpleParticleDistribution()
    sig_distrib.particles = stats.uniform.rvs(loc=0, scale=5 * np.sqrt(m_vars[0]),
                                              size=num_parts, random_state=rng)
    sig_distrib.num_parts_per_ind = np.ones(num_parts)
    z_distrib = gdistrib.SimpleParticleDistribution()
    z_distrib.rng = rng
    z_distrib.particles = np.nan * np.ones(num_parts)
    for ii, v in enumerate(df_distrib.particles):
        z_distrib.particles[ii] = stats.invgamma.rvs(v / 2, scale=1 / (2 / v),
                                                     random_state=rng)
    z_distrib.num_parts_per_ind = np.ones(num_parts)
    mf.particleDistribution.particles = z_distrib.particles * sig_distrib.particles**2
    mf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
    mf.rng = rng

    filt._meas_noise_filters[0] = mf

    mf = gfilts.BootstrapFilter()

    def import_dist_fnc(parts, _rng):
        idx = 1
        n_parts = parts.num_particles
        df = stats.uniform.rvs(loc=1, scale=4,
                               size=n_parts, random_state=_rng)
        z = np.nan * np.ones(n_parts)
        for ii, v in enumerate(df):
            z[ii] = stats.invgamma.rvs(v / 2, scale=1 / (2 / v),
                                       random_state=_rng)

        sig = stats.uniform.rvs(loc=0, scale=5 * np.sqrt(m_vars[idx]),
                                size=n_parts, random_state=_rng)

        return z * sig**2

    mf.importance_dist_fnc = import_dist_fnc
    mf.importance_weight_fnc = import_w_fnc
    mf.particleDistribution = gdistrib.SimpleParticleDistribution()
    df_distrib = gdistrib.SimpleParticleDistribution()
    df_distrib.particles = stats.uniform.rvs(loc=1, scale=4, size=num_parts,
                                             random_state=rng)
    df_distrib.num_parts_per_ind = np.ones(num_parts)
    sig_distrib = gdistrib.SimpleParticleDistribution()
    sig_distrib.particles = stats.uniform.rvs(loc=0, scale=5 * np.sqrt(m_vars[1]),
                                              size=num_parts, random_state=rng)
    sig_distrib.num_parts_per_ind = np.ones(num_parts)
    z_distrib = gdistrib.SimpleParticleDistribution()
    z_distrib.rng = rng
    z_distrib.particles = np.nan * np.ones(num_parts)
    for ii, v in enumerate(df_distrib.particles):
        z_distrib.particles[ii] = stats.invgamma.rvs(v / 2, scale=1 / (2 / v),
                                                     random_state=rng)
    z_distrib.num_parts_per_ind = np.ones(num_parts)
    mf.particleDistribution.particles = z_distrib.particles * sig_distrib.particles**2
    mf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
    mf.rng = rng

    filt._meas_noise_filters[1] = mf

    # define QKF specific parameters for core filter
    filt.points_per_axis = 3

    # test save/load filter
    filt_state = filt.save_filter_state()
    filt = gfilts.QKFGaussianScaleMixtureFilter()
    filt.load_filter_state(filt_state)

    if debug_figs:
        figs = {}
        steps = (1e-1, 1e-5)
        bnds = (3000, 0.03)
        lbls = ('Range (m)', 'Az (rad))')
        for ii in range(len(m_vars)):
            ttl = 'Init {} Measurement Noise Particles'.format(lbls[ii])
            key = 'meas_noise_particles_{:02d}'.format(ii)
            figs.update(filt.plot_particles(ii, title=ttl))
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs[key].axes[0].plot(x, y, color='b')
            figs[key].axes[0].grid(True)
            figs[key].axes[0].set_xlim((0, bnds[ii]))

    # sim loop
    time = np.arange(t0, t1, dt)
    t_states = np.nan * np.ones((time.size, 6))
    t_states[0, :] = np.array([2000, 2000, 20, 20, 0, 0])
    states = t_states.copy()
    meas_lst = np.nan * np.ones((time.size, 2))
    stds = np.nan * np.ones((time.size, 6))
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    m_stds = np.nan * np.ones((time.size, 2))
    m_stds[0, :] = np.array([np.sqrt(np.mean(f.particleDistribution.particles))
                             for f in filt._meas_noise_filters]).ravel()
    # m_stds[0, :] = np.array([np.sqrt(f._calc_state())
    #                          for f in filt._meas_noise_filters]).ravel()

    for kk, t in enumerate(time[:-1]):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(t))
            sys.stdout.flush()

        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((6, 1))).ravel()

        p_noise = rng.multivariate_normal(np.zeros(6), proc_cov)
        t_states[kk + 1, :] = (state_mat @ t_states[[kk], :].T).ravel() + 0*p_noise
        meas = meas_fun(t, (t_states[kk + 1, :] + p_noise).reshape((6, 1)))
        for ii, (df, var) in enumerate(zip(m_dfs, m_vars)):
            meas[ii, 0] += stats.t.rvs(df, scale=np.sqrt(var), random_state=rng)

        meas_lst[kk + 1, :] = meas.ravel()

        states[kk + 1, :] = filt.correct(t, meas,
                                         states[kk + 1, :].reshape((6, 1)))[0].ravel()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))
        m_stds[kk + 1, :] = np.sqrt(np.diag(filt.meas_noise))

    errs = t_states - states

    if debug_figs:
        figs = {}

        # plot measurement standard
        figs['m_stds'] = plt.figure()
        ttl = 'Estimated Measurement Noise Scalings (sqrt(z))'
        y_lbls = ('range (m)', 'az (deg)')
        for ii in range(m_stds.shape[1]):
            figs['m_stds'].add_subplot(2, 1, ii + 1)
            if ii == 1:
                c_fac = 180 / np.pi
            else:
                c_fac = 1
            figs['m_stds'].axes[ii].scatter(time, m_stds[:, ii] * c_fac)
            figs['m_stds'].axes[ii].grid(True)
            figs['m_stds'].axes[ii].set_ylabel(y_lbls[ii])
        figs['m_stds'].axes[-1].set_xlabel('Time (s)')
        figs['m_stds'].suptitle(ttl)

        figs['m_pdfs'] = plt.figure()
        ttl = 'Estimated vs True PDF'
        for ii in range(len(m_vars)):
            bnd = 6 * np.sqrt(m_vars[ii])
            gauss = stats.norm.rvs(size=time.size,
                                   random_state=rnd.default_rng(global_seed))
            gsm = m_stds[:, ii] * gauss
            figs['m_pdfs'].add_subplot(1, 2, ii + 1)
            figs['m_pdfs'].axes[ii].hist(gsm, bins='auto', density=True,
                                         histtype='stepfilled', alpha=0.2, color='b')
            x = np.arange(-bnd, bnd, 1e-3)
            y = stats.t.pdf(x, m_dfs[ii], scale=np.sqrt(m_vars[ii]))
            figs['m_pdfs'].axes[ii].plot(x, y, color='b')
            figs['m_pdfs'].axes[ii].grid(True)
            figs['m_pdfs'].axes[ii].set_xlabel('value')
            figs['m_pdfs'].axes[ii].set_xlim((-bnd, bnd))
        figs['m_pdfs'].axes[0].set_ylabel('probability')
        figs['m_pdfs'].suptitle(ttl)

        steps = (1e-1, 1e-5)
        bnds = (3000, 0.03)
        lbls = ('Range (m)', 'Az (rad))')
        for ii in range(len(m_vars)):
            ttl = '{} Measurement Noise Particles'.format(lbls[ii])
            key = 'meas_noise_particles_{:02d}'.format(ii)
            figs.update(filt.plot_particles(ii, title=ttl))
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs[key].axes[0].plot(x, y, color='b')
            figs[key].axes[0].grid(True)
            figs[key].axes[0].set_xlim((0, bnds[ii]))

        figs['mix_dist_samps'] = plt.figure()
        ttl = 'Mixing Distribution Estimates'
        steps = (1e-1, 1e-5)
        bnds = (500, 0.02)
        for ii in range(len(m_vars)):
            figs['mix_dist_samps'].add_subplot(1, len(m_vars), ii + 1)
            figs['mix_dist_samps'].axes[ii].hist(m_stds[:, ii]**2, bins='auto',
                                                 density=True, histtype='stepfilled',
                                                 alpha=0.2, color='b')
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs['mix_dist_samps'].axes[ii].plot(x, y, color='b')
            figs['mix_dist_samps'].axes[ii].grid(True)
            figs['mix_dist_samps'].axes[ii].set_xlabel('value')
            figs['mix_dist_samps'].axes[ii].set_xlim((0, bnds[ii]))
        figs['mix_dist_samps'].axes[0].set_ylabel('probability')
        figs['mix_dist_samps'].suptitle(ttl)

        figs['pos'] = plt.figure()
        ttl = 'Estimated vs True Position'
        figs['pos'].add_subplot(1, 1, 1)
        figs['pos'].axes[0].plot(t_states[:, 0], t_states[:, 1], label='true',
                                 marker='.')
        figs['pos'].axes[0].plot(states[:, 0], states[:, 1], label='estimated',
                                 marker='.')
        x_pos = np.nan * np.ones(time.size)
        y_pos = np.nan * np.ones(time.size)
        for ii, meas in enumerate(meas_lst):
            if np.isnan(meas[0]):
                continue
            x_pos[ii] = meas[0] * np.cos(meas[1])
            y_pos[ii] = meas[0] * np.sin(meas[1])
        figs['pos'].axes[0].scatter(x_pos, y_pos, marker='^', alpha=0.5, c='g',
                                    label='measurement')
        figs['pos'].axes[0].legend()
        figs['pos'].axes[0].grid(True)
        figs['pos'].axes[0].set_ylabel('y position (m)')
        figs['pos'].axes[0].set_xlabel('x position (m)')
        figs['pos'].suptitle(ttl)

        figs['pos_err'] = plt.figure()
        ttl = 'Position Errors'
        y_lbls = ('x pos (m)', 'y pos (m)')
        for ii in range(2):
            figs['pos_err'].add_subplot(2, 1, ii + 1)
            figs['pos_err'].axes[ii].plot(time, np.abs(errs[:, ii]), color='r')
            figs['pos_err'].axes[ii].plot(time, stds[:, ii], color='k',
                                          label='Standard deviation')
            figs['pos_err'].axes[ii].grid(True)
            figs['pos_err'].axes[ii].set_ylabel(y_lbls[ii])

        figs['pos_err'].axes[-1].set_xlabel('Time (s)')
        figs['pos_err'].suptitle(ttl)

        print('State bounding:')
        print(np.sum(np.abs(errs) <= stds, axis=0) / time.size)


def test_SQKF_GSM_dyn_fnc():
    print('Test SQKF-GSM')

    dt = 1
    t0, t1 = 0, 100 + dt

    rng = rnd.default_rng(global_seed)

    # define state and measurement models
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.array([[np.sqrt(x[0, 0]**2 + x[1, 0]**2)],
                         [np.arctan2(x[1, 0], x[0, 0])]])

    m_dfs = (2, 2)
    m_vars = (100, 0.001)

    # define base GSM parameters
    filt = gfilts.SQKFGaussianScaleMixtureFilter()
    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun=meas_fun)
    filt.cov = np.diag((5 * 10**4, 5 * 10**4, 8, 8, 0.02, 0.02))

    # define measurement noise filters
    def range_rvs():
        idx = 0
        return np.array([[stats.invgamma.rvs(m_dfs[idx] / 2,
                                             scale=m_vars[idx] / (2 / m_dfs[idx]),
                                             random_state=rng)]])

    def angle_rvs():
        idx = 1
        return np.array([[stats.invgamma.rvs(m_dfs[idx] / 2,
                                             scale=m_vars[idx] / (2 / m_dfs[idx]),
                                             random_state=rng)]])

    filt.set_meas_noise_model(500, [range_rvs, angle_rvs],
                              rng=rng)

    # define SQKF specific parameters for core filter
    filt.points_per_axis = 3

    # test save/load filter
    filt_state = filt.save_filter_state()
    filt = gfilts.SQKFGaussianScaleMixtureFilter()
    filt.load_filter_state(filt_state)

    # sim loop
    time = np.arange(t0, t1, dt)
    t_states = np.nan * np.ones((time.size, 6))
    t_states[0, :] = np.array([2000, 2000, 20, 20, 0, 0])
    states = t_states.copy()
    meas_lst = np.nan * np.ones((time.size, 2))
    stds = np.nan * np.ones((time.size, 6))
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    m_stds = np.nan * np.ones((time.size, 2))
    m_stds[0, :] = np.array([np.sqrt(f._calc_state())
                             for f in filt._meas_noise_filters]).ravel()

    for kk, t in enumerate(time[:-1]):
        if np.mod(kk, int(10 / dt)) == 0:
            print('\t\t{:.2f}'.format(t))
            sys.stdout.flush()

        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((6, 1))).ravel()

        p_noise = rng.multivariate_normal(np.zeros(6), proc_cov)
        t_states[kk + 1, :] = (state_mat @ t_states[[kk], :].T).ravel()
        meas = meas_fun(t, (t_states[kk + 1, :] + p_noise).reshape((6, 1)))
        for ii, (df, var) in enumerate(zip(m_dfs, m_vars)):
            meas[ii, 0] += stats.t.rvs(df, scale=np.sqrt(var), random_state=rng)

        meas_lst[kk + 1, :] = meas.ravel()

        states[kk + 1, :] = filt.correct(t, meas,
                                         states[kk + 1, :].reshape((6, 1)))[0].ravel()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))
        m_stds[kk + 1, :] = np.sqrt(np.diag(filt.meas_noise))

    errs = t_states - states

    if debug_figs:
        figs = {}

        # plot measurement standard
        figs['m_stds'] = plt.figure()
        ttl = 'Estimated Measurement Noise Scalings (sqrt(z))'
        y_lbls = ('range (m)', 'az (deg)')
        for ii in range(m_stds.shape[1]):
            figs['m_stds'].add_subplot(2, 1, ii + 1)
            if ii == 1:
                c_fac = 180 / np.pi
            else:
                c_fac = 1
            figs['m_stds'].axes[ii].scatter(time, m_stds[:, ii] * c_fac)
            figs['m_stds'].axes[ii].grid(True)
            figs['m_stds'].axes[ii].set_ylabel(y_lbls[ii])
        figs['m_stds'].axes[-1].set_xlabel('Time (s)')
        figs['m_stds'].suptitle(ttl)

        figs['m_pdfs'] = plt.figure()
        ttl = 'Estimated vs True PDF'
        for ii in range(len(m_vars)):
            bnd = 6 * np.sqrt(m_vars[ii])
            gauss = stats.norm.rvs(size=time.size,
                                   random_state=rnd.default_rng(global_seed))
            gsm = m_stds[:, ii] * gauss
            figs['m_pdfs'].add_subplot(1, 2, ii + 1)
            figs['m_pdfs'].axes[ii].hist(gsm, bins='auto', density=True,
                                         histtype='stepfilled', alpha=0.2, color='b')
            x = np.arange(-bnd, bnd, 1e-3)
            y = stats.t.pdf(x, m_dfs[ii], scale=np.sqrt(m_vars[ii]))
            figs['m_pdfs'].axes[ii].plot(x, y, color='b')
            figs['m_pdfs'].axes[ii].grid(True)
            figs['m_pdfs'].axes[ii].set_xlabel('value')
            figs['m_pdfs'].axes[ii].set_xlim((-bnd, bnd))
        figs['m_pdfs'].axes[0].set_ylabel('probability')
        figs['m_pdfs'].suptitle(ttl)

        steps = (1e-1, 1e-5)
        bnds = (3000, 0.03)
        lbls = ('Range (m)', 'Az (rad))')
        for ii in range(len(m_vars)):
            ttl = '{} Measurement Noise Particles'.format(lbls[ii])
            key = 'meas_noise_particles_{:02d}'.format(ii)
            figs.update(filt.plot_particles(ii, title=ttl))
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs[key].axes[0].plot(x, y, color='b')
            figs[key].axes[0].grid(True)
            figs[key].axes[0].set_xlim((0, bnds[ii]))

        figs['mix_dist_samps'] = plt.figure()
        ttl = 'Mixing Distribution Estimates'
        steps = (1e-1, 1e-5)
        bnds = (500, 0.02)
        for ii in range(len(m_vars)):
            figs['mix_dist_samps'].add_subplot(1, len(m_vars), ii + 1)
            figs['mix_dist_samps'].axes[ii].hist(m_stds[:, ii]**2, bins='auto',
                                                 density=True, histtype='stepfilled',
                                                 alpha=0.2, color='b')
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs['mix_dist_samps'].axes[ii].plot(x, y, color='b')
            figs['mix_dist_samps'].axes[ii].grid(True)
            figs['mix_dist_samps'].axes[ii].set_xlabel('value')
            figs['mix_dist_samps'].axes[ii].set_xlim((0, bnds[ii]))
        figs['mix_dist_samps'].axes[0].set_ylabel('probability')
        figs['mix_dist_samps'].suptitle(ttl)

        figs['pos'] = plt.figure()
        ttl = 'Estimated vs True Position'
        figs['pos'].add_subplot(1, 1, 1)
        figs['pos'].axes[0].plot(t_states[:, 0], t_states[:, 1], label='true',
                                 marker='.')
        figs['pos'].axes[0].plot(states[:, 0], states[:, 1], label='estimated',
                                 marker='.')
        x_pos = np.nan * np.ones(time.size)
        y_pos = np.nan * np.ones(time.size)
        for ii, meas in enumerate(meas_lst):
            if np.isnan(meas[0]):
                continue
            x_pos[ii] = meas[0] * np.cos(meas[1])
            y_pos[ii] = meas[0] * np.sin(meas[1])
        figs['pos'].axes[0].scatter(x_pos, y_pos, marker='^', alpha=0.5, c='g',
                                    label='measurement')
        figs['pos'].axes[0].legend()
        figs['pos'].axes[0].grid(True)
        figs['pos'].axes[0].set_ylabel('y position (m)')
        figs['pos'].axes[0].set_xlabel('x position (m)')
        figs['pos'].suptitle(ttl)

        figs['pos_err'] = plt.figure()
        ttl = 'Position Errors'
        y_lbls = ('x pos (m)', 'y pos (m)')
        for ii in range(2):
            figs['pos_err'].add_subplot(2, 1, ii + 1)
            figs['pos_err'].axes[ii].plot(time, np.abs(errs[:, ii]), color='r')
            figs['pos_err'].axes[ii].plot(time, stds[:, ii], color='k',
                                          label='Standard deviation')
            figs['pos_err'].axes[ii].grid(True)
            figs['pos_err'].axes[ii].set_ylabel(y_lbls[ii])

        figs['pos_err'].axes[-1].set_xlabel('Time (s)')
        figs['pos_err'].suptitle(ttl)

        print('State bounding:')
        print(np.sum(np.abs(errs) <= stds, axis=0) / time.size)


def test_UKF_GSM_dyn_fnc():
    print('Test UKF-GSM')

    dt = 1
    t0, t1 = 0, 100 + dt
    x0 = np.array([2000, 2000, 20, 20, 0, 0]).reshape((6, 1))
    alpha = 0.5
    kappa = 1
    beta = 1.5

    rng = rnd.default_rng(global_seed)

    # define state and measurement models
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun_range(t, x, *args):
        return np.sqrt(x[0, 0]**2 + x[1, 0]**2)

    def meas_fun_bearing(t, x, *args):
        return np.arctan2(x[1, 0], x[0, 0])

    meas_fun_lst = [meas_fun_range, meas_fun_bearing]

    m_dfs = (2, 2)
    m_vars = (100, 0.001)

    # define base GSM parameters
    filt = gfilts.UKFGaussianScaleMixtureFilter()
    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun_lst=meas_fun_lst)
    filt.cov = np.diag((5 * 10**4, 5 * 10**4, 8, 8, 0.02, 0.02))

    # define measurement noise filters
    def range_rvs():
        idx = 0
        return np.array([[stats.invgamma.rvs(m_dfs[idx] / 2,
                                             scale=m_vars[idx] / (2 / m_dfs[idx]),
                                             random_state=rng)]])

    def angle_rvs():
        idx = 1
        return np.array([[stats.invgamma.rvs(m_dfs[idx] / 2,
                                             scale=m_vars[idx] / (2 / m_dfs[idx]),
                                             random_state=rng)]])

    filt.set_meas_noise_model(500, [range_rvs, angle_rvs],
                              rng=rng)

    # define UKF specific parameters for core filter
    filt.init_sigma_points(x0, alpha, kappa, beta=beta)

    # test save/load filter
    filt_state = filt.save_filter_state()
    filt = gfilts.UKFGaussianScaleMixtureFilter()
    filt.load_filter_state(filt_state)

    # sim loop
    time = np.arange(t0, t1, dt)
    t_states = np.nan * np.ones((time.size, 6))
    t_states[0, :] = x0.ravel().copy()
    states = t_states.copy()
    meas_lst = np.nan * np.ones((time.size, 2))
    stds = np.nan * np.ones((time.size, 6))
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    m_stds = np.nan * np.ones((time.size, 2))
    m_stds[0, :] = np.array([np.sqrt(f._calc_state())
                             for f in filt._meas_noise_filters]).ravel()

    for kk, t in enumerate(time[:-1]):
        if np.mod(kk, int(10 / dt)) == 0:
            print('\t\t{:.2f}'.format(t))
            sys.stdout.flush()

        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((6, 1))).ravel()

        p_noise = rng.multivariate_normal(np.zeros(6), proc_cov)
        t_states[kk + 1, :] = (state_mat @ t_states[[kk], :].T).ravel()
        meas = np.nan * np.ones((len(meas_fun_lst), 1))
        for ii, meas_fun in enumerate(meas_fun_lst):
            meas[ii] = meas_fun(t, (t_states[kk + 1, :] + p_noise).reshape((6, 1)))

        for ii, (df, var) in enumerate(zip(m_dfs, m_vars)):
            meas[ii, 0] += stats.t.rvs(df, scale=np.sqrt(var), random_state=rng)

        meas_lst[kk + 1, :] = meas.ravel()

        states[kk + 1, :] = filt.correct(t, meas,
                                         states[kk + 1, :].reshape((6, 1)))[0].ravel()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))
        m_stds[kk + 1, :] = np.sqrt(np.diag(filt.meas_noise))

    errs = t_states - states

    if debug_figs:
        figs = {}

        # plot measurement standard
        figs['m_stds'] = plt.figure()
        ttl = 'Estimated Measurement Noise Scalings (sqrt(z))'
        y_lbls = ('range (m)', 'az (deg)')
        for ii in range(m_stds.shape[1]):
            figs['m_stds'].add_subplot(2, 1, ii + 1)
            if ii == 1:
                c_fac = 180 / np.pi
            else:
                c_fac = 1
            figs['m_stds'].axes[ii].scatter(time, m_stds[:, ii] * c_fac)
            figs['m_stds'].axes[ii].grid(True)
            figs['m_stds'].axes[ii].set_ylabel(y_lbls[ii])
        figs['m_stds'].axes[-1].set_xlabel('Time (s)')
        figs['m_stds'].suptitle(ttl)

        figs['m_pdfs'] = plt.figure()
        ttl = 'Estimated vs True PDF'
        for ii in range(len(m_vars)):
            bnd = 6 * np.sqrt(m_vars[ii])
            gauss = stats.norm.rvs(size=time.size,
                                   random_state=rnd.default_rng(global_seed))
            gsm = m_stds[:, ii] * gauss
            figs['m_pdfs'].add_subplot(1, 2, ii + 1)
            figs['m_pdfs'].axes[ii].hist(gsm, bins='auto', density=True,
                                         histtype='stepfilled', alpha=0.2, color='b')
            x = np.arange(-bnd, bnd, 1e-3)
            y = stats.t.pdf(x, m_dfs[ii], scale=np.sqrt(m_vars[ii]))
            figs['m_pdfs'].axes[ii].plot(x, y, color='b')
            figs['m_pdfs'].axes[ii].grid(True)
            figs['m_pdfs'].axes[ii].set_xlabel('value')
            figs['m_pdfs'].axes[ii].set_xlim((-bnd, bnd))
        figs['m_pdfs'].axes[0].set_ylabel('probability')
        figs['m_pdfs'].suptitle(ttl)

        steps = (1e-1, 1e-5)
        bnds = (3000, 0.03)
        lbls = ('Range (m)', 'Az (rad))')
        for ii in range(len(m_vars)):
            ttl = '{} Measurement Noise Particles'.format(lbls[ii])
            key = 'meas_noise_particles_{:02d}'.format(ii)
            figs.update(filt.plot_particles(ii, title=ttl))
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs[key].axes[0].plot(x, y, color='b')
            figs[key].axes[0].grid(True)
            figs[key].axes[0].set_xlim((0, bnds[ii]))

        figs['mix_dist_samps'] = plt.figure()
        ttl = 'Mixing Distribution Estimates'
        steps = (1e-1, 1e-5)
        bnds = (500, 0.02)
        for ii in range(len(m_vars)):
            figs['mix_dist_samps'].add_subplot(1, len(m_vars), ii + 1)
            figs['mix_dist_samps'].axes[ii].hist(m_stds[:, ii]**2, bins='auto',
                                                 density=True, histtype='stepfilled',
                                                 alpha=0.2, color='b')
            x = np.arange(0, bnds[ii], steps[ii])
            y = stats.invgamma.pdf(x, m_dfs[ii] / 2, scale=m_vars[ii] / (2 / m_dfs[ii]))
            figs['mix_dist_samps'].axes[ii].plot(x, y, color='b')
            figs['mix_dist_samps'].axes[ii].grid(True)
            figs['mix_dist_samps'].axes[ii].set_xlabel('value')
            figs['mix_dist_samps'].axes[ii].set_xlim((0, bnds[ii]))
        figs['mix_dist_samps'].axes[0].set_ylabel('probability')
        figs['mix_dist_samps'].suptitle(ttl)

        figs['pos'] = plt.figure()
        ttl = 'Estimated vs True Position'
        figs['pos'].add_subplot(1, 1, 1)
        figs['pos'].axes[0].plot(t_states[:, 0], t_states[:, 1], label='true',
                                 marker='.')
        figs['pos'].axes[0].plot(states[:, 0], states[:, 1], label='estimated',
                                 marker='.')
        x_pos = np.nan * np.ones(time.size)
        y_pos = np.nan * np.ones(time.size)
        for ii, meas in enumerate(meas_lst):
            if np.isnan(meas[0]):
                continue
            x_pos[ii] = meas[0] * np.cos(meas[1])
            y_pos[ii] = meas[0] * np.sin(meas[1])
        figs['pos'].axes[0].scatter(x_pos, y_pos, marker='^', alpha=0.5, c='g',
                                    label='measurement')
        figs['pos'].axes[0].legend()
        figs['pos'].axes[0].grid(True)
        figs['pos'].axes[0].set_ylabel('y position (m)')
        figs['pos'].axes[0].set_xlabel('x position (m)')
        figs['pos'].suptitle(ttl)

        figs['pos_err'] = plt.figure()
        ttl = 'Position Errors'
        y_lbls = ('x pos (m)', 'y pos (m)')
        for ii in range(2):
            figs['pos_err'].add_subplot(2, 1, ii + 1)
            figs['pos_err'].axes[ii].plot(time, np.abs(errs[:, ii]), color='r')
            figs['pos_err'].axes[ii].plot(time, stds[:, ii], color='k',
                                          label='Standard deviation')
            figs['pos_err'].axes[ii].grid(True)
            figs['pos_err'].axes[ii].set_ylabel(y_lbls[ii])

        figs['pos_err'].axes[-1].set_xlabel('Time (s)')
        figs['pos_err'].suptitle(ttl)

        print('State bounding:')
        print(np.sum(np.abs(errs) <= stds, axis=0) / time.size)


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
    # test_UPF_dynObj()
    # test_MCMC_UPF_dyn_fnc()
    # test_MCMC_UPF_dynObj()
    # test_MCUPF_dyn_fnc()
    # test_MCMC_MCUPF_dyn_fnc()

    # test_QKF_dynObj()
    # test_SQKF_dynObj()

    test_QKF_GSM_dyn_fnc()
    # test_SQKF_GSM_dyn_fnc()
    # test_UKF_GSM_dyn_fnc()

    end = timer()
    print('{:.2f} s'.format(end - start))
    print('Close all plots to exit')
    plt.show()
