def main():
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt

    import gncpy.plotting as gplot
    from gncpy.filters import ExtendedKalmanFilter
    from gncpy.dynamics.basic import CoordinatedTurn

    d2r = np.pi / 180
    r2d = 1 / d2r

    # set measurement and process noise values
    rng = rnd.default_rng(29)

    p_posx_std = 0.2
    p_posy_std = 0.2
    p_turn_std = 0.1 * d2r

    m_posx_std = 0.2
    m_posy_std = 0.2
    m_turn_std = 0.2 * d2r

    # create time vector
    dt = 0.01
    t0, t1 = 0, 10
    time = np.arange(t0, t1, dt)

    # create filter
    coordTurn = CoordinatedTurn(dt=dt)

    filt = ExtendedKalmanFilter()
    filt.cov = 0.03 ** 2 * np.eye(5)
    filt.cov[4, 4] = (0.3 * d2r) ** 2

    filt.set_state_model(dyn_obj=coordTurn)
    filt.proc_noise = coordTurn.get_dis_process_noise_mat(
        dt, p_posx_std, p_posy_std, p_turn_std
    )

    m_mat = np.eye(5)
    filt.set_measurement_model(meas_mat=m_mat)
    filt.meas_noise = (
        np.diag([m_posx_std, m_posx_std, m_posy_std, m_posy_std, m_turn_std]) ** 2
    )

    # set variables to save states
    states = np.nan * np.ones((time.size, 5))
    stds = np.nan * np.ones(states.shape)
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    states[0, :] = np.array([10, 0, 0, 10, 25 * d2r])
    t_states = states.copy()

    gamma = np.array(
        [[dt ** 2 / 2, 0, 0], [dt, 0, 0], [0, dt ** 2 / 2, 0], [0, dt, 0], [0, 0, 1]]
    )
    Q = np.diag([p_posx_std ** 2, p_posy_std ** 2, p_turn_std ** 2])
    for kk, t in enumerate(time[:-1]):
        # prediction
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((5, 1))).flatten()

        # propagate truth and get measurement
        t_states[kk + 1, :] = coordTurn.propagate_state(
            t, t_states[kk, :].reshape((5, 1))
        ).flatten()
        meas = m_mat @ (
            t_states[kk + 1, :].reshape((5, 1))
            + gamma @ np.sqrt(np.diag(Q)).reshape((3, 1)) * rng.standard_normal(1)
        )
        meas = meas + (
            np.sqrt(np.diag(filt.meas_noise)) * rng.standard_normal(meas.size)
        ).reshape(meas.shape)

        # correction
        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((5, 1)))[
            0
        ].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    # plot states
    plt_opts = gplot.init_plotting_opts()
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].plot(states[:, 0], states[:, 2])
    fig.axes[0].grid(True)
    gplot.set_title_label(
        fig, 0, plt_opts, x_lbl="x pos (m)", y_lbl="y pos (m)", ttl="Position"
    )
    fig.tight_layout()

    return fig


def run():
    import os

    fig = main()

    fig.savefig(
        os.path.join(
            os.path.dirname(__file__), "{}.png".format(os.path.basename(__file__)[:-3])
        )
    )
