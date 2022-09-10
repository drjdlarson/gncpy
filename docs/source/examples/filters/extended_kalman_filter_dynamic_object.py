def main():
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt

    import gncpy.plotting as gplot
    from gncpy.filters import ExtendedKalmanFilter
    from gncpy.dynamics.basic import CoordinatedTurnUnknown, CoordinatedTurnKnown

    d2r = np.pi / 180
    r2d = 1 / d2r

    # set measurement and process noise values
    rng = rnd.default_rng(29)

    p_posx_std = 0.2
    p_posy_std = 0.2
    p_velx_std = 0.3
    p_vely_std = 0.3
    p_turn_std = 0.2 * d2r

    m_posx_std = 0.1
    m_posy_std = 0.1

    # create time vector
    dt = 0.01
    t0, t1 = 0, 10
    time = np.arange(t0, t1, dt)

    # create true dynamics object
    trueDyn = CoordinatedTurnKnown(turn_rate=18 * d2r)

    # create filter
    coordTurn = CoordinatedTurnUnknown(dt=dt, turn_rate_cor_time=300)

    filt = ExtendedKalmanFilter(cont_cov=True)
    filt.cov = 0.5 ** 2 * np.eye(5)
    filt.cov[4, 4] = (0.3 * d2r) ** 2

    filt.set_state_model(dyn_obj=coordTurn)
    filt.proc_noise = (
        np.diag([p_posx_std, p_posy_std, p_velx_std, p_vely_std, p_turn_std]) ** 2
    )

    m_mat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.meas_noise = np.diag([m_posx_std, m_posy_std]) ** 2

    # set variables to save states
    states = np.nan * np.ones((time.size, 5))
    stds = np.nan * np.ones(states.shape)
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    # set radius to be 10 m using v = omega r
    states[0, :] = np.array([10, 0, 0, 10 * trueDyn.turn_rate, trueDyn.turn_rate])
    t_states = states.copy()

    Q = np.array([p_posx_std, p_posy_std, p_velx_std, p_vely_std, p_turn_std])
    for kk, t in enumerate(time[:-1]):
        # prediction
        states[kk + 1, :] = filt.predict(t, states[kk, :].reshape((5, 1))).flatten()

        # propagate truth and get measurement
        t_states[kk + 1, :] = trueDyn.propagate_state(
            t, t_states[kk, :].reshape((5, 1)), state_args=(dt,)
        ).flatten()
        meas = m_mat @ (t_states[kk + 1, :] + (Q * rng.standard_normal(5)))
        meas += (
            np.sqrt(np.diag(filt.meas_noise)) * rng.standard_normal(meas.size)
        ).reshape(meas.shape)

        # correction
        states[kk + 1, :] = filt.correct(
            t, meas.reshape((-1, 1)), states[kk + 1, :].reshape((5, 1))
        )[0].ravel()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    # plot states
    plt_opts = gplot.init_plotting_opts()
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].set_aspect("equal", adjustable="box")
    fig.axes[0].plot(states[:, 0], states[:, 1], linestyle='--')
    fig.axes[0].plot(
        t_states[:, 0], t_states[:, 1], color="k", zorder=1000
    )
    fig.axes[0].grid(True)
    gplot.set_title_label(
        fig, 0, plt_opts, x_lbl="x pos (m)", y_lbl="y pos (m)", ttl="Position"
    )
    fig.tight_layout()

    return fig


def run():
    import os

    print("Generating EKF examples")

    fout = os.path.join(
        os.path.dirname(__file__), "{}.png".format(os.path.basename(__file__)[:-3])
    )
    if not os.path.isfile(fout):
        fig = main()
        fig.savefig(fout)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")

    run()

    plt.show()
