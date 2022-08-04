def main():
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt

    import gncpy.plotting as gplot
    from gncpy.filters import KalmanFilter
    from gncpy.dynamics.basic import DoubleIntegrator

    # set measurement and process noise values
    m_noise = 0.02
    p_noise = 0.2
    rng = rnd.default_rng(29)

    # define the simulation time
    dt = 0.01
    t0, t1 = 0, 10
    time = np.arange(t0, t1, dt)

    # setup the filter
    filt = KalmanFilter()
    filt.cov = 0.25 * np.eye(4)

    filt.set_state_model(dyn_obj=DoubleIntegrator())
    filt.proc_noise = p_noise ** 2 * np.eye(4)

    m_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.meas_noise = m_noise ** 2 * np.eye(2)

    # setup variables to save states and truth data
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    states[0, :] = np.array([0, 0, 2, 1])
    stds[0, :] = np.sqrt(np.diag(filt.cov))

    t_states = states.copy()

    # Run filter
    A = DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(
            t, states[kk, :].reshape((4, 1)), state_mat_args=(dt,)
        ).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        n_state = m_mat @ (
            t_states[kk + 1, :]
            + np.sqrt(np.diag(filt.proc_noise)) * rng.standard_normal(1)
        ).reshape((4, 1))
        meas = n_state + m_noise * rng.standard_normal(n_state.size).reshape(
            n_state.shape
        )

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[
            0
        ].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

    # plot states
    fig = plt.figure()
    for ii, s in enumerate(DoubleIntegrator().state_names):
        fig.add_subplot(4, 1, ii + 1)
        fig.axes[ii].plot(time, states[:, ii], color="b", label="est")
        fig.axes[ii].plot(time, t_states[:, ii], color="k", label="true")
        fig.axes[ii].plot(time, states[:, ii] + stds[:, ii], color="r")
        fig.axes[ii].plot(time, states[:, ii] - stds[:, ii], color="r")
        fig.axes[ii].grid(True)
        fig.axes[ii].set_ylabel(s)

    plt_opts = gplot.init_plotting_opts()
    gplot.set_title_label(fig, -1, plt_opts, ttl="States", x_lbl="Time (s)")
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
