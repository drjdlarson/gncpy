def main():
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt

    import gncpy.plotting as gplot
    import serums.models as smodels
    from gncpy.filters import StudentsTFilter
    from gncpy.dynamics.basic import DoubleIntegrator

    # define measurement noise
    m_dof = 3
    m_cov = 0.07 ** 2 * np.eye(2)
    measNoise = smodels.StudentsT(mean=np.zeros((2, 1)), scale=((m_dof - 2) / m_dof) * m_cov, dof=m_dof)
    rng = rnd.default_rng(69)

    # define process noise
    p_dof = 5
    p_cov = 0.5 ** 2 * np.eye(4)
    procNoise = smodels.StudentsT(mean=np.zeros((4,1)), scale=((p_dof - 2) / p_dof) * p_cov, dof=p_dof)

    # define simulation time
    dt = 0.01
    t0, t1 = 0, 10
    time = np.arange(t0, t1, dt)

    # set up the filter
    filt = StudentsTFilter()
    filt.dof = 3
    covariance = 0.7 * np.eye(4)
    filt.scale = ((filt.dof - 2) / filt.dof) * covariance

    # set up variables to save states and truth data
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    states[0, :] = np.array([0, 0, 2, 1])
    stds[0, :] = np.sqrt(np.diag(filt.cov))

    t_states = states.copy()

    # set up dynamics model
    model = DoubleIntegrator()

    # set up state model
    filt.set_state_model(dyn_obj=model)
    filt.proc_noise = procNoise.scale.copy()
    filt.proc_noise_dof = procNoise.degrees_of_freedom

    # set up measurement model
    m_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    filt.set_measurement_model(meas_mat=m_mat)
    filt.meas_noise = measNoise.scale.copy()
    filt.meas_noise_dof = measNoise.degrees_of_freedom


    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(
            t, states[kk, :].reshape((4, 1)), state_mat_args=(dt,)
        ).flatten()

        t_states[kk + 1, :] = model.propagate_state(t, t_states[kk, :].reshape((4, 1)), state_args=(dt,)).flatten()

        meas = m_mat @ t_states[kk + 1, :].reshape((-1, 1)) + measNoise.sample(rng=rng).reshape((-1, 1))

        states[kk + 1, :] = filt.correct(t, meas, states[kk + 1, :].reshape((4, 1)))[0].flatten()

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

    print("Generating StudentsT examples")

    fout = os.path.join(
        os.path.dirname(__file__), "{}.png".format(os.path.basename(__file__)[:-3])
    )
    if not os.path.isfile(fout):
        fig = main()
        fig.savefig(fout)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()