def main():
    import numpy as np
    import numpy.random as rnd
    import matplotlib.pyplot as plt

    import gncpy.plotting as gplot
    from gncpy.filters import KalmanFilter
    from gncpy.filters import InteractingMultipleModel
    from gncpy.dynamics.basic import CoordinatedTurnKnown

    # set measurement and process noise values
    m_noise = 0.002
    p_noise = 0.004

    dt = 0.01
    t0, t1 = 0, 9.5 + dt
    rng = rnd.default_rng(69)

    dyn_obj1 = CoordinatedTurnKnown(turn_rate=0)
    dyn_obj2 = CoordinatedTurnKnown(turn_rate=5 * np.pi / 180)

    in_filt1 = KalmanFilter()
    in_filt1.set_state_model(dyn_obj=dyn_obj1)
    m_mat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
    in_filt1.set_measurement_model(meas_mat=m_mat)
    in_filt1.cov = np.diag([0.25, 0.25, 3.0, 3.0, 0.25])
    gamma = np.array([0, 0, 1, 1, 0]).reshape((5, 1))
    # in_filt1.proc_noise = gamma @ np.array([[p_noise ** 2]]) @ gamma.T
    in_filt1.proc_noise = np.eye(5) * np.array([[p_noise ** 2]])
    in_filt1.meas_noise = m_noise ** 2 * np.eye(m_mat.shape[0])

    in_filt2 = KalmanFilter()
    in_filt2.set_state_model(dyn_obj=dyn_obj2)
    in_filt2.set_measurement_model(meas_mat=m_mat)
    in_filt2.cov = np.diag([0.25, 0.25, 3.0, 3.0, 0.25])
    in_filt2.proc_noise = np.eye(5) * np.array([[p_noise ** 2]])
    # in_filt2.proc_noise = gamma @ np.array([[p_noise ** 2]]) @ gamma.T
    in_filt2.meas_noise = m_noise ** 2 * np.eye(m_mat.shape[0])

    # vx0 = 2
    # vy0 = 1
    v = np.sqrt(2 ** 2 + 1 ** 2)
    angle = 60 * np.pi / 180
    vx0 = v * np.cos(angle)
    vy0 = v * np.sin(angle)

    filt_list = [in_filt1, in_filt2]

    model_trans = np.array([[1 - 1 / 200, 1 / 200], [1 / 200, 1 - 1 / 200]])

    init_means = np.array([[0, 0, vx0, vy0, 0], [0, 0, vx0, vy0, 0]]).T
    init_covs = np.array([in_filt1.cov, in_filt2.cov])

    filt = InteractingMultipleModel()
    filt.set_models(
        filt_list, model_trans, init_means, init_covs, init_weights=[0.5, 0.5]
    )

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 5))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, vx0, vy0, 0])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    t_states = states.copy()

    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(t, state_mat_args=(dt,)).flatten()
        if t < 5:
            t_states[kk + 1, :] = dyn_obj1.propagate_state(
                t, t_states[kk, :].reshape((5, 1)), state_args=(dt,)
            ).flatten()
        else:
            t_states[kk + 1, :] = dyn_obj2.propagate_state(
                t, t_states[kk, :].reshape((5, 1)), state_args=(dt,)
            ).flatten()
        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        n_state = m_mat @ (
            t_states[kk + 1, :].reshape((5, 1))
            + gamma * p_noise * rng.standard_normal(1)
        )

        meas = (
            n_state + m_noise * rng.standard_normal(n_state.size).reshape(n_state.shape)
        ).reshape((-1, 1))

        states[kk + 1, :] = filt.correct(t, meas)[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))
    errs = states - t_states

# plot states
    fig = plt.figure()
    for ii, s in enumerate(CoordinatedTurnKnown().state_names):
        fig.add_subplot(5, 1, ii + 1)
        fig.axes[ii].plot(time, states[:, ii], color="b")
        fig.axes[ii].plot(time, t_states[:, ii], color="r")
        fig.axes[ii].grid(True)
        fig.axes[ii].set_ylabel(s)
    fig.axes[-1].set_xlabel("time (s)")
    fig.suptitle("States (mat)")
    fig.tight_layout()

    # # plot stds
    # fig2 = plt.figure()
    # for ii, s in enumerate(CoordinatedTurnKnown().state_names):
    #     fig2.add_subplot(5, 1, ii + 1)
    #     fig2.axes[ii].plot(time, stds[:, ii], color="b")
    #     fig2.axes[ii].plot(time, np.abs(errs[:, ii]), color="r")
    #     # fig.axes[ii].plot(time, pre_stds[:, ii], color="g")
    #     fig2.axes[ii].grid(True)
    #     fig2.axes[ii].set_ylabel(s + " std")
    # fig2.axes[-1].set_xlabel("time (s)")
    # fig2.suptitle("Filter standard deviations (mat)")
    # fig2.tight_layout()
    return fig

def run():
    import os

    print("Generating KF examples")

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
