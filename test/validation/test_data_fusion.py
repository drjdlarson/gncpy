import numpy as np
import scipy.stats as stats
import gncpy.data_fusion as gdf
import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import matplotlib.pyplot as plt

debug_figs = False


def test_GCI_meas_fusion():
    m_noise = 0.02
    p_noise = 0.2

    dt = 0.01
    t0, t1 = 0, 10 + dt

    def meas_fun2(t, meas_fun_args):
        x = meas_fun_args[0] + 10
        y = meas_fun_args[1]
        return np.array(
            [
                [(x) / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)],
                [-y / (x**2 + y**2), (x) / (x**2 + y**2)],
            ]
        ).reshape(2, 2)

    def meas_fun0(t, state):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def meas_fun1(t, state):
        return np.array([[1, 0], [0, 1]])

    meas_model_list = [meas_fun1, meas_fun2]
    filt = gfilts.KalmanFilter()
    filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    m_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    filt.set_measurement_model(meas_fun=meas_fun0)
    filt.cov = 0.25 * np.eye(4)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    filt.meas_noise = m_noise**2 * np.eye(2)

    time = np.arange(t0, t1, dt)
    states = np.nan * np.ones((time.size, 4))
    stds = np.nan * np.ones(states.shape)
    pre_stds = stds.copy()
    states[0, :] = np.array([0, 0, 1, 2])
    stds[0, :] = np.sqrt(np.diag(filt.cov))
    pre_stds[0, :] = stds[0, :]

    t_states = states.copy()

    filt_state = filt.save_filter_state()
    filt = gfilts.KalmanFilter()
    filt.load_filter_state(filt_state)

    A = gdyn.DoubleIntegrator().get_state_mat(0, dt)
    for kk, t in enumerate(time[:-1]):
        states[kk + 1, :] = filt.predict(
            t, states[kk, :].reshape((4, 1)), state_mat_args=(dt,)
        ).flatten()
        t_states[kk + 1, :] = (A @ t_states[kk, :].reshape((4, 1))).flatten()

        pre_stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))

        meas2 = np.array(
            [
                [np.sqrt((t_states[kk + 1][0] + 10) ** 2 + t_states[kk + 1][1] ** 2)],
                [np.arctan2(t_states[kk + 1][1], t_states[kk + 1][0] + 10)],
            ]
        )
        meas1 = meas_fun0(t, states[kk + 1]) @ states[kk + 1].reshape(4, 1)
        meas1_cov = np.eye(2)
        meas2_cov = np.array([[1, 0], [0, 1 * np.pi / 180]])
        meas_list = [meas1, meas2]
        meas_cov_list = [meas1_cov, meas2_cov]

        meas, meas_cov, new_weight_list = gdf.GeneralizedCovarianceIntersection(
            meas_list, meas_cov_list, [0.5, 0.5], meas_model_list
        )
        meas_fun_args = (states[kk + 1, :].reshape((4, 1)),)
        states[kk + 1, :] = filt.correct(
            t, meas, states[kk + 1, :].reshape((4, 1)), meas_fun_args=meas_fun_args
        )[0].flatten()
        stds[kk + 1, :] = np.sqrt(np.diag(filt.cov))
    errs = states - t_states

    if debug_figs:
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, states[:, ii], color="b")
            fig.axes[ii].plot(time, t_states[:, ii], color="r")
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s)
        fig.axes[-1].set_xlabel("time (s)")
        fig.suptitle("States (obj)")
        fig.tight_layout()

        # plot stds
        fig = plt.figure()
        for ii, s in enumerate(gdyn.DoubleIntegrator().state_names):
            fig.add_subplot(4, 1, ii + 1)
            fig.axes[ii].plot(time, stds[:, ii], color="b")
            fig.axes[ii].plot(time, pre_stds[:, ii], color="r")
            fig.axes[ii].grid(True)
            fig.axes[ii].set_ylabel(s + " std")
        fig.axes[-1].set_xlabel("time (s)")
        fig.suptitle("Filter standard deviations (obj)")
        fig.tight_layout()
    sig_num = 1
    bounding = np.sum(np.abs(errs) < sig_num * stds, axis=0) / time.size
    assert all(
        bounding > (stats.norm.sf(-sig_num) - stats.norm.sf(sig_num))
    ), "bounding failed"


if __name__ == "__main__":
    from timeit import default_timer as timer

    plt.close("all")
    debug_figs = True

    start = timer()

    test_GCI_meas_fusion()

    end = timer()
    print("{:.2f} s".format(end - start))
    print("Close all plots to exit")
    plt.show()
