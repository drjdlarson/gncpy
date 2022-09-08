import matplotlib.pyplot as plt

def linear_inf_horizon():
    import numpy as np

    import gncpy.dynamics.basic as gdyn
    from gncpy.control import LQR

    cur_time = 0
    dt = 0.1
    cur_state = np.array([0.5, 1, 0, 0]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0]).reshape((-1, 1))
    time_horizon = float("inf")

    # Create dynamics object
    dynObj = gdyn.DoubleIntegrator()
    dynObj.control_model = lambda _t, *_args: np.array(
        [[0, 0], [0, 0], [1, 0], [0, 1]]
    )
    state_args = (dt,)

    # Setup LQR Object
    u_nom = np.zeros((2, 1))
    Q = 10 * np.eye(len(dynObj.state_names))
    R = 0.5 * np.eye(u_nom.size)
    lqr = LQR(time_horizon=time_horizon)
    # need to set dt here so the controller can generate a state trajectory
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)
    lqr.set_cost_model(Q, R)

    u, cost, state_traj, ctrl_signal = lqr.calculate_control(
        cur_time,
        cur_state,
        end_state=end_state,
        end_state_tol=0.1,
        check_inds=[0, 1],
        state_args=state_args,
        provide_details=True,
    )

    return state_traj, end_state


def linear_finite_horizon():
    import numpy as np

    import gncpy.dynamics.basic as gdyn
    from gncpy.control import LQR

    cur_time = 0
    dt = 0.1
    cur_state = np.array([0.5, 1, 0, 0]).reshape((-1, 1))
    end_state = np.array([2, 4, 0, 0]).reshape((-1, 1))
    time_horizon = 3 * dt

    # Create dynamics object
    dynObj = gdyn.DoubleIntegrator()
    dynObj.control_model = lambda _t, *_args: np.array(
        [[0, 0], [0, 0], [1, 0], [0, 1]]
    )
    state_args = (dt,)
    inv_state_args = (-dt, )

    # Setup LQR Object
    u_nom = np.zeros((2, 1))
    Q = 200 * np.eye(len(dynObj.state_names))
    R = 0.01 * np.eye(u_nom.size)
    lqr = LQR(time_horizon=time_horizon)
    # need to set dt here so the controller can generate a state trajectory
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)
    lqr.set_cost_model(Q, R)

    u, cost, state_traj, ctrl_signal = lqr.calculate_control(
        cur_time,
        cur_state,
        end_state=end_state,
        state_args=state_args,
        inv_state_args=inv_state_args,
        provide_details=True,
    )

    return state_traj, end_state


def nonlin_finite_hor():
    import numpy as np

    import gncpy.dynamics.basic as gdyn
    from gncpy.control import LQR

    d2r = np.pi / 180

    cur_time = 0
    dt = 0.1
    cur_state = np.array([0.5, 0, 1, 0, 45 * d2r]).reshape((-1, 1))
    end_state = np.array([2, 0, 4, 0, 0]).reshape((-1, 1))
    time_horizon = 50 * dt

    def ctrl02(t, x, u, *args):
        return 0

    def ctrl1(t, x, u, *args):
        return (2 * np.min([np.max([u[0, 0], -1]), 1]) * np.cos(x[4])).item()

    def ctrl3(t, x, u, *args):
        return (2 * np.min([np.max([u[0, 0], -1]), 1]) * np.sin(x[4])).item()

    def ctrl4(t, x, u, *args):
        return (5 * d2r * np.min([np.max([u[1, 0], -1]), 1])).item()

    # Create dynamics object
    dynObj = gdyn.CoordinatedTurn()
    dynObj.control_model = [ctrl02, ctrl1, ctrl02, ctrl3, ctrl4]

    # Setup LQR Object
    u_nom = np.zeros((2, 1))
    Q = np.diag([200, 0.0001, 2000, 0.0001, 2000])
    R = 0.01 * np.eye(u_nom.size)
    lqr = LQR(time_horizon=time_horizon)
    # need to set dt here so the controller can generate a state trajectory
    lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)
    lqr.set_cost_model(Q, R)

    u, cost, state_traj, ctrl_signal = lqr.calculate_control(
        cur_time,
        cur_state,
        end_state=end_state,
        provide_details=True,
    )

    return state_traj, end_state


def make_plot(x_data, y_data, x_lbl, y_lbl, ttl, end):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].grid(True)
    fig.axes[0].plot(x_data, y_data)
    fig.axes[0].scatter(x_data[0], y_data[0], marker="o", color="g")
    fig.axes[0].scatter(end[0], end[1], marker="x", color="r")
    fig.axes[0].set_xlabel(x_lbl)
    fig.axes[0].set_ylabel(y_lbl)
    fig.suptitle(ttl)
    fig.tight_layout()

    return fig


def run():
    import os

    print("Generating LQR examples")

    fout = os.path.join(
        os.path.dirname(__file__),
        "{}_linear_inf_horizon.png".format(os.path.basename(__file__)[:-3]),
    )
    if not os.path.isfile(fout):
        state_traj, end_state = linear_inf_horizon()
        fig = make_plot(
            state_traj[:, 0],
            state_traj[:, 1],
            "x pos",
            "y pos",
            "Double Integrator inf Horizon LQR",
            end_state[0:2, 0],
        )

        fig.savefig(fout)

    fout = os.path.join(
        os.path.dirname(__file__),
        "{}_linear_finite_horizon.png".format(os.path.basename(__file__)[:-3]),
    )
    if not os.path.isfile(fout):
        state_traj, end_state = linear_finite_horizon()
        fig = make_plot(
            state_traj[:, 0],
            state_traj[:, 1],
            "x pos",
            "y pos",
            "Double Integrator Finite Horizon LQR",
            end_state[0:2, 0],
        )

        fig.savefig(fout)

    fout = os.path.join(
        os.path.dirname(__file__),
        "{}_nonlinear_finite_horizon.png".format(os.path.basename(__file__)[:-3]),
    )
    if not os.path.isfile(fout):
        state_traj, end_state = nonlin_finite_hor()
        fig = make_plot(
            state_traj[:, 0],
            state_traj[:, 2],
            "x pos",
            "y pos",
            "Coordinated Turn Finite Horizon LQR",
            end_state[[0, 2], 0],
        )

        fig.savefig(fout)

if __name__ == "__main__":
    plt.close('all')

    run()

    plt.show()
