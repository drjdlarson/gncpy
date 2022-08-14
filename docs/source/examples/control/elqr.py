def main():
    import numpy as np

    import gncpy.control as gcontrol
    from gncpy.dynamics.basic import IRobotCreate

    # define starting and ending state for control calculation
    tt = 0  # current time when calculating control
    u_nom = np.array([0.25, 0.25]).reshape((2, 1))

    # define dynamics
    dynObj = IRobotCreate(0.258, dt=1 / 30)

    # define some circular obstacles with center pos and radius (x, y, radius)
    obstacles = np.array([])

    # define Q and R weights for using standard cost function
    Q = 50 * np.eye(len(dynObj.state_names))
    R = 0.6 * np.eye(u_nom.size)

    # define non-quadratic term for cost function
    # has form: (tt, state, ctrl_input, end_state, is_initial, is_final, *args)
    def non_quadratic_cost(
        tt,
        state,
        ctrl_input,
        end_state,
        is_initial,
        is_final,
        obstacles,
        obs_factor,
        scale_factor,
    ):
        cost = 0
        # cost for obstacles
        for obs in obstacles:
            diff = state[0:2, 0] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            signed_dist = (dist - dynObj.radius) - obs[2]
            cost += obs_factor * np.exp(-scale_factor * signed_dist)

        # add cost for going out of bounds

        return cost

    # create control obect
    elqr = gcontrol.ELQR()
    elqr.set_state_model(u_nom, dynObj=dynObj)
    elqr.set_cost_model(Q=Q, R=R, non_quadratic_fun=non_quadratic_cost)

    # calculate control
    elqr.calculate_control(tt, cur_state, end_state)
