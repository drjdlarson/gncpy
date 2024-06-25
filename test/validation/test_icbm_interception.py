import pytest
import numpy as np
import numpy.testing as test

import gncpy.dynamics.basic as gdyn
import gncpy.control as gctrl
import gncpy.measurements as sensors

# FIXME eventually want this to be an actual E2E test that asserts the interceptor hit the icbm

def test_icbm_interception():
    dt = 0.1
    t1 = 70
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    target = gdyn.ReentryVehicle(dt=dt)
    chaser = gdyn.ReentryVehicle(dt=dt)

    # initialize icbm as coming from NE towards origin, just entering atmosphere
    # parameters based on "Data for ICBM entries"
    t_state = np.zeros((time.size, len(target.state_names)))
    reentry_speed = 7500 # 24300 ft/s
    reentry_angle = np.deg2rad(10)
    reentry_heading = np.deg2rad(45+180) # assume 45 degree heading angle towards origin
    reentry_velocity = np.array([reentry_speed*np.cos(reentry_heading)*np.cos(reentry_angle), 
                                 reentry_speed*np.sin(reentry_heading)*np.cos(reentry_angle),
                                 -reentry_speed*np.sin(reentry_angle)
                                 ])
    t_state[0] = np.concatenate(([300000, 300000, 100000], reentry_velocity))

    # init interceptor ('chaser') vehicle at origin with marginal launch velocity in direction
    # of icbm at 30 degree up angle
    c_state = np.zeros((time.size, len(chaser.state_names)))
    launch_speed = 10 # m/s
    launch_angle = np.deg2rad(30)
    launch_heading = np.deg2rad(45) # assume 45 degree heading angle away from origin
    launch_velocity = np.array([launch_speed*np.cos(launch_heading)*np.cos(launch_angle), 
                                 launch_speed*np.sin(launch_heading)*np.cos(launch_angle),
                                 launch_speed*np.sin(launch_angle)
                                 ])
    c_state[0] = np.concatenate(([0, 0, 0], launch_velocity))
    chaser_specific_thrust = 200 # m/s (aka about 20g's acceleration at launch)

    R = np.zeros((2,2))
    seeker = sensors.Seeker(R)
    pronav = gctrl.Pronav()
    hit_distance_threshold = 10 # meters

    # simulate with no icbm evasion control and pronav chaser control
    # this simulation has no sensor noise (perfect knowledge of states)
    for kk, tt in enumerate(time[:-1]):
        # compute control command for interceptor
        relative_state = seeker.calc_relative_state(c_state[kk,:], t_state[kk,:])
        t_go = pronav.estimate_tgo(relative_state[:3], relative_state[3:])
        u_ENU = pronav.PN(relative_state[:3], relative_state[3:], t_go) # 3D pronav control
        u_ENU[2] = u_ENU[2] + 9.81 # account (roughly/flat earth assumption) for gravity in control vector; akin to setting u nominal
        u_VTC = pronav.enu2vtc(c_state[kk, :], u_ENU) # convert pronav control in ENU frame to VTC frame
        u_VTC[0] = u_VTC[0] + 100 # add specific thrust to pronav control FIXME this assumes thrust is variable as commanded by pronav
        
        # propogate interceptor and target forward
        c_state[kk + 1, :] = chaser.propagate_state(tt, c_state[kk].reshape((-1, 1)), u=u_VTC).flatten()
        t_state[kk + 1, :] = chaser.propagate_state(tt, t_state[kk].reshape((-1, 1)), u=np.zeros(3)).flatten()
        
        # check for hit:
        # iff sign change in separations, since only then may vehicle have passed nearest distance point
        separation_now = c_state[kk + 1, :3] - t_state[kk+1, :3]
        separation_last = c_state[kk, :3] - t_state[kk, :3]
        now_signs = np.sign(separation_now)
        next_signs = np.sign(separation_last)
        if np.any(now_signs - next_signs):
            line_vector = separation_now - separation_last
            origin_vector =  -separation_last
            
            # Projection of the origin vector onto the line vector
            projection_length = np.dot(origin_vector, line_vector) / np.dot(line_vector, line_vector)
            if projection_length < -1:
                closest_point = separation_last
            elif projection_length > 1:
                closest_point = separation_now
            else:
                closest_point = separation_last + projection_length * line_vector
            nearest_distance = np.linalg.norm(closest_point)
            if(nearest_distance < hit_distance_threshold):
                print('interception! Touchdown Alabama. Nearest distance: ', nearest_distance, ' m')


    # debug plots
    if DEBUG:
        # --------------------- position-time plots ----------------------
        # Create subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Plot on each subplot
        ax1.plot(time, t_state[:, 0], label='target')
        ax1.plot(time, c_state[:, 0], label='interceptor')
        ax1.set_ylabel("E-pos (m)")
        ax1.grid(True)

        ax2.plot(time, t_state[:, 1], label='target')
        ax2.plot(time, c_state[:, 1], label='interceptor')
        ax2.set_ylabel("N-pos (m)")
        ax2.grid(True)

        ax3.plot(time, t_state[:, 2], label='target')
        ax3.plot(time, c_state[:, 2], label='interceptor')
        ax3.set_ylabel("U-pos (m)")
        ax3.set_xlabel("time (s)")
        ax3.grid(True)
 
        fig.suptitle("Uncontrolled Reentry vehicle vs. Interceptor simulation")

        # --------------------- velocity-time plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        fig.add_subplot(3, 1, 3)

        fig.axes[0].plot(time, t_state[:, 3], label='target')
        fig.axes[0].plot(time, c_state[:, 3], label='interceptor')
        fig.axes[0].set_ylabel("E-vel (m/s)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, t_state[:, 4], label='target')
        fig.axes[1].plot(time, c_state[:, 4], label='interceptor')
        fig.axes[1].set_ylabel("N-vel (m/s)")
        fig.axes[1].grid(True)

        fig.axes[2].plot(time, t_state[:, 5], label='target')
        fig.axes[2].plot(time, c_state[:, 5], label='interceptor')
        fig.axes[2].set_ylabel("U-vel (m/s)")
        fig.axes[2].set_xlabel("time (s)")
        fig.axes[2].grid(True)

        fig.suptitle("Uncontrolled Reentry vehicle vs. Interceptor simulation")

        # --------------------- x-y, y-z plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        
        fig.axes[0].plot(t_state[:,0], t_state[:, 1], label='target')
        fig.axes[0].plot(c_state[:,0], c_state[:, 1], label='interceptor')
        fig.axes[0].set_ylabel("North position (m)")
        fig.axes[0].grid(True)
        #fig.axes[0].set_aspect('equal', adjustable='box')

        fig.axes[1].plot(t_state[:,0], t_state[:, 2], label='target')
        fig.axes[1].plot(c_state[:,0], c_state[:, 2], label='interceptor')
        fig.axes[1].set_ylabel("Up position (m)")
        fig.axes[1].grid(True)
        fig.axes[1].set_xlabel("East position (m)")
        #fig.axes[1].set_aspect('equal', adjustable='box')


        fig.suptitle("Uncontrolled Reentry vehicle vs. Interceptor simulation")


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("WebAgg")

        plt.close("all")

    test_icbm_interception()

    if DEBUG:
        plt.show()
        