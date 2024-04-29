import pytest
import numpy as np
import numpy.testing as test
import gncpy.math as math
import gncpy.dynamics.basic as gdyn
import gncpy.control as gcont

# FIXME want to put some asserts in here to actually function as a test, not just a debug/plotting function

DEBUG = False

def test_rv_prop():
    dt = 0.1
    t1 = 100
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    dynObj = gdyn.ReentryVehicle(dt=dt)
    state = np.zeros((time.size, len(dynObj.state_names)))
    reentry_speed = 7500 # 24300 ft/s
    reentry_angle = np.deg2rad(10)
    reentry_heading = np.deg2rad(45+180) # assume 45 degree heading angle towards origin
    reentry_velocity = np.array([reentry_speed*np.cos(reentry_heading)*np.cos(reentry_angle), 
                                 reentry_speed*np.sin(reentry_heading)*np.cos(reentry_angle),
                                 -reentry_speed*np.sin(reentry_angle)
                                 ])
    state[0] = np.concatenate(([300000, 300000, 100000], reentry_velocity))

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1, :] = dynObj.propagate_state(tt, state[kk].reshape((-1, 1))).flatten()

    if DEBUG:
        # Create subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Plot on each subplot
        ax1.plot(time, state[:, 0])
        ax1.set_ylabel("E-pos (m)")
        ax1.grid(True)

        ax2.plot(time, state[:, 1])
        ax2.set_ylabel("N-pos (m)")
        ax2.grid(True)

        ax3.plot(time, state[:, 2])
        ax3.set_ylabel("U-pos (m)")
        ax3.set_xlabel("time (s)")
        ax3.grid(True)

        fig.suptitle("Reentry vehicle simulation without control input")
        
        
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        fig.add_subplot(3, 1, 3)

        fig.axes[0].plot(time, state[:, 3])
        fig.axes[0].set_ylabel("E-vel (m/s)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 4])
        fig.axes[1].set_ylabel("N-vel (m/s)")
        fig.axes[1].grid(True)

        fig.axes[2].plot(time, state[:, 5])
        fig.axes[2].set_ylabel("U-vel (m/s)")
        fig.axes[2].set_xlabel("time (s)")
        fig.axes[2].grid(True)

        fig.suptitle("Reentry vehicle simulation without control input")

        # --------------------- x-y, y-z plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        
        fig.axes[0].plot(state[:,0], state[:, 1])
        fig.axes[0].set_ylabel("North position (m)")
        fig.axes[0].grid(True)
        fig.axes[1].set_xlabel("East position (m)")
        

        fig.axes[1].plot(state[:,0], state[:, 2])
        fig.axes[1].set_ylabel("Up position (m)")
        fig.axes[1].grid(True)
        fig.axes[1].set_xlabel("East position (m)")

        fig.suptitle("Reentry vehicle simulation without control input")


def test_rv_control():
    dt = 0.1
    t1 = 100
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    dynObj = gdyn.ReentryVehicle(dt=dt)

    # Control model already set up for RV: input constant 50 m/s2 right hand turn
    u = np.array([0,-100,0]) 

    # simulate for some time
    state = np.zeros((time.size, len(dynObj.state_names)))
    reentry_speed = 7500 # 24300 ft/s
    reentry_angle = np.deg2rad(10)
    reentry_heading = np.deg2rad(45+180) # assume 45 degree heading angle towards origin
    reentry_velocity = np.array([reentry_speed*np.cos(reentry_heading)*np.cos(reentry_angle), 
                                 reentry_speed*np.sin(reentry_heading)*np.cos(reentry_angle),
                                 -reentry_speed*np.sin(reentry_angle)
                                 ])
    state[0] = np.concatenate(([300000, 300000, 100000], reentry_velocity))

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1, :] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), u=u
        ).flatten()

    # debug plots
    if DEBUG:
        # --------------------- position-time plots ----------------------
        # Create subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Plot on each subplot
        ax1.plot(time, state[:, 0])
        ax1.set_ylabel("E-pos (m)")
        ax1.grid(True)

        ax2.plot(time, state[:, 1])
        ax2.set_ylabel("N-pos (m)")
        ax2.grid(True)

        ax3.plot(time, state[:, 2])
        ax3.set_ylabel("U-pos (m)")
        ax3.set_xlabel("time (s)")
        ax3.grid(True)
 
        fig.suptitle("Reentry vehicle simulation with 10m/s2 RH turn")

        # --------------------- velocity-time plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        fig.add_subplot(3, 1, 3)

        fig.axes[0].plot(time, state[:, 3])
        fig.axes[0].set_ylabel("E-vel (m/s)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 4])
        fig.axes[1].set_ylabel("N-vel (m/s)")
        fig.axes[1].grid(True)

        fig.axes[2].plot(time, state[:, 5])
        fig.axes[2].set_ylabel("U-vel (m/s)")
        fig.axes[2].set_xlabel("time (s)")
        fig.axes[2].grid(True)

        fig.suptitle("Reentry vehicle simulation with RH 10m/s2 turn")

        # --------------------- x-y, y-z plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        
        fig.axes[0].plot(state[:,0], state[:, 1])
        fig.axes[0].set_ylabel("North position (m)")
        fig.axes[0].grid(True)
        #fig.axes[0].set_aspect('equal', adjustable='box')

        fig.axes[1].plot(state[:,0], state[:, 2])
        fig.axes[1].set_ylabel("Up position (m)")
        fig.axes[1].grid(True)
        fig.axes[1].set_xlabel("East position (m)")
        #fig.axes[1].set_aspect('equal', adjustable='box')


        fig.suptitle("Reentry vehicle simulation with RH 10m/s2 turn")

# function to test that the Reentry vehicle class works for simulating a launched missile to intercept as well

def test_interceptor_control():
    dt = 0.1
    t1 = 50
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    dynObj = gdyn.ReentryVehicle(dt=dt)

    # Control model already set up for interceptor: 
    # Constant specific thrust of 100 m/s2 and constant 30 m/s2 right hand turn
    # need 10 m/s climb command to counteract gravity
    u = np.array([100,0,9.8]) 

    # simulate for some time
    state = np.zeros((time.size, len(dynObj.state_names)))
    launch_speed = 10 # m/s
    launch_angle = np.deg2rad(30)
    launch_heading = np.deg2rad(45) # assume 45 degree heading angle away from origin
    launch_velocity = np.array([launch_speed*np.cos(launch_heading)*np.cos(launch_angle), 
                                 launch_speed*np.sin(launch_heading)*np.cos(launch_angle),
                                 launch_speed*np.sin(launch_angle)
                                 ])
    state[0] = np.concatenate(([0, 0, 0], launch_velocity))

    for kk, tt in enumerate(time[:-1]):
        state[kk + 1, :] = dynObj.propagate_state(
            tt, state[kk].reshape((-1, 1)), u=u
        ).flatten()

    # debug plots
    if DEBUG:
        # --------------------- position-time plots ----------------------
        # Create subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Plot on each subplot
        ax1.plot(time, state[:, 0])
        ax1.set_ylabel("E-pos (m)")
        ax1.grid(True)

        ax2.plot(time, state[:, 1])
        ax2.set_ylabel("N-pos (m)")
        ax2.grid(True)

        ax3.plot(time, state[:, 2])
        ax3.set_ylabel("U-pos (m)")
        ax3.set_xlabel("time (s)")
        ax3.grid(True)
 
        fig.suptitle("Interceptor simulation")

        # --------------------- velocity-time plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        fig.add_subplot(3, 1, 3)

        fig.axes[0].plot(time, state[:, 3])
        fig.axes[0].set_ylabel("E-vel (m/s)")
        fig.axes[0].grid(True)

        fig.axes[1].plot(time, state[:, 4])
        fig.axes[1].set_ylabel("N-vel (m/s)")
        fig.axes[1].grid(True)

        fig.axes[2].plot(time, state[:, 5])
        fig.axes[2].set_ylabel("U-vel (m/s)")
        fig.axes[2].set_xlabel("time (s)")
        fig.axes[2].grid(True)

        fig.suptitle("Interceptor simulation")

        # --------------------- x-y, y-z plots ----------------------
        fig = plt.figure()
        fig.add_subplot(3, 1, 1)
        fig.add_subplot(3, 1, 2)
        
        fig.axes[0].plot(state[:,0], state[:, 1])
        fig.axes[0].set_ylabel("North position (m)")
        fig.axes[0].grid(True)
        #fig.axes[0].set_aspect('equal', adjustable='box')

        fig.axes[1].plot(state[:,0], state[:, 2])
        fig.axes[1].set_ylabel("Up position (m)")
        fig.axes[1].grid(True)
        fig.axes[1].set_xlabel("East position (m)")
        #fig.axes[1].set_aspect('equal', adjustable='box')


        fig.suptitle("Interceptor simulation")

if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("WebAgg")

        plt.close("all")

    test_rv_prop()
    test_rv_control()
    test_interceptor_control()

    if DEBUG:
        plt.show()
        