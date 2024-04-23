import pytest
import numpy as np
import numpy.testing as test
import gncpy.math as math
import gncpy.dynamics.basic as gdyn
import gncpy.control as gcont

DEBUG = False

def test_rv_prop():
    dt = 0.1
    t1 = 10
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    dynObj = gdyn.ReentryVehicle(dt=dt)
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([10000, 10000, 10000, -1000, -1000, -100])

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
        fig.axes[2].set_ylabel("N-vel (m/s)")
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

        fig.suptitle("Reentry vehicle simulation with RH 10m/s2 turn")


def test_rv_control():
    dt = 0.1
    t1 = 10
    time = np.arange(0, t1 + dt, dt)

    # Create dynamics object
    dynObj = gdyn.ReentryVehicle(dt=dt)

    # Control model already set up for RV: input constant 50 m/s2 right hand turn
    u = np.array([0,-100,0]) # FIXME integration fails for high values; works for 10

    # simulate for some time
    state = np.zeros((time.size, len(dynObj.state_names)))
    state[0] = np.array([10000, 10000, 10000, -1000, -1000, -100])

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
        fig.axes[2].set_ylabel("N-vel (m/s)")
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


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("WebAgg")

        plt.close("all")

    #test_rv_prop()
    test_rv_control()

    if DEBUG:
        plt.show()
        