"""Implements variations of the RRT* algorithm."""
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import deepcopy
from PIL import Image
from sys import exit

import gncpy.control as gcontrol
import gncpy.plotting as gplot


class Node:  # Each node on the tree has these properties
    """Helper class for nodes in the tree.

    Attributes
    ----------
    sv : N numpy array
        State vector
    u : Nu numpy array
        Control vector
    path : N x Np numpy array
        All state vectors in the path
    parent : :class:`.Node`
        Parent node.
    cost : float
        Node cost value.
    """

    def __init__(self, state):
        """Initialize an object.

        Parameters
        ----------
        state : numpy array
            state vector for the node.
        """
        self.sv = state.ravel()
        self.u = []
        self.path = []
        self.parent = []
        self.cost = 0


class LQRRRTStar:
    """Implements the LQR-RRT* algorithm.

    Attributes
    ----------
    rng : numpy random number generator
        Instance of a random number generator to use.
    start : N x 1 numpy array
        Starting state
    end : N x 1 numpy array
        Ending state
    node_list : list
        List of nodes in the tree.
    min_rand : Np numpy array
        Minimum position values when generating random nodes
    max_rand : Np numpy array
        Maximum position values when generating radnom nodes
    pos_inds : list
        List of position indices in the state vector
    sampling_fun : callable
        Function that returns a random sample from the state space. Must
        take rng, pos_inds, min_rand, max_rand as inputs and return a numpy array
        of the same size as the state vector.
    goal_sample_rate : float
    max_iter : int
        Maximum iterations to search for.
    connect_circle_dist : float
    step_size : float
    expand_dis : float
    ell_con : float
        Ellipsoidal constraint value
    Nobs : int
        Number of obstacles.
    obstacle_list : Nobs x (3 or 4) numpy array
        Each row describes an obstacle; x pos, y pos, (z pos), radius.
    P : (2 or 3) x (2 or 3) x Nobs numpy array
        States of each obstacle for collision checking.
    planner : :class:`gncpy.control.LQR`
        Planner class instance for predicting paths.
    S : N x N numpy array
        Solution to the discrete time algebraic riccatti equation.
    """

    def __init__(
        self,
        planner=None,
        goal_sample_rate=10,
        max_iter=300,
        connect_circle_dist=2,
        step_size=1,
        expand_dis=1,
        rng=None,
        sampling_fun=None,
    ):

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.start = None
        self.end = None
        self.node_list = []
        self.min_rand = np.array([])
        self.max_rand = np.array([])
        self.pos_inds = []

        self.sampling_fun = sampling_fun

        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.step_size = step_size
        self.expand_dis = expand_dis

        # Obstacles
        self.ell_con = None
        self.Nobs = 0
        self.obstacle_list = np.array([])
        self.P = np.zeros((self.numPos, self.numPos, self.Nobs))

        # setup LQR planner
        self.planner = planner
        self.S = np.array([[]])
        self.planner_args = {}

        # plotting helpers
        self._fig = None
        self._plt_inds = None
        self._show_planner = False
        self._frame_list = []
        self._fig_h = None
        self._fig_w = None
        self._plt_opts = gplot.init_plotting_opts()

    @property
    def numPos(self):
        return len(self.pos_inds)

    def set_control_model(self, planner, pos_inds, controller_args=None):
        if not isinstance(planner, gcontrol.LQR):
            raise TypeError("Must specify an LQR instance")
        self.planner = planner
        self.pos_inds = pos_inds
        if controller_args is not None:
            self.planner_args = controller_args

    def set_environment(self, search_area=None, obstacles=None):
        if search_area is not None:
            self.min_rand = search_area[0, :]
            self.max_rand = search_area[1, :]

        if obstacles is not None:
            self.ell_con = 1
            self.Nobs = obstacles.shape[0]
            self.obstacle_list = obstacles

            dim = int(max(obstacles.shape[1] - 1, 0))

            self.P = np.zeros((dim, dim, self.Nobs))
            for k, r in enumerate(self.obstacle_list[:, -1]):
                self.P[:, :, k] = r ** (-2) * np.eye(dim)

    def dx(
        self, x1, x2
    ):  # x1 goes to x2. Difference between current state to Reference State
        return (x1 - x2).reshape((-1, 1))

    def _make_sphere(self, xc, yc, zc, r):
        u = np.linspace(0, 2 * np.pi, 39)
        v = np.linspace(0, np.pi, 21)

        x = r * np.outer(np.cos(u), np.sin(v)) + xc
        y = r * np.outer(np.sin(u), np.sin(v)) + yc
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zc

        return x, y, z

    def draw_obstacles(self, ax):
        for obs in self.obstacle_list:
            if self.numPos == 2:
                c = Circle(obs[:2], radius=obs[-1], color="k", zorder=1000)
                self._fig.axes[ax].add_patch(c)

            elif self.numPos == 3:
                self._fig.axes[ax].plot_surface(
                    *self._make_sphere(*obs), rstride=3, cstride=3, color="k"
                )

    def draw_start(self, start, ax):
        self._fig.axes[ax].scatter(
            *start.ravel()[self._plt_inds], color="g", marker="o", zorder=1000,
        )

    def save_frame(self):
        plt.pause(0.005)
        with io.BytesIO() as buff:
            self._fig.savefig(buff, format="raw")
            buff.seek(0)
            img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                (self._fig_h, self._fig_w, -1)
            )
        self._frame_list.append(Image.fromarray(img))

    def reset_controller_plot(self):
        self._fig.axes[1].clear()

        self._fig.axes[1].set_aspect("equal", adjustable="box")
        self._fig.axes[1].set_xlim((self.min_rand[0], self.max_rand[0]))
        self._fig.axes[1].set_ylim((self.min_rand[1], self.max_rand[1]))

        gplot.set_title_label(
            self._fig, 1, self._plt_opts, x_lbl="x pos", use_local=True,
        )

        self.draw_obstacles(1)

        plt.pause(0.05)

    def plan(
        self,
        cur_time,
        cur_state,
        end_state,
        use_first_traj=True,
        use_convergence=False,
        rtol=1e-3,
        state_args=None,
        ctrl_args=None,
        provide_details=False,
        disp=True,
        show_animation=False,
        save_animation=False,
        plt_opts=None,
        ttl=None,
        fig=None,
        plt_inds=None,
        show_planner=True,
    ):
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if plt_inds is None:
            plt_inds = [0, 1]

        if use_convergence:
            use_first_traj = False

        self._plt_inds = plt_inds
        self._fig = fig

        self.start = Node(cur_state.reshape((-1, 1)))
        self.end = Node(end_state.reshape((-1, 1)))
        self.node_list = [
            deepcopy(self.start),
        ]

        self._frame_list = []
        self._show_planner = False
        if show_animation:
            self._show_planner = show_planner and isinstance(
                self.planner, gcontrol.ELQR
            )
            if self._show_planner:
                n_plts = 2
            else:
                n_plts = 1

            if self._fig is None:
                self._fig = plt.figure()
                if self.numPos == 2:
                    for ii in range(n_plts):
                        self._fig.add_subplot(1, n_plts, ii + 1)
                    self._fig.axes[0].set_aspect("equal", adjustable="box")
                    self._fig.axes[0].set_xlim((self.min_rand[0], self.max_rand[0]))
                    self._fig.axes[0].set_ylim((self.min_rand[1], self.max_rand[1]))
                elif self.numPos == 3:
                    self._fig.add_subplot(1, 1, 1, projection="3d")
                    self._fig.axes[0].set_xlim((self.min_rand[0], self.max_rand[0]))
                    self._fig.axes[0].set_ylim((self.min_rand[1], self.max_rand[1]))
                    self._fig.axes[0].set_zlim((self.min_rand[2], self.max_rand[2]))

                if plt_opts is None:
                    self._plt_opts = gplot.init_plotting_opts(f_hndl=self._fig)
                if ttl is None:
                    ttl = "LQR-RRT*"
                gplot.set_title_label(
                    self._fig, 0, self._plt_opts, ttl=ttl, x_lbl="x pos", y_lbl="y pos"
                )

                self.draw_start(self.start.sv, 0)
                self.draw_obstacles(0)

                if self._show_planner:
                    self.reset_controller_plot()
                    gplot.set_title_label(
                        self._fig, 0, self._plt_opts, ttl="Tree", use_local=True
                    )

            self._fig.axes[0].scatter(
                *self.end.sv[plt_inds], color="r", marker="x", zorder=1000,
            )

            if self._show_planner:
                self.planner_args.update(
                    {
                        "show_animation": True,
                        "save_animation": save_animation,
                        "ax_num": 1,
                        "fig": self._fig,
                        "plt_inds": plt_inds,
                        "plt_opts": self._plt_opts,
                    }
                )

            self._fig.tight_layout()
            plt.pause(0.1)

            # for stopping simulation with the esc key.
            self._fig.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            self._fig_w, self._fig_h = self._fig.canvas.get_width_height()

            # save first frame of animation
            if save_animation:
                self.save_frame()

        if disp:
            print("Starting LQR-RRT* Planning...")

        last_cost = float("inf")
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            if disp:
                print(
                    "\tIter: {:4d}, number of nodes: {:5d}, last cost: {:10.4f}".format(
                        i, len(self.node_list), last_cost
                    )
                )

            nearest_ind = self.get_nearest_node_index(
                cur_time, rnd, state_args, ctrl_args
            )
            new_node = self.steer(
                cur_time, self.node_list[nearest_ind], rnd, state_args, ctrl_args, show_controller=True,
                planner_ttl="Searching",
            )
            if new_node is None:
                continue
            if self.check_collision(new_node):
                near_indices = self.find_near_nodes(new_node, nearest_ind)
                new_node = self.choose_parent(
                    cur_time, new_node, near_indices, state_args, ctrl_args
                )
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(cur_time, new_node, near_indices, state_args, ctrl_args)

                    if show_animation and new_node is not None:
                        if self.numPos == 2:
                            self._fig.axes[0].plot(
                                new_node.path[plt_inds[0], :],
                                new_node.path[plt_inds[1], :],
                                color=(0.5, 0.5, 0.5),
                                alpha=0.15,
                                zorder=-10,
                            )
                        elif self.numPos == 3:
                            self._fig.axes[0].plot(
                                new_node.path[plt_inds[0], :],
                                new_node.path[plt_inds[1], :],
                                new_node.path[plt_inds[2], :],
                                color=(0.5, 0.5, 0.5),
                                alpha=0.1,
                                zorder=-10,
                            )

                        plt.pause(0.005)
                        if save_animation:
                            self.save_frame()

            if new_node:
                last_index = self.search_best_goal_node()
                if last_index:
                    traj, u_traj = self.generate_final_course(last_index)
                    cost = self.node_list[last_index].cost
                    converged = use_convergence and np.abs(np.abs(last_cost - cost) / cost) < rtol
                    last_cost = cost
                    if use_first_traj or converged:
                        if show_animation:
                            if self.numPos == 2:
                                self._fig.axes[0].plot(
                                    traj[plt_inds[0], :], traj[plt_inds[1], :], color="g",
                                )
                            elif self.numPos == 3:
                                self._fig.axes[0].plot(
                                    traj[plt_inds[0], :],
                                    traj[plt_inds[1], :],
                                    traj[plt_inds[2], :],
                                    color="g",
                                )
                            plt.pause(0.01)
                            if save_animation:
                                self.save_frame()

                        details = (u_traj, self._fig, self._frame_list)
                        return (traj, *details) if provide_details else traj

        if disp:
            print("\tReached Max Iteration!!")

        last_index = self.search_best_goal_node()
        if last_index:
            traj, u_traj = self.generate_final_course(last_index)
            if show_animation:
                if self.numPos == 2:
                    self._fig.axes[0].plot(
                        traj[plt_inds[0], :], traj[plt_inds[1], :], color="g",
                    )
                elif self.numPos == 3:
                    self._fig.axes[0].plot(
                        traj[plt_inds[0], :],
                        traj[plt_inds[1], :],
                        traj[plt_inds[2], :],
                        color="g",
                    )
                plt.pause(0.01)
                if save_animation:
                    self.save_frame()
        else:
            traj = np.array([[]])
            u_traj = np.array([[]])
            if disp:
                print("\tCannot find path!!")  # Undo Print

        details = (u_traj, self._fig, self._frame_list)
        return (traj, *details) if provide_details else traj

    def generate_final_course(self, goal_index):  # Generate Final Course
        path = self.end.sv.reshape(-1, 1)
        u_path = np.array([])
        node = self.node_list[goal_index]
        while node.parent:
            i = np.flip(node.path, 1)
            path = np.append(path, i, axis=1)

            j = np.flip(node.u, 1)
            if u_path.size == 0:
                u_path = j
            else:
                u_path = np.append(u_path, j, axis=1)

            node = node.parent
        path = np.flip(path, 1)
        u_path = np.flip(u_path, 1)
        return path, u_path

    def search_best_goal_node(self):  # Finds Node closest to Goal Node
        dist_to_goal_list = [self.calc_dist_to_goal(node.sv) for node in self.node_list]
        goal_inds = [
            dist_to_goal_list.index(i)
            for i in dist_to_goal_list
            if i <= self.expand_dis
        ]
        if not goal_inds:
            return None
        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost:
                return i
        return None

    def calc_dist_to_goal(self, sv):  # Calculate distance between Node and the Goal
        dist = np.sqrt(
            self.dx(sv, self.end.sv)[self.pos_inds].T
            @ self.dx(sv, self.end.sv)[self.pos_inds]
        )
        return dist

    def rewire(
        self, cur_time, new_node, near_inds, state_args, ctrl_args
    ):  # Rewires the Nodes
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(
                cur_time,
                new_node,
                near_node,
                state_args,
                ctrl_args,
                show_controller=True,
                planner_ttl="Rewiring"
            )
            if edge_node is None:
                continue
            edge_node.cost = self.calc_new_cost(
                cur_time, new_node, near_node, state_args, ctrl_args
            )
            no_collision = self.check_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                self.propagate_cost_to_leaves(cur_time, new_node, state_args, ctrl_args)

    def propagate_cost_to_leaves(
        self, cur_time, parent_node, state_args, ctrl_args
    ):  # Re-computes cost from rewired Nodes
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(
                    cur_time, parent_node, node, state_args, ctrl_args
                )
                self.propagate_cost_to_leaves(cur_time, node, state_args, ctrl_args)

    def choose_parent(
        self, cur_time, new_node, near_inds, state_args, ctrl_args
    ):  # Chooses a parent node with lowest cost

        if not near_inds:
            return None
        costs = np.inf * np.ones(len(near_inds))
        for kk, i in enumerate(near_inds):
            near_node = self.node_list[i]
            t_node = self.steer(cur_time, near_node, new_node, state_args, ctrl_args)
            if t_node and self.check_collision(t_node):
                costs[kk] = self.calc_new_cost(
                    cur_time, near_node, new_node, state_args, ctrl_args
                )

        min_ind = np.argmin(costs)
        min_cost = costs[min_ind]

        if np.isinf(min_cost):
            print("\t\tNo Path - Infinite Cost")
            return None

        new_node = self.steer(
            cur_time,
            self.node_list[near_inds[min_ind]],
            new_node,
            state_args,
            ctrl_args,
            show_controller=True,
            planner_ttl="Generating Node",
        )
        new_node.parent = self.node_list[near_inds[min_ind]]
        new_node.cost = min_cost
        return new_node

    def calc_new_cost(
        self, cur_time, from_node, to_node, state_args, ctrl_args
    ):  # Calculates cost of node
        x_sim, u_sim = self.call_planner(
            cur_time, from_node.sv, to_node.sv, state_args, ctrl_args
        )
        x_sim_sample, u_sim_sample, course_lens = self.sample_path(x_sim.T, u_sim.T)
        if len(x_sim_sample) == 0:
            return float("inf")
        return from_node.cost + sum(course_lens)

    def find_near_nodes(
        self, new_node, nearest_ind
    ):  # Finds near nodes close to new_node
        nnode = len(self.node_list) + 1
        dist_list = [
            self.dx(node.sv, new_node.sv).T @ self.S @ self.dx(node.sv, new_node.sv)
            for node in self.node_list
        ]
        r = (
            self.connect_circle_dist
            * np.amin(dist_list)
            * (np.log(nnode) / nnode) ** (1 / (new_node.sv.size - 1))
        )
        ind = [dist_list.index(i) for i in dist_list if i <= r]
        if not ind:
            ind = [nearest_ind]
        return ind

    def check_collision(self, node):  # Check for collisions with Ellipsoids
        if len(self.obstacle_list) == 0:
            return False

        for k, xobs in enumerate(self.obstacle_list):
            distxyz = (node.path[self.pos_inds, :].T - xobs[:-1].reshape((1, -1))).T
            d_list = np.einsum("ij,ij->i", (distxyz.T @ self.P[:, :, k]), distxyz.T)
            if -min(d_list) + self.ell_con >= 0.0:
                return False
        return True

    def get_random_node(self):  # Find a random node from the state space
        if self.rng.integers(0, high=100) > self.goal_sample_rate:
            rand_x = self.sampling_fun(
                self.rng, self.pos_inds, self.min_rand, self.max_rand
            )
            rnd = Node(rand_x)
        else:  # goal point sampling
            rnd = Node(self.end.sv.reshape((-1, 1)))
        return rnd

    def get_nearest_node_index(
        self, cur_time, rnd_node, state_args, ctrl_args
    ):  # Get nearest node index in tree
        self.S = self.planner.solve_dare(
            cur_time,
            rnd_node.sv.reshape((-1, 1)),
            state_args=state_args,
            ctrl_args=ctrl_args,
        )[0]
        dlist = np.array(
            [
                (
                    self.dx(node.sv, rnd_node.sv).T
                    @ self.S
                    @ self.dx(node.sv, rnd_node.sv)
                ).item()
                for node in self.node_list
            ]
        )
        return np.argmin(dlist)

    def call_planner(
        self, cur_time, from_state, to_state, state_args, ctrl_args, **kwargs
    ):
        extra_args = dict()
        extra_args.update(**self.planner_args)
        extra_args.update(**kwargs)

        if self._show_planner and extra_args["show_animation"]:
            self.reset_controller_plot()
            self.draw_start(from_state, 1)
            plt.pause(0.01)

            _, _, x_traj, u_traj, _, f_lst = self.planner.calculate_control(
                cur_time,
                from_state.reshape((-1, 1)),
                end_state=to_state.reshape((-1, 1)),
                provide_details=True,
                state_args=state_args,
                ctrl_args=ctrl_args,
                **extra_args,
            )

            self._frame_list.extend(f_lst)
            return x_traj, u_traj

        return self.planner.calculate_control(
            cur_time,
            from_state.reshape((-1, 1)),
            end_state=to_state.reshape((-1, 1)),
            provide_details=True,
            state_args=state_args,
            ctrl_args=ctrl_args,
            **extra_args,
        )[2:4]

    def steer(
        self,
        cur_time,
        from_node,
        to_node,
        state_args,
        ctrl_args,
        show_controller=False,
        planner_ttl=None,
    ):  # Obtain trajectory between from_node to to_node using LQR and save trajectory
        if self._show_planner:
            self.planner_args["show_animation"] = show_controller
        if self.planner_args.get("show_animation", False):
            self.planner_args["ttl"] = (
                planner_ttl if planner_ttl is not None else "Controller"
            )

        x_sim, u_sim = self.call_planner(
            cur_time, from_node.sv, to_node.sv, state_args, ctrl_args
        )

        x_sim_sample, u_sim_sample, course_lens = self.sample_path(x_sim.T, u_sim.T)
        if len(x_sim_sample) == 0:
            return None
        newNode = Node(x_sim_sample[:, -1].reshape((-1, 1)))
        newNode.u = u_sim_sample
        newNode.path = x_sim_sample
        newNode.cost = from_node.cost + np.sum(np.abs(course_lens))
        newNode.parent = from_node
        return newNode

    def sample_path(self, x_sim, u_sim):  # Interpolate path obtained by LQR
        x_sim_sample = []
        u_sim_sample = []
        if x_sim.size == 0:
            clen = []
            return x_sim_sample, u_sim_sample, clen
        for i in range(x_sim.shape[1] - 1):
            for t in np.arange(0.0, 1.0, self.step_size):
                u_sim_sample.append(u_sim[:, i].tolist())
                x_sample = (t * x_sim[:, i + 1] + (1.0 - t) * x_sim[:, i]).reshape(
                    (-1, 1)
                )[:, 0]
                x_sim_sample.append(x_sample.tolist())

        # enforce shape if x_sim_sample is empty list
        x_sim_sample = np.array(x_sim_sample).T.reshape((x_sim.shape[0], -1))
        u_sim_sample = np.array(u_sim_sample).T.reshape((u_sim.shape[0], -1))

        # diff_x_sim=np.diff(x_sim_sample);
        diff_x_sim2 = [
            self.dx(x_sim_sample[:, k + 1], x_sim_sample[:, k])[:, 0]
            for k in range(x_sim_sample.shape[1] - 1)
        ]
        diff_x_sim = np.array(diff_x_sim2).T
        if diff_x_sim.size == 0:
            return [], [], []
        clen = np.einsum("ij,ij->i", (diff_x_sim.T @ self.S), diff_x_sim.T)
        return x_sim_sample, u_sim_sample, clen


class ExtendedLQRRRTStar(LQRRRTStar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inv_state_args = None
        self._inv_ctrl_args = None
        self._cost_args = None

    def call_planner(
        self, cur_time, from_state, to_state, state_args, ctrl_args, **kwargs
    ):
        extra_args = dict(disp=False)
        extra_args.update(**self.planner_args)
        extra_args.update(**kwargs)
        extra_args.update(
            dict(
                cost_args=self._cost_args,
                inv_state_args=self._inv_state_args,
                inv_ctrl_args=self._inv_ctrl_args,
            )
        )
        return super().call_planner(
            cur_time, from_state, to_state, state_args, ctrl_args, **extra_args
        )

    def plan(
        self,
        cur_time,
        cur_state,
        end_state,
        inv_state_args=None,
        inv_ctrl_args=None,
        cost_args=None,
        **kwargs
    ):
        self._inv_state_args = inv_state_args
        self._inv_ctrl_args = inv_ctrl_args
        self._cost_args = cost_args
        return super().plan(cur_time, cur_state, end_state, **kwargs)
