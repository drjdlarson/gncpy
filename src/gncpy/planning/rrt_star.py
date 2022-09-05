import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import deepcopy
from PIL import Image
from sys import exit

import gncpy.dynamics as gdyn
import gncpy.control as gcontrol
import gncpy.plotting as gplot


class Node:  # Each node on the tree has these properties
    def __init__(self, state):
        self.sv = state[:, 0]
        self.u = []
        self.path = []
        self.parent = []
        # Parent node this node corresponds to
        self.cost = 0
        # Node cost


class LQRRRTStar:
    def __init__(
        self,
        lqr=None,
        goal_sample_rate=10,
        max_iter=300,
        connect_circle_dist=2,
        step_size=1,
        expand_dis=1,
        update_plot=1,
        rng=None,
    ):

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.start = None
        self.end = None
        self.node_list = []
        self.numPos = 0
        self.min_rand = np.array([])
        # Min State space to sample RRT* Paths

        self.max_rand = np.array([])
        # Max State space to sample RRT* Paths

        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.step_size = step_size
        self.expand_dis = expand_dis
        self.update_plot = update_plot

        # Obstacles
        self.ell_con = None
        self.Nobs = 0
        self.obstacle_list = np.array([])
        # States of each Obstacle
        self.P = np.zeros((self.numPos, self.numPos, self.Nobs))

        self.pos_orient = list(range(0, self.numPos))

        # setup LQR planner
        self.lqr = lqr
        self.S = np.array([[]])

    def set_control_model(self, lqr):
        if not isinstance(lqr, gcontrol.LQR):
            raise TypeError("Must specify an LQR instance")
        self.lqr = lqr
        self.lqr.time_horizon = float("inf")  # must generate trajecotries to end state

    def set_environment(self, search_area=None, obstacles=None):
        if search_area is not None:
            self.numPos = int(search_area.shape[0] / 2)
            self.min_rand = search_area[0 : self.numPos]
            self.max_rand = search_area[self.numPos :]
            self.pos_orient = list(range(0, self.numPos))

        if obstacles is not None:
            self.ell_con = 1
            self.Nobs = obstacles.shape[0]
            self.obstacle_list = obstacles

            self.P = np.zeros((self.numPos, self.numPos, self.Nobs))
            for k, r in enumerate(self.obstacle_list[:, -1]):
                self.P[:, :, k] = r ** (-2) * np.eye(self.numPos)

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

    def plan(
        self,
        cur_time,
        cur_state,
        end_state,
        search_until_max_iter=False,
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
    ):
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if plt_inds is None:
            plt_inds = [0, 1]

        self.start = Node(cur_state.reshape((-1, 1)))
        self.end = Node(end_state.reshape((-1, 1)))
        self.node_list = [
            deepcopy(self.start),
        ]

        frame_list = []
        if show_animation:
            if fig is None:
                fig = plt.figure()
                if self.numPos == 2:
                    fig.add_subplot(1, 1, 1)
                    fig.axes[0].set_aspect("equal", adjustable="box")
                    fig.axes[0].set_xlim((self.min_rand[0], self.max_rand[0]))
                    fig.axes[0].set_ylim((self.min_rand[1], self.max_rand[1]))
                elif self.numPos == 3:
                    fig.add_subplot(1, 1, 1, projection="3d")
                    fig.axes[0].set_xlim((self.min_rand[0], self.max_rand[0]))
                    fig.axes[0].set_ylim((self.min_rand[1], self.max_rand[1]))
                    fig.axes[0].set_zlim((self.min_rand[2], self.max_rand[2]))

                if plt_opts is None:
                    plt_opts = gplot.init_plotting_opts(f_hndl=fig)
                if ttl is None:
                    ttl = "LQR-RRT*"
                gplot.set_title_label(fig, 0, plt_opts, ttl=ttl)

                fig.axes[0].scatter(
                    *self.start.sv[plt_inds], color="g", marker="o", zorder=1000,
                )

                for obs in self.obstacle_list:
                    if self.numPos == 2:
                        c = Circle(obs[:2], radius=obs[-1], color="k", zorder=1000)
                        fig.axes[0].add_patch(c)

                    elif self.numPos == 3:
                        fig.axes[0].plot_surface(
                            *self._make_sphere(*obs), rstride=3, cstride=3, color="k"
                        )

            fig.axes[0].scatter(
                *self.end.sv[plt_inds], color="r", marker="x", zorder=1000,
            )
            fig.tight_layout()
            plt.pause(0.1)

            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            fig_w, fig_h = fig.canvas.get_width_height()

            # save first frame of animation
            if save_animation:
                with io.BytesIO() as buff:
                    fig.savefig(buff, format="raw")
                    buff.seek(0)
                    img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                        (fig_h, fig_w, -1)
                    )
                frame_list.append(Image.fromarray(img))

        if disp:
            print("Starting LQR-RRT* Planning...")

        for i in range(self.max_iter):
            rnd = self.get_random_node()
            if disp:
                print(
                    "\tIter: {:4d}, number of nodes: {:6d}".format(
                        i, len(self.node_list)
                    )
                )

            nearest_ind = self.get_nearest_node_index(
                cur_time, rnd, state_args, ctrl_args
            )
            new_node = self.steer(
                cur_time, self.node_list[nearest_ind], rnd, state_args, ctrl_args
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

            if show_animation and (i % self.update_plot == 0):
                last_index = self.search_best_goal_node()
                if last_index:
                    traj = self.generate_final_course(last_index)[0]

                    if self.numPos == 2:
                        fig.axes[0].plot(
                            traj[plt_inds[0], :],
                            traj[plt_inds[1], :],
                            color=(0.5, 0.5, 0.5),
                            alpha=0.1,
                            zorder=-10,
                        )
                    elif self.numPos == 3:
                        fig.axes[0].plot(
                            traj[plt_inds[0], :],
                            traj[plt_inds[1], :],
                            traj[plt_inds[2], :],
                            color=(0.5, 0.5, 0.5),
                            alpha=0.1,
                            zorder=-10,
                        )
                    plt.pause(0.001)
                    if save_animation:
                        with io.BytesIO() as buff:
                            fig.savefig(buff, format="raw")
                            buff.seek(0)
                            img = np.frombuffer(
                                buff.getvalue(), dtype=np.uint8
                            ).reshape((fig_h, fig_w, -1))
                        frame_list.append(Image.fromarray(img))

            if (not search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index:
                    traj, u_traj = self.generate_final_course(last_index)
                    if show_animation:
                        if self.numPos == 2:
                            fig.axes[0].plot(
                                traj[plt_inds[0], :], traj[plt_inds[1], :], color="g",
                            )
                        elif self.numPos == 3:
                            fig.axes[0].plot(
                                traj[plt_inds[0], :],
                                traj[plt_inds[1], :],
                                traj[plt_inds[2], :],
                                color="g",
                            )
                        plt.pause(0.01)
                        if save_animation:
                            with io.BytesIO() as buff:
                                fig.savefig(buff, format="raw")
                                buff.seek(0)
                                img = np.frombuffer(
                                    buff.getvalue(), dtype=np.uint8
                                ).reshape((fig_h, fig_w, -1))
                            frame_list.append(Image.fromarray(img))
                    details = (u_traj, fig, frame_list)
                    return (traj, *details) if provide_details else traj

        if disp:
            print("\tReached Max Iteration!!")

        last_index = self.search_best_goal_node()
        if last_index:
            traj, u_traj = self.generate_final_course(last_index)
            if show_animation:
                if self.numPos == 2:
                    fig.axes[0].plot(
                        traj[plt_inds[0], :], traj[plt_inds[1], :], color="g",
                    )
                elif self.numPos == 3:
                    fig.axes[0].plot(
                        traj[plt_inds[0], :],
                        traj[plt_inds[1], :],
                        traj[plt_inds[2], :],
                        color="g",
                    )
                plt.pause(0.01)
                if save_animation:
                    with io.BytesIO() as buff:
                        fig.savefig(buff, format="raw")
                        buff.seek(0)
                        img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                            (fig_h, fig_w, -1)
                        )
                    frame_list.append(Image.fromarray(img))
        else:
            traj = np.array([[]])
            u_traj = np.array([[]])
            if disp:
                print("\tCannot find path!!")  # Undo Print

        details = (u_traj, fig, frame_list)
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
            self.dx(sv, self.end.sv)[self.pos_orient].T
            @ self.dx(sv, self.end.sv)[self.pos_orient]
        )
        return dist

    def rewire(
        self, cur_time, new_node, near_inds, state_args, ctrl_args
    ):  # Rewires the Nodes
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(cur_time, new_node, near_node, state_args, ctrl_args)
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
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(cur_time, near_node, new_node, state_args, ctrl_args)
            if t_node and self.check_collision(t_node):
                costs.append(
                    self.calc_new_cost(
                        cur_time, near_node, new_node, state_args, ctrl_args
                    )
                )
            else:
                costs.append(float("inf"))
        min_cost = min(costs)

        if min_cost == float("inf"):
            NoPathInf_str = "No Path - Infinite Cost"
            # print(NoPathInf_str);#Undo Print

            return None
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(
            cur_time, self.node_list[min_ind], new_node, state_args, ctrl_args
        )
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def calc_new_cost(
        self, cur_time, from_node, to_node, state_args, ctrl_args
    ):  # Calculates cost of node
        x_sim, u_sim = self.lqr.calculate_control(
            cur_time,
            from_node.sv.reshape((-1, 1)),
            end_state=to_node.sv.reshape((-1, 1)),
            provide_details=True,
            state_args=state_args,
            ctrl_args=ctrl_args,
        )[2:4]
        x_sim = x_sim.T
        u_sim = u_sim.T
        x_sim_sample, u_sim_sample, course_lens = self.sample_path(x_sim, u_sim)
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
        for k, xobs in enumerate(self.obstacle_list):
            distxyz = (node.path[: self.numPos, :].T - xobs[:-1].reshape((1, -1))).T
            d_list = np.einsum("ij,ij->i", (distxyz.T @ self.P[:, :, k]), distxyz.T)
            if -min(d_list) + self.ell_con >= 0.0:
                return False
        return True

    def get_random_node(self):  # Find a random node from the state space
        if self.rng.integers(0, high=100) > self.goal_sample_rate:
            rand_Pos = np.zeros((self.numPos, 1))
            for k in range(self.numPos):
                rand_Pos[k, 0] = self.rng.uniform(self.min_rand[k], self.max_rand[k])
            rand_x = np.block([[rand_Pos], [np.zeros((2, 1))]])
            rnd = Node(rand_x)
        else:  # goal point sampling
            rnd = Node(self.end.sv.reshape((-1, 1)))
        return rnd

    def get_nearest_node_index(
        self, cur_time, rnd_node, state_args, ctrl_args
    ):  # Get nearest node index in tree
        self.S = self.lqr.solve_dare(
            cur_time,
            rnd_node.sv.reshape((-1, 1)),
            state_args=state_args,
            ctrl_args=ctrl_args,
        )[0]
        # dlist=float('inf')*np.ones((len(self.node_list),1));
        dlist = [
            np.matmul(
                np.matmul(self.dx(node.sv, rnd_node.sv).T, self.S),
                self.dx(node.sv, rnd_node.sv),
            )
            for node in self.node_list
        ]
        minind = dlist.index(min(dlist))
        return minind

    def steer(
        self, cur_time, from_node, to_node, state_args, ctrl_args,
    ):  # Obtain trajectory between from_node to to_node using LQR and save trajectory
        x_sim, u_sim = self.lqr.calculate_control(
            cur_time,
            from_node.sv.reshape((-1, 1)),
            end_state=to_node.sv.reshape((-1, 1)),
            provide_details=True,
            state_args=state_args,
            ctrl_args=ctrl_args,
        )[2:4]

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
