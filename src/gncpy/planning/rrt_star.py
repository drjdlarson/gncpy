import numpy as np
import scipy.linalg

import gncpy.dynamics as gdyn
import gncpy.control as gcontrol


class Node:  # Each node on the tree has these properties
    def __init__(self, state):
        self.sv = state[:, 0]
        self.u = []
        self.path = []
        self.parent = []
        # Parent node this node corresponds to
        self.cost = 0
        # Node cost


class RRTStar:
    def __init__(self, x0, xdes, obstacles, randArea, Q, R, dynObj, dt):
        self.start = Node(x0)
        # Start Node

        self.end = Node(xdes)
        # End Node

        self.numPos = int(randArea.shape[0] / 2)
        self.min_rand = randArea[0 : self.numPos]
        # Min State space to sample RRT* Paths

        self.max_rand = randArea[self.numPos :]
        # Max State space to sample RRT* Paths

        self.goal_sample_rate = 10
        # Samples goal 10% of time

        self.max_iter = 300
        # Max RRT* Iteration

        self.connect_circle_dist = 2
        # Circular Radius of to Calculate Near Nodes

        self.step_size = 1
        # Step size of Interpolation between k and k+1

        self.expand_dis = 1
        # To find CLosest Node to the end Node

        self.update_plot = 20
        # Update plot Iteration

        self.d = len(x0) - 1
        # Dimension of State minus one

        # Obstacles
        self.ell_con = 1
        # Ellipsoid Obstacle Model for 2d
        if self.numPos == 2:
            self.Nobs = obstacles.shape[0]
            # Number of Obstacles
            self.obstacle_list = obstacles[:, 0:-1]
            # States of each Obstacle
            self.P = np.zeros((2, 2, self.Nobs))
            k = 0
            for r in obstacles[:, -1]:
                self.P[:, :, k] = r ** (-2) * np.array([[1, 0], [0, 1]])
                # Shape of Obstacle
                k = k + 1
        elif self.numPos == 3:
            self.Nobs = obstacles.shape[0]
            # Number of Obstacles
            self.obstacle_list = obstacles[:, 0:-1]
            # States of each Obstacle
            self.P = np.zeros((3, 3, self.Nobs))
            k = 0
            for r in obstacles[:, -1]:
                self.P[:, :, k] = r ** (-2) * np.array(
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                )
                # Shape of Obstacle
                k = k + 1

        # setup LQR planner
        self.lqr = gcontrol.LQR(time_horizon=100 * dt)
        self.lqr.set_state_model(u_nom, dynObj=dynObj, dt=dt)
        self.lqr.set_cost_model(Q, R)

        self.GOAL_DIST = 0.1
        # Max Distance from xdes for LQR Convergence

        self.eps = 1e-8
        # eps for finite differences

        self.pos_orient = list(range(0, self.numPos))
        self.nx = len(dynObj.state_names)
        # Number of states
        self.nu = R.shape[0]
        # Number of Control Inputs

    def E(self, x):
        G = np.array(
            [
                [x[9], -x[8], x[7]],
                [x[8], x[9], -x[6]],
                [-x[7], x[6], x[9]],
                [-x[6], -x[7], -x[8]],
            ]
        )
        Eout = scipy.linalg.block_diag(self.eye3, self.eye3, G, self.eye3)
        return Eout

    def slerp(self, q1, q2, t):  # Quaternion Interpolation
        lamb = np.dot(q1, q2)
        if lamb < 0:
            q2 = -q2
            lamb = -lamb
        if np.sqrt((1 - lamb) ** 2) < 0.000001:
            r = 1 - t
            s = t
        else:
            alpha = np.arccos(lamb)
            gamma = 1 / np.sin(alpha)
            r = np.sin((1 - t) * alpha) * gamma
            s = np.sin(t * alpha) * gamma
        qi = r * q1 + s * q2
        Q = qi / np.linalg.norm(qi)
        return Q

    def qinv(self, q1):  # Quaternion inverse
        qout = scipy.linalg.block_diag(-np.identity(3), 1) @ q1
        return qout

    def qprod(self, q2, q1):  # q1 goes to q2 (ref), Quaternion Multiplication
        Psi = np.array(
            [
                [q1[3], q1[2], -q1[1]],
                [-q1[2], q1[3], q1[0]],
                [q1[1], -q1[0], q1[3]],
                [-q1[0], -q1[1], -q1[2]],
            ]
        )
        qdot = np.bmat([Psi, q1.reshape((4, 1))]).A
        qout = qdot @ q2
        return qout

    def psiinv(self, q):  # Quaternion to 3 variable representation
        q = q / np.linalg.norm(q)
        out = q[0:3] / q[3]
        return out

    def dx(
        self, x1, x2
    ):  # x1 goes to x2. Difference between current state to Reference State
        return (x1 - x2).reshape((-1, 1))

    def plan(self, disp=True):  # LQR-RRT* Planner
        search_until_max_iter = 0
        self.node_list = [self.start]

        if disp:
            print("Starting LQRRRT* Planning...")

        for i in range(self.max_iter):
            rnd = self.get_random_node()
            if disp and (i % self.update_plot == 0):
                print(
                    "\tIter: {:4d}, number of nodes: {:6d}".format(
                        i, len(self.node_list)
                    )
                )

                last_index = self.search_best_goal_node()
                traj = None
                if last_index:
                    traj, u_traj = self.generate_final_course(last_index)
                # self.draw_plot(rnd,traj)

            nearest_ind = self.get_nearest_node_index(rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)
            if new_node is None:
                continue
            if self.check_collision(new_node) and self.check_vel(new_node):
                near_indices = self.find_near_nodes(new_node, nearest_ind)
                new_node = self.choose_parent(new_node, near_indices)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indices)
            # if (i)%self.update_plot==0:
            # last_index=self.search_best_goal_node();
            # traj=None;
            # if last_index:
            #    traj, u_traj=self.generate_final_course(last_index);
            # self.draw_plot(rnd,traj);

            if (not search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index:
                    traj, u_traj = self.generate_final_course(last_index)
                    return traj, u_traj, self
        MaxIter_str = "Reached Max Iteration"
        print(MaxIter_str)

        last_index = self.search_best_goal_node()
        if last_index:
            traj, u_traj = self.generate_final_course(last_index)
            return traj, u_traj, self
        else:
            NoPath_str = "Cannot find path"
            # print(NoPath_str) #Undo Print

        return None, None

    def generate_final_course(self, goal_index):  # Generate Final Course
        path = self.end.sv[0:13].reshape(-1, 1)
        u_path = np.empty((6, 0))
        node = self.node_list[goal_index]
        while node.parent:
            i = np.flip(node.path, 1)
            path = np.append(path, i, axis=1)

            j = np.flip(node.u, 1)
            u_path = np.append(u_path, j, axis=1)

            node = node.parent
        path = np.flip(path, 1)
        u_path = np.flip(u_path, 1)
        # path.append([self.start.sv]);
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

    # def draw_plot(self,rnd=None,traj=None): # Draws 3D Trajectory using LQR-RRT*
    #    plt.clf()
    #    ax = plt.axes(projection="3d");
    #    #if rnd is not None:
    #    #    ax.scatter3D(rnd.x,rnd.y,rnd.z,c="black");
    #    for node in self.node_list:
    #        if node.parent:
    #            ax.plot3D(node.path_x,node.path_y,node.path_z,"-g");
    #    ax.scatter3D(self.start.x, self.start.y, c="red")
    #    ax.scatter3D(self.end.x, self.end.y, c="limegreen")
    #    if traj is not None:
    #        ax.plot3D(traj[0,:],traj[1,:],traj[2,:],"-r");
    #    ax.set_xlim3d(-2, 8);
    #    ax.set_ylim3d(-2,8);
    #    ax.set_zlim3d(-2,2);
    #    ax.grid(True);
    #    plt.pause(0.01);
    def rewire(self, new_node, near_inds):  # Rewires the Nodes
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if edge_node is None:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)
            no_collision = self.check_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(
        self, parent_node
    ):  # Re-computes cost from rewired Nodes
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def choose_parent(
        self, new_node, near_inds
    ):  # Chooses a parent node with lowest cost

        if not near_inds:
            return None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)

        if min_cost == float("inf"):
            NoPathInf_str = "No Path - Infinite Cost"
            # print(NoPathInf_str);#Undo Print

            return None
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def calc_new_cost(self, from_node, to_node):  # Calculates cost of node
        x_sim, u_sim = self.LQR_planning(from_node.sv, to_node.sv)
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
            * (np.log(nnode) / nnode) ** (1 / (self.d))
        )
        ind = [dist_list.index(i) for i in dist_list if i <= r]
        if not ind:
            ind = [nearest_ind]
        return ind

    def check_collision(self, node):  # Check for collisions with Ellipsoids
        k = 0
        for xobs in self.obstacle_list.T:
            distxyz = (node.path[0:3, :].T - xobs.T).T
            d_list = np.einsum("ij,ij->i", (distxyz.T @ self.P[:, :, k]), distxyz.T)
            k = k + 1
            if -min(d_list) + self.ell_con >= 0.0:
                return False
        return True

    def check_vel(self, node):
        for k in range(node.path.shape[1]):
            vel_bool = np.abs(node.path[3:6, [k]]) < self.vel_max_array
            # angvel_bool=np.abs(node.path[10:13,[k]])<self.angvel_max_array;
            for i in range(3):
                if vel_bool[i, 0] == False:  # or angvel_bool[i,0]==False
                    # print(vel_bool)
                    # rospy.loginfo(i)
                    # print(angvel_bool)
                    # print('---------')
                    return False
        return True

    def get_random_node(self):  # Find a random node from the state space
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rand_trans = np.array(
                [
                    [
                        np.random.uniform(self.min_rand[0], self.max_rand[0]),
                        np.random.uniform(self.min_rand[1], self.max_rand[1]),
                        np.random.uniform(self.min_rand[2], self.max_rand[2]),
                        0.0,
                        0.0,
                        0.0,
                    ]
                ]
            ).T
            s = np.random.uniform()
            sigma1 = np.sqrt(1 - s)
            sigma2 = np.sqrt(s)
            theta1 = 2 * 3.1415 * np.random.uniform()
            theta2 = 2 * 3.1415 * np.random.uniform()
            rand_rot = np.array(
                [
                    [
                        np.sin(theta1) * sigma1,
                        np.cos(theta1) * sigma1,
                        np.sin(theta2) * sigma2,
                        np.cos(theta2) * sigma2,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ]
            ).T
            rnd = Node(np.bmat([[rand_trans], [rand_rot]]).A)
        else:  # goal point sampling
            rnd = Node(self.end.sv.reshape((self.nx, 1)))
        return rnd

    def get_nearest_node_index(self, rnd_node):  # Get nearest node index in tree
        Ad, Bd = self.get_system_model(rnd_node.sv)
        S = self.solve_dare(Ad, Bd)
        # dlist=float('inf')*np.ones((len(self.node_list),1));
        dlist = np.array(
            [
                (self.dx(node.sv, rnd_node.sv).T @ S) @ self.dx(node.sv, rnd_node.sv)
                for node in self.node_list
            ]
        )
        minind = np.argmin(dlist)[0]
        # minind = dlist.index(min(dlist))
        return minind

    def solve_dare(self, A, B):  # Solve discrete Ricatti Equation for LQR
        Q, R = self.lqr.Q, self.lqr.R
        X = Q
        Xn = Q
        for i in range(self.MAX_ITER_LQR_Cost):
            Xn = (
                A.T @ X @ A
                - (((A.T @ X @ B) @ np.linalg.pinv(R + B.T @ X @ B)) @ B.T @ X @ A)
                + Q
            )
            if (abs(Xn - X)).max() < self.EPS:
                break
            X = Xn
        return Xn

    def steer(
        self, from_node, to_node
    ):  # Obtain trajectory between from_node to to_node using LQR and save trajectory
        x_sim, u_sim = self.lqr.calculate_control(
            cur_time,
            from_node.sv.reshape((-1, 1)),
            end_state=to_node.sv.reshape((-1, 1)),
            provide_details=True,
        )[2:4]
        # TODO: S used to be calculated in the LQR_planning function that used to be called above and is used in sample_path
        x_sim_sample, u_sim_sample, course_lens = self.sample_path(x_sim, u_sim)
        if len(x_sim_sample) == 0:
            return None
        newNode = Node()
        newNode.sv = x_sim_sample[:, -1]
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
                u_sim_sample.append(u_sim[:, i])
                x_trans = (t * x_sim[0:6, i + 1] + (1.0 - t) * x_sim[0:6, i]).reshape(
                    (-1, 1)
                )[:, 0]
                q1 = x_sim[6:10, i]
                q2 = x_sim[6:10, i + 1]
                x_rot = self.slerp(q1, q2, t)
                x_rot_vel = (
                    t * x_sim[10:13, i + 1] + (1.0 - t) * x_sim[10:13, i]
                ).reshape((-1, 1))[:, 0]
                x_sim_sample.append(np.concatenate((x_trans, x_rot, x_rot_vel), axis=0))
        x_sim_sample = np.array(x_sim_sample).T
        u_sim_sample = np.array(u_sim_sample).T
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

    def get_system_model(
        self, tt, xdes, state_args, ctrl_args
    ):  # Get Discrete model of LTV system
        return self.lqr.get_state_space(
            tt, xdes.reshape((-1, 1)), self.lqr.u_nom, state_args, ctrl_args
        )

    def cfinitediff(self, xtraj, utraj):
        xdotdot = np.zeros((xtraj.shape[0], xtraj.shape[1]))
        for k in range(0, xtraj.shape[1]):
            if k + 1 != xtraj.shape[1]:
                xdotplus = xtraj[:, [k]] + self.eps * self.fn_dyn(
                    xtraj[:, [k]], utraj[:, [k]], self.m, self.I
                )
                xdotminus = xtraj[:, [k]] - self.eps * self.fn_dyn(
                    xtraj[:, [k]], utraj[:, [k]], self.m, self.I
                )
                xdotdot[:, [k]] = (xdotplus - xdotminus) / (2 * self.eps)
            else:
                uzeros = np.zeros((6, 1))
                xdotplus = xtraj[:, [k]] + self.eps * self.fn_dyn(
                    xtraj[:, [k]], uzeros, self.m, self.I
                )
                xdotminus = xtraj[:, [k]] - self.eps * self.fn_dyn(
                    xtraj[:, [k]], uzeros, self.m, self.I
                )
                xdotdot[:, [k]] = (xdotplus - xdotminus) / (2 * self.eps)
        return xdotdot


def main():
    dt = 0.01

    # define dynamics
    dynObj = gdyn.DoubleIntegrator()
    uSize = 2

    # define starting and ending state for control calculation
    xdes = np.array([0, 2.5, 0, 0]).reshape((4, 1))
    x0 = np.array([0, -2.5, 0, 0]).reshape((4, 1))

    # define some circular obstacles with center pos and radius (x, y, radius)
    obstacles = np.array(
        [
            [0, -1.35, 0.2],
            [1.0, -0.5, 0.2],
            [-0.95, -0.5, 0.2],
            [-0.2, 0.3, 0.2],
            [0.8, 0.7, 0.2],
            [1.1, 2.0, 0.2],
            [-1.2, 0.8, 0.2],
            [-1.1, 2.1, 0.2],
            [-0.1, 1.6, 0.2],
            [-1.1, -1.9, 0.2],
            [1.0 + np.sqrt(2), -1.5 - np.sqrt(2), 0.2],
        ]
    )

    # define Q and R weights for using standard cost function
    Q = 50 * np.eye(len(dynObj.state_names))
    R = 0.6 * np.eye(uSize)

    # define enviornment bounds for the robot
    minxy = np.array([-2.0, -3])
    maxxy = np.array([2, 3])
    randArea = np.concatenate((minxy, maxxy))

    # Initialize LQR-RRT* Planner
    param = rrtStar(x0, xdes, obstacles, randArea, Q, R, dynObj, dt)

    a = 3


if __name__ == "__main__":
    main()
