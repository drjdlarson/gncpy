"""Implements the A* algorithm and several variations."""
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sys import exit

import gncpy.plotting as gplot


class Node:
    """Helper class for grid nodes in A* planning.

    Attributes
    ----------
    indices : 2 numpy array
        row/column index into grid.
    cost : float
        cost of the path up to this node.
    parent_idx : int
        index of the parent node in the closed set
    """

    def __init__(self, indices, cost, parent_idx):
        """Initialize an object.

        Parameters
        ----------
        indices : 2 numpy array
            row/column index into grid.
        cost : float
            cost of the path up to this node.
        parent_idx : int
            index of the parent node in the closed set
        """
        self.indices = indices
        self.cost = cost
        self.parent_idx = parent_idx


class AStar:
    """Implements various forms of the A* gird search algorithm.

    This is based on the example implementation found
    `here <https://github.com/AtsushiSakai/PythonRobotics>`_. It currently
    implements

        - Normal A*
        - Beam search
        - Weighted

    Attributes
    ----------
    resolution : 2 numpy array
        Real distance per gird square for x and y dimensions. Should not be set
        directly, use the :meth:`.set_map` function.
    min : 2 numpy array
        Minimum x and y positions in real units. Should not be set directly, use
        the :meth:`.set_map` function.
    max : 2 numpy array
        Maximum x/y positions in real units. Should not be set directly, use
        the :meth:`.set_map` function.
    weight : float
        Constant weighting factor applied to the heuristic. The default is 1 and
        is only used if the weighting funtion is not overwritten.
    use_beam_search : bool
        Flag indicating if the beam search variation is used.
    beam_search_max_nodes : int
        Maximum nuber of grid nodes to keep when using the beam search variation.
    motion : 8 x 3 numpy array
        Each row represents a potential action, with the first column being row
        motion in gird squares, second column being column motion, and third
        being the cost of moving.
    """

    def __init__(self, use_beam_search=False, beam_search_max_nodes=30):
        """Initialize an object.

        Parameters
        ----------
        use_beam_search : bool, optional
            Flag indicating if the beam search variation is used. The default
            is False.
        beam_search_max_nodes : int, optional
            Maximum nuber of grid nodes to keep when using the beam search
            variation. The default is 30.
        """
        self.resolution = np.nan * np.ones(2)
        self.min = np.nan * np.ones(2)
        self.max = np.nan * np.ones(2)

        self.weight = 1
        self.use_beam_search = use_beam_search
        self.beam_search_max_nodes = beam_search_max_nodes

        self.motion = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [-1, -1, np.sqrt(2)],
                [-1, 1, np.sqrt(2)],
                [1, -1, np.sqrt(2)],
                [1, 1, np.sqrt(2)],
            ]
        )

        self._map = np.array([[]])
        self._obstacles = None
        self._hazards = None

    def ind_to_pos(self, indices):
        """Convert a set of row/colum indices to real positions.

        Note that rows are y positions and columns are x positions, this function
        handles the conversion.

        Parameters
        ----------
        indices : 2 numpy array
            row/column index into grid.

        Returns
        -------
        2 numpy array
            real x/y position.
        """
        # NOTE: row is y, column is x
        return indices[[1, 0]] * self.resolution + self.min + self.resolution / 2

    def pos_to_ind(self, pos):
        """Convert a set of x/y positions to grid indices.

        Note that rows are y positions and columns are x positions, this function
        handles the conversion. This does not bounds check the indices.

        Parameters
        ----------
        pos : 2 numpy array
            Real x/y position.

        Returns
        -------
        2 numpy array
            grid row/column index.
        """
        # NOTE: x is column, y is row, bound to given area
        p = np.max(np.vstack((pos.ravel(), self.min.ravel())), axis=0)
        p = np.min(np.vstack((p, self.max.ravel())), axis=0)
        inds = np.floor(
            (p - self.min.ravel() + self.resolution / 2) / self.resolution.ravel()
        )[[1, 0]]

        return inds.astype(int)

    def ravel_ind(self, multi_index):
        """Convert row/column indices into a single index into flattened array.

        Parameters
        ----------
        multi_index : 2 numpy array
            Row/col indices into the gird.

        Returns
        -------
        float
            Flattened index
        """
        return np.ravel_multi_index(multi_index, self._map.shape).item()

    def final_path(self, endNode, closed_set):
        """Calculate the final path and total cost.

        Parameters
        ----------
        endNode : :class`.Node`
            Ending position.
        closed_set : dict
            Dictionary of path nodes. The keys are flattened indices and the
            values are :class:`.Node`.

        Returns
        -------
        N x 2 numpy array
            Real x/y positions of the path nodes.
        float
            total cost of the path.
        """
        path = np.nan * np.ones((len(closed_set) + 1, endNode.indices.size))
        path[0, :] = self.ind_to_pos(endNode.indices)
        parent_idx = endNode.parent_idx
        p_ind = 1
        while parent_idx > 0:
            curNode = closed_set[parent_idx]
            path[p_ind, :] = self.ind_to_pos(curNode.indices)
            p_ind += 1
            parent_idx = curNode.parent_idx

        # remove all the extras
        path = path[:p_ind, :]

        # flip so first row is starting point
        return path[::-1, :], endNode.cost

    def calc_weight(self, node):
        """Calculates the weight of applied to the heuristic.

        This can be overriddent to calculate custom weights.

        Parameters
        ----------
        node : :class:`.Node`
            Current node.

        Returns
        -------
        float
            weight value.
        """
        return self.weight

    def calc_heuristic(self, endNode, curNode):
        """Calculates the heuristic cost.

        Parameters
        ----------
        endNode : :class:`.Node`
            Goal node.
        curNode : :class:`.Node`
            Current node.

        Returns
        -------
        float
            heuristic cost.
        """
        diff = self.ind_to_pos(endNode.indices) - self.ind_to_pos(curNode.indices)
        return np.sqrt(np.sum(diff * diff))

    def is_valid(self, node):
        """Checks if the node is valid.

        Bounds checks the indices and determines  if node corresponds to a wall.

        Parameters
        ----------
        node : :class:`.Node`
            Node to check.

        Returns
        -------
        bool
            True if the node is valid.
        """
        pos = self.ind_to_pos(node.indices)

        # bounds check position
        if np.any(pos < self.min) or np.any(pos > self.max):
            return False

        # obstacles are inf
        if np.isinf(self._map[node.indices[0], node.indices[1]]):
            return False

        return True

    def draw_start(self, fig, startNode):
        """Draw starting node on the figure.

        Parameters
        ----------
        fig : matplotlib figure
            Figure to draw on.
        startNode : :class:`.Node`
            Starting node.
        """
        pos = self.ind_to_pos(startNode.indices) - self.resolution / 2
        fig.axes[0].add_patch(
            Rectangle(
                pos, self.resolution[0], self.resolution[1], color="g", zorder=1000
            ),
        )

    def draw_end(self, fig, endNode):
        """Draw the ending node on the figure.

        Parameters
        ----------
        fig : matplotlib figure
            Figure to draw on.
        endNode : :class:`.Node`
            Ending node.
        """
        pos = self.ind_to_pos(endNode.indices) - self.resolution / 2
        fig.axes[0].add_patch(
            Rectangle(
                pos, self.resolution[0], self.resolution[1], color="r", zorder=1000
            ),
        )

    def draw_map(self, fig):
        """Draws the map to the figure.

        Parameters
        ----------
        fig : matplotlib figure
            Figure to draw on.
        """
        rows, cols = np.where(np.isinf(self._map))
        inds = np.vstack((rows, cols)).T
        for ii in inds:
            pos = self.ind_to_pos(ii)
            fig.axes[0].add_patch(
                Rectangle(
                    pos - self.resolution / 2,
                    self.resolution[0],
                    self.resolution[1],
                    facecolor="k",
                )
            )

        rows, cols = np.where(self._map > 0)
        inds = np.vstack((rows, cols)).T
        for ii in inds:
            pos = self.ind_to_pos(ii)
            fig.axes[0].add_patch(
                Rectangle(
                    pos - self.resolution / 2,
                    self.resolution[0],
                    self.resolution[1],
                    facecolor=(255 / 255, 140 / 255, 0 / 255),
                    zorder=-1000,
                )
            )

    def set_map(self, min_pos, max_pos, grid_res, obstacles=None, hazards=None):
        """Sets up the map with obstacles and hazards.

        Parameters
        ----------
        min_pos : 2 numpy array
            Min x/y position in real units.
        max_pos : 2 numpy array
            Max x/y position in real units.
        grid_res : 2 numpy array
            Real distance per gird square for x/y positions.
        obstacles : N x 4 numpy array, optional
            Locations of walls, each row is one wall. First column is x position
            second column is y position, third is width, and fourth is height.
            The position is the location of the center. All distances are in
            real units. The default is None.
        hazards : N x 5 numpy array, optional
            Locations of walls, each row is one wall. First column is x position
            second column is y position, third is width, fourth is height, and
            the last is the cost of being on that node. The position is the
            location of the center. All distances are in real units. The
            default is None.
        """
        self.resolution = grid_res.ravel()
        self.min = min_pos.ravel() - self.resolution / 2
        self.max = max_pos.ravel() + self.resolution / 2

        self._obstacles = obstacles
        self._hazards = hazards

        max_inds = self.pos_to_ind(self.max)
        self._map = np.zeros([ii + 1 for ii in max_inds.tolist()])

        if self._obstacles is not None:
            for obs in self._obstacles:
                width2 = np.array([max(obs[2], self.resolution[0]) / 2, 0])
                height2 = np.array([0, max(obs[3], self.resolution[1]) / 2])
                left = max(self.pos_to_ind(obs[:2] - width2)[1], 0)
                right = min(self.pos_to_ind(obs[:2] + width2)[1], max_inds[1])
                top = min(self.pos_to_ind(obs[:2] + height2)[0], max_inds[0])
                bot = max(self.pos_to_ind(obs[:2] - height2)[0], 0)

                # TODO: more efficent indexing?
                for row in range(bot, top):
                    for col in range(left, right):
                        self._map[row, col] = np.inf

        if self._hazards is not None:
            for haz in self._hazards:
                width2 = np.array([max(haz[2], self.resolution[0]) / 2, 0])
                height2 = np.array([0, max(haz[3], self.resolution[1]) / 2])
                left = max(self.pos_to_ind(haz[:2] - width2)[1], 0)
                right = min(self.pos_to_ind(haz[:2] + width2)[1], max_inds[1])
                top = min(self.pos_to_ind(haz[:2] + height2)[0], max_inds[0])
                bot = max(self.pos_to_ind(haz[:2] - height2)[0], 0)

                # TODO: more efficent indexing?
                for row in range(bot, top):
                    for col in range(left, right):
                        self._map[row, col] += haz[4]

    def get_map_cost(self, indices):
        """Returns the cost of being at the given map indices.

        Parameters
        ----------
        indices : 2 numpy array
            Row/col indices.

        Returns
        -------
        float
            Cost of the gird node.
        """
        return self._map[indices[0], indices[1]]

    def plan(
        self,
        start_pos,
        end_pos,
        show_animation=False,
        save_animation=False,
        plt_opts=None,
        ttl=None,
        fig=None,
    ):
        """Runs the search algorithm.

        The setup functoins should be called prior to this.

        Parameters
        ----------
        start_pos : 2 numpy array
            Starting x/y pos in real units.
        end_pos : 2 numpy array
            ending x/y pos in real units.
        show_animation : bool, optional
            Flag indicating if an animated plot should be shown during planning.
            The default is False. If shown, escape key can be used to quit.
        save_animation : bool, optional
            Flag for saving each frame of the animation as a PIL image. This
            can later be saved to a gif. The default is False. The animation must
            be shown for this to have an effect.
        plt_opts : dict, optional
            Additional options for the plot from
            :meth:`gncpy.plotting.init_plotting_opts`. The default is None.
        ttl : string, optional
            title string of the plot. The default is None which gives a generic
            name of the algorithm.
        fig : matplotlib figure, optional
            Figure object to plot on. The default is None which makes a new
            figure. If a figure is provided, then only the ending state,
            searched nodes, and best path are added to the figure.

        Returns
        -------
        path : Nx2 numpy array
            Real positions of the path.
        cost : float
            total cost of the path.
        fig : matplotlib figure
            Figure that was drawn, None if the animation is not shown.
        frame_list : list
            Each element is a PIL image corresponding to an animation frame.
        """
        startNode = Node(
            self.pos_to_ind(start_pos),
            self.get_map_cost(self.pos_to_ind(start_pos)),
            -1,
        )
        endNode = Node(
            self.pos_to_ind(end_pos), self.get_map_cost(self.pos_to_ind(end_pos)), -1
        )

        if not self.is_valid(startNode) or not self.is_valid(endNode):
            return np.array((0, start_pos.size)), float('inf'), fig, [] 

        frame_list = []

        if show_animation:
            if fig is None:
                fig = plt.figure()
                fig.add_subplot(1, 1, 1)

                # fig.axes[0].grid(True)
                fig.axes[0].set_aspect("equal", adjustable="box")

                fig.axes[0].set_xlim((self.min[0], self.max[0]))
                fig.axes[0].set_ylim((self.min[1], self.max[1]))

                if plt_opts is None:
                    plt_opts = gplot.init_plotting_opts(f_hndl=fig)

                if ttl is None:
                    ttl = "A* Pathfinding"
                    if self.use_beam_search:
                        ttl += " with Beam Search"

                gplot.set_title_label(fig, 0, plt_opts, ttl=ttl)

                # draw map
                self.draw_start(fig, startNode)
                self.draw_map(fig)
                fig.tight_layout()

            self.draw_end(fig, endNode)
            plt.pause(0.01)

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

        else:
            fig = None

        open_set = {}
        closed_set = {}
        ind = self.ravel_ind(startNode.indices)
        open_set[ind] = startNode

        while len(open_set) > 0:
            s_inds = np.argsort(
                [
                    n.cost + self.calc_weight(n) * self.calc_heuristic(endNode, n)
                    for n in open_set.values()
                ]
            )
            keys = list(open_set.keys())
            c_ind = keys[s_inds[0]]
            curNode = open_set[c_ind]
            if self.use_beam_search and s_inds.size > self.beam_search_max_nodes:
                for ii in s_inds[: self.beam_search_max_nodes - 1 : -1]:
                    del open_set[keys[ii]]

            del open_set[c_ind]

            if show_animation:
                pos = self.ind_to_pos(curNode.indices)
                fig.axes[0].scatter(pos[0], pos[1], marker="x", color="c")
                if len(closed_set) % 10 == 0:
                    plt.pause(0.001)
                    # save frame after pause to make sure frame is drawn
                    if save_animation:
                        with io.BytesIO() as buff:
                            fig.savefig(buff, format="raw")
                            buff.seek(0)
                            img = np.frombuffer(
                                buff.getvalue(), dtype=np.uint8
                            ).reshape((fig_h, fig_w, -1))
                        frame_list.append(Image.fromarray(img))

            if np.all(curNode.indices == endNode.indices):
                endNode.parent_idx = curNode.parent_idx
                endNode.cost = curNode.cost
                break

            closed_set[c_ind] = curNode

            for m in self.motion:
                new_indices = curNode.indices + m[:2].astype(int)
                node = Node(new_indices, 0, c_ind)
                if not self.is_valid(node):
                    continue
                node.cost = curNode.cost + m[2] + self.get_map_cost(new_indices)

                n_ind = self.ravel_ind(node.indices)

                if n_ind in closed_set:
                    continue

                if n_ind not in open_set or open_set[n_ind].cost > node.cost:
                    open_set[n_ind] = node

        path, cost = self.final_path(endNode, closed_set)

        if show_animation:
            fig.axes[0].plot(path[:, 0], path[:, 1], linestyle="-", color="g")
            plt.pause(0.001)
            if save_animation:
                with io.BytesIO() as buff:
                    fig.savefig(buff, format="raw")
                    buff.seek(0)
                    img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                        (fig_h, fig_w, -1)
                    )
                frame_list.append(Image.fromarray(img))

        return path, cost, fig, frame_list
