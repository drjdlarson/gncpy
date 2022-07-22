"""Implements RL environments for the SimpleUAV2d game.

This follows the new format for the OpenAI Gym environment API. They provide
default wrappers for backwards compatibility with some learning libraries.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from warnings import warn

from gncpy.games.SimpleUAV2d import SimpleUAV2d as UAVGame


class SimpleUAV2d(gym.Env):
    """RL environment for the :class:`gncpy.games.SimpleUAV2d.SimpleUAV2d` game.

    Attributes
    ----------
    render_mode : string
        Mode to render in. See :attr:`.metadata` for available modes.
    game : :class:`gncpy.games.SimpleUAV2d.SimpleUAV2d`
        Main game to play.
    fig : matplotlib figure
        For legacy support of rendering function
    obs_type : string
        Observation type to use. Options are :code:`'image'` or :code:`'player_state'`.
    aux_use_n_targets : bool
        Flag indicating if auxilary state uses the number of targets.
    aux_use_time : bool
        Flag indicating if auxilary state uses the current time.
    max_time : float
        Maximum time in real units for the environment. It is recommended to
        set the game to have unlimited time and use this instead as this allows
        the RL algorithms more visibility. Once this time is surpassed the
        episode is truncated and appropriate flags are set.
    observation_space : :class:`gym.spaces.Box` or :class:`gym.spaces.Dict`
        Observation space. This depends on the observation type and auxilary
        flags.
    """

    metadata = {"render_modes": ["human", "single_rgb_array"], "render_fps": 60}
    """Additional metadata for the class."""

    action_space = spaces.Box(
        low=-np.ones(2), high=np.ones(2), dtype=np.float32
    )
    """Space for available actions."""

    def __init__(
        self,
        config_file="SimpleUAV2d.yaml",
        render_mode="single_rgb_array",
        obs_type="player_state",
        max_time=10,
        aux_use_n_targets=False,
        aux_use_time=False,
    ):
        """Initialize an object.

        Parameters
        ----------
        config_file : string, optional
            Full path of the configuration file. The default is "SimpleUAV2d.yaml".
        render_mode : string, optional
            Render mode to use. Must be specified at initialization time, then
            the render function does not need to be called. The default is
            "single_rgb_array".
        obs_type : string, optional
            Observation type to use. The default is "player_state".
        max_time : float, optional
            Maximum time for an episode in game's real units. The default is 10.
        aux_use_n_targets : bool, optional
            Flag indicating if auxilary state uses the number of targets. The
            default is False.
        aux_use_time : bool, optional
            Flag indicating if auxilary state uses the current time. The default
            is False.
        """
        super().__init__()

        if render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode
        else:
            self.render_mode = self.metadata["render_modes"][0]
            warn(
                "Invalid render mode ({}) defaulting to {}".format(
                    render_mode, self.render_mode
                )
            )

        # self.action_space = spaces.Box(
        #     low=-np.ones(2), high=np.ones(2), dtype=np.float32
        # )

        self.game = UAVGame(config_file, self.render_mode, rng=self.np_random)
        self.game.setup()
        self.game.step(self.gen_act_map(np.zeros_like(self.action_space.low)))

        self.fig = None  # for legacy support of render function
        self.obs_type = obs_type
        self.aux_use_n_targets = aux_use_n_targets
        self.aux_use_time = aux_use_time
        self.max_time = max_time

        self.observation_space = self.calc_obs_space()

        self.metadata["render_fps"] = self.game.render_fps

    def step(self, action):
        """Perform one iteration of the game loop.

        Parameters
        ----------
        action : numpy array
            Action to take in the game.

        Returns
        -------
        observation : :class:`gym.spaces.Box` or :class:`gym.spaces.Dict`
            Current observation of the game.
        reward : float
            Reward from current step.
        done : bool
            Flag indicating if the episode met the end conditions.
        truncated : bool
            Flag indicating if the episode has ended due to time constraints.
        info : dict
            Extra debugging info.
        """
        truncated = False
        info = self.game.step(self.gen_act_map(action))

        if self.max_time is None:
            truncated = False
        else:
            truncated = self.game.elapsed_time > self.max_time

        return (self._get_obs(), self.game.score, self.game.game_over, truncated, info)

    def render(self, mode=None):
        """Deprecated. Handles rendering a frame of the environment.

        This is deprecated and the render mode should instead be set at
        initialization.

        Parameters
        ----------
        mode : string, optional
            The rendering mode to use. The default is None, which does nothing.
        """
        if mode is None:
            return
        elif self.render_mode is not None:
            DeprecationWarning(
                "Calling render directly is deprecated, "
                + "specify the render mode during initialization instead."
            )

        if mode == "single_rgb_array":
            return self.game.img.copy()
        elif self.render_mode == "human":
            return
        elif mode == "human":
            if self.fig is None:
                px2in = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
                orig = plt.rcParams["toolbar"]
                plt.rcParams["toolbar"] = "None"
                self.fig = plt.figure(
                    figsize=(
                        px2in * self.game.img.shape[0],
                        px2in * self.game.img.shape[1],
                    )
                )
                self.fig.add_axes([0, 0, 1, 1], frame_on=False, rasterized=True)
                plt.rcParams["toolbar"] = orig

            self.fig.clear()
            self.fig.imshow(self.game.img)
            plt.pause(1 / self.game.render_fps)

        else:
            warn("Invalid render mode: {}".format(mode))

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the environment to an initial state.

        This method can reset the environment’s random number generator(s)
        if seed is an integer or if the environment has not yet initialized a
        random number generator. If the environment already has a random number
        generator and reset() is called with seed=None, the RNG should not be
        reset. Moreover, reset() should (in the typical use case) be called
        with an integer seed right after initialization and then never again.

        Parameters
        ----------
        seed : int, optional
            The seed that is used to initialize the environment’s PRNG. If the
            environment does not already have a PRNG and :code:`seed=None`
            (the default option) is passed, a seed will be chosen from some
            source of entropy (e.g. timestamp or /dev/urandom). However, if the
            environment already has a PRNG and :code:`seed=None` is passed, the
            PRNG will not be reset. If you pass an integer, the PRNG will be
            reset even if it already exists. Usually, you want to pass an
            integer right after the environment has been initialized and then
            never again. The default is None.
        return_info : bool, optional
            If true, return additional information along with initial
            observation. This info should be analogous to the info returned in
            :meth:`.step`. The default is False.
        options : dict, optional
            Not used by this environment. The default is None.

        Returns
        -------
        observation : :class:`gym.spaces.Box` or :class:`gym.spaces.Dict`
            Initial observation of the environment.
        info : dict, optional
            Additonal debugging info, only returned if :code:`return_info=True`.
        """
        seed = super().reset(seed=seed)

        self.game.reset(rng=self.np_random)
        info = self.game.step(self.gen_act_map(np.zeros_like(self.action_space.low)))
        observation = self._get_obs()

        return (observation, info) if return_info else observation

    def close(self):
        """Nicely shuts down the environment."""
        self.game.close()
        super().close()

    def calc_obs_space(self):
        """Determines the observation space based on specified options.

        If a dictionary space is used, the keys are :code:`'img'` for the image
        of the game screen and :code:`'aux'` for the auxilary state vector. Both
        values are boxes.

        Raises
        ------
        RuntimeError
            Invalid observation type specified.

        Returns
        -------
        out : :class:`gym.spaces.Box` or :class:`gym.spaces.Dict`
            Observation space.
        """
        # determine main state
        main_state = None
        if self.obs_type == "image":
            shape = (*self.game.get_image_size(), 3)
            main_state = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        elif self.obs_type == "player_state":
            state_bnds = self.game.get_player_state_bounds()
            main_state = spaces.Box(
                low=state_bnds[0], high=state_bnds[1], dtype=np.float32
            )

        else:
            raise RuntimeError("Invalid observation type ({})".format(self.obs_type))

        # create aux state if needed
        aux_state_low = np.array([])
        aux_state_high = np.array([])

        if self.aux_use_n_targets:
            aux_state_low = np.append(aux_state_low, 0)
            aux_state_high = np.append(aux_state_high, np.inf)

        if self.aux_use_time:
            aux_state_low = np.append(aux_state_low, 0)
            aux_state_high = np.append(aux_state_high, np.inf)

        # combine into final space
        if self.obs_type == "image":
            if aux_state_low.size > 0:
                aux_state = spaces.Box(aux_state_low, aux_state_high, dtype=np.float32)
                out = spaces.Dict({"img": main_state, "aux": aux_state})

            else:
                out = main_state

        else:
            if aux_state_low.size > 0:
                low = np.concatenate((main_state.low, aux_state_low))
                high = np.concatenate((main_state.high, aux_state_high))
                out = spaces.Box(low, high, dtype=np.float32)

            else:
                out = main_state

        return out

    def gen_act_map(self, action):
        """Maps actions to entity ids for the game.

        This assumes there is only 1 player and if there are more then all
        players get the same action.

        Parameters
        ----------
        action : numpy array
            Action to take in the game.

        Returns
        -------
        act_map : dict
            Each key is an entity id and each value is a numpy array.
        """
        # Note: should only have 1 player
        ids = self.game.get_player_ids()
        if len(ids) > 1:
            warn(
                "Multi-player environment not supported, "
                + "all players using same action."
            )

        act_map = {}
        for _id in ids:
            act_map[_id] = action
        return act_map

    def _get_obs(self):
        """Generates an observation."""
        # get main state
        if self.obs_type == "image":
            main_state = self.game.img.copy()
        elif self.obs_type == "player_state":
            p_states = self.game.get_players_state()
            if len(p_states) == 0:
                raise RuntimeError("No players alive")
            main_state = p_states[list(p_states.keys())[0]]
        else:  # catch all in case a new case is forgotten
            msg = "Failed to generate observation for type {}".format(self.obs_type)
            raise NotImplementedError(msg)

        # get aux state, if any
        aux_state = np.array([])
        if self.aux_use_n_targets:
            aux_state = np.append(aux_state, self.game.get_num_targets())

        if self.aux_use_time:
            aux_state = np.append(aux_state, self.game.elapsed_time)

        # combine into output
        if self.obs_type == "image":
            if aux_state.size > 0:
                return dict(img=main_state, aux=aux_state)
            else:
                return main_state
        else:
            if aux_state.size > 0:
                return np.concatenate((main_state, aux_state), dtype=np.float32)
            else:
                return main_state.astype(np.float32)


class SimpleUAVHazards2d(SimpleUAV2d):
    """Simple 2d UAV environment with hazards.

    This follows the same underlying game logic as the :class:`.SimpleUAV2d`
    environment but has some hazards added to its default configuration.
    """

    def __init__(self, config_file="SimpleUAVHazards2d.yaml", **kwargs):
        """Initialize an object."""
        super().__init__(config_file=config_file, **kwargs)
