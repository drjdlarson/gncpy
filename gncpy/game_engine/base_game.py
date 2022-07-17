"""Defines base game engine classes.

These define common functions, properties, and interfaces for all games.
"""
from abc import ABC, abstractmethod
import os
import pathlib
import numpy as np
from ruamel.yaml import YAML

import gncpy.game_engine.rendering2d as grender2d
import gncpy.game_engine.components as gcomp
from gncpy.game_engine.rendering2d import Shape2dParams
from gncpy.game_engine.physics2d import Physics2dParams, Collision2dParams
from gncpy.game_engine.entities import EntityManager


yaml = YAML()
"""Global yaml interpretter, should be used when parsing any came configs."""

library_config_dir = os.path.join(
    pathlib.Path(__file__).parent.parent.resolve(), "games", "configs"
)
"""Directory of the libraries game configs where the default yaml files live."""


class WindowParams:
    """Parameters of the game window to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    width : int
        Width of the window in pixels.
    height : int
        Height of the window in pixels.
    """

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0


class BaseParams:
    """Main parameter class, may be inherited from to add custom attributes.

    Attributes
    ----------
    window : :class:`.WindowParams`
        Parameters of the window.
    """

    def __init__(self):
        super().__init__()
        self.window = WindowParams()


class Base2dParams(BaseParams):
    """Base parameters for 2d games, can be inherited from.

    Attributes
    ----------
    physics : :class:`.physics2d.Physics2dParams`
        Physics system parameters.
    start_time : float
        Starting time in game.
    max_time : float
        Maximum time for the game to run. The default is infinity if not set
        in the config file. To manually set unlimited time then supply a negative
        value in the config file.
    """

    def __init__(self):
        super().__init__()
        self.physics = Physics2dParams()
        self.start_time = 0.0
        self.max_time = np.inf


class BaseGame(ABC):
    """Base class for defining games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities. It must be subclassed and defines the expected interface of games.

    Attributes
    ----------
    config_file : string
        Full file path of the configuration file.
    entityManager : :class:`.entities.EntityManager`
        Factory class for making and managing game entities.
    render_mode : string
        Method of rendering the game.
    render_fps : int
        Frame rate to render the game at.
    current_frame : int
        Number of the current frame.
    score : float
        Score accumulated in the game.
    game_over : bool
        Flag indicating if the game has ended.
    params : :class:`BaseParams`
        Parameters for the game, read from yaml file.
    rng : numpy random generator
        Random number generator.
    seed_val : int
        Optional seed used in the random generator.
    """

    def __init__(
        self,
        config_file,
        render_mode,
        render_fps=None,
        use_library_config=False,
        seed=0,
        rng=None,
    ):
        """Initialize the object.

        Parameters
        ----------
        config_file : string
            Full path of the configuration file. This can be the name of a one
            of the libraries default files but the use_library_config flag
            should also be set.
        render_mode : string
            Mode to render the game.
        render_fps : int, optional
            FPS to render the game at. The default is None.
        use_library_config : bool, optional
            Flag indicating if the config file is in the default library location.
            The default is False.
        seed : int, optional
            Seed for the random number generator. The default is 0.
        rng : numpy random generator, optional
            Instance of the random generator to use. The default is None and will
            cause one to be created.
        """
        super().__init__()

        self.config_file = self.validate_config_file(
            config_file, use_library=use_library_config
        )
        self.entityManager = EntityManager()
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.current_frame = -1
        self.score = 0
        self.game_over = False
        self.params = None
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.seed_val = seed

    def setup(self):
        """Sets up the game by parsing the config file and checking max time.

        This should be called before any other game functions. It should be
        extended by child classes. If overridden then the developer is responsible
        for registering the parameters for the yaml parsing and parsing the config
        file.
        """
        global yaml
        self.register_params(yaml)

        self.parse_config_file()
        if self.params.max_time < 0:
            self.params.max_time = np.inf

    def register_params(self, yaml):
        """Register classes with the yaml parser.

        This should be extended by inherited classes.

        Parameters
        ----------
        yaml : ruamel.yaml YAML object
            yaml parser to use, should be the global parser.
        """
        yaml.register_class(WindowParams)
        yaml.register_class(BaseParams)

    def validate_config_file(self, config_file, use_library):
        """Validate that the config file exists.

        First checks if the file exists as provided, then checks the library
        directory if the use_library flag is true or it failed to find the file
        as provided.

        Parameters
        ----------
        config_file : string
            Full path to the config file.
        use_library : bool
            Flag indicating if the library directory will be checked.

        Raises
        ------
        RuntimeError
            If the file cannot be found.

        Returns
        -------
        cf : string
            full path to the config file that was found.
        """
        succ = os.path.isfile(config_file)
        if use_library or not succ:
            cf = os.path.join(library_config_dir, config_file)
            succ = os.path.isfile(cf)
        else:
            cf = config_file

        if not succ:
            raise RuntimeError("Failed to find config file {}".format(config_file))

        return cf

    def parse_config_file(self):
        """Parses the config file and saves the parameters."""
        builtins = (int, float, bool, str, dict)

        def helper(item):
            """Set default class values before being overriding from file."""
            item_cls = type(item)
            if item_cls in builtins or item_cls == tuple:
                return item
            true_item = item_cls()
            for field in dir(item):
                if field[0:2] == "__" or field[0] == "_":
                    continue
                # skip properties, assume they are read only
                try:
                    if isinstance(getattr(item_cls, field), property):
                        continue
                except AttributeError:
                    pass
                val = getattr(item, field)
                if callable(val):
                    continue
                val_type = type(getattr(true_item, field))
                if val_type in builtins:
                    setattr(true_item, field, val)
                elif val_type == list:
                    if not isinstance(val, list):
                        val = list([val,])
                    lst = []
                    for lst_item in val:
                        lst.append(helper(lst_item))
                    setattr(true_item, field, lst)
                elif val_type == tuple:
                    if not isinstance(val, tuple):
                        val = tuple(val)
                    setattr(true_item, field, val)
                elif val_type == np.ndarray:
                    try:
                        if not isinstance(val, list):
                            val = list([val,])
                        arr_val = np.array(val, dtype=float)
                    except ValueError:
                        raise RuntimeError(
                            "Failed to convert {:s} to numpy array ({}).".format(
                                field, val
                            )
                        )
                    setattr(true_item, field, arr_val)
                else:
                    setattr(true_item, field, helper(val))

            return true_item

        with open(self.config_file, "r") as fin:
            self.params = helper(yaml.load(fin))

    def reset(self, seed=None, rng=None):
        """Resets to the base state.

        If a random generator is provided then that is used. Otherwise the seed
        value is used to create a new generator. If neither is provided, but a
        seed had previously been provided then the old seed is used to recreate
        the generator. If all else fails, then a new default generator is initialized.

        Parameters
        ----------
        seed : int, optional
            seed for the random number generator. The default is None.
        rng : numpy random generator, optional
            Instance of the random number generator to use. The default is None.
        """
        self.entityManager = EntityManager()
        self.current_frame = 0
        self.game_over = False
        self.score = 0

        if rng is not None:
            self.rng = rng
        elif seed is None:
            if self.seed_val is None:
                self.rng = np.random.default_rng()
            else:
                self.rng = np.random.default_rng(self.seed_val)
        else:
            self.rng = np.random.default_rng(seed)
            self.seed_val = seed

    @abstractmethod
    def s_movement(self, action):
        """Abstract method for moving entities according to their dynamics.

        Parameters
        ----------
        action : numpy array, int, bool, dict, etc.
            action to take in the game.
        """
        raise NotImplementedError()

    @abstractmethod
    def s_collision(self):
        """Abstract method to check for collisions between entities.

        May return extra info useful by the step function.
        """
        raise NotImplementedError()

    @abstractmethod
    def s_game_over(self):
        """Abstract method to check for game over conditions."""
        raise NotImplementedError()

    @abstractmethod
    def s_score(self):
        """Abstact method to calculate the score.

        Returns
        -------
        info : dict
            Extra info for debugging.
        """
        raise NotImplementedError()

    @abstractmethod
    def s_input(self, *args):
        """Abstract method to turn user inputs into game actions."""
        raise NotImplementedError()

    @abstractmethod
    def step(self, *args):
        """Abstract method defining what to do each frame.

        Returns
        -------
        info : dict
            Extra infomation for debugging.
        """
        raise NotImplementedError


class BaseGame2d(BaseGame):
    """Base class for defining 2d games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities. It assumes the rendering will be done by pygame.

    Attributes
    ----------
    clock : pygame clock
        Clock for the rendering system.
    window : pygame window
        Window for drawing to.
    img : H x W x 3 numpy array
        Pixel values of the screen image.
    current_update_count : int
        Number of update steps taken, this may be more than the frame count as
        multiple physics updates may be made per frame.
    dist_per_pix : numpy array
        Real distance per pixel in x (width) and y (height) order.
    """

    def __init__(self, config_file, render_mode, **kwargs):
        super().__init__(config_file, render_mode, **kwargs)
        self.clock = None
        self.window = None
        self.img = np.array([])
        self.current_update_count = -1
        self.dist_per_pix = np.array([])  # width, height

    @property
    def current_time(self):
        """Current time in real units."""
        return self.elapsed_time + self.params.start_time

    @property
    def elapsed_time(self):
        """Amount of time elapsed in game in real units."""
        return self.params.physics.update_dt * self.current_update_count

    def setup(self):
        """Sets up the game and should be called before any game functions.

        Sets up the physics and rendering system and resets to the base state.
        """
        super().setup()

        # update render fps if not set
        if self.render_fps is None:
            self.render_fps = 1 / self.params.physics.dt

        self.clock = grender2d.init_rendering_system()
        self.window = grender2d.init_window(
            self.render_mode, self.params.window.width, self.params.window.height
        )

        self.dist_per_pix = np.array(
            [
                self.params.physics.dist_width / self.window.get_width(),
                self.params.physics.dist_height / self.window.get_height(),
            ]
        )

        self.reset()

    def register_params(self, yaml):
        """Register custom classes for this game with the yaml parser.

        Parameters
        ----------
        yaml : ruamel.yaml YAML object
            yaml parser to use, should be the global parser.
        """
        super().register_params(yaml)
        yaml.register_class(Shape2dParams)
        yaml.register_class(Collision2dParams)
        yaml.register_class(Physics2dParams)
        yaml.register_class(Base2dParams)

    def get_image_size(self):
        """Gets the size of the window.

        Returns
        -------
        tuple
            first is the height next is the width, in pixels.
        """
        sz = self.window.get_size()
        return sz[1], sz[0]

    def append_name_to_keys(self, in_dict, prefix):
        """Append a prefix to every key in a dictionary.

        A dot is placed between the prefix and the original key.

        Parameters
        ----------
        in_dict : dict
            Original dictionary.
        prefix : string
            string to prepend.

        Returns
        -------
        out : dict
            updated dictionary.
        """
        out = {}
        for key, val in in_dict.items():
            n_key = "{:s}.{:s}".format(prefix, key)
            out[n_key] = val
        return out

    def reset(self, **kwargs):
        """Resets to the base state."""
        super().reset(**kwargs)
        self.img = 255 * np.ones((*self.get_image_size(), 3), dtype=np.uint8)
        self.current_update_count = 0

    def step(self, user_input):
        """Perform one iteration of the game loop.

        Multiple physics updates can be made between rendering calls.

        Parameters
        ----------
        user_input : dict
            Each key is an integer representing the entity id that the action
            applies to. Each value is an action to take.

        Returns
        -------
        info : dict
            Extra infomation for debugging.
        """
        info = {}
        self.current_frame += 1

        self.score = 0
        reached_target = False
        for ii in range(self.params.physics.step_factor):
            self.current_update_count += 1
            self.entityManager.update()

            # clear events for entities
            for e in self.entityManager.get_entities():
                if e.has_component(gcomp.CEvents):
                    e.get_component(gcomp.CEvents).events = []

            action = self.s_input(user_input)
            self.s_movement(action)
            hit_tar = self.s_collision()
            reached_target = reached_target or hit_tar
            self.s_game_over()
            score, s_info = self.s_score()
            self.score += score
        self.score /= self.params.physics.step_factor

        info["reached_target"] = reached_target

        self.s_render()

        info.update(self.append_name_to_keys(s_info, "Reward"))

        return info

    def s_render(self):
        """Render a frame of the game."""
        self.img = grender2d.render(
            grender2d.get_drawable_entities(self.entityManager.get_entities()),
            self.window,
            self.clock,
            self.render_mode,
            self.render_fps,
        )

    def close(self):
        """Shutdown the rendering system."""
        grender2d.shutdown(self.window)
