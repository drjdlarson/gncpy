"""Defines base game engine classes.

These define common functions, properties, and interfaces for all games.
"""
from abc import ABC, abstractmethod
import os
import pathlib
import numpy as np
from ruamel.yaml import YAML
import pygame
from sys import exit

import gncpy.game_engine.rendering2d as grender2d
from gncpy.game_engine.physics2d import Physics2dParams
import gncpy.game_engine.components as gcomp
from gncpy.game_engine.entities import EntityManager


yaml = YAML()
library_config_dir = os.path.join(
    pathlib.Path(__file__).parent.parent.resolve(), "games", "configs"
)


class WindowParams:
    def __init__(self):
        self.width = 0
        self.height = 0


class Shape2dParams:
    def __init__(self):
        self.type = ""
        self.width = 0
        self.height = 0
        self.color = ()


class Collision2dParams:
    def __init__(self):
        self.width = 0
        self.height = 0


class BaseParams:
    def __init__(self):
        self.window = WindowParams()


class Base2dParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.physics = Physics2dParams()
        self.start_time = 0
        self.max_time = np.inf


class BaseGame(ABC):
    """Base class for defining games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities. It must be subclassed and defines the expected interface of games.

    Attributes
    ----------
    """

    def __init__(
        self,
        config_file,
        render_mode,
        render_fps=None,
        use_library_config=False,
        seed=0,
    ):
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
        self.config = None
        self.params = None
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.seed_val = seed

    @abstractmethod
    def setup(self):
        pass

    def register_params(self, yaml):
        yaml.register_class(WindowParams)
        yaml.register_class(BaseParams)

    def validate_config_file(self, config_file, use_library):
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
                            val = list([val, ])
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

    def reset(self, seed=None):
        """Resets to the base state."""
        self.entityManager = EntityManager()
        self.current_frame = 0
        self.game_over = False
        self.score = 0

        if seed is None:
            if self.seed_val is None:
                self.rng = np.random.default_rng()
            else:
                self.rng = np.random.default_rng(self.seed_val)
        else:
            self.rng = np.random.default_rng(seed)
            self.seed_val = seed

    @abstractmethod
    def s_movement(self, action):
        """Move entities according to their dynamics.

        Parameters
        ----------
        action : numpy array, int, bool, etc.
            action to take in the game.
        """
        raise NotImplementedError()

    @abstractmethod
    def s_collision(self):
        """Check for collisions between entities."""
        raise NotImplementedError()

    @abstractmethod
    def s_game_over(self):
        """Checks for game over conditions."""
        raise NotImplementedError()

    @abstractmethod
    def s_score(self):
        """Calculates the score.

        Returns
        -------
        info : dict
            Extra info for debugging.
        """
        raise NotImplementedError()

    @abstractmethod
    def s_input(self, *args):
        """Turns user inputs into game actions."""
        raise NotImplementedError()

    @abstractmethod
    def step(self, *args):
        """Abstract method defining what to do each frame.

        This must be implemented by the child class.

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
        return self.params.physics.update_dt * self.current_update_count + self.params.start_time

    def setup(self):
        global yaml
        self.register_params(yaml)

        self.parse_config_file()

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

        Parameters
        ----------

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
        grender2d.shutdown(self.window)
