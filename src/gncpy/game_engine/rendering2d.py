"""Handles low level calls to 2D graphics libraries."""
import os
import numpy as np
from sys import exit
from warnings import warn

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa

import gncpy.game_engine.components as gcomp  # noqa


class Shape2dParams:
    """Parameters of a 2d shape object.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    type : string
        Pygame object type to create, see :class:`.components.CShape` for options.
    width : int
        Width in real units of the shape.
    height : int
        Height in real units of the shape.
    color : tuple
        RGB triplet of the shape, in range [0, 255].
    file : string
        Full file path to image if needed.
    """

    def __init__(self):
        super().__init__()
        self.type = ""
        self.width = 0
        self.height = 0
        self.color = ()
        self.file = ""


def init_rendering_system():
    """Initialize the rendering system.

    Returns
    -------
    pygame clock
        pygame clock instance
    """
    pygame.init()
    clock = pygame.time.Clock()

    return clock


def init_window(render_mode, width, height):
    """Initialize the main window.

    Parameters
    ----------
    render_mode : string
        Render mode to use, if :code:`'human'` is given then the window will be
        shown. All other values are ignored.
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.

    Returns
    -------
    window : pygame window
        Main window object for drawing.
    """
    extra = {}
    if render_mode != "human":
        extra["flags"] = pygame.HIDDEN

    window = pygame.display.set_mode((int(width), int(height)), **extra)
    return window


def get_drawable_entities(entities):
    """Get all the drawable entities.

    Parameters
    ----------
    entities : list
        List of all entities from the :class:`.entities.EntityManager` class.

    Returns
    -------
    list
        list of all entities that can be drawn.
    """

    def _can_draw(_e):
        return (
            _e.active
            and _e.has_component(gcomp.CShape)
            and _e.has_component(gcomp.CTransform)
        )

    drawable = list(filter(_can_draw, entities))
    drawable.sort(key=lambda _e: _e.get_component(gcomp.CShape).zorder)

    return drawable


def render(drawable, window, clock, mode, fps):
    """Draw all entities to the window.

    Parameters
    ----------
    drawable : list
        List of drawable entities.
    window : pygame window
        Window to draw to.
    clock : pygame clock
        main clock for the game.
    mode : string
        Rendering mode, if :code:`'human'` then the screen is updated and events
        are checked for the close button.
    fps : int
        Frame rate to render at.

    Returns
    -------
    numpy array
        pixel values of the window in HxWx3 order.
    """
    window.fill((255, 255, 255))
    offset = window.get_size()[1]
    for e in drawable:
        e_shape = e.get_component(gcomp.CShape)
        e_trans = e.get_component(gcomp.CTransform)

        if np.any(np.isnan(e_trans.pos)) or np.any(np.isinf(e_trans.pos)):
            continue

        # flip in the vertical direction
        c_pos = (e_trans.pos[0].item(), offset - e_trans.pos[1].item())
        # e_shape.shape.centerx = e_trans.pos[0].item()
        # # flip in the vertical direction
        # e_shape.shape.centery = offset - e_trans.pos[1].item()

        if e_shape.type == "rect":
            e_shape.shape.centerx = c_pos[0]
            e_shape.shape.centery = c_pos[1]
            pygame.draw.rect(window, e_shape.color, e_shape.shape)

        elif e_shape.type == "sprite":
            window.blit(e_shape.shape, e_shape.shape.get_rect(center=c_pos))

        else:
            warn("No rendering method for this shape")

    if mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shutdown(window)
                exit()
        clock.tick(fps)
        pygame.display.update()

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(window), dtype=np.uint8), axes=(1, 0, 2)
    )


def shutdown(window):
    """Nicely close pygame if window was initialized.

    Parameters
    ----------
    window : pygame window
        Main window object.
    """
    if window is not None:
        pygame.quit()
