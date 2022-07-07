"""Handles low level calls to 2D graphics libraries."""
import os
import numpy as np
from sys import exit
from warnings import warn

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa

import gncpy.game_engine.components as gcomp  # noqa


def init_rendering_system():
    pygame.init()
    clock = pygame.time.Clock()

    return clock


def init_window(render_mode, width, height):
    extra = {}
    if render_mode != "human":
        extra["flags"] = pygame.HIDDEN

    window = pygame.display.set_mode((int(width), int(height)), **extra)
    return window



def get_drawable_entities(entities):
    def _can_draw(_e):
        return _e.has_component(gcomp.CShape) and _e.has_component(gcomp.CTransform)

    drawable = list(filter(_can_draw, entities))
    drawable.sort(key=lambda _e: _e.get_component(gcomp.CShape).zorder)

    return drawable


def render(drawable, window, clock, mode, fps):
    window.fill((255, 255, 255))
    offset = window.get_size()[1]
    for e in drawable:
        e_shape = e.get_component(gcomp.CShape)
        e_trans = e.get_component(gcomp.CTransform)

        if np.any(np.isnan(e_trans.pos)) or np.any(np.isinf(e_trans.pos)):
            continue

        e_shape.shape.centerx = e_trans.pos[0].item()
        # flip in the vertical direction
        e_shape.shape.centery = offset - e_trans.pos[1].item()

        if isinstance(e_shape.shape, pygame.Rect):
            pygame.draw.rect(window, e_shape.color, e_shape.shape)
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
    if window is not None:
        pygame.quit()
