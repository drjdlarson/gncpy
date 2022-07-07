"""Defines physics engine for 2d games."""
import os
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa


class Physics2dParams:
    def __init__(self):
        self.dt = 0
        self.step_factor = 1
        self.min_pos = np.array([])
        self.dist_width = 0
        self.dist_height = 0

    @property
    def update_dt(self):
        return self.dt / self.step_factor


def check_collision2d(bb1, bb2):
    return pygame.Rect.colliderect(bb1, bb2)


def _get_overlap2d(pt1, pt2, bb1, bb2):
    delta = (abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1]))
    ox = bb1.width / 2 + bb2.width / 2 - delta[0]
    oy = bb1.height / 2 + bb2.height / 2 - delta[1]
    return ox, oy


def resolve_collision2d(bb1, bb2, trans1, trans2):
    ox, oy = _get_overlap2d(
        (bb1.centerx, bb1.centery), (bb2.centerx, bb2.centery), bb1, bb2
    )
    opx, opy = _get_overlap2d(
        trans1.last_pos.ravel(), trans2.last_pos.ravel(), bb1, bb2
    )

    if opy > 0:
        trans1.vel[0] = 0
        if trans1.last_pos[0] < trans1.pos[0]:
            bb1.centerx -= ox
        else:
            bb1.centerx += ox
    elif opx > 0:
        trans1.vel[1] = 0
        if trans1.last_pos[1] < trans1.pos[1]:
            bb1.centery -= oy
        else:
            bb1.centery += oy

    trans1.pos[0] = bb1.centerx
    trans1.pos[1] = bb1.centery


def clamp_window_bounds2d(bb, trans, width, height):
    out_side = False
    if bb.left < 0:
        bb.left = 0
        trans.vel[0] = 0
        out_side = True
    elif bb.right > width:
        bb.right = width
        trans.vel[0] = 0
        out_side = True

    out_top = False
    if bb.top < 0:
        bb.top = 0
        trans.vel[1] = 0
        out_top = True
    elif bb.bottom > height:
        bb.bottom = height
        trans.vel[1] = 0
        out_top = True

    trans.pos[0] = bb.centerx
    trans.pos[1] = bb.centery

    return out_side, out_top


def pixels_to_dist(pt, dist_per_pix, min_pos=None):
    """Convert pixel units to real units.

    Parameters
    ----------

    Returns
    -------
    numpy array, float
        converted point
    """
    if min_pos is not None:
        if isinstance(pt, np.ndarray):
            res = pt.ravel() * dist_per_pix.ravel() + min_pos.ravel()
        else:
            res = pt * dist_per_pix + min_pos
    else:
        if isinstance(pt, np.ndarray):
            res = pt.ravel() * dist_per_pix.ravel()
        else:
            res = pt * dist_per_pix

    if isinstance(res, np.ndarray):
        if res.size > 1:
            return res.reshape(pt.shape)
        else:
            return res.item()
    else:
        return res


def dist_to_pixels(pt, dist_per_pix, min_pos=None):
    """Convert real units to pixel units.

    Parameters
    ----------

    Returns
    -------
    numpy array, float
        converted point
    """
    if min_pos is not None:
        if isinstance(pt, np.ndarray):
            res = (pt.ravel() - min_pos.ravel()) / dist_per_pix.ravel()
        else:
            res = (pt - min_pos) / dist_per_pix
    else:
        if isinstance(pt, np.ndarray):
            res = pt.ravel() / dist_per_pix.ravel()
        else:
            res = pt / dist_per_pix

    if isinstance(res, np.ndarray):
        if res.size > 1:
            return res.reshape(pt.shape).astype(int)
        else:
            return res.astype(int).item()
    else:
        return int(res)
