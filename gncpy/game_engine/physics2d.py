"""Defines physics engine for 2d games."""
import os
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa


class Physics2dParams:
    """Parameters for the 2d physics system to be parsed by the config parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    dt : float
        Main dt of the game, this is the rendering dt and may be the same as
        the update dt.
    step_factor : int
        Factor to divide the base dt by to get the update dt. Must be positive.
        Allows multiple incremental physics steps inbetween the main rendering
        calls. Helps ensure objects don't pass through each other from moving too
        far in a given frame.
    min_pos : numpy array
        Minimium position in real units in the order x, y.
    dist_width : float
        Distance in real units for the width of the game world.
    dist_height : float
        Distance in real units for the height of the game world.
    """

    def __init__(self):
        super().__init__()
        self.dt = 0
        self.step_factor = 1
        self.min_pos = np.array([])
        self.dist_width = 0.0
        self.dist_height = 0.0

    @property
    def update_dt(self):
        """Timestep for physics updates."""
        return self.dt / self.step_factor


class Collision2dParams:
    """Parameters of an axis aligned bounding box for 2d objects.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    width : int
        Width in pixels of the bounding box.
    height : int
        Height in pixels of the bounding box.
    """

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0


def check_collision2d(bb1, bb2):
    """Check for a collision between 2 bounding boxes.

    Parameters
    ----------
    bb1 : pygame rect
        First axis aligned bounding box.
    bb2 : pygame rect
        Second axis aligned bounding box.

    Returns
    -------
    bool
        Flag indicating if there was a collision.
    """
    return pygame.Rect.colliderect(bb1, bb2)


def _get_overlap2d(pt1, pt2, bb1, bb2):
    """Find the overlap between 2 bounding boxes with some previous position."""
    delta = (abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1]))
    ox = bb1.width / 2 + bb2.width / 2 - delta[0]
    oy = bb1.height / 2 + bb2.height / 2 - delta[1]
    return ox, oy


def resolve_collision2d(bb1, bb2, trans1, trans2):
    """Resolve a collision between 2 bounding boxes by moving one.

    This assumes that the two bounding boxes are colliding. This should first be
    checked.

    Parameters
    ----------
    bb1 : pygame rect
        Bounding box to move to resolve collision.
    bb2 : pygame rect
        Colliding bounding box.
    trans1 : :class:`.components.CTransform`
        Transform component associated with the first bounding box.
    trans2 : :class:`.components.CTransform`
        Transform component associated with the second bounding box.
    """
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
    """Checks for the bounding box leaving the window and halts it.

    Parameters
    ----------
    bb : pygame rect
        Axis aligned bounding box.
    trans : :class:`.components.CTransform`
        Transform component associated with the bounding box.
    width : int
        Width of the window in pixels.
    height : int
        Height of the window in pixels.

    Returns
    -------
    out_side : bool
        Flag for if there was a collision with the side.
    out_top : bool
        Flag for if there was a collision with the top or bottom.
    """
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
    pt : numpy array, float
        Point to convert to real distance.
    dist_per_pix : numpy array, float
        real distance per pixel.
    min_pos : numpy array or float, optional
        Minimum position to use for translation. The default is None meaning no
        translation is applied (i.e. velocity transform).

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
    pt : numpy array, float
        Point to convert to pixels.
    dist_per_pix : numpy array, float
        real distance per pixel.
    min_pos : numpy array or float, optional
        Minimum position to use for translation (real units). The default is
        None meaning no translation is applied (i.e. velocity transform).

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
