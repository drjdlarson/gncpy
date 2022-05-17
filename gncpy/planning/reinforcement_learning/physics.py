import pygame
import numpy as np


def check_collision2d(bb1, bb2):
    return pygame.Rect.colliderect(bb1, bb2)


def _get_overlap2d(pt1, pt2, bb1, bb2):
    delta = (abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1]))
    ox = bb1.width / 2 + bb2.width / 2 - delta[0]
    oy = bb1.height / 2 + bb2.height / 2 - delta[1]
    return ox, oy


def resolve_collision2d(bb1, bb2, trans1, trans2):
    ox, oy = _get_overlap2d((bb1.centerx, bb1.centery), (bb2.centerx, bb2.centery),
                            bb1, bb2)
    opx, opy = _get_overlap2d(trans1.last_pos.ravel(), trans2.last_pos.ravel(),
                              bb1, bb2)

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
