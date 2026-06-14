"""
create_dataset.py
Procedurally generates 16x16 binary pixel art patterns for training.
Categories:
  - Symmetric: hearts, skulls, butterflies, mushrooms, anchors, crowns, aliens
  - Chess: rook, bishop, pawn, king
  - Completion: letters, numbers, arrows, houses, animals (fish, bird, cat)
  - Structural: mazes, checkerboards, frames, diamonds, hourglasses, crosses, stars
"""

import numpy as np
import json
import os
import random

GRID_SIZE = 16


def blank():
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)


def pixels_to_grid(pixel_list, ox=0, oy=0):
    """Helper: list of (row, col) tuples -> grid."""
    g = blank()
    for r, c in pixel_list:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def filled_circle(cx, cy, radius, grid=None, ox=0, oy=0):
    g = grid if grid is not None else blank()
    for r in range(16):
        for c in range(16):
            if (r - cy) ** 2 + (c - cx) ** 2 <= radius ** 2:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def filled_rect(r0, c0, r1, c1, grid=None, ox=0, oy=0):
    g = grid if grid is not None else blank()
    for r in range(r0, r1):
        for c in range(c0, c1):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def clear_rect(r0, c0, r1, c1, grid, ox=0, oy=0):
    for r in range(r0, r1):
        for c in range(c0, c1):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                grid[rr][cc] = 0
    return grid


# ======================================================
#  SYMMETRIC PATTERNS
# ======================================================

def make_heart(ox=0, oy=0):
    g = blank()
    rows = [
        [4, 5, 10, 11],
        [3, 4, 5, 6, 9, 10, 11, 12],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [4, 5, 6, 7, 8, 9, 10, 11],
        [5, 6, 7, 8, 9, 10],
        [6, 7, 8, 9],
        [7, 8],
    ]
    for r, cols in enumerate(rows):
        for c in cols:
            rr, cc = r + 3 + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_skull(ox=0, oy=0):
    g = filled_circle(7.5, 6.0, 5.5, None, ox, oy)
    for r in range(12, 16):
        for c in range(16):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 0
    clear_rect(4, 5, 7, 7, g, ox, oy)
    clear_rect(4, 9, 7, 11, g, ox, oy)
    filled_rect(12, 4, 14, 12, g, ox, oy)
    for c in [5, 7, 9]:
        rr, cc = 13 + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 0
    return g


def make_butterfly(ox=0, oy=0):
    g = blank()
    for r in range(2, 7):
        w = 4 - abs(r - 4)
        for c in range(3 - w, 4 + w):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in range(2, 7):
        w = 4 - abs(r - 4)
        for c in range(12 - w, 13 + w):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in range(8, 13):
        w = 3 - abs(r - 10)
        for c in range(4 - w, 5 + w):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in range(8, 13):
        w = 3 - abs(r - 10)
        for c in range(11 - w, 12 + w):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in range(2, 14):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for (r, c) in [(1, 6), (1, 9), (0, 5), (0, 10)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_mushroom(ox=0, oy=0):
    g = filled_circle(7.5, 5.0, 5.0, None, ox, oy)
    for r in range(9, 16):
        for c in range(16):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 0
    filled_rect(9, 6, 15, 10, g, ox, oy)
    return g


def make_anchor(ox=0, oy=0):
    g = blank()
    for r in range(1, 5):
        for c in range(6, 10):
            dist = np.sqrt((r - 2.5) ** 2 + (c - 7.5) ** 2)
            if 1.0 <= dist <= 2.2:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    for r in range(4, 13):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(7, 5, 8, 11, g, ox, oy)
    for c in range(2, 14):
        rr, cc = 13 + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    for (r, c) in [(12, 2), (12, 3), (12, 12), (12, 13), (11, 2), (11, 13)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_crown(ox=0, oy=0):
    g = blank()
    filled_rect(9, 2, 13, 14, g, ox, oy)
    for r in range(3, 9):
        for c in [3, 4]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
        for c in [11, 12]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in range(5, 9):
        depth = r - 5
        for c in range(5 - depth, 5 + depth + 1):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
        for c in range(9 - depth, 9 + depth + 1):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for c in [5, 8, 11]:
        rr, cc = 10 + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 0
    return g


def make_tree(ox=0, oy=0):
    g = blank()
    for r in range(1, 9):
        width = r
        for c in range(8 - width, 8 + width):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in range(9, 15):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


# ======================================================
#  PIXEL ART CHARACTERS
# ======================================================

def make_alien1(ox=0, oy=0):
    pixels = [
        (2, 5), (2, 10),
        (3, 6), (3, 9),
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
        (5, 4), (5, 5), (5, 7), (5, 8), (5, 10), (5, 11),
        (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12),
        (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 12),
        (8, 3), (8, 5), (8, 10), (8, 12),
        (9, 6), (9, 7), (9, 8), (9, 9),
        (10, 5), (10, 6), (10, 9), (10, 10),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_alien2(ox=0, oy=0):
    pixels = [
        (3, 7), (3, 8),
        (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 4), (6, 5), (6, 7), (6, 8), (6, 10), (6, 11),
        (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
        (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12),
        (9, 3), (9, 5), (9, 10), (9, 12),
        (10, 4), (10, 5), (10, 10), (10, 11),
        (11, 6), (11, 7), (11, 8), (11, 9),
        (12, 5), (12, 6), (12, 9), (12, 10),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_alien3(ox=0, oy=0):
    pixels = [
        (2, 7), (2, 8),
        (3, 6), (3, 7), (3, 8), (3, 9),
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
        (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
        (6, 4), (6, 6), (6, 7), (6, 8), (6, 9), (6, 11),
        (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11),
        (8, 5), (8, 6), (8, 9), (8, 10),
        (9, 4), (9, 5), (9, 10), (9, 11),
        (10, 3), (10, 4), (10, 11), (10, 12),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_pacman(ox=0, oy=0):
    g = filled_circle(7.5, 7.5, 6.5, None, ox, oy)
    for r in range(5, 11):
        for c in range(8, 15):
            dist_from_center = abs(r - 7.5)
            if c > 8 + (3 - dist_from_center):
                continue
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 0
    rr, cc = 5 + oy, 9 + ox
    if 0 <= rr < 16 and 0 <= cc < 16:
        g[rr][cc] = 0
    return g


def make_ghost(ox=0, oy=0):
    g = filled_circle(7.5, 6.0, 5.5, None, ox, oy)
    for r in range(11, 16):
        for c in range(16):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 0
    filled_rect(6, 2, 13, 14, g, ox, oy)
    for c in range(2, 14):
        if (c // 2) % 2 == 0:
            rr, cc = 13 + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
        rr, cc = 12 + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    for (r, c) in [(5, 5), (5, 6), (5, 9), (5, 10), (6, 5), (6, 6), (6, 9), (6, 10)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 0
    for (r, c) in [(5, 6), (5, 10), (6, 6), (6, 10)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


# ======================================================
#  CHESS PIECES
# ======================================================

def make_rook(ox=0, oy=0):
    pixels = [
        (2, 4), (2, 5), (2, 7), (2, 8), (2, 10), (2, 11),
        (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 6), (6, 7), (6, 8), (6, 9),
        (7, 6), (7, 7), (7, 8), (7, 9),
        (8, 6), (8, 7), (8, 8), (8, 9),
        (9, 6), (9, 7), (9, 8), (9, 9),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
        (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11),
        (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_bishop(ox=0, oy=0):
    pixels = [
        (1, 7), (1, 8),
        (2, 6), (2, 7), (2, 8), (2, 9),
        (3, 6), (3, 7), (3, 8), (3, 9),
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
        (5, 5), (5, 6), (5, 8), (5, 9), (5, 10),
        (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
        (7, 6), (7, 7), (7, 8), (7, 9),
        (8, 6), (8, 7), (8, 8), (8, 9),
        (9, 6), (9, 7), (9, 8), (9, 9),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
        (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11),
        (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_pawn(ox=0, oy=0):
    pixels = [
        (3, 7), (3, 8),
        (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 6), (5, 7), (5, 8), (5, 9),
        (6, 7), (6, 8),
        (7, 7), (7, 8),
        (8, 6), (8, 7), (8, 8), (8, 9),
        (9, 6), (9, 7), (9, 8), (9, 9),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
        (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11),
        (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_king(ox=0, oy=0):
    pixels = [
        (1, 7), (1, 8),
        (2, 6), (2, 7), (2, 8), (2, 9),
        (3, 7), (3, 8),
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
        (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
        (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10),
        (8, 6), (8, 7), (8, 8), (8, 9),
        (9, 6), (9, 7), (9, 8), (9, 9),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
        (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11),
        (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12),
    ]
    return pixels_to_grid(pixels, ox, oy)


# ======================================================
#  LETTERS AND NUMBERS
# ======================================================

def make_letter_A(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [5, 6, 9, 10]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(2, 6, 4, 10, g, ox, oy)
    filled_rect(7, 5, 9, 11, g, ox, oy)
    return g


def make_letter_T(ox=0, oy=0):
    g = blank()
    filled_rect(2, 3, 4, 13, g, ox, oy)
    for r in range(4, 14):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_letter_H(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [4, 5, 10, 11]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(7, 4, 9, 12, g, ox, oy)
    return g


def make_letter_E(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [4, 5]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(2, 4, 4, 12, g, ox, oy)
    filled_rect(7, 4, 9, 11, g, ox, oy)
    filled_rect(12, 4, 14, 12, g, ox, oy)
    return g


def make_letter_L(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [4, 5]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(12, 4, 14, 12, g, ox, oy)
    return g


def make_letter_O(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [4, 5, 10, 11]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(2, 4, 4, 12, g, ox, oy)
    filled_rect(12, 4, 14, 12, g, ox, oy)
    return g


def make_letter_X(ox=0, oy=0):
    g = blank()
    for i in range(2, 14):
        for d in [-1, 0, 1]:
            c1 = i + d
            c2 = 15 - i + d
            rr = i + oy
            if 0 <= rr < 16:
                if 0 <= c1 + ox < 16:
                    g[rr][c1 + ox] = 1
                if 0 <= c2 + ox < 16:
                    g[rr][c2 + ox] = 1
    return g


def make_letter_I(ox=0, oy=0):
    g = blank()
    filled_rect(2, 5, 4, 11, g, ox, oy)
    for r in range(4, 12):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(12, 5, 14, 11, g, ox, oy)
    return g


def make_letter_U(ox=0, oy=0):
    g = blank()
    for r in range(2, 12):
        for c in [4, 5, 10, 11]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(12, 4, 14, 12, g, ox, oy)
    return g


def make_letter_C(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [4, 5]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(2, 4, 4, 12, g, ox, oy)
    filled_rect(12, 4, 14, 12, g, ox, oy)
    return g


def make_number_1(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for (r, c) in [(2, 6), (3, 6)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    filled_rect(12, 5, 14, 11, g, ox, oy)
    return g


def make_number_0(ox=0, oy=0):
    return make_letter_O(ox, oy)


def make_number_7(ox=0, oy=0):
    g = blank()
    filled_rect(2, 4, 4, 12, g, ox, oy)
    for r in range(4, 14):
        c = 11 - (r - 4) * 6 // 10
        for dc in [0, 1]:
            rr, cc = r + oy, c + dc + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_number_4(ox=0, oy=0):
    g = blank()
    for r in range(2, 9):
        for c in [4, 5]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(8, 4, 10, 12, g, ox, oy)
    for r in range(2, 14):
        for c in [10, 11]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_number_3(ox=0, oy=0):
    g = blank()
    filled_rect(2, 4, 4, 12, g, ox, oy)
    filled_rect(7, 4, 9, 12, g, ox, oy)
    filled_rect(12, 4, 14, 12, g, ox, oy)
    for r in range(2, 14):
        for c in [10, 11]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_number_8(ox=0, oy=0):
    g = make_letter_O(ox, oy)
    filled_rect(7, 4, 9, 12, g, ox, oy)
    return g


# ======================================================
#  ARROWS (all four directions)
# ======================================================

def make_arrow_up(ox=0, oy=0):
    g = blank()
    for r in range(5, 14):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    head = [(2, 7), (2, 8), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10)]
    for r, c in head:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_arrow_down(ox=0, oy=0):
    g = blank()
    for r in range(2, 11):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    head = [(11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
            (12, 6), (12, 7), (12, 8), (12, 9), (13, 7), (13, 8)]
    for r, c in head:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_arrow_left(ox=0, oy=0):
    g = blank()
    for c in range(5, 14):
        for r in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    head = [(7, 2), (8, 2), (6, 3), (7, 3), (8, 3), (9, 3),
            (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4)]
    for r, c in head:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_arrow_right(ox=0, oy=0):
    g = blank()
    for c in range(2, 11):
        for r in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    head = [(7, 13), (8, 13), (6, 12), (7, 12), (8, 12), (9, 12),
            (5, 11), (6, 11), (7, 11), (8, 11), (9, 11), (10, 11)]
    for r, c in head:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


# ======================================================
#  BUILDINGS, OBJECTS, WEAPONS
# ======================================================

def make_house(ox=0, oy=0):
    g = blank()
    for r in range(2, 8):
        width = r - 1
        for c in range(8 - width, 8 + width):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(8, 3, 14, 13, g, ox, oy)
    clear_rect(10, 7, 14, 9, g, ox, oy)
    clear_rect(9, 4, 11, 6, g, ox, oy)
    clear_rect(9, 10, 11, 12, g, ox, oy)
    return g


def make_smiley(ox=0, oy=0):
    g = blank()
    cx, cy = 7.5, 7.5
    for r in range(16):
        for c in range(16):
            dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
            if 5.5 <= dist <= 7.0:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    for (r, c) in [(5, 5), (5, 10)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    for c in [4, 5, 6, 7, 8, 9, 10, 11]:
        rr, cc = 11 + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    for (r, c) in [(10, 4), (10, 11)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_sword(ox=0, oy=0):
    g = blank()
    for r in range(1, 9):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    rr, cc = 0 + oy, 7 + ox
    if 0 <= rr < 16 and 0 <= cc < 16:
        g[rr][cc] = 1
    filled_rect(9, 4, 11, 12, g, ox, oy)
    for r in range(11, 14):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    filled_rect(14, 6, 15, 10, g, ox, oy)
    return g


def make_shield(ox=0, oy=0):
    g = blank()
    filled_rect(2, 3, 4, 13, g, ox, oy)
    for r in range(4, 14):
        shrink = (r - 4) * 5 // 10
        filled_rect(r, 3 + shrink, r + 1, 13 - shrink, g, ox, oy)
    return g


def make_rocket(ox=0, oy=0):
    g = blank()
    for (r, c) in [(1, 7), (1, 8), (2, 6), (2, 7), (2, 8), (2, 9)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    filled_rect(3, 5, 11, 11, g, ox, oy)
    clear_rect(5, 7, 7, 9, g, ox, oy)
    for r in range(9, 13):
        for c in [3, 4, 11, 12]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for c in [6, 7, 8, 9]:
        for r in [11, 12]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for c in [7, 8]:
        for r in [13, 14]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_music_note(ox=0, oy=0):
    g = blank()
    for r in range(2, 12):
        rr, cc = r + oy, 9 + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    for (r, c) in [(2, 10), (2, 11), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10), (5, 11)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    filled_circle(7.0, 12.0, 2.2, g, ox, oy)
    return g


def make_lightning(ox=0, oy=0):
    pixels = [
        (1, 8), (1, 9), (1, 10),
        (2, 7), (2, 8), (2, 9),
        (3, 6), (3, 7), (3, 8),
        (4, 5), (4, 6), (4, 7),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
        (7, 7), (7, 8), (7, 9),
        (8, 7), (8, 8), (8, 9),
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 8), (11, 9), (11, 10),
        (12, 7), (12, 8), (12, 9),
        (13, 6), (13, 7), (13, 8),
        (14, 5), (14, 6), (14, 7),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_key(ox=0, oy=0):
    g = blank()
    # Ring
    for r in range(2, 7):
        for c in range(3, 8):
            dist = np.sqrt((r - 4.0) ** 2 + (c - 5.0) ** 2)
            if 1.5 <= dist <= 2.8:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    # Shaft
    for c in range(7, 14):
        for r in [4, 5]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    # Teeth
    for (r, c) in [(6, 11), (6, 12), (6, 13), (7, 13)]:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


# ======================================================
#  ANIMALS
# ======================================================

def make_fish(ox=0, oy=0):
    pixels = [
        (6, 2), (7, 2), (8, 2), (9, 2),
        (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
        (6, 4), (7, 4), (8, 4), (9, 4),
        (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5),
        (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6), (11, 6),
        (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7),
        (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (11, 8),
        (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9), (11, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10),
        (6, 11), (7, 11), (8, 11), (9, 11),
        (7, 12), (8, 12),
    ]
    g = pixels_to_grid(pixels, ox, oy)
    rr, cc = 6 + oy, 10 + ox
    if 0 <= rr < 16 and 0 <= cc < 16:
        g[rr][cc] = 0
    return g


def make_bird(ox=0, oy=0):
    pixels = [
        (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10),
        (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11),
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
        (6, 10), (6, 11), (7, 11), (7, 12),
        (7, 13), (7, 14),
        (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
        (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
        (3, 2), (3, 3), (3, 4), (3, 5),
        (8, 2), (8, 3), (9, 2),
        (10, 6), (10, 9),
        (11, 6), (11, 9),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_cat(ox=0, oy=0):
    pixels = [
        (2, 4), (2, 11),
        (3, 4), (3, 5), (3, 10), (3, 11),
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
        (6, 4), (6, 5), (6, 7), (6, 8), (6, 10), (6, 11),
        (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11),
        (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10),
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
        (12, 5), (12, 6), (12, 9), (12, 10),
        (13, 5), (13, 6), (13, 9), (13, 10),
        (11, 11), (10, 12), (9, 13), (8, 13),
    ]
    return pixels_to_grid(pixels, ox, oy)


def make_dog(ox=0, oy=0):
    pixels = [
        # Ears
        (2, 3), (2, 4), (2, 11), (2, 12),
        (3, 3), (3, 4), (3, 11), (3, 12),
        # Head
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
        (6, 4), (6, 5), (6, 7), (6, 8), (6, 10), (6, 11),  # eyes at 6,6 and 6,9
        (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11),
        # Snout
        (8, 6), (8, 7), (8, 8), (8, 9),
        # Body
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
        (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11),
        (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11),
        # Legs
        (12, 4), (12, 5), (12, 10), (12, 11),
        (13, 4), (13, 5), (13, 10), (13, 11),
        # Tail
        (9, 11), (8, 12), (7, 13),
    ]
    return pixels_to_grid(pixels, ox, oy)


# ======================================================
#  STRUCTURAL / ABSTRACT
# ======================================================

def make_diamond(ox=0, oy=0):
    g = blank()
    cx, cy = 7.5, 7.5
    for r in range(16):
        for c in range(16):
            if abs(r - cy) + abs(c - cx) <= 6.5:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_cross(ox=0, oy=0):
    g = blank()
    filled_rect(2, 6, 14, 10, g, ox, oy)
    filled_rect(6, 2, 10, 14, g, ox, oy)
    return g


def make_hourglass(ox=0, oy=0):
    g = blank()
    for r in range(16):
        half = abs(r - 7.5)
        w = int(half) + 1
        for c in range(8 - w, 8 + w):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_star(ox=0, oy=0):
    g = blank()
    pixels = [
        (1, 7), (1, 8),
        (2, 7), (2, 8),
        (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
        (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
        (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
        (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11),
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
        (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11),
        (11, 3), (11, 4), (11, 7), (11, 8), (11, 11), (11, 12),
        (12, 2), (12, 3), (12, 7), (12, 8), (12, 12), (12, 13),
        (13, 1), (13, 2), (13, 7), (13, 8), (13, 13), (13, 14),
    ]
    for r, c in pixels:
        rr, cc = r + oy, c + ox
        if 0 <= rr < 16 and 0 <= cc < 16:
            g[rr][cc] = 1
    return g


def make_circle(ox=0, oy=0, radius=None):
    radius = radius or random.uniform(4.0, 6.5)
    return filled_circle(7.5, 7.5, radius, None, ox, oy)


def make_rectangle(ox=0, oy=0, w=None, h=None):
    g = blank()
    w = w or random.randint(6, 12)
    h = h or random.randint(6, 12)
    r0 = (16 - h) // 2
    c0 = (16 - w) // 2
    filled_rect(r0, c0, r0 + h, c0 + w, g, ox, oy)
    return g


def make_checkerboard(ox=0, oy=0, size=None):
    g = blank()
    size = size or random.choice([2, 4])
    for r in range(16):
        for c in range(16):
            if ((r // size) + (c // size)) % 2 == 0:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_frame(ox=0, oy=0, thickness=None):
    g = blank()
    t = thickness or random.choice([2, 3])
    for r in range(16):
        for c in range(16):
            if r < t or r >= 16 - t or c < t or c >= 16 - t:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_frame_with_cross(ox=0, oy=0):
    g = make_frame(ox, oy, thickness=2)
    for r in range(4, 12):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for r in [7, 8]:
        for c in range(4, 12):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_frame_with_diamond(ox=0, oy=0):
    g = make_frame(ox, oy, thickness=2)
    cx, cy = 7.5, 7.5
    for r in range(3, 13):
        for c in range(3, 13):
            if abs(r - cy) + abs(c - cx) <= 3.5:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_maze1(ox=0, oy=0):
    g = blank()
    filled_rect(0, 0, 16, 2, g, ox, oy)
    filled_rect(0, 14, 16, 16, g, ox, oy)
    filled_rect(0, 0, 2, 16, g, ox, oy)
    filled_rect(14, 0, 16, 16, g, ox, oy)
    filled_rect(4, 4, 6, 10, g, ox, oy)
    filled_rect(8, 2, 10, 8, g, ox, oy)
    filled_rect(6, 10, 12, 12, g, ox, oy)
    return g


def make_maze2(ox=0, oy=0):
    g = blank()
    filled_rect(0, 0, 16, 2, g, ox, oy)
    filled_rect(0, 14, 16, 16, g, ox, oy)
    filled_rect(0, 0, 2, 16, g, ox, oy)
    filled_rect(14, 0, 16, 16, g, ox, oy)
    filled_rect(2, 4, 8, 6, g, ox, oy)
    filled_rect(4, 8, 6, 14, g, ox, oy)
    filled_rect(8, 6, 14, 8, g, ox, oy)
    filled_rect(10, 10, 12, 14, g, ox, oy)
    return g


def make_spiral(ox=0, oy=0):
    g = blank()
    filled_rect(2, 2, 4, 14, g, ox, oy)
    filled_rect(2, 2, 14, 4, g, ox, oy)
    filled_rect(12, 2, 14, 14, g, ox, oy)
    filled_rect(2, 12, 12, 14, g, ox, oy)
    filled_rect(6, 4, 8, 10, g, ox, oy)
    filled_rect(6, 4, 10, 6, g, ox, oy)
    filled_rect(8, 8, 10, 10, g, ox, oy)
    return g


def make_stripes_h(ox=0, oy=0):
    g = blank()
    for r in range(16):
        if (r // 2) % 2 == 0:
            for c in range(16):
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_stripes_v(ox=0, oy=0):
    g = blank()
    for r in range(16):
        for c in range(16):
            if (c // 2) % 2 == 0:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_diagonal_stripes(ox=0, oy=0):
    g = blank()
    for r in range(16):
        for c in range(16):
            if ((r + c) // 3) % 2 == 0:
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_triangle_up(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        width = r - 2
        for c in range(8 - width, 8 + width):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_triangle_down(ox=0, oy=0):
    g = blank()
    for r in range(2, 14):
        width = 13 - r
        for c in range(8 - width, 8 + width):
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_plus_sign(ox=0, oy=0):
    """Thin plus sign."""
    g = blank()
    for r in range(3, 13):
        for c in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    for c in range(3, 13):
        for r in [7, 8]:
            rr, cc = r + oy, c + ox
            if 0 <= rr < 16 and 0 <= cc < 16:
                g[rr][cc] = 1
    return g


def make_target(ox=0, oy=0):
    """Concentric rings target."""
    g = blank()
    cx, cy = 7.5, 7.5
    for r in range(16):
        for c in range(16):
            dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
            if dist <= 2.0 or (4.0 <= dist <= 5.5) or (6.5 <= dist <= 7.5):
                rr, cc = r + oy, c + ox
                if 0 <= rr < 16 and 0 <= cc < 16:
                    g[rr][cc] = 1
    return g


def make_yin_yang(ox=0, oy=0):
    """Simplified yin-yang."""
    g = blank()
    cx, cy = 7.5, 7.5
    for r in range(16):
        for c in range(16):
            dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
            if dist <= 7.0:
                # Left half filled
                if c <= 7:
                    rr, cc = r + oy, c + ox
                    if 0 <= rr < 16 and 0 <= cc < 16:
                        g[rr][cc] = 1
                # Top-right small circle
                dist_top = np.sqrt((r - 4.0) ** 2 + (c - 7.5) ** 2)
                if dist_top <= 2.5:
                    rr, cc = r + oy, c + ox
                    if 0 <= rr < 16 and 0 <= cc < 16:
                        g[rr][cc] = 1
                # Bottom-left clear
                dist_bot = np.sqrt((r - 11.0) ** 2 + (c - 7.5) ** 2)
                if dist_bot <= 2.5:
                    rr, cc = r + oy, c + ox
                    if 0 <= rr < 16 and 0 <= cc < 16:
                        g[rr][cc] = 0
    return g


# ======================================================
#  PATTERN REGISTRY
# ======================================================

PATTERN_GENERATORS = [
    # Symmetric
    ("heart", make_heart),
    ("skull", make_skull),
    ("butterfly", make_butterfly),
    ("mushroom", make_mushroom),
    ("anchor", make_anchor),
    ("crown", make_crown),
    ("tree", make_tree),
    # Pixel art characters
    ("alien1", make_alien1),
    ("alien2", make_alien2),
    ("alien3", make_alien3),
    ("pacman", make_pacman),
    ("ghost", make_ghost),
    # Chess
    ("rook", make_rook),
    ("bishop", make_bishop),
    ("pawn", make_pawn),
    ("king", make_king),
    # Letters
    ("letter_A", make_letter_A),
    ("letter_T", make_letter_T),
    ("letter_H", make_letter_H),
    ("letter_E", make_letter_E),
    ("letter_L", make_letter_L),
    ("letter_O", make_letter_O),
    ("letter_X", make_letter_X),
    ("letter_I", make_letter_I),
    ("letter_U", make_letter_U),
    ("letter_C", make_letter_C),
    # Numbers
    ("number_0", make_number_0),
    ("number_1", make_number_1),
    ("number_3", make_number_3),
    ("number_4", make_number_4),
    ("number_7", make_number_7),
    ("number_8", make_number_8),
    # Arrows
    ("arrow_up", make_arrow_up),
    ("arrow_down", make_arrow_down),
    ("arrow_left", make_arrow_left),
    ("arrow_right", make_arrow_right),
    # Objects
    ("house", make_house),
    ("smiley", make_smiley),
    ("sword", make_sword),
    ("shield", make_shield),
    ("rocket", make_rocket),
    ("music_note", make_music_note),
    ("lightning", make_lightning),
    ("key", make_key),
    # Animals
    ("fish", make_fish),
    ("bird", make_bird),
    ("cat", make_cat),
    ("dog", make_dog),
    # Structural / abstract
    ("diamond", make_diamond),
    ("cross", make_cross),
    ("hourglass", make_hourglass),
    ("star", make_star),
    ("circle", make_circle),
    ("rectangle", make_rectangle),
    ("checkerboard", make_checkerboard),
    ("frame", make_frame),
    ("frame_cross", make_frame_with_cross),
    ("frame_diamond", make_frame_with_diamond),
    ("maze1", make_maze1),
    ("maze2", make_maze2),
    ("spiral", make_spiral),
    ("stripes_h", make_stripes_h),
    ("stripes_v", make_stripes_v),
    ("diagonal_stripes", make_diagonal_stripes),
    ("triangle_up", make_triangle_up),
    ("triangle_down", make_triangle_down),
    ("plus_sign", make_plus_sign),
    ("target", make_target),
    ("yin_yang", make_yin_yang),
]


# ======================================================
#  DATASET GENERATION
# ======================================================

def add_noise(grid, noise_prob=0.02):
    noisy = grid.copy()
    for r in range(16):
        for c in range(16):
            if random.random() < noise_prob:
                noisy[r][c] = 1 - noisy[r][c]
    return noisy


def generate_dataset(num_per_pattern=6, noise_variants=2):
    dataset = []
    for name, gen_fn in PATTERN_GENERATORS:
        for i in range(num_per_pattern):
            ox = random.randint(-1, 1)
            oy = random.randint(-1, 1)
            grid = gen_fn(ox=ox, oy=oy)
            grid = np.clip(grid, 0, 1)

            dataset.append({
                "name": f"{name}_{i}",
                "label": name,
                "grid": grid.tolist()
            })

            for j in range(noise_variants):
                noisy = add_noise(grid, noise_prob=0.03)
                dataset.append({
                    "name": f"{name}_{i}_noisy_{j}",
                    "label": name,
                    "grid": noisy.tolist()
                })

    random.shuffle(dataset)
    return dataset


def mask_grid(grid, mask_fraction=None):
    if mask_fraction is None:
        mask_fraction = random.uniform(0.3, 0.75)

    grid = np.array(grid, dtype=np.float32)
    masked = grid.copy()
    mask = np.random.random((16, 16)) < mask_fraction
    masked[mask] = -1.0
    return masked, grid


def print_grid(grid):
    for row in grid:
        line = ""
        for val in row:
            if val == 1:
                line += "██"
            elif val == -1:
                line += "░░"
            else:
                line += "  "
        print(line)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    dataset = generate_dataset(num_per_pattern=6, noise_variants=2)
    print(f"Generated {len(dataset)} patterns")
    print(f"Pattern types: {len(PATTERN_GENERATORS)}")

    output_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f)
    print(f"Saved to {output_path}")

    from collections import Counter
    counts = Counter(d["label"] for d in dataset)
    print(f"\nExamples per pattern:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    # Preview all base patterns
    print("\n" + "=" * 40)
    print("ALL BASE PATTERNS:")
    print("=" * 40)
    for name, gen_fn in PATTERN_GENERATORS:
        print(f"\n--- {name} ---")
        g = gen_fn()
        print_grid(g)