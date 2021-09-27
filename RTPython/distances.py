#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:35:56 2021

@author: Simon Krekels

RTPython.distances - provides functions for handling periodic boundary
conditions in 2D
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def pbc_dist(x1, y1, x2, y2, L):
    '''
    Return the distance with PBC between coordinates (p1) = (x1, y1) and
    (p2) = (x2, y2). The shortest distance is that between (p1) and the
    closest mirror image of (p2).


    Parameters
    ----------
    x(y)1(2) : float
        coordinate sets 1 and 2

    L : Float
        Size of periodic box.

    Returns
    -------
    float
        Closest mirror image distance between (p1) and (p2).

    '''

    xdiff = np.abs(x1 - x2)
    if xdiff > (L / 2):
        # if |Δx| > L/2 there is a closer mirror image
        xdiff = L - xdiff

    ydiff = np.abs(y1 - y2)
    if ydiff > (L / 2):
        # if |Δy| > L/2 there is a closer mirror image
        ydiff = L - ydiff

    return np.sqrt(xdiff**2 + ydiff**2)


@nb.jit(nopython=True)
def pbc_metric(u, v, L):
    '''
    The same function as RTPython.distances.pbc_dist but accepting 2-vectors
    u and v instead of components.

    '''
    x1, y1 = u
    x2, y2 = v

    return pbc_dist(x1, y1, x2, y2, L)


@nb.jit(nopython=True)
def pbc_vec(x1, y1, x2, y2, L):
    '''
    Returns the vector from (p2) = (x2, y2) to the closest mirror image of
    (p1) = (x1, y1).

    Parameters
    ----------
    x(y)1(2) : float
        coordinate sets 1 and 2

    L : Float
        Size of periodic box.

    Returns
    -------
    ndarray
        2-vector pointing from (p2) to p(1) using closest mirror image

    '''
    xdiff = x1 - x2
    ydiff = y1 - y2

    # a stores information on the nearest mirror image
    a = np.zeros(2)

    if xdiff > L/2:
        # nearest mirror image is to the left
        a += np.array((1, 0))

    elif xdiff < -L/2:
        # nearest mirror image is to the right
        a += np.array((-1, 0))

    if ydiff > L/2:
        # nearest mirror image is to the bottom
        a += np.array((0, 1))

    elif ydiff < -L/2:
        # nearest mirror image is to the top
        a += np.array((0, -1))

    return np.array((xdiff, ydiff)) - L*a
