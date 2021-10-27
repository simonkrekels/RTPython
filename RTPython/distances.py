#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 27 15:35:56 2021

@author: Simon Krekels

RTPython.distances - provides functions for handling periodic boundary
conditions in 2D
'''

import numba as nb
import numpy as np

# 'box_shape' takes values of either 'rect' or 'square', and determines the
# shape of the box. Also for 'rect', the 'L' to be passed to pbc functions
# should be an iterable (L = (Lx, Ly)).
box_shape = 'square'


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
    if (box_shape == 'square'):
        xdiff = np.abs(x1 - x2)
        if xdiff > (L / 2):
            # if |Δx| > L/2 there is a closer mirror image
            xdiff = L - xdiff

        ydiff = np.abs(y1 - y2)
        if ydiff > (L / 2):
            # if |Δy| > L/2 there is a closer mirror image
            ydiff = L - ydiff

        return np.sqrt(xdiff**2 + ydiff**2)

    elif (box_shape == 'rect'):
        Lx, Ly = L
        xdiff = np.abs(x1 - x2)
        if xdiff > (Lx / 2):
            # if |Δx| > L/2 there is a closer mirror image
            xdiff = Lx - xdiff

        ydiff = np.abs(y1 - y2)
        if ydiff > (Ly / 2):
            # if |Δy| > L/2 there is a closer mirror image
            ydiff = Ly - ydiff

        return np.sqrt(xdiff**2 + ydiff**2)

    else:
        raise ValueError("RTPython.distances.box_shape must be either \
                         'square' or 'rect'.")


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
    if (box_shape == 'square'):
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

    if (box_shape == 'rect'):

        Lx, Ly = L

        xdiff = x1 - x2
        ydiff = y1 - y2

        # a stores information on the nearest mirror image
        a = np.zeros(2)

        if xdiff > Lx/2:
            # nearest mirror image is to the left
            a += np.array((1, 0))

        elif xdiff < -Lx/2:
            # nearest mirror image is to the right
            a += np.array((-1, 0))

        if ydiff > Ly/2:
            # nearest mirror image is to the bottom
            a += np.array((0, 1))

        elif ydiff < -Ly/2:
            # nearest mirror image is to the top
            a += np.array((0, -1))

        return np.array((xdiff, ydiff)) - L*a

    else:
        raise ValueError("RTPython.distances.box_shape must be either \
                         'square' or 'rect'.")


@nb.jit(nopython=True)
def euclid_dist(x1, y1, x2, y2):
    '''
    Euclidean distance function

    Parameters
    ----------
    x(y)1(2) : float
        coordinate sets 1 and 2

    Returns
    -------
    float
        2D Euclidean distance: √(dx² + dy²).

    '''
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


@nb.jit(nopython=True)
def euclid_metric(u, v):
    '''
    The same function as RTPython.distances.euclid_dist but accepting 2-vectors
    u and v instead of components.

    '''

    x1, y1 = u
    x2, y2 = v

    return euclid_dist(x1, y1, x2, y2)


@nb.jit(nopython=True)
def euclid_vec(x1, y1, x2, y2):
    '''
    Return the vector from (p2) to (p1)

    Parameters
    ----------
    x(y)1(2) : float
        coordinate sets 1 and 2

    Returns
    -------
    ndarray
        2D vector from (p2) to (p1).

    '''
    return np.array((x1 - x2, y1 - y2))


@nb.jit(nopython=True)
def pbc_dist_1D(x1, x2, L):
    """
    Calculate distance between x1, x2 on a periodic 1D line

    Parameters
    ----------
    x1(2) : float
        positions 1 and 2 on the real line.
    L : float
        Length of periodic line segment.

    Returns
    -------
    dist : float
        shortest distance between x1, x2 on the periodic line.

    """

    dist = np.abs(x1 - x2)

    if (dist > L/2):
        return L - dist
    else:
        return dist


@nb.jit(nopython=True)
def pbc_vec_1D(x1, x2, L):
    """
    Calculate *signed* distance between x1, x2 on a periodic 1D line, looking
    from x2 to x1

    Parameters
    ----------
    x1(2) : float
        positions 1 and 2 on the real line.
    L : float
        Length of periodic line segment.

    Returns
    -------
    dist : float
        shortest distance between x1, x2 on the periodic line. The sign gives
        distance looking from x2 to x1

    """

    dist = x1 - x2

    if (dist > L/2):
        return dist - L
    elif (dist < -L/2):
        return dist + L
    else:
        return dist
