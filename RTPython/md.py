#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:22:10 2021

@author: Simon Krekels
"""
import numpy as np
import numba as nb
import scipy.spatial as spatial
# from . import forces
from . import distances


def interaction_periodic(r, force, cutoff, boxsize, eps, a):
    """
    Calculates the interactions between particles in a 2D periodic box.

    Parameters
    ----------
    r : ndarray
        Particle positions in 2D. Must be an N x 2 array.
    force : function
        A force function which takes (r, a, eps) as arguments. See module
        'forces'.
    cutoff : float
        Cutoff distance for the interaction.
    boxsize : float
        Boxsize to be passed to cKDTree. Must be a float (e.g. L for an L x L
        box).
    eps : float
        Modulates the strength of the interaction potential.
    a : float
        Particle interaction range

    Returns
    -------
        Finds interacting pairs and passes them to calc_int, which returns the
        interaction forces.

    """

    # Consrtuct a k-d tree to quickly find interacting (nearby) pairs
    kd = spatial.cKDTree(r, boxsize=boxsize)

    pairs = kd.query_pairs(cutoff, output_type='ndarray')

    return calc_int(r, pairs, force, cutoff, eps, boxsize, a)


@nb.jit(nopython=True)
def calc_int(r, pairs, force, cutoff, eps, boxsize, a):
    """
    Calculates forces between particle pairs.

    Parameters
    ----------
    r : ndarray
        Particle positions in 2D. Must be an N x 2 array.
    pairs : ndarray
        Array of pairs of indices of interacting particles.
    force : function
        A force function which takes (r, a, eps) as arguments. See module
        'forces'.
     cutoff : float
        Cutoff distance for the interaction.
   eps : float
        Modulates the strength of the interaction potential.
    boxsize : float
        Boxsize to be passed to cKDTree. Must be a float (e.g. L for an L x L
        box).
     a : float
        Particle interaction range

    Returns
    -------
    v: ndarray
        Force vectors for each particle


    """

    # initialize force vector as zeros
    v = np.zeros((len(r), 2))

    # loop over interacting pairs
    for k in range(len(pairs)):
        i, j = pairs[k]

        # calculate distance between particles
        r_dist = distances.pbc_dist(r[i][0],
                                    r[i][1],
                                    r[j][0],
                                    r[j][1], boxsize)

        # Calculate unit vector from (j) to (i)
        rr = distances.pbc_vec(r[i][0],
                               r[i][1],
                               r[j][0],
                               r[j][1], boxsize) / r_dist

        # Calculate magnitude of force between (i) and (j)
        F = force(r_dist, cutoff, eps)

        # Add force vectors to the resulting force
        v[i] += F * rr
        v[j] += -F * rr

    return v
