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


def colloid_interaction_periodic(rtp, force, cutoff, boxsize, eps, a):
    """
    Calculates the interactions between particles in a 2D periodic box.

    Parameters
    ----------
    rtp : dict
        Dictionary specifying particles; see RTPython.initialize
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
    r = rtp['r']

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


def colloid_probe_interaction(rtp, probes, force, cutoff, boxsize, eps, a):
    """
    Extracts relevant data from dicts and passes to calc func.
    """
    return calc_colloid_probe_int(rtp['r'],
                                  probes['r'],
                                  force, cutoff, boxsize, eps, a)


@nb.jit(nopython=True)
def calc_colloid_probe_int(r, rp, force, cutoff, boxsize, eps, a):
    """
    Calculates the interaction between probes in the colloid-bath. Uses a boun-
    ding box to exclude colliods from interaction.

    Parameters
    ----------
    rtp : dict
        DESCRIPTION.
    probes : TYPE
        DESCRIPTION.
    force : TYPE
        DESCRIPTION.
    cutoff : TYPE
        DESCRIPTION.
    boxsize : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    fp : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    """

    # determine which particles are 'close' to fixed probes. This is
    # useful since probes don't move and will often be clustered close together

    # determine 'box' containing all probes (numba doesn't support 'axis' args)
    max_xy = np.array((np.max(rp[:, 0]), np.max(rp[:, 1]))) + cutoff
    min_xy = np.array((np.min(rp[:, 0]), np.min(rp[:, 1]))) - cutoff

    # determine where coordinates are inside box
    temp = np.logical_and(r > min_xy, r < max_xy)
    # determine where both x and y are inside box
    close = np.where(np.logical_and(temp[:, 0], temp[:, 1]))

    # n° of pairs unknown -> use flexible python list
    pairs = []
    dists = []
    for i in range(len(rp)):
        # only loop over RTPs in box
        for j in close[0]:
            dist = distances.pbc_metric(rp[i], r[j], boxsize)
            if dist < cutoff:
                # store pair, dist if overlap
                pairs.append((i, j))
                dists.append(dist)

    # handle overlapping pairs
    fp = np.zeros(np.shape(rp))
    v = np.zeros(np.shape(r))
    for k in range(len(pairs)):
        i, j = pairs[k]
        r_dist = dists[k]
        r_vec = distances.pbc_vec(rp[i][0],
                                  rp[i][1],
                                  r[j][0],
                                  r[j][1], boxsize)/r_dist
        F = force(r_dist, a, eps)
        fp[i] += F * r_vec
        v[j] += -F * r_vec

    return fp, v
