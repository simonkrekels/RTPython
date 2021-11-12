#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:22:10 2021

@author: Simon Krekels
"""
import numpy as np
import numba as nb
import scipy.spatial as spatial
from . import forces
from . import distances


def colloid_interaction_periodic(r, force, cutoff, eps, boxsize, a):
    """
    Calculates the interactions between particles in a 2D periodic box.

    Parameters
    ----------
    r : ndarray
        N×2 array containing particle coordinates
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

    return v,


def colloid_probe_interaction_bb(rtp, probes, force, cutoff, eps, boxsize, a):
    """
    Extracts relevant data from dicts and passes to calc func.
    """
    return calc_colloid_probe_int_bb(rtp['r'],
                                     probes['r'],
                                     force, cutoff, eps, boxsize, a)


@nb.jit(nopython=True)
def calc_colloid_probe_int_bb(r, rp, force, cutoff, eps, boxsize, a):
    """
    Calculates the interaction between probes in the colloid-bath. Uses a boun-
    ding box to exclude colliods from interaction.

    The bounding box does not perform well with periodic boundary conditions,
    and in situations where there are many probes, and thus not many colloids
    to be excluded.

    Parameters
    ----------
    rtp : dict
        dict containing info on colloids; see RTPython.initialize.
    probes : dict
        dict containing info on probes; see RTPython.initialize.
    force : func
        function which evaluates the pair potential force between probes and
        colloids.
    cutoff : float
        cutoff radius for the probe-colloid interaction.
    boxsize : float/ndarray
        boxsize to use when evaluating distances using PBCs.
    eps : float
        interaction strength to pass to 'force'.
    a : float
        interaction radius to pass to 'force'.

    Returns
    -------
    fp : ndarray
        forces on the probes.
    v : ndarray
        forces on the colloids.

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

    return v, fp


def colloid_probe_interaction(rtp, probes, force, cutoff, eps, boxsize, a):
    """
    Extracts relevant data from dicts and passes to calc func.
    """
    return calc_colloid_probe_int_bb(rtp['r'],
                                     probes['r'],
                                     force, cutoff, eps, boxsize, a)


@nb.jit(nopython=True)
def calc_colloid_probe_int(r, rp, force, cutoff, eps, boxsize, a):
    """
    Calculates the interaction between probes in the colloid-bath.

    Parameters
    ----------
    r : ndarray
        Nx2 array containing colloid coordinates.
    rp : ndarray
        Npx2 array containing probe coordinates.
    force : func
        function which evaluates the pair potential force between probes and
        colloids.
    cutoff : float
        cutoff radius for the probe-colloid interaction.
    boxsize : float/ndarray
        boxsize to use when evaluating distances using PBCs.
    eps : float
        interaction strength to pass to 'force'.
    a : float
        interaction radius to pass to 'force'.

    Returns
    -------
    fp : ndarray
        forces on the probes.
    v : ndarray
        forces on the colloids.

    """

    # n° of pairs unknown -> use flexible python list
    pairs = []
    dists = []
    N = len(r)
    for i in range(len(rp)):
        for j in range(N):
            d = distances.pbc_metric(r[j], rp[i], boxsize)
            if (d < cutoff):
                pairs.append((i, j))
                dists.append(d)

    # handle overlapping pairs
    fp = np.zeros(rp.shape)
    v = np.zeros(r.shape)
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

    return v, fp


@nb.jit(nopython=True)
def colloid_probe_int_1D(x, xp, force, cutoff, eps, boxsize, a):
    """

    """
    pairs = []
    dists = []
    for i in range(len(xp)):
        for j in range(len(x)):
            dist = distances.pbc_vec_1D(xp[i], x[j], boxsize)
            if (np.abs(dist) < cutoff):
                pairs.append((i, j))
                dists.append(dist)

    fp = np.zeros(xp.shape)
    f = np.zeros(x.shape)

    for k in range(len(pairs)):
        l, m = pairs[k]
        dist = dists[k]
        vec = np.sign(dist)
        F = force(np.abs(dist), a, eps)
        fp[l] += F * vec
        f[m] += -F * vec

    return f, fp


@nb.jit(nopython=True)
def hwall_int(r, sigma, eps, boxsize):
    """
    Repulsive WCA-type interaction with top and bottom of box.

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.
    boxsize : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    f = np.zeros(r.shape)
    # below
    low = np.where(r[:, 1] < 2**(1/6)*sigma)[0]
    for i in low:
        f[i, 1] += forces.wca(r[i, 1], sigma, eps)
    # above
    high = np.where(r[:, 1] > boxsize[1] - 2**(1/6)*sigma)[0]
    for i in high:
        f[i, 1] -= forces.wca(boxsize[1]-r[i, 1], sigma, eps)
    return f
