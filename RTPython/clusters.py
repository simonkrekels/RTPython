#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:35:52 2021

@author: Simon Krekels

Analysis of clusters in simulation results
"""

import numpy as np
import numba as nb
from scipy.spatial import cKDTree


def get_clusters(r, sigma, L):
    """
    Get all clusters of a given configuration

    Parameters
    ----------
    r : ndarray
        Particle positions in 2D.
    sigma : float
        Particle interaction radius.
    L : float/ndarray
        Boxsize to be passed to scipy.distance.cKDTree.

    Returns
    -------
    clusters : list
        List of arrays containing indices of individual clusters.

    """

    # Find 'overlaps' using kd-tree
    kd = cKDTree(r, boxsize=L)
    # overlap range slightly larger than cutoff range
    overlaps = kd.query_pairs(1.05*2**(1/6)*sigma, output_type='ndarray')

    # initialize list
    clusters = []

    # variable 'index' keeps track of indices of particles not yet included in
    # a cluster
    index = np.arange(len(r))

    while len(index) > 0:

        # start with last index
        i = index[-1]

        # retrieve cluster and add to list
        clusters.append(cluster(overlaps, i))

        # remove indices of particles just added to cluster
        index = np.setdiff1d(index, clusters[-1])

    return clusters


@nb.jit(nopython=True)
def cluster(overlaps, i=0):
    """
    Return separate clusters given a list of all overlapping pairs and a
    particle index "i" to start at

    Parameters
    ----------
    overlaps : ndarray
        array of index-pairs of neighboring ("overlapping") particles within
        each other's interaction cutoff.
    i : int
        index of particle whose cluster to determine.

    Returns
    -------
    incluster : ndarray
        array of indices of particles in the same cluster as particle i.

    """

    # initialize empty list
    incluster = []

    # use stack-like datastructure
    stack = [i]

    while len(stack) > 0:

        # pop from stack
        current = stack.pop()

        # add current to cluster
        incluster.append(current)

        # determine which particles overlap with 'current'
        nbs = overlaps[np.where(overlaps.T[0] == current)].T[1]

        # repeat recursively for neighbours of 'current'
        for j in nbs:
            # avoid double counting: check if j already handled or in queue
            if j not in incluster and j not in stack:
                stack.append(j)

        # repeat process for when 'current' is the second in pair
        nbs2 = overlaps[np.where((overlaps.T[1] == current))].T[0]
        for j in nbs2:
            # avoid double counting: check if j already handled or in queue
            if j not in incluster and j not in stack:
                stack.append(j)

    return np.array(incluster)


def N_clustered(clusters, N_rtp, eps=0.1):
    """
    Return the number of parcticles in clusters of size 0.10*N_rtp or larger

    Parameters
    ----------
    clusters : list
        list of lists containing indices of 'clustered' particles.
    N_rtp : int
        Number of particles in the simulation.
    eps : float
        Number between 0 and 1 indicating the raction of total particles a
        cluster must contain to be counted.

    Returns
    -------
    n : int
        Number of particles in large enough clusters.

    """

    n = 0

    for c in clusters:

        if (size := len(c)) > eps*N_rtp:

            n += size

    return n


def largest_cluster_size(clusters):
    """
    Return the size of the largest cluster in the given list of clusters

    Parameters
    ----------
    clusters : list
        list of lists containing indices of 'clustered' particles.

    Returns
    -------
    largest : int
        size of the largest cluster.

    """

    largest = 0

    for c in clusters:

        if len(c) > largest:

            largest = len(c)

    return largest
