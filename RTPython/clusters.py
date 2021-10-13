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
    kd = cKDTree(r, boxsize=(L, L))
    overlaps = kd.query_pairs(1.05*2**(1/6)*sigma, output_type='ndarray')
    clusters = []
    index = np.arange(len(r))
    while len(index) > 0:
        i = index[-1]
        clusters.append(cluster(overlaps, i))
        index = np.setdiff1d(index, clusters[-1])
    return clusters


@nb.jit(nopython=True)
def cluster(overlaps, i=0):
    """
    Return separate clusters given a list of all overlapping pairs and a
    particle index "i" to start at

    Parameters
    ----------
    overlaps : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    incluster = []
    stack = [i]
    while len(stack) > 0:
        current = stack.pop()
        incluster.append(current)
        nbs = overlaps[np.where(overlaps.T[0] == current)].T[1]
        for j in nbs:
            # avoid double counting: check if j already handled or in queue
            if j not in incluster and j not in stack:
                stack.append(j)
        nbs2 = overlaps[np.where((overlaps.T[1] == current))].T[0]
        for j in nbs2:
            # avoid double counting: check if j already handled or in queue
            if j not in incluster and j not in stack:
                stack.append(j)
    return np.array(incluster)


def N_clustered(clusters, N_rtp):
    """
    Return the number of parcticles in clusters of size 0.10*N_rtp or larger

    Parameters
    ----------
    clusters : TYPE
        DESCRIPTION.
    N_rtp : TYPE
        DESCRIPTION.

    Returns
    -------
    n : TYPE
        DESCRIPTION.

    """

    n = 0

    for c in clusters:

        if (size := len(c)) > 0.1*N_rtp:

            n += size

    return n


def largest_cluster_size(clusters):
    """
    Return the size of the largest cluster in the given list of clusters

    Parameters
    ----------
    clusters : TYPE
        DESCRIPTION.

    Returns
    -------
    largest : TYPE
        DESCRIPTION.

    """

    largest = 0

    for c in clusters:

        if len(c) > largest:

            largest = len(c)

    return largest
