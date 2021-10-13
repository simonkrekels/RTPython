#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:18:17 2021

@author: Simon Krekels
"""
import numpy as np


def simple_grid(L, phi, sigma):
    """
    Initiates a simple rectangular grid of colliods.

    Parameters
    ----------
    L : float
        Box size.
    phi : float
        Density (packing fraction). Should be a number between 0 and 1; between
        0 and π/4 ≈ 0.785 if hard sphere potential is used.
    sigma : float
        Interaction range (diameter) of colloids.

    Returns
    -------
    ndarray
        list of 2D coordinates of the colloids

    """
    # Some consistency checks
    if (L < 0):
        raise ValueError("Box size 'L' must be positive.")

    if (phi < 0) or (phi > 1):
        raise ValueError("Packing fraction 'phi' must be a number between 0 \
                         and 1.")

    if (sigma < 0):
        raise ValueError("Interaction range 'sigma' must be positive.")

    # calculate number of particles needed
    N = int(np.round(L**2 * phi / (np.pi * (sigma/2)**2)))

    N_sq = int(np.ceil(np.sqrt(N)))

    coords = np.array(list(np.ndindex(N_sq, N_sq)))[:-(N_sq**2-N)]

    coords = L / N_sq * coords

    return coords


def probes_grid(L, phi, sigma, probes):
    """


    Parameters
    ----------
    L : float
        Box size.
    phi : float
        Density (packing fraction). Should be a number between 0 and 1; between
        0 and π/4 ≈ 0.785 if hard sphere potential is used.
    sigma : float
        Interaction range (diameter) of colloids.
    probes : dict
        dict containing probe data. See RTPython.initialize.init_probes

    Returns
    -------
    coords : ndarray
        Grid coordinates with space left out for probes..

    """
    coords = simple_grid(L, phi, sigma)

    for r in probes['r']:
        index = np.where(np.sqrt(np.sum((coords - r)**2, axis=1))
                         < (sigma + probes['size']) / 2)
        coords = np.delete(coords, index, axis=0)

    return coords


def init_rtp(r0, rate, velocity, sigma, name='A'):
    """
    Initialize a collection of RTPs with identical features.

    Parameters
    ----------
    r0 : ndarray
        N×2 array specifying the positions of the RTPs.
    rate : float
        RTP tumble rate.
    velocity : float
        RTP propulsion velocity.
    sigma : float
        RTP interaction range (diameter).
    name : str
        label for this collection.

    Returns
    -------
    dict
        Dictionary containing keys:
            - r: (ndarray) current RTP positions.
            - v: (ndarray) current RTP self-propulsion velocities.
            - t: (ndarray) RTP tumble times
            - rate: (float) RTP tumble rate
            - velocity: (float) RTP self-propulsion velocity; |v| = velocity
            - size: (float) RTP interaction range (diameter)
            - time: (float) simulation clock
            - name: (str) label for this collection
            - type: (str) type of particles described

    """

    # Find # of RTPs from number of specified coordinates
    N_rtp = len(r0)

    # Initialize velocities in random directions
    ang0 = 2 * np.pi * np.random.random(N_rtp)
    v0 = velocity * np.array((np.cos(ang0), np.sin(ang0))).T

    # pull tumble times from exponential distribution
    t0 = np.random.exponential(1/rate, N_rtp)

    return {'r': r0,
            'v': v0,
            't': t0,
            'rate': rate,
            'velocity': velocity,
            'size': sigma,
            'time': 0.0,
            'name': name,
            'type': 'rtp'}


def init_abp():
    pass


def init_probes(r0, sigma, name='B'):
    """
    Initialize a collection of identical (non-moving) probes.

    Parameters
    ----------
    r0 : ndarray
        probe positions.
    sigma : float
        probe diameter.
    name : str
        label for this collection.

    Returns
    -------
    dict
        Dictionary containing keys:
            - r: (ndarray) probe positions
            - size: (float) probe diameter
            - name: (str) label for this collection
            - type (str) type of particles described

    """
    return {'r': r0,
            'size': sigma,
            'name': name,
            'type': 'probe'}


def init_walls():
    pass


def semi_circle(position, rad, rot, num):
    """
    Generate a semi-circular set of coordinates.

    Parameters
    ----------
    position : iterable
        2×1 array specifying the center of the semicircle.
    rad : float
        Radius of the semicircle.
    rot : float
        Rotation (in radians) of the semicircle.
    num : int
        Number of coordinates to generate on the semicircle.

    Returns
    -------
    ndarray
        num×2-sized array of coordinates.

    """

    # distribute angles linearly along the semicircle
    angles = np.linspace(-np.pi/2 + rot, np.pi/2 + rot, num)

    # convert to cartesian coordinates
    r = np.array((position[0] + rad * np.cos(angles),
                  position[1] + rad * np.sin(angles)))

    return r.T
