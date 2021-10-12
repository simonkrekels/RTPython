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


def probes_grid():
    pass


def init_rtp():
    pass


def init_abp():
    pass


def init_probes():
    pass


def init_walls():
    pass
