#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:35:56 2021

@author: Simon Krekels

RTPython.forces - Offers several repulsive forces for use between APs and
probes.
"""

import numba as nb


@nb.jit(nopython=True)
def quadratic(r, a, eps):
    '''
    Force derived from a quadratic potential.

        V(r) = eps * (1 - r/(2a))**2

    The resulting force is linear in r.

    ----------
    Parameters
    ----------
    r : float
        distance between objects to calculate force between

    a : float
        radius of interaction; assumes symmetric interaction

    eps : float
        modulation of interaction strength

    Returns
    -------
    float
        the magnitude of the quadratic force between the objects at distance r

    '''

    return eps/2 * (1-r/(2*a))/a


@nb.jit(nopython=True)
def wca(r, a, eps):
    '''
    Force derived from a Weeks-Chandler-Anderson potential.

        V(r) = 4 * eps * ( (σ/r)**12 - (σ/r)**6) + eps

    Parameters
    ----------
    r : float
        distance between objects to calculate force between

    a : float
        radius of interaction; assumes symmetric interaction

    eps : float
        modulation of interaction strength

    Returns
    -------
    float
        the magnitude of the quadratic force between the objects at distance r

    '''

    sigma = 2*a

    return 4*eps*(12*sigma**12/r**13 - 6*sigma**6/r**7)
