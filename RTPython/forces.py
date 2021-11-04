#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:35:56 2021

@author: Simon Krekels

RTPython.forces - Offers several repulsive forces for use between APs and
probes.
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def quadratic(r, a, eps):
    '''
    Force derived from a quadratic potential; F = -∇V

        V(r) = eps * (1 - r/a)**2

    The resulting force is linear in r.

    ----------
    Parameters
    ----------
    r : float
        distance between objects to calculate force between

    a : float
        range of interaction; assumes symmetric interaction

    eps : float
        modulation of interaction strength

    Returns
    -------
    float
        the magnitude of the quadratic force between the objects at distance r

    '''

    return eps * (1-r/a)/a


def soft_bump(r, a, eps):
    '''
    Force derived from a quadratic potential; F = -∇V

        V(r) = eps * (1 - (r/a)**2)**2


    ----------
    Parameters
    ----------
    r : float
        distance between objects to calculate force between

    a : float
        range of interaction; assumes symmetric interaction

    eps : float
        modulation of interaction strength

    Returns
    -------
    float
        the magnitude of the quadratic force between the objects at distance r

    '''

    return 4*eps*r/a * (1-(r/a)**2)


@nb.jit(nopython=True)
def wca(r, a, eps):
    '''
    Force derived from a Weeks-Chandler-Anderson (WCA) potential; F = -∇V

        V(r) = 4 * eps * ( (σ/r)**12 - (σ/r)**6) + eps

    Parameters
    ----------
    r : float
        distance between objects to calculate force between

    a : float
        range of interaction; assumes symmetric interaction

    eps : float
        modulation of interaction strength

    Returns
    -------
    float
        the magnitude of the WCA force between the objects at distance r

    '''

    return 4*eps*(12*a**12/r**13 - 6*a**6/r**7)


def force_sum(particles, interactions):
    """
    Calculates total forces exerted by 'particles' on other 'particles',
    mediated by 'interactions'.

    Parameters
    ----------
    particles : dict
        dict containing dicts specifying particle collections (see
        RTPython.initialize) keys are the 'names' of the particles
    interactions : iterable
        iterables whose entries contain:
            - (func) the interaction function (RTPython.md) to use
            - (str, str) 'names' of interacting particles
            - (iterable) interaction arguments to pass to the forces

    Returns
    -------
    list
        a list of ndarrays corresponding to the forces on the 'particles'
        supplied.

    """

    F_res = {particles[p]['name']: np.zeros(particles[p]['r'].shape)
             for p in particles}

    for current_int in interactions:
        F = current_int[0](*[particles[a] for a in current_int[1]],
                           *current_int[2:])
        for i, f in enumerate(F):
            F_res[current_int[1][i]] += f

    return F_res
