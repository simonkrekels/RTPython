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
    Force derived from a quadratic potential; F = -∇V

        V(r) = eps * (1 - r/(2a))**2

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

    return eps * (1-r/(a))/a


@nb.jit(nopython=True)
def wca(r, a, eps):
    '''
    Force derived from a Weeks-Chandler-Anderson potential; F = -∇V

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
        the magnitude of the quadratic force between the objects at distance r

    '''

    return 4*eps*(12*a**12/r**13 - 6*a**6/r**7)


def force_sum(particles, interactions):
    """
    Calculates total forces exerted by 'particles' on other 'particles',
    mediated by 'interactions'.

    Parameters
    ----------
    particles : iterable
        iterable containing dicts specifying particle collections (see
        RTPython.initialize)
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
    pass
