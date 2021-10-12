#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:55:43 2021

@author: Simon Krekels
"""


def srk_rtp_nn(rtp, dt, interactions):
    """
    Stochastic Runge-Kutta integration for RTPs (no noise). Employs a variable
    time step to ensure stability. Acts on a given dict of RTPs in-place.

    Parameters
    ----------
    rtp : dict
        Collection of RTPs (see rtpy.initialize.init_rtp).
    dt : float
        Maximum time step.
    interactions : iterable
        interactions to be passed to force_sum.

    Returns
    -------
    float
        The time step taken

    """
    pass


def srk_abp(abp, dt, interactions):
    pass
