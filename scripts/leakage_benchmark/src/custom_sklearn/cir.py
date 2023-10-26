# -*- coding: utf-8 -*-
"""
from https://github.com/assaforon/pycir/blob/main/pycir/cir.py

Centered isotonic regression, generic but more tailored to dose-response and dose-finding data

Assaf Oron, recoded from R 'cir' package
"""

import numpy as np
import pandas as pd


def cirPAVA(y, x=None, wt=None, interiorStrict=True, strict=False, ybounds=None, full=False, dec=False):
    """
    Perform regression.

    Args:
        x,y,wt: vectors of equal length with the doses, mean y values (usually response rates in [0,1]), and weights (often sample size)
        outx: ...
    """
    if ybounds is None:
        ybounds = np.asarray([0.0, 1.0])

    y = np.asarray(y)
    if y.ndim != 1:
        raise Exception(f"You supplied an array with {y.ndim} dimensions, but must be a vector (i.e. 1 dimension)")

    m = len(y)
    _y = y.copy()
    _x = x.copy()
    _wt = wt.copy()

    if dec:
        _y = -_y

    while True:
        viol = np.diff(_y) < 0

        if interiorStrict:
            equals = np.diff(_y) == 0
            for i in range(0, m - 1):
                if _y[i] in ybounds and _y[i + 1] in ybounds:
                    equals[i] = False
            viol = viol | equals

        # strict flag overrides interior-strict nuance
        if strict:
            viol = np.diff(_y) <= 0

        if not (any(viol)):
            break

        i = np.min(np.where(viol))

        _y[i] = (_y[i] * _wt[i] + _y[i + 1] * _wt[i + 1]) / (_wt[i] + _wt[i + 1])
        _x[i] = (_x[i] * _wt[i] + _x[i + 1] * _wt[i + 1]) / (_wt[i] + _wt[i + 1])
        _wt[i] = _wt[i] + _wt[i + 1]
        _y = np.delete(_y, i + 1)
        _x = np.delete(_x, i + 1)
        _wt = np.delete(_wt, i + 1)

        m -= 1
        if m <= 1:
            break

    # extending back to original boundaries if needed
    if _x[0] > x[0]:
        _x = np.insert(_x, 0, x[0])
        _y = np.insert(_y, 0, y[0])
        _wt = np.insert(_wt, 0, wt[0])
        _y[0] = _y[1]
        _wt[0] = 0  # The weight is spoken for though
    if np.max(_x) < np.max(x):
        _x = np.insert(_x, -1, x[0])
        _y = np.insert(_y, -1, y[0])
        _wt = np.insert(_wt, -1, wt[0])
        _wt[_x == np.max(_x)] = 0  # The weight is spoken for though
        _y[_x == np.max(_x)] = np.max(_y)

    # Finish up
    outy = np.interp(x=x, xp=_x, fp=_y)
    if not full:
        return outy

