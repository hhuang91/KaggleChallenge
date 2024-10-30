# -*- coding: utf-8 -*-
"""
3D hermite cubic spline interpolation

@author: H. Huang
"""

import torch as T


def h_poly_helper(tt):
    A = T.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=tt[-1].dtype)
    return [
        sum(A[i, j]*tt[j] for j in range(4))
        for i in range(4)]


def h_poly(t):
    tt = [None for _ in range(4)]
    tt[0] = T.ones_like(t)
    for i in range(1, 4):
        tt[i] = tt[i-1]*t
    return h_poly_helper(tt)


def interp_func(x, F, dim):

    "Returns interpolating function"
    """
    x, 1D, float Tensor, sampling location along dim
    F, 3D, float Tensor, sampled value
    dim, single number, int, sampling dimension along which x is located
    """
    assert len(x) > 1, 'Not implemented for signle number interpolation'
    dF = T.narrow(F, dim, 1, len(x)-1) - T.narrow(F, dim, 0, len(x)-1)
    dx = x[1:] - x[:-1]
    newShape = [1, 1, 1]
    newShape[dim] = len(dx)
    newShape = tuple(newShape)
    dx = dx.view(newShape)
    m = dF/dx
    m0 = T.narrow(m, dim, 0, 1)
    mMiddle = T.narrow(m, dim, 1, len(x)-1-1) + T.narrow(m, dim, 0, len(x)-1-1)
    mEnd = T.narrow(m, dim, -1, 1)
    # with shape similar as F but +1 len at dim
    m = T.cat((m0, mMiddle/2, mEnd), dim=dim)
    # pad F to avoid selection out of range
    F = T.cat((F, T.narrow(F, dim, -1, 1)), dim=dim)

    def func(xs):
        I = T.searchsorted(x[1:], xs)
        dxi = (x[I+1]-x[I])
        hh = h_poly((xs-x[I])/dxi)
        newShape = [1, 1, 1]
        newShape[dim] = len(I)
        newShape = tuple(newShape)
        hh0 = hh[0].view(newShape) * T.index_select(F, dim, I)
        hh1 = hh[1].view(newShape) * T.index_select(m,
                                                    dim, I) * dxi.view(newShape)
        # <-- reason for padding F, I+1 could be out of range
        hh2 = hh[2].view(newShape) * T.index_select(F, dim, I+1)
        hh3 = hh[3].view(newShape) * T.index_select(m,
                                                    dim, I+1) * dxi.view(newShape)
        return hh0 + hh1 + hh2 + hh3
    return func


def interp(x, y, z, Fxyz, xs, ys, zs):
    """
    x,y,z,t: 1-D Tensors, specifying sampling grid
    xs,ys,zs,ts: 1-D Tensors, specifying interpolating grid
    """
    Fyz = interp_func(x, Fxyz, 0)(xs)
    Fz = interp_func(y, Fyz, 1)(ys)
    F = interp_func(z, Fz, 2)(zs)
    return F
