from firedrake import *

import numpy as np
import numpy.linalg as la

import utils.adaptivity as adap
import utils.interpolation as inte


def helmholtzSolve(mesh_H, space, p):
    """
    Solve Helmholtz' equation in CG2 space. By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y).
    
    :param mesh: mesh of [0,1]x[0,1] upon which problem is solved.
    :param space: type of test space considered.
    :param p: degree of polynomials used.
    :return: approximation on ``mesh``, under ``space`` of order ``p``, along with exact solution.
    """

    # Establish numerical solution
    x, y = SpatialCoordinate(mesh_H)
    V = FunctionSpace(mesh_H, space, p)
    f = Function(V).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
    u_h = Function(V)
    v = TestFunction(V)
    B = (inner(grad(u_h), grad(v)) + u_h*v)*dx
    L = f*v*dx
    F = B - L
    solve(F == 0, u_h)

    # Establish analytic solution
    u = Function(V).interpolate(cos(2*pi*x)*cos(2*pi*y))

    return u_h, u


# Fixed mesh runs
print("Refinement   Space       Error")
for space in ("CG", "DG"):
    for p in range(1, 4):
        for i in range(8):
            n = pow(2, i)
            mesh = UnitSquareMesh(n, n)
            u_h, u = helmholtzSolve(mesh, space, p)
            err = la.norm(u_h.dat.data-u.dat.data)
            print("     %d      %s  %d          %.4f" % (i, space, p, err))
