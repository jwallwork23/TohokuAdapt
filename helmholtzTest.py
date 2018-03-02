from firedrake import *

import numpy.linalg as la
from time import clock

import utils.adaptivity as adap
import utils.mesh as msh
import utils.options as opt


op = opt.Options(hmin=1e-3,
                 hmax=0.5)

def helmholtzSolve(mesh_H, p, space="CG"):
    """
    Solve Helmholtz' equation in CG2 space. By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y).
    
    :param mesh: mesh of [0,1]x[0,1] upon which problem is solved.
    :param p: degree of polynomials used.
    :param space: type of test space considered.
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


# Adaptive runs
print("Adaptive runs")
for i in range(8):
    print("\nInitial mesh %d" % i)
    tic = clock()
    n = pow(2, i)
    mesh = UnitSquareMesh(n, n)
    nVerT = 0.85 * msh.meshStats(mesh)[0]
    u_h, u = helmholtzSolve(mesh, 1)
    err = la.norm(u_h.dat.data - u.dat.data)
    print("%d   %.4f" % (0, err))
    for cnt in range(3):
        M = adap.computeSteadyMetric(u_h, nVerT=nVerT, op=op)
        mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
        u_h, u = helmholtzSolve(mesh, 1)
        err = la.norm(u_h.dat.data - u.dat.data)
        print("%d   %.4f" % (cnt+1, err))
    print("Timing: %.2fs" % (clock()-tic))

# TODO: Run multiple tests:
# TODO: * Number of adaption iterations
# TODO: * Methods of normalisation
# TODO: * Methods of Hessian reconstruction
# TODO: * Scaling parameters (i.e. nVerT)
# TODO: * Comparison with isotropic case

# TODO: Further, create a wave equation test case based on this
# TODO: * Test metric advection

# Fixed mesh runs
print("\nFixed mesh runs\nRefinement   Degree    Error   Timing")
for p in range(1, 4):
    for i in range(8):
        tic = clock()
        n = pow(2, i)
        mesh = UnitSquareMesh(n, n)
        u_h, u = helmholtzSolve(mesh, p)
        err = la.norm(u_h.dat.data-u.dat.data)
        print("     %d          %d      %.4f    %.2fs" % (i, p, err, clock() - tic))
