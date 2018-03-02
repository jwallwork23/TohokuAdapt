from firedrake import *

import numpy as np
import numpy.linalg as la
from time import clock
import matplotlib.pyplot as plt

import utils.adaptivity as adap
import utils.mesh as msh
import utils.options as opt


op = opt.Options(hmin=1e-3,
                 hmax=0.5)

def helmholtzSolve(mesh, p, space="CG"):
    """
    Solve Helmholtz' equation in CG2 space. By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y).
    
    :param mesh: mesh of [0,1]x[0,1] upon which problem is solved.
    :param p: degree of polynomials used.
    :param space: type of test space considered.
    :return: approximation on ``mesh``, under ``space`` of order ``p``, along with exact solution.
    """

    # Establish numerical solution
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, space, p)
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


def fixedTests():
    print("\nFixed mesh runs\nRefinement   Degree    Error   Timing")
    for p in range(1, 4):
        for i in range(8):
            tic = clock()
            n = pow(2, i)
            mesh = UnitSquareMesh(n, n)
            u_h, u = helmholtzSolve(mesh, p)
            err = la.norm(u_h.dat.data - u.dat.data)
            print("     %d          %d      %.4f    %.2fs" % (i, p, err, clock() - tic))

def adaptiveTests(meshIterations=3, op=op):
    errors = []
    nEls = []
    times = []
    for i in range(7):
        tic = clock()
        n = pow(2, i)
        mesh = UnitSquareMesh(n, n)
        nVerT = op.vscale * msh.meshStats(mesh)[1]
        u_h, u = helmholtzSolve(mesh, 1)
        err = la.norm(u_h.dat.data - u.dat.data)
        for cnt in range(meshIterations):
            M = adap.computeSteadyMetric(u_h, nVerT=nVerT, op=op)
            mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            u_h, u = helmholtzSolve(mesh, 1)
            err = la.norm(u_h.dat.data - u.dat.data)
        nEle = msh.meshStats(mesh)[0]
        tic = clock()-tic
        print("Initial mesh %d   Error: %.4f    #Elements: %d     Timing: %.2fs" % (i, err, nEle, tic))
        errors.append(err)
        nEls.append(nEle)
        times.append(tic)

    return errors, nEls, times

# TODO: Run more tests:
# TODO: * Comparison with isotropic case

# TODO: Further, create a wave equation test case based on this
# TODO: * Test metric advection


if __name__ == '__main__':
    mode = input("Choose parameter to vary from {'meshIterations', 'hessMeth', 'ntype', 'vscale'}: ") \
           or 'meshIterations'

    errors = []
    nEls = []
    times = []

    if mode == 'meshIterations':
        S = range(1, 4)
        for i in S:
            print("\nTesting use of %d mesh iterations\n" % i)
            err, nEle, tic = adaptiveTests(i, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'hessMeth':
        S = ('parts', 'dL2')
        for hessMeth in S:
            print("\nTesting use of %s Hessian reconstruction\n" % hessMeth)
            op.hessMeth = hessMeth
            err, nEle, tic = adaptiveTests(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'ntype':
        S = ('lp', 'manual')
        for ntype in S:
            print("\nTesting use of %s metric normalisation\n" % ntype)
            op.ntype = ntype
            err, nEle, tic = adaptiveTests(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'vscale':
        S = np.linspace(0.25, 1., 6)
        for vscale in S:
            print("\nTesting metric rescaling by %.2f\n" % vscale)
            op.vscale = vscale
            err, nEle, tic = adaptiveTests(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    styles = ('s', '^', 'x', 'o', 'h', '*', '+')
    for i in range(len(S)):
        if mode == 'vscale':
            plt.loglog(nEls[i], errors[i], label=str(S[i]), marker=styles[i])
        else:
            plt.semilogx(nEls[i], errors[i], label=str(S[i]), marker=styles[i])
    plt.title('Experiment: '+mode)
    plt.legend()
    plt.savefig('outdata/outputs/helmholtz_'+mode+'_errors.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
    for i in range(len(S)):
        plt.loglog(nEls[i], times[i], label=str(S[i]), marker=styles[i])
    plt.title('Experiment: '+mode)
    plt.legend()
    plt.savefig('outdata/outputs/helmholtz_'+mode+'_times.pdf', bbox_inches='tight')
    plt.show()
