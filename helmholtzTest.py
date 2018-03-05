from firedrake import *

import numpy as np
import numpy.linalg as la
from time import clock
import matplotlib.pyplot as plt

import utils.adaptivity as adap
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt


op = opt.Options(hmin=1e-3,
                 hmax=0.5)

def helmholtzSolve(mesh_H, p, f=None, space="CG"):
    """
    Solve Helmholtz' equation in CG2 space. By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y).
    
    :param mesh_H: mesh of [0,1]x[0,1] upon which problem is solved.
    :param p: degree of polynomials used.
    :param f: RHS function.
    :param space: type of test space considered.
    :return: approximation on ``mesh_H``, under ``space`` of order ``p``, along with exact solution.
    """

    # Establish numerical solution
    x, y = SpatialCoordinate(mesh_H)
    V = FunctionSpace(mesh_H, space, p)
    if not f:
        f = Function(V).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
    u_H = Function(V)
    v = TestFunction(V)
    B = (inner(grad(u_H), grad(v)) + u_H*v)*dx
    L = f*v*dx
    F = B - L
    solve(F == 0, u_H)

    # Establish analytic solution
    u = Function(V).interpolate(cos(2*pi*x)*cos(2*pi*y))

    return u_H, u, f


def fixed():
    print("\nFixed mesh_H runs\nRefinement   Degree    Error   Timing")
    for p in range(1, 4):
        for i in range(8):
            tic = clock()
            n = pow(2, i)
            mesh_H = UnitSquareMesh(n, n)
            u_H, u = helmholtzSolve(mesh_H, p)[:2]
            err = la.norm(u_H.dat.data - u.dat.data)
            print("     %d          %d      %.4f    %.2fs" % (i, p, err, clock() - tic))

def adaptive(mesh_HIterations=3, degree=1, approach='hessianBased', op=op):
    errors = []
    nEls = []
    times = []
    for i in range(8):
        tic = clock()
        n = pow(2, i)
        mesh_H = UnitSquareMesh(n, n)
        nVerT = op.vscale * msh.meshStats(mesh_H)[1]
        u_H, u, f = helmholtzSolve(mesh_H, degree)
        err = la.norm(u_H.dat.data - u.dat.data)
        for cnt in range(mesh_HIterations):
            if approach == 'hessianBased':
                M = adap.computeSteadyMetric(u_H, nVerT=nVerT, op=op)
            elif approach == 'higherOrderResidual':
                x, y = SpatialCoordinate(mesh_H)
                V_oi = FunctionSpace(mesh_H, "CG", degree+1)
                u_H_oi = Function(V_oi).interpolate(u_H)
                f_oi = Function(V_oi).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
                v_DG0 = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                R_oi = assemble(v_DG0 * (- div(grad(u_H_oi)) + u_H_oi - f_oi) * dx)
                M = adap.isotropicMetric(R_oi, invert=False, nVerT=nVerT, op=op)
            elif approach == 'refinedResidual':
                mesh_h = adap.isoP2(mesh_H)
                x, y = SpatialCoordinate(mesh_h)
                u_h = inte.interp(mesh_h, u_H)[0]
                V_h = FunctionSpace(mesh_h, "CG", degree)
                f_h = Function(V_h).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
                v_DG0 = TestFunction(FunctionSpace(mesh_h, "DG", 0))
                R_h = assemble(v_DG0 * (- div(grad(u_h)) + u_h - f_h) * dx)
                M = adap.isotropicMetric(inte.interp(mesh_H, R_h)[0], invert=False, nVerT=nVerT, op=op)
            else:
                raise NotImplementedError
            if op.gradate:
                adap.metricGradation(M, op=op)
            mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
            f = inte.interp(mesh_H, f)[0]
            u_H, u, f = helmholtzSolve(mesh_H, degree, f)
            err = la.norm(u_H.dat.data - u.dat.data)
        nEle = msh.meshStats(mesh_H)[0]
        tic = clock()-tic
        print("Initial mesh_H %d   Error: %.4f    #Elements: %d     Timing: %.2fs" % (i, err, nEle, tic))
        errors.append(err)
        nEls.append(nEle)
        times.append(tic)

    return errors, nEls, times


if __name__ == '__main__':
    mode = input("""Choose parameter to vary from 
                        {'mesh_HIterations', 'hessMeth', 'ntype', 'p', 'vscale', 'gradate', 'approach'}: """)\
           or 'mesh_HIterations'

    errors = []
    nEls = []
    times = []

    if mode == 'mesh_HIterations':
        S = range(1, 4)
        for i in S:
            print("\nTesting use of %d mesh_H iterations\n" % i)
            err, nEle, tic = adaptive(i, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'hessMeth':
        S = ('parts', 'dL2')
        for hessMeth in S:
            print("\nTesting use of %s Hessian reconstruction\n" % hessMeth)
            op.hessMeth = hessMeth
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'ntype':
        S = ('lp', 'manual')
        for ntype in S:
            print("\nTesting use of %s metric normalisation\n" % ntype)
            op.ntype = ntype
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'p':
        S = (1, 2, 3)
        for p in S:
            print("\nTesting Lp metric normalisation with p = %d\n" % p)
            op.p = p
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'vscale':
        S = np.linspace(0.25, 1., 6)
        for vscale in S:
            print("\nTesting metric rescaling by %.2f\n" % vscale)
            op.vscale = vscale
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'gradate':
        S = (True, False)
        for gradate in S:
            print("\nTesting metric gradation: ", gradate, "\n")
            op.gradate = gradate
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'approach':
        S = ('hessianBased', 'higherOrderResidual', 'refinedResidual')
        for approach in S:
            print("\nTesting use of error estimator %s\n" % approach)
            err, nEle, tic = adaptive(approach=approach, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    styles = ('s', '^', 'x', 'o', 'h', '*', '+')
    outputs = {'errors': errors, 'times': times}
    for output in outputs:
        for i in range(len(S)):
            plt.loglog(nEls[i], outputs[output][i], label=str(S[i]), marker=styles[i])
        plt.title('Experiment: '+mode)
        plt.legend()
        plt.savefig('outdata/outputs/helmholtz_'+mode+'_'+output+'.pdf', bbox_inches='tight')
        plt.show()
        plt.clf()

# TODO: Further, create a wave equation test case based on this
# TODO: * Test metric advection
