from firedrake import *

import numpy as np
from time import clock
import matplotlib.pyplot as plt

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt


op = opt.Options(hmin=1e-3,
                 hmax=0.5)


def helmholtzSolve(mesh_H, p, f=None, space="CG", normType='L2', region=2, adjoint=False):
    """
    Solve Helmholtz' equation in CG2 space. By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y).
    
    :param mesh_H: mesh of [0,1]x[0,1] upon which problem is solved.
    :param p: degree of polynomials used.
    :param f: RHS function.
    :param space: type of test space considered.
    :param normType: type of error norm considered. The 'OF' option measures error in objective computation.
    :param adjoint: toggle use of primal or dual equations.
    :return: approximation on ``mesh_H``, under ``space`` of order ``p``, along with exact solution.
    """

    # Establish numerical solution
    x, y = SpatialCoordinate(mesh_H)
    V = FunctionSpace(mesh_H, space, p)
    if adjoint or normType == 'OF':
        k = form.indicator(V, mode='helmholtz1' if region == 1 else 'helmholtz2')
    if adjoint:
        f = Function(V).assign(k)   # Note Helmholtz equation is self-adjoint
    elif not f:
        f = Function(V).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
    u_H = Function(V)
    v = TestFunction(V)
    B = (inner(grad(u_H), grad(v)) + u_H*v)*dx
    L = f*v*dx
    F = B - L
    solve(F == 0, u_H)

    # Establish analytic solution and compute (relative) error
    u = Function(V).interpolate(cos(2*pi*x)*cos(2*pi*y))
    if normType in ('L2', 'H1', 'Hdiv', 'Hcurl'):
        err = errornorm(u, u_H, norm_type=normType) / norm(u, norm_type=normType)
    elif normType == 'OF':
        J = -0.0285 if region == 1 else 0.0033  # Given by OF evaluated on exact solution on a fine, high degree space
        err = np.abs(J - assemble(k * u_H * dx)) / np.abs(J)
        # err = assemble(k * u * dx)

    return u_H, u, f, err


def fixed():
    print("\nFixed mesh_H runs\nRefinement   Degree    Error   Timing")
    for p in range(1, 4):
        for i in range(10):
            tic = clock()
            n = pow(2, i)
            mesh_H = UnitSquareMesh(n, n)
            u_H, u, f, err = helmholtzSolve(mesh_H, p, normType='OF')
            print("     %d          %d      %.8f    %.2fs" % (i, p, err, clock() - tic))

def adaptive(meshIterations=3, numMeshes=9, degree=1, normType='OF', redefine=False, approach='hessianBased', region=1,
             op=op):
    errors = []
    nEls = []
    times = []
    for i in range(numMeshes):
        tic = clock()
        n = pow(2, i)
        mesh_H = UnitSquareMesh(n, n)
        primalFile = File("plots/helmholtz/solution.pvd")
        dualFile = File("plots/helmholtz/adjoint.pvd")

        # Solve primal equation on initial mesh
        nVerT = op.vscale * msh.meshStats(mesh_H)[1]
        u_H, u, f, err = helmholtzSolve(mesh_H, degree, normType=normType, region=region)

        # Solve primal equation adaptively
        if approach != 'fixedMesh':
            for cnt in range(meshIterations):

                # Solve dual problem on initial mesh
                if approach in ('adjoint', 'DWR', 'DWF', 'refinedDWR', 'refinedDWE',
                                'higherOrderDWR', 'higherOrderDWE', 'lowerOrderDWR', 'lowerOrderDWE'):
                    l_H = helmholtzSolve(mesh_H, degree, adjoint=True, region=region)[0]
                    l_H.rename('Adjoint')
                    dualFile.write(l_H, time=float(i))

                if approach == 'hessianBased':
                    M = adap.computeSteadyMetric(u_H, nVerT=nVerT, op=op)
                elif approach == 'fluxJump':
                    v_DG0 = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    j_bdy = assemble(dot(v_DG0 * grad(u_H), FacetNormal(mesh_H)) * ds)
                    j_int = assemble(jump(v_DG0 * grad(u_H), n=FacetNormal(mesh_H)) * dS)
                    M = adap.isotropicMetric(assemble(v_DG0 * (j_bdy * j_bdy + j_int * j_int) * dx), invert=False,
                                             nVerT=nVerT, op=op)
                elif approach in ('higherOrderResidual', 'higherOrderImplicit', 'higherOrderExplicit',
                                  'higherOrderDWR', 'higherOrderDWE', 'lowerOrderResidual', 'lowerOrderImplicit',
                                  'lowerOrderExplicit', 'lowerOrderDWR', 'lowerOrderDWE'):
                    deg = degree+1 if \
                        approach in ('higherOrderResidual', 'higherOrderImplicit', 'higherOrderExplicit',
                                     'higherOrderDWR', 'higherOrderDWE') else degree-1
                    x, y = SpatialCoordinate(mesh_H)
                    V_oi = FunctionSpace(mesh_H, "CG", deg)
                    u_H_oi = Function(V_oi).interpolate(u_H)
                    f_oi = Function(V_oi).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
                    v_DG0 = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    if approach in ('higherOrderResidual', 'higherOrderExplicit',
                                    'lowerOrderResidual', 'lowerOrderExplicit'):
                        R_oi = assemble(v_DG0 * (- div(grad(u_H_oi)) + u_H_oi - f_oi) * dx)
                        if approach in ('higherOrderResidual', 'lowerOrderResidual'):
                            M = adap.isotropicMetric(R_oi, invert=False, nVerT=nVerT, op=op)
                        else:
                            hk = CellSize(mesh_H)
                            resTerm = assemble(v_DG0 * hk * hk * R_oi * R_oi * dx)
                            j_bdy = assemble(dot(v_DG0 * grad(u_H_oi), FacetNormal(mesh_H)) * ds)
                            j_int = assemble(jump(v_DG0 * grad(u_H_oi), n=FacetNormal(mesh_H)) * dS)
                            jumpTerm = assemble(v_DG0 * hk * (j_bdy * j_bdy + j_int * j_int) * dx)
                            M = adap.isotropicMetric(assemble(sqrt(resTerm + jumpTerm)), invert=False, nVerT=nVerT, op=op)
                    elif approach in ('higherOrderImplicit', 'lowerOrderImplicit', 'lowerOrderDWE', 'higherOrderDWE'):
                        v_oi = TestFunction(V_oi)
                        e = Function(V_oi)
                        Be = (inner(grad(e), grad(v_oi)) + e * v_oi) * dx
                        Bu = (inner(grad(u_H_oi), grad(v_oi)) + u_H_oi * v_oi) * dx
                        L_oi = f_oi * v_oi * dx
                        I = form.interelementTerm(v_oi * grad(u_H_oi), n=FacetNormal(mesh_H)) * dS
                        F_oi = Be - L_oi + Bu - I
                        solve(F_oi == 0, e)
                        if approach in ('higherOrderImplicit', 'lowerOrderImplicit'):
                            M = adap.isotropicMetric(e, invert=False, nVerT=nVerT, op=op)
                        else:
                            l_oi = Function(V_oi).interpolate(l_H)
                            el = assemble(v_DG0 * e * l_oi * dx)
                            M = adap.isotropicMetric(el, invert=False, nVerT=nVerT, op=op)
                    elif approach in ('lowerOrderDWR', 'higherOrderDWR'):
                        l_oi = Function(V_oi).interpolate(l_H)
                        Rl = assemble(v_DG0 * (- div(grad(u_H_oi)) + u_H_oi - f_oi) * l_oi * dx)
                        M = adap.isotropicMetric(Rl, invert=False, nVerT=nVerT, op=op)
                elif approach in ('refinedResidual', 'refinedImplicit', 'refinedExplicit', 'refinedDWR', 'refinedDWE'):
                    mesh_h = adap.isoP2(mesh_H)
                    x, y = SpatialCoordinate(mesh_h)
                    u_h = inte.interp(mesh_h, u_H)[0]
                    V_h = FunctionSpace(mesh_h, "CG", degree)
                    f_h = Function(V_h).interpolate((1+8*pi*pi)*cos(2*pi*x)*cos(2*pi*y))
                    v_DG0 = TestFunction(FunctionSpace(mesh_h, "DG", 0))
                    if approach in ('refinedResidual', 'refinedExplicit'):
                        R_h = assemble(v_DG0 * (- div(grad(u_h)) + u_h - f_h) * dx)
                        if approach == 'refinedResidual':
                            M = adap.isotropicMetric(inte.interp(mesh_H, R_h)[0], invert=False, nVerT=nVerT, op=op)
                        else:
                            hk = CellSize(mesh_h)
                            resTerm = assemble(v_DG0 * hk * hk * R_h * R_h * dx)
                            j_bdy = assemble(dot(v_DG0 * grad(u_h), FacetNormal(mesh_h)) * ds)
                            j_int = assemble(jump(v_DG0 * grad(u_h), n=FacetNormal(mesh_h)) * dS)
                            jumpTerm = assemble(v_DG0 * hk * (j_bdy * j_bdy + j_int * j_int) * dx)
                            eps = assemble(sqrt(resTerm + jumpTerm))
                            M = adap.isotropicMetric(inte.interp(mesh_H, eps)[0], invert=False, nVerT=nVerT, op=op)
                    elif approach == 'refinedDWR':
                        Rl = assemble(v_DG0 * (- div(grad(u_h)) + u_h - f_h) * inte.interp(mesh_h, l_H)[0] * dx)
                        M = adap.isotropicMetric(inte.interp(mesh_H, Rl)[0], invert=False, nVerT=nVerT, op=op)
                    else:
                        v_h = TestFunction(V_h)
                        e = Function(V_h)
                        Be = (inner(grad(e), grad(v_h)) + e * v_h) * dx
                        Bu = (inner(grad(u_h), grad(v_h)) + u_h * v_h) * dx
                        L_h = f_h * v_h * dx
                        I = form.interelementTerm(v_h * grad(u_h), n=FacetNormal(mesh_h)) * dS
                        F_h = Be - L_h + Bu - I
                        solve(F_h == 0, e)
                        if approach == 'refinedDWE':
                            el = assemble(v_DG0 * e * inte.interp(mesh_h, l_H)[0] * dx)
                            M = adap.isotropicMetric(inte.interp(mesh_H, el)[0], invert=False, nVerT=nVerT, op=op)
                        else:
                            M = adap.isotropicMetric(inte.interp(mesh_H, e)[0], invert=False, nVerT=nVerT, op=op)
                elif approach in ('residual', 'explicit', 'DWR'):
                    v_DG0 = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    R_H = assemble(v_DG0 * (- div(grad(u_H)) + u_H - f) * dx)
                    if approach == 'residual':
                        M = adap.isotropicMetric(R_H, invert=False, nVerT=nVerT, op=op)
                    elif approach == 'explicit':
                        hk = CellSize(mesh_H)
                        resTerm = assemble(v_DG0 * hk * hk * R_H * R_H * dx)
                        j_bdy = assemble(dot(v_DG0 * grad(u_H), FacetNormal(mesh_H)) * ds)
                        j_int = assemble(jump(v_DG0 * grad(u_H), n=FacetNormal(mesh_H)) * dS)
                        jumpTerm = assemble(v_DG0 * hk * (j_bdy * j_bdy + j_int * j_int) * dx)
                        M = adap.isotropicMetric(assemble(sqrt(resTerm + jumpTerm)), invert=False, nVerT=nVerT, op=op)
                    else:
                        M = adap.isotropicMetric(assemble(v_DG0 * R_H * l_H * dx), invert=False, nVerT=nVerT, op=op)
                elif approach == 'DWF':
                    M = adap.isotropicMetric(assemble(u_H * l_H), invert=False, nVerT=nVerT, op=op)
                elif approach == 'adjoint':
                    M = adap.isotropicMetric(l_H, invert=False, nVerT=nVerT, op=op)
                else:
                    raise NotImplementedError
                if op.gradate:
                    adap.metricGradation(M, op=op)
                mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                f = None if redefine else inte.interp(mesh_H, f)[0]
                u_H, u, f, err = helmholtzSolve(mesh_H, degree, f, normType=normType, region=region)
        u_H.rename('Numerical solution')
        primalFile.write(u_H, time=float(i))
        nEle = msh.meshStats(mesh_H)[0]
        tic = clock()-tic
        print("Initial mesh_H %d   Error: %.4f    #Elements: %d     Timing: %.2fs" % (i, err, nEle, tic))
        errors.append(err)
        nEls.append(nEle)
        times.append(tic)

    return errors, nEls, times


if __name__ == '__main__':
    mode = input("""Choose parameter to vary from {'meshIterations', 'redefine', 'hessMeth', 'ntype', 'p', 'vscale', 
                                                   'gradate', 'region', 'normType', 'approach', 'order'}: """)\
           or 'meshIterations'
    errors = []
    nEls = []
    times = []

    if mode == 'test':              # Perform a generic test
        tester = input('Method?: ')
        print("\nTesting approach %s\n" % tester)
        err, nEle, tic = adaptive(approach=tester, region=1, op=op)
        exit(23)
    elif mode == 'meshIterations':  # Multiple mesh optimisation steps
        S = range(1, 4)
        for i in S:
            print("\nTesting %d mesh_H iterations\n" % i)
            err, nEle, tic = adaptive(i, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'redefine':        # Avoid interpolation error and costs
        S = (True, False)
        for tf in S:
            print("\nTesting with RHS redefinition: ", tf, "\n")
            err, nEle, tic = adaptive(redefine=tf, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'hessMeth':        # Mode of Hessian recovery
        S = ('parts', 'dL2')
        for hessMeth in S:
            print("\nTesting %s Hessian reconstruction\n" % hessMeth)
            op.hessMeth = hessMeth
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'ntype':           # Mode of metric normalisation
        S = ('lp', 'manual')
        for ntype in S:
            print("\nTesting %s metric normalisation\n" % ntype)
            op.ntype = ntype
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'p':               # Type of Lp norm used in normalisation step
        S = (1, 2, 3)
        for p in S:
            print("\nTesting Lp metric normalisation with p = %d\n" % p)
            op.p = p
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'vscale':          # Scale target number of vertices
        S = np.linspace(0.25, 1., 6)
        for vscale in S:
            print("\nTesting metric rescaling by %.2f\n" % vscale)
            op.vscale = vscale
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'normType':        # Quantify errors in different norms
        S = ('L2 fixedMesh', 'H1 fixedMesh', 'OF fixedMesh', 'L2 hessianBased', 'H1 hessianBased', 'OF hessianBased')
        for Si in S:
            normType, approach = Si.split()
            print("\nTesting norm type %s for %s\n" % (normType, approach))
            err, nEle, tic = adaptive(approach=approach, normType=normType, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'gradate':         # Smoothen mesh using gradation
        S = (True, False)
        for gradate in S:
            print("\nTesting metric gradation: ", gradate, "\n")
            op.gradate = gradate
            err, nEle, tic = adaptive(op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'region':         # Change region of importance
        S = (1, 2)
        for r in S:
            print("\nTesting in region ", r, "\n")
            err, nEle, tic = adaptive(region=r, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'approach':        # Experiment with different error estimators
        experiment = int(input("""Choose experiment from list:
0: Non-adjoint approaches
1: Residual approximations
2: Implicit approximations
3: Explicit approximations
4: Higher order approximations
5: Refined approximations
6: Adjoint approximations
7: Main approaches\n"""))
        A = ('fixedMesh', 'hessianBased', 'fluxJump', 'residual', 'explicit',
             'higherOrderResidual', 'higherOrderImplicit', 'higherOrderExplicit', 'higherOrderDWR', 'higherOrderDWE',
             'refinedResidual', 'refinedImplicit', 'refinedExplicit', 'refinedDWR', 'refinedDWE', 'DWF'
             )
        E = {0: (A[0], A[1], A[2], A[3], A[4], A[6]),
             1: (A[0], A[3], A[5], A[10]),
             2: (A[0], A[6], A[11]),
             3: (A[0], A[4], A[7], A[12]),
             4: (A[0], A[5], A[6], A[7]),
             5: (A[0], A[10], A[11], A[12]),
             6: (A[0], A[8], A[9], A[13], A[14], A[15]),
             7: (A[0], A[1], A[8], A[13], A[15])}
        S = E[experiment]
        for approach in S:
            print("\nTesting use of %s error estimation\n" % approach)
            err, nEle, tic = adaptive(approach=approach, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    elif mode == 'order':           # Approximate errors and residuals in lower, higher or the same space
        experiment = int(input("""Choose experiment from list:
0: Residual approximations
1: Implicit approximations
2: Explicit approximations\n"""))
        A = ('fixedMesh',
             'lowerOrderResidual', 'residual', 'higherOrderResidual',
             'lowerOrderImplicit', 'higherOrderImplicit',
             'lowerOrderExplicit', 'explicit', 'higherOrderExplicit',
             'lowerOrderDWR', 'DWR', 'higherOrderDWR',
             'lowerOrderDWE', 'lowerOrderDWE'
             )
        E = {0: (A[0], A[1], A[2], A[3], A[9], A[10], A[11]),
             1: (A[0], A[4], A[5], A[12], A[13]),
             2: (A[0], A[6], A[7], A[8])}
        S = E[experiment]
        for approach in S:
            print("\nTesting use of %s error estimation\n" % approach)
            err, nEle, tic = adaptive(approach=approach, numMeshes=7, degree=2, op=op)
            errors.append(err)
            nEls.append(nEle)
            times.append(tic)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    styles = ('s', '^', 'x', 'o', 'h', '*', '+', '.')
    outputs = {'errors': errors, 'times': times}
    for output in outputs:
        for i in range(len(S)):
            plt.loglog(nEls[i], outputs[output][i], label=str(S[i]), marker=styles[i])
        title = 'Experiment: '+mode
        if mode in ('approach', 'order'):
            title += ' ' + str(experiment)
        plt.title(title)
        plt.legend()
        plt.xlabel('#Elements')
        plt.ylabel('CPU time' if output == 'times' else 'Error')
        filename = 'outdata/outputs/helmholtz/helmholtz_'+mode+'_'+output
        if mode in ('approach', 'order'):
            filename += '_experiment' + str(experiment)
        filename += '.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.clf()

# TODO: Further, create a wave equation test case based on this
# TODO: * Test metric advection

# TODO: Even more basic test cases to check Hessian recovery does what it should. i.e. consider a C^2 function
