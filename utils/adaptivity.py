from firedrake import *

import numpy as np
import numpy
from numpy import linalg as la
from scipy import linalg as sla


def constructHessian(mesh, V, sol, op=None):
    """
    Reconstructs the hessian of a scalar solution field with respect to the current mesh. The code for the integration 
    by parts reconstruction approach is based on the Monge-Amp\`ere tutorial provided in the Firedrake website 
    documentation.

    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param sol: P1 solution field defined on ``mesh``.
    :param op: Options class object providing min/max cell size values.
    :return: reconstructed Hessian associated with ``sol``.
    """
    if op == None:
        from . import options

        op = options.Options()

    H = Function(V)
    tau = TestFunction(V)
    nhat = FacetNormal(mesh)  # Normal vector
    params = {'snes_rtol': 1e8,
              'ksp_rtol': 1e-5,
              'ksp_gmres_restart': 20,
              'pc_type': 'sor'}
    if op.mtype == 'parts':
        Lh = (inner(tau, H) + inner(div(tau), grad(sol))) * dx
        Lh -= (tau[0, 1] * nhat[1] * sol.dx(0) + tau[1, 0] * nhat[0] * sol.dx(1)) * ds
        Lh -= (tau[0, 0] * nhat[1] * sol.dx(0) + tau[1, 1] * nhat[0] * sol.dx(1)) * ds  # Term not in Firedrake tutorial
    else:
        W = VectorFunctionSpace(mesh, 'CG', 1)
        g = Function(W)
        psi = TestFunction(W)
        Lg = (inner(g, psi) - inner(grad(sol), psi)) * dx
        NonlinearVariationalSolver(NonlinearVariationalProblem(Lg, g), solver_parameters=params).solve()
        Lh = (inner(tau, H) + inner(div(tau), g)) * dx
        Lh -= (tau[0, 1] * nhat[1] * g[0] + tau[1, 0] * nhat[0] * g[1]) * ds
        Lh -= (tau[0, 0] * nhat[1] * g[0] + tau[1, 1] * nhat[0] * g[1]) * ds
    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters=params)
    H_solv.solve()

    return H


def computeSteadyMetric(mesh, V, H, sol, nVerT=1000., iError=1000., op=None):
    """
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function ``computeSteadyMetric``, from 
    ``adapt.py``, 2016.

    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param H: reconstructed Hessian, usually chosen to be associated with ``sol``.
    :param sol: P1 solution field defined on ``mesh``.
    :param nVerT: target number of vertices, in the case of Lp normalisation.
    :param iError: inverse of the target error, in the case of manual normalisation.
    :param op: Options class object providing min/max cell size values.
    :return: steady metric associated with Hessian H.
    """
    if op == None:
        from . import options

        op = options.Options()

    ia2 = 1. / pow(op.a, 2)         # Inverse square aspect ratio
    ihmin2 = 1. / pow(op.hmin, 2)   # Inverse square minimal side-length
    ihmax2 = 1. / pow(op.hmax, 2)   # Inverse square maximal side-length
    M = Function(V)
    if op.ntype == 'manual':
        sol_min = 1e-3  # Minimum tolerated value for the solution field

        for i in range(mesh.topology.num_vertices()):

            # Generate local Hessian
            H_loc = H.dat.data[i] * iError / (max(np.sqrt(assemble(sol * sol * dx)), sol_min))  # Avoid round-off error
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs and truncate eigenvalues
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
            lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)

            # Reconstruct edited Hessian
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    else:
        detH = Function(FunctionSpace(mesh, 'CG', 1))
        for i in range(mesh.topology.num_vertices()):
            # Generate local Hessian
            H_loc = H.dat.data[i]
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs of Hessian and truncate eigenvalues
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = max(abs(lam[0]), 1e-10)  # \ To avoid round-off error
            lam2 = max(abs(lam[1]), 1e-10)  # /
            det = lam1 * lam2

            # Reconstruct edited Hessian and rescale
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
            M.dat.data[i] *= pow(det, -1. / (2 * op.p + 2))
            detH.dat.data[i] = pow(det, op.p / (2. * op.p + 2))

        detH_integral = assemble(detH * dx)
        M *= nVerT / detH_integral  # Scale by the target number of vertices
        for i in range(mesh.topology.num_vertices()):
            # Find eigenpairs of metric and truncate eigenvalues
            lam, v = la.eig(M.dat.data[i])
            v1, v2 = v[0], v[1]
            lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
            lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)

            # Reconstruct edited Hessian
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    return M


def isotropicMetric(V, f, bdy=False, op=None):
    """
    :param V: tensor function space on which metric will be defined.
    :param f: (scalar) function to adapt to.
    :param bdy: toggle boundary metric.
    :param op: Options class object providing min/max cell size values.
    :return: isotropic metric corresponding to the scalar function.
    """
    if op == None:
        from . import options

        op = options.Options()

    hmin2 = pow(op.hmin, 2)
    hmax2 = pow(op.hmax, 2)
    M = Function(V)
    g = Function(FunctionSpace(V.mesh(), 'CG', 1))
    family = f.ufl_element().family()
    deg = f.ufl_element().degree()
    if (family == 'Lagrange') & (deg == 1):
        g.assign(f)
    else:
        print("""Field for adaption is degree %d %s. Interpolation is required to 
get a degree 1 Lagrange metric.""" % (deg, family))
        g.interpolate(f)
    for i in DirichletBC(V, 0, 'on_boundary').nodes if bdy else range(len(g.dat.data)):
        ig2 = 1. / max(hmin2, min(pow(g.dat.data[i], 2), hmax2))
        # print('#### istropicMetic DEBUG: 1/g^2 = ', ig2)
        M.dat.data[i][0, 0] = ig2
        M.dat.data[i][1, 1] = ig2
    return M


def metricGradation(mesh, M, beta=1.4, iso=False):
    """
    Perform anisotropic metric gradation in the method described in Alauzet 2010, using linear interpolation. Python
    code based on Nicolas Barral's function ``DMPlexMetricGradation2d_Internal`` in ``plex-metGradation.c``, 2017.

    :param mesh: current mesh on which variables are defined.
    :param metric: metric to be gradated.
    :param beta: scale factor used.
    :param iso: specify whether isotropic or anisotropic mesh adaptivity is being used.
    :return: gradated ``metric``.
    """

    # Get vertices and edges of mesh
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)  # Vertices
    eStart, eEnd = plex.getDepthStratum(1)  # Edges
    numVer = vEnd - vStart
    xy = mesh.coordinates.dat.data

    # Establish arrays for storage and a list of tags for vertices
    v12 = v21 = np.zeros(2)
    # TODO: work only with the upper triangular part for speed
    verTag = np.zeros(numVer) + 1
    correction = True
    i = 0
    ln_beta = np.log(beta)

    while correction & (i < 500):
        i += 1
        correction = False

        # Loop over edges of mesh
        for e in range(eStart, eEnd):
            cone = plex.getCone(e)      # Get vertices associated with edge e
            iVer1 = cone[0] - vStart    # Vertex 1 index
            iVer2 = cone[1] - vStart    # Vertex 2 index
            if (verTag[iVer1] < i) & (verTag[iVer2] < i):
                continue

            # Assemble local metrics and calculate edge lengths
            met1 = M.dat.data[iVer1]
            met2 = M.dat.data[iVer2]
            v12[0] = xy[iVer2][0] - xy[iVer1][0]
            v12[1] = xy[iVer2][1] - xy[iVer1][1]
            v21[0] = - v12[0]
            v21[1] = - v12[1]

            if iso:
                eta2_12 = 1. / pow(1 + (v12[0] * v12[0] + v12[1] * v12[1]) * ln_beta / met1[0, 0], 2)
                eta2_21 = 1. / pow(1 + (v21[0] * v21[0] + v21[1] * v21[1]) * ln_beta / met2[0, 0], 2)
                redMet1 = eta2_12 * met2
                redMet2 = eta2_21 * met1
            else:
                # Intersect metric with a scaled 'grown' metric to get reduced metric
                eta2_12 = 1. / pow(1 + symmetricProduct(met1, v12) * ln_beta, 2)
                eta2_21 = 1. / pow(1 + symmetricProduct(met2, v21) * ln_beta, 2)
                # print('#### metricGradation DEBUG: determinants', la.det(met1), la.det(met2))
                redMet1 = localMetricIntersection(met1, eta2_21 * met2)
                redMet2 = localMetricIntersection(met2, eta2_12 * met1)

            # Calculate difference in order to ascertain whether the metric is modified
            diff = np.abs(met1[0, 0] - redMet1[0, 0]) + np.abs(met1[0, 1] - redMet1[0, 1]) \
                   + np.abs(met1[1, 1] - redMet1[1, 1])
            diff /= (np.abs(met1[0, 0]) + np.abs(met1[0, 1]) + np.abs(met1[1, 1]))
            if diff > 1e-3:
                M.dat.data[iVer1][0, 0] = redMet1[0, 0]
                M.dat.data[iVer1][0, 1] = redMet1[0, 1]
                M.dat.data[iVer1][1, 0] = redMet1[1, 0]
                M.dat.data[iVer1][1, 1] = redMet1[1, 1]
                verTag[iVer1] = i + 1
                correction = True

            # Repeat above process
            diff = np.abs(met2[0, 0] - redMet2[0, 0]) + np.abs(met2[0, 1] - redMet2[0, 1]) \
                   + np.abs(met2[1, 1] - redMet2[1, 1])
            diff /= (np.abs(met2[0, 0]) + np.abs(met2[0, 1]) + np.abs(met2[1, 1]))
            if diff > 1e-3:
                M.dat.data[iVer2][0, 0] = redMet2[0, 0]
                M.dat.data[iVer2][0, 1] = redMet2[0, 1]
                M.dat.data[iVer2][1, 0] = redMet2[1, 0]
                M.dat.data[iVer2][1, 1] = redMet2[1, 1]
                verTag[iVer2] = i + 1
                correction = True


def localMetricIntersection(M1, M2):
    """
    Intersect two metrics (i.e. two 2x2 matrices).

    :param M1: first metric to be intersected.
    :param M2: second metric to be intersected.
    :return: intersection of metrics M1 and M2.
    """
    # print('#### localMetricIntersection DEBUG: attempting to compute sqrtm of matrix with determinant ', la.det(M1))
    sqM1 = sla.sqrtm(M1)
    sqiM1 = la.inv(sqM1)    # Note inverse and square root commute whenever both are defined
    lam, v = la.eig(np.transpose(sqiM1) * M2 * sqiM1)
    return np.transpose(sqM1) * v * [[max(lam[0], 1), 0], [0, max(lam[1], 1)]] * np.transpose(v) * sqM1


def metricIntersection(mesh, V, M1, M2, bdy=False):
    """
    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on current mesh.
    :param M1: first metric to be intersected.
    :param M2: second metric to be intersected.
    :param bdy: when True, intersection with M2 only contributes on the domain boundary.
    :return: intersection of metrics M1 and M2.
    """
    for i in DirichletBC(V, 0, 'on_boundary').nodes if bdy else range(mesh.topology.num_vertices()):
        M1.dat.data[i] = localMetricIntersection(M1.dat.data[i], M2.dat.data[i])
        # print('#### metricIntersection DEBUG: det(Mi) = ', la.det(M1.dat.data[i]))
    return M1


def symmetricProduct(A, b):
    """
    :param A: symmetric, 2x2 matrix / metric field.
    :param b: 2-vector / vector field.
    :return: product b^T * A * b.
    """
    # assert(isinstance(A, numpy.ndarray) | isinstance(A, Function))
    # assert(isinstance(b, list) | isinstance(b, numpy.ndarray) | isinstance(b, Function))

    def bAb(A, b):
        return b[0] * A[0, 0] * b[0] + 2 * b[0] * A[0, 1] * b[1] + b[1] * A[1, 1] * b[1]

    if isinstance(A, numpy.ndarray) | isinstance(A, list):
        if isinstance(b, list) | isinstance(b, numpy.ndarray):
            return bAb(A, b)
        else:
            return [bAb(A, b.dat.data[i]) for i in range(len(b.dat.data))]
    else:
        if isinstance(b, list) | isinstance(b, numpy.ndarray):
            return [bAb(A.dat.data[i], b) for i in range(len(A.dat.data))]
        else:
            return [bAb(A.dat.data[i], b.dat.data[i]) for i in range(len(A.dat.data))]


def pointwiseMax(f, g):
    """
    :param f: first field to be considered.
    :param g: second field to be considered.
    :return: field taking pointwise maximal values in modulus.
    """
    fdat = f.dat.data
    gdat = g.dat.data
    try:
        assert(len(fdat) == len(gdat))
    except:
        raise ValueError("Function space mismatch: ", f.function_space().ufl_element().family(),
                         f.function_space().ufl_element().degree(), " vs. ", g.function_space().ufl_element().family(),
                         g.function_space().ufl_element().degree())
    for i in range(len(fdat)):
        if np.abs(gdat[i]) > np.abs(fdat[i]):
            f.dat.data[i] = gdat[i]
    return f


def advectMetric(M_, w, dt, n=1, outfile=None, bc=None, diffusion=False, timestepper='ImplicitEuler'):
    """
    'Advect' metric with finest resolution in direction of fluid velocity/wind field.
    
    :param M_: metric field defined on current mesh, at current timestep.
    :param w: (vector) velocity field on current mesh. Can be Function, list or ndarray.
    :param dt: timestep.
    :param n: number of timesteps to advect over.
    :param outfile: toggle metric output and location.
    :param diffusion: toggle inclusion of metric diffusion.
    :param timestepper: time integration scheme used.
    :param bc: boundary condition on Tensor advection PDE problem.
    """
    if outfile != None:
        Mfile = File(outfile)
        Mfile.write(M_, time=0)

    # Get FunctionSpace data and select timestepping scheme
    V = M_.function_space()
    mesh = V.mesh()
    sigma = TestFunction(V)
    M = Function(V)
    if timestepper == 'CrankNicolson':
        Mm = 0.5 * (M + M_)
    elif timestepper == 'ImplicitEuler':
        Mm = M
    elif timestepper == 'ExplicitEuler':
        Mm = M_
    else:
        raise NotImplementedError

    # Set up Tensor advection FEM problem
    F = (inner(M - M_, sigma) + dt * inner(dot(w, nabla_grad(Mm)), sigma)) * dx
    if diffusion:
        F -= inner(grad(M), grad(sigma)) * dx       # TODO: what does this mean?
    prob = NonlinearVariationalProblem(F, M)
    solv = NonlinearVariationalSolver(prob, bc=bc)

    # Time integrate
    for i in range(1, n+1):
        solv.solve()
        M_.assign(metricIntersection(mesh, V, M_, M))
        if outfile != None:
            Mfile.write(M_, time=i)

    return M

    # TODO: include metric advection form in forms.py


if __name__ == '__main__':

    mesh = RectangleMesh(64, 16, 4, 1)
    V = TensorFunctionSpace(mesh, "CG", 1)
    M = Function(V, name="Metric").interpolate(Expression([['2+sin(pi * x[0] / 2)', 0], [0, '1']]))
    w = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(Expression([1, 0]))
    advectMetric(M, w, 0.05, 20, outfile='plots/tests/utils/meshAdvect.pvd')