from firedrake import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla

from . import options

def constructHessian(mesh, V, sol, method='dL2'):
    """
    Reconstructs the hessian of a scalar solution field with respect to the current mesh. The code for the integration 
    by parts reconstruction approach is based on the Monge-Amp\`ere tutorial provided in the Firedrake website 
    documentation.

    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param sol: P1 solution field defined on ``mesh``.
    :param method: mode of Hessian reconstruction; either a double L2 projection ('dL2') or as integration by parts 
    ('parts').
    :return: reconstructed Hessian associated with ``sol``.
    """

    # Construct functions:
    H = Function(V)
    tau = TestFunction(V)
    nhat = FacetNormal(mesh)  # Normal vector
    params = {'snes_rtol': 1e8,
              'ksp_rtol': 1e-5,
              'ksp_gmres_restart': 20,
              'pc_type': 'sor',
              'snes_monitor': False,
              'snes_view': False,
              'ksp_monitor_true_residual': False,
              'snes_converged_reason': False,
              'ksp_converged_reason': False, }

    if method == 'parts':
        # Hessian reconstruction using integration by parts:
        Lh = (inner(tau, H) + inner(div(tau), grad(sol))) * dx
        Lh -= (tau[0, 1] * nhat[1] * sol.dx(0) + tau[1, 0] * nhat[0] * sol.dx(1)) * ds
        Lh -= (tau[0, 0] * nhat[1] * sol.dx(0) + tau[1, 1] * nhat[0] * sol.dx(1)) * ds  # Term not in tutorial
    elif method == 'dL2':
        # Hessian reconstruction using a double L2 projection:
        V = VectorFunctionSpace(mesh, 'CG', 1)
        g = Function(V)
        psi = TestFunction(V)
        Lg = (inner(g, psi) - inner(grad(sol), psi)) * dx
        g_prob = NonlinearVariationalProblem(Lg, g)
        g_solv = NonlinearVariationalSolver(g_prob, solver_parameters=params)
        g_solv.solve()
        Lh = (inner(tau, H) + inner(div(tau), g)) * dx
        Lh -= (tau[0, 1] * nhat[1] * g[0] + tau[1, 0] * nhat[0] * g[1]) * ds
        Lh -= (tau[0, 0] * nhat[1] * g[0] + tau[1, 1] * nhat[0] * g[1]) * ds
    else:
        raise ValueError('Hessian reconstruction method ``%s`` not recognised' % method)

    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters=params)
    H_solv.solve()

    return H


def computeSteadyMetric(mesh, V, H, sol, hmin=0.005, hmax=0.1, a=100., normalise='lp', p=2, num=1000., ieps=1000.):
    """
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function ``computeSteadyMetric``, from 
    ``adapt.py``, 2016.

    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param H: reconstructed Hessian, usually chosen to be associated with ``sol``.
    :param sol: P1 solution field defined on ``mesh``.
    :param hmin: minimum tolerated side-lengths.
    :param hmax: maximum tolerated side-lengths.
    :param a: maximum tolerated aspect ratio.
    :param normalise: mode of normalisation; either a manual rescaling ('manual') or an Lp approach ('Lp').
    :param p: norm order in the Lp normalisation approach, where ``p => 1`` and ``p = infty`` is an option.
    :param num: target number of vertices, in the case of Lp normalisation.
    :param ieps: inverse of the target error, in the case of manual normalisation.
    :return: steady metric associated with Hessian H.
    """

    ia2 = 1. / pow(a, 2)  # Inverse square aspect ratio
    ihmin2 = 1. / pow(hmin, 2)  # Inverse square minimal side-length
    ihmax2 = 1. / pow(hmax, 2)  # Inverse square maximal side-length
    M = Function(V)

    if normalise == 'manual':
        for i in range(mesh.topology.num_vertices()):
            sol_min = 1e-3  # Minimum tolerated value for the solution field

            # Generate local Hessian:
            H_loc = H.dat.data[i] * ieps / (max(np.sqrt(assemble(sol * sol * dx)), sol_min))  # To avoid round-off error
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs and truncate eigenvalues:
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
            lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)

            # Reconstruct edited Hessian:
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]

    elif normalise == 'lp':
        detH = Function(FunctionSpace(mesh, 'CG', 1))
        for i in range(mesh.topology.num_vertices()):
            # Generate local Hessian:
            H_loc = H.dat.data[i]
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs of Hessian and truncate eigenvalues:
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = max(abs(lam[0]), 1e-10)  # \ To avoid round-off error
            lam2 = max(abs(lam[1]), 1e-10)  # /
            det = lam1 * lam2

            # Reconstruct edited Hessian and rescale:
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
            M.dat.data[i] *= pow(det, -1. / (2 * p + 2))
            detH.dat.data[i] = pow(det, p / (2. * p + 2))

        detH_integral = assemble(detH * dx)
        M *= num / detH_integral  # Scale by the target number of vertices
        for i in range(mesh.topology.num_vertices()):
            # Find eigenpairs of metric and truncate eigenvalues:
            lam, v = la.eig(M.dat.data[i])
            v1, v2 = v[0], v[1]
            lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
            lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)

            # Reconstruct edited Hessian:
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    else:
        raise ValueError('Normalisation method ``%s`` not recognised.' % normalise)

    return M


def isotropicMetric(V, f, bdy=False, op=options.Options()):
    """
    :param V: tensor function space on which metric will be defined.
    :param f: (scalar) function to adapt to.
    :param bdy: toggle boundary metric.
    :param op: Options class object providing min/max cell size values.
    :return: isotropic metric corresponding to the scalar function.
    """
    hmin2 = pow(op.hmin, 2)
    hmax2 = pow(op.hmax, 2)
    M = Function(V)
    try:
        assert(len(M.dat.data[0]) == len(f.dat.data))
    except:
        raise NotImplementedError('CG Tensor field and DG scalar field are currently at odds.')
    for i in DirichletBC(V, 0, 'on_boundary').nodes if bdy else range(len(f.dat.data)):
        if2 = 1. / max(hmin2, min(pow(f.dat.data[i], 2), hmax2))
        M.dat.data[i][0, 0] = if2
        M.dat.data[i][1, 1] = if2
    return M


def localMetricIntersection(M1, M2):
    """
    Intersect two metrics (i.e. two 2x2 matrices).

    :param M1: first metric to be intersected.
    :param M2: second metric to be intersected.
    :return: intersection of metrics M1 and M2.
    """
    sqM1 = sla.sqrtm(M1)
    sqiM1 = la.inv(sqM1)    # Note inverse and square root commute whenever both are defined
    lam, v = la.eig(np.transpose(sqiM1) * M2 * sqiM1)
    return np.transpose(sqM1) * v * [[max(lam[0], 1), 0], [0, max(lam[1], 1)]] * np.transpose(v) * sqM1


def metricGradation(mesh, metric, beta=1.4, isotropic=False):
    """
    Perform anisotropic metric gradation in the method described in Alauzet 2010, using linear interpolation. Python
    code based on Nicolas Barral's function ``DMPlexMetricGradation2d_Internal`` in ``plex-metGradation.c``, 2017.

    :param mesh: current mesh on which variables are defined.
    :param metric: metric to be gradated.
    :param beta: scale factor used.
    :param isotropic: specify whether isotropic or anisotropic mesh adaptivity is being used.
    :return: gradated ``metric``.
    """

    # Get vertices and edges of mesh:
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    numVer = vEnd - vStart
    eStart, eEnd = plex.getDepthStratum(1)
    xy = mesh.coordinates.dat.data

    # Establish arrays for storage:
    v12 = np.zeros(2)
    v21 = np.zeros(2)
    grownMet1 = np.zeros((2, 2))  # TODO: work only with the upper triangular part for speed
    grownMet2 = np.zeros((2, 2))
    M = metric.dat.data

    # Create a list of tags for vertices:
    verTag = np.zeros(numVer)
    for v in range(numVer):
        verTag[v] = 1
    correction = True
    i = 0
    ln_beta = np.log(beta)

    while correction & (i < 500):
        i += 1
        correction = False

        # Loop over edges of mesh:
        for e in range(eStart, eEnd):
            cone = plex.getCone(e)  # Get vertices associated with edge e
            iVer1 = cone[0] - vStart  # Vertex 1 index
            iVer2 = cone[1] - vStart  # Vertex 2 index

            if (verTag[iVer1] < i) & (verTag[iVer2] < i):
                continue

            # Assemble local metrics:
            met1 = M[iVer1]
            met2 = M[iVer2]

            # Calculate edge lengths and scale factor:
            v12[0] = xy[iVer2][0] - xy[iVer1][0]
            v12[1] = xy[iVer2][1] - xy[iVer1][1]
            v21[0] = - v12[0]
            v21[1] = - v12[1]

            if isotropic:
                redMet1 = np.zeros((2, 2))
                redMet2 = np.zeros((2, 2))
                ih12 = 1. / met1[0, 0]
                ih21 = 1. / met2[0, 0]
                eta2_12 = 1. / pow(1 + np.dot(v12) * ih12 * ln_beta, 2)
                eta2_21 = 1. / pow(1 + np.dot(v21) * ih21 * ln_beta, 2)
                for j in range(2):
                    redMet1[j, j] = eta2_12 * met2[j, j]
                    redMet2[j, j] = eta2_21 * met1[j, j]
            else:
                edgLen1 = symmetricProduct(met1, v12)
                edgLen2 = symmetricProduct(met2, v21)
                eta2_12 = 1. / pow(1 + edgLen1 * ln_beta, 2)
                eta2_21 = 1. / pow(1 + edgLen2 * ln_beta, 2)

                # Scale to get 'grown' metric:
                for j in range(2):
                    for k in range(2):
                        grownMet1[j, k] = eta2_12 * met1[j, k]
                        grownMet2[j, k] = eta2_21 * met2[j, k]

                # Intersect metric with grown metric to get reduced metric:
                redMet1 = localMetricIntersection(met1, grownMet2)
                redMet2 = localMetricIntersection(met2, grownMet1)

            # Calculate difference in order to ascertain whether the metric is modified:
            diff = np.abs(met1[0, 0] - redMet1[0, 0]) + np.abs(met1[0, 1] - redMet1[0, 1]) \
                   + np.abs(met1[1, 1] - redMet1[1, 1])
            diff /= (np.abs(met1[0, 0]) + np.abs(met1[0, 1]) + np.abs(met1[1, 1]))
            if diff > 1e-3:
                M[iVer1][0, 0] = redMet1[0, 0]
                M[iVer1][0, 1] = redMet1[0, 1]
                M[iVer1][1, 0] = redMet1[1, 0]
                M[iVer1][1, 1] = redMet1[1, 1]
                verTag[iVer1] = i + 1
                correction = True

            # Repeat above process:
            diff = np.abs(met2[0, 0] - redMet2[0, 0]) + np.abs(met2[0, 1] - redMet2[0, 1]) \
                   + np.abs(met2[1, 1] - redMet2[1, 1])
            diff /= (np.abs(met2[0, 0]) + np.abs(met2[0, 1]) + np.abs(met2[1, 1]))
            if diff > 1e-3:
                M[iVer2][0, 0] = redMet2[0, 0]
                M[iVer2][0, 1] = redMet2[0, 1]
                M[iVer2][1, 0] = redMet2[1, 0]
                M[iVer2][1, 1] = redMet2[1, 1]
                verTag[iVer2] = i + 1
                correction = True

    metric.dat.data[:] = M


def metricIntersection(mesh, V, M1, M2, bdy=False):
    """
    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on current mesh.
    :param M1: first metric to be intersected.
    :param M2: second metric to be intersected.
    :param bdy: when True, intersection with M2 only contributes on the domain boundary.
    :return: intersection of metrics M1 and M2.
    """
    M12 = Function(V)
    for i in DirichletBC(V, 0, 'on_boundary').nodes if bdy else range(mesh.topology.num_vertices()):
        M = M1.dat.data[i]
        iM = la.inv(M)
        lam, v = la.eig(np.transpose(sla.sqrtm(iM)) * M2.dat.data[i] * sla.sqrtm(iM))
        M12.dat.data[i] = v * [[max(lam[0], 1), 0], [0, max(lam[1], 1)]] * np.transpose(v)
        M12.dat.data[i] = np.transpose(sla.sqrtm(M)) * M12.dat.data[i] * sla.sqrtm(M)
    return M12


def symmetricProduct(A, b):
    """
    :param A: symmetric, 2x2 matrix.
    :param b: 2-vector.
    :return: product b^T * A * b.
    """
    return b[0] * A[0, 0] * b[0] + 2 * b[0] * A[0, 1] * b[1] + b[1] * A[1, 1] * b[1]


def meshStats(mesh):
    """
    :param mesh: current mesh.
    :return: number of cells and vertices on the mesh.
    """
    plex = mesh._plex
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    return cEnd - cStart, vEnd - vStart
