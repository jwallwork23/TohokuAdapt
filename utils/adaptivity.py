from thetis import *

import numpy as np
import numpy
from numpy import linalg as la
from scipy import linalg as sla

from .options import Options


__all__ = ["constructGradient", "constructHessian", "steadyMetric", "isotropicMetric", "isoP2", "anisoRefine",
           "metricGradation", "localMetricIntersection", "metricIntersection", "metricConvexCombination",
           "symmetricProduct", "pointwiseMax", "metricComplexity", "normaliseIndicator"]


def constructGradient(f):
    """
    Assuming the function `f` is P1 (piecewise linear and continuous), direct differentiation will give a gradient which
    is P0 (piecewise constant and discontinuous). Since we would prefer a smooth gradient, an L2 projection gradient 
    recovery technique is performed, which makes use of the Cl\'ement interpolation operator.

    :arg f: (scalar) P1 solution field.
    :return: reconstructed gradient associated with `f`.
    """
    W = VectorFunctionSpace(f.function_space().mesh(), "CG", 1)
    g = Function(W)
    psi = TestFunction(W)
    Lg = (inner(g, psi) - inner(grad(f), psi)) * dx
    NonlinearVariationalSolver(NonlinearVariationalProblem(Lg, g), solver_parameters={'snes_rtol': 1e8,
                                                                                      'ksp_rtol': 1e-5,
                                                                                      'ksp_gmres_restart': 20,
                                                                                      'pc_type': 'sor'}).solve()
    return g


def constructHessian(f, g=None, op=Options()):
    """
    Assuming the smooth solution field has been approximated by a function `f` which is P1, all second derivative
    information has been lost. As such, the Hessian of `f` cannot be directly computed. We provide two means of
    recovering it, as follows.
    
    (1) "Integration by parts" ('parts'):
    This involves solving the PDE $H = \nabla^T\nabla f$ in the weak sense. Code is based on the Monge-Amp\`ere 
    tutorial provided on the Firedrake website: https://firedrakeproject.org/demos/ma-demo.py.html.
    
    (2) "Double L2 projection" ('dL2'):
    This involves two applications of the L2 projection operator given by `computeGradient`, above.

    :arg f: P1 solution field.
    :kwarg g: gradient (if already computed).
    :param op: Options class object providing min/max cell size values.
    :return: reconstructed Hessian associated with ``f``.
    """
    mesh = f.function_space().mesh()
    V = TensorFunctionSpace(mesh, "CG", 1)
    H = Function(V)
    tau = TestFunction(V)
    nhat = FacetNormal(mesh)  # Normal vector
    if op.hessMeth == 'parts':
        Lh = (inner(tau, H) + inner(div(tau), grad(f))) * dx
        Lh -= (tau[0, 1] * nhat[1] * f.dx(0) + tau[1, 0] * nhat[0] * f.dx(1)) * ds
        Lh -= (tau[0, 0] * nhat[1] * f.dx(0) + tau[1, 1] * nhat[0] * f.dx(1)) * ds  # Term not in Firedrake tutorial
    elif op.hessMeth == 'dL2':
        if g is None:
            g = constructGradient(f)
        Lh = (inner(tau, H) + inner(div(tau), g)) * dx
        Lh -= (tau[0, 1] * nhat[1] * g[0] + tau[1, 0] * nhat[0] * g[1]) * ds
        Lh -= (tau[0, 0] * nhat[1] * g[0] + tau[1, 1] * nhat[0] * g[1]) * ds
    H_prob = NonlinearVariationalProblem(Lh, H)
    NonlinearVariationalSolver(H_prob, solver_parameters={'snes_rtol': 1e8,
                                                          'ksp_rtol': 1e-5,
                                                          'ksp_gmres_restart': 20,
                                                          'pc_type': 'sor'}).solve()
    return H


def steadyMetric(f, H=None, op=Options()):
    """
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function ``computeSteadyMetric``, from 
    ``adapt.py``, 2016.

    :arg f: P1 solution field.
    :arg H: reconstructed Hessian associated with `f` (if already computed).
    :param op: Options class object providing min/max cell size values.
    :return: steady metric associated with Hessian H.
    """
    if H is None:
        H = constructHessian(f, op=op)
    V = H.function_space()
    mesh = V.mesh()

    ia2 = 1. / pow(op.maxAnisotropy, 2)     # Inverse square max aspect ratio
    ihmin2 = 1. / pow(op.hmin, 2)           # Inverse square minimal side-length
    ihmax2 = 1. / pow(op.hmax, 2)           # Inverse square maximal side-length
    M = Function(V)

    if op.normalisation == 'manual':
        f_min = 1e-6  # Minimum tolerated value for the solution field

        for i in range(mesh.topology.num_vertices()):

            # Generate local Hessian
            H_loc = H.dat.data[i] * op.nVerT / max(np.sqrt(assemble(f * f * dx)), f_min)    # Avoid round-off error
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
            if (lam[0] >= 0.9999 * ihmin2) or (lam[1] >= 0.9999 * ihmin2):
                print("WARNING: minimum element size reached as %.2e" % np.sqrt(min(1./lam[0], 1./lam[1])))

            # Reconstruct edited Hessian
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    else:
        detH = Function(FunctionSpace(mesh, "CG", 1))
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
            M.dat.data[i] *= pow(det, -1. / (2 * op.normOrder + 2))
            detH.dat.data[i] = pow(det, op.normOrder / (2. * op.normOrder + 2))

        M *= op.nVerT / assemble(detH * dx)    # Scale by the target number of vertices and Hessian complexity
        for i in range(mesh.topology.num_vertices()):
            # Find eigenpairs of metric and truncate eigenvalues
            lam, v = la.eig(M.dat.data[i])
            v1, v2 = v[0], v[1]
            lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
            lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)
            if (lam[0] >= 0.9999 * ihmin2) or (lam[1] >= 0.9999 * ihmin2):
                print("WARNING: minimum element size reached as %.2e" % np.sqrt(min(1./lam[0], 1./lam[1])))

            # Reconstruct edited Hessian
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    return M


def normaliseIndicator(f, op=Options()):        # TODO: What if metric is constant?
    """
    Normalise error indicator `f` using procedure defined by `op`.
    
    :arg f: error indicator to normalise.
    :param op: option parameters object.
    :return: normalised indicator.
    """
    f.dat.data[:] = np.abs(f.dat.data)
    if len(f.ufl_element().value_shape()) == 0:
        gnorm = max(np.abs(assemble(f * dx)), op.minNorm)           # NOTE this changes in 3D case
    else:
        gnorm = max(assemble(sqrt(inner(f, f)) * dx), op.minNorm)   # Equivalent thresholded metric complexity
    scaleFactor = min(op.nVerT / gnorm, op.maxScaling)              # Cap error estimate, also computational cost
    if scaleFactor == op.maxScaling:
        print("WARNING: maximum scaling for error estimator reached as %.2e" % (op.nVerT / gnorm))
    # print("#### DEBUG: Complexity = %.4e" % gnorm)
    f.dat.data[:] = np.abs(f.dat.data) * scaleFactor

    return f


def isotropicMetric(f, bdy=None, invert=True, op=Options()):
    """
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.
    
    :arg f: function to adapt to.
    :param bdy: specify domain boundary to compute metric on.
    :param invert: when True, the inverse square of field `f` is considered, as in anisotropic mesh adaptivity.
    :param op: Options class object providing min/max cell size values.
    :return: isotropic metric corresponding to `f`.
    """
    hmin2 = pow(op.hmin, 2)
    hmax2 = pow(op.hmax, 2)
    scalar = len(f.ufl_element().value_shape()) == 0
    mesh = f.function_space().mesh()
    g = Function(FunctionSpace(mesh, "CG", 1) if scalar else VectorFunctionSpace(mesh, "CG", 1))
    if (f.ufl_element().family() == 'Lagrange') & (f.ufl_element().degree() == 1):
        g.assign(f)
    else:
        g.interpolate(f)

    # Establish metric
    V = TensorFunctionSpace(mesh, "CG", 1)
    M = Function(V)
    for i in DirichletBC(V, 0, bdy).nodes if bdy is not None else range(len(g.dat.data)):
        if scalar:
            if invert:
                alpha = 1. / max(hmin2, min(pow(g.dat.data[i], 2), hmax2))
            else:
                alpha = max(1. / hmax2, min(g.dat.data[i], 1. / hmin2))
            beta = alpha
        else:
            if invert:
                alpha = 1. / max(hmin2, min(pow(g.dat.data[i, 0], 2), hmax2))
                beta = 1. / max(hmin2, min(pow(g.dat.data[i, 1], 2), hmax2))
            else:
                alpha = max(1. / hmax2, min(g.dat.data[i, 0], 1. / hmin2))
                beta = max(1. / hmax2, min(g.dat.data[i, 1], 1. / hmin2))
        M.dat.data[i][0, 0] = alpha
        M.dat.data[i][1, 1] = beta

        if (alpha >= 0.9999 / hmin2) or (beta >= 0.9999 / hmin2):
            print("WARNING: minimum element size reached as %.2e" % np.sqrt(min(1./alpha, 1./beta)))

    return M


def isoP2(mesh):
    """
    Uniformly refine a mesh (in each canonical direction) using an iso-P2 refinement. That is, nodes of a quadratic 
    element on the initial mesh become vertices of the new mesh.
    """
    return MeshHierarchy(mesh, 1).__getitem__(1)


def anisoRefine(M, direction=0):
    """
    (Anisotropically) refine a mesh (or, more precisely, the metric field `M` associated with a mesh) in such a way as 
    to approximately half the element size in a canonical direction (x- or y-), by scaling of the corresponding 
    eigenvalue.
       
    :param M: metric to refine.
    :param direction: 0 or 1, corresponding to x- or y-direction, respectively.
    :return: anisotropically refined metric.
    """
    for i in range(len(M.dat.data)):
        lam, v = la.eig(M.dat.data[i])
        v1, v2 = v[0], v[1]
        lam[direction] *= 4
        M.dat.data[i][0, 0] = lam[0] * v1[0] * v1[0] + lam[1] * v2[0] * v2[0]
        M.dat.data[i][0, 1] = lam[0] * v1[0] * v1[1] + lam[1] * v2[0] * v2[1]
        M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
        M.dat.data[i][1, 1] = lam[0] * v1[1] * v1[1] + lam[1] * v2[1] * v2[1]
    return M


def metricGradation(M, op=Options()):   # TODO: Implement this in pyop2
    """
    Perform anisotropic metric gradation in the method described in Alauzet 2010, using linear interpolation. Python
    code found here is based on the C code of Nicolas Barral's function ``DMPlexMetricGradation2d_Internal``, found in 
    ``plex-metGradation.c``, 2017.

    :arg M: metric to be gradated.
    :param op: Options class object providing parameter values.
    :return: gradated metric.
    """
    ln_beta = np.log(op.maxGrowth)

    # Get vertices and edges of mesh
    V = M.function_space()
    M_grad = Function(V).assign(M)
    mesh = V.mesh()
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)  # Vertices
    eStart, eEnd = plex.getDepthStratum(1)  # Edges
    numVer = vEnd - vStart
    xy = mesh.coordinates.dat.data

    # Establish arrays for storage and a list of tags for vertices
    v12 = np.zeros(2)
    v21 = np.zeros(2)   # Could work only with the upper triangular part for speed
    verTag = np.zeros(numVer) + 1
    correction = True
    i = 0

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
            met1 = M_grad.dat.data[iVer1]
            met2 = M_grad.dat.data[iVer2]
            v12[0] = xy[iVer2][0] - xy[iVer1][0]
            v12[1] = xy[iVer2][1] - xy[iVer1][1]
            v21[0] = - v12[0]
            v21[1] = - v12[1]

            # if op.iso:
            #     eta2_12 = 1. / pow(1 + (v12[0] * v12[0] + v12[1] * v12[1]) * ln_beta / met1[0, 0], 2)
            #     eta2_21 = 1. / pow(1 + (v21[0] * v21[0] + v21[1] * v21[1]) * ln_beta / met2[0, 0], 2)
            #     # print('#### metricGradation DEBUG: 1,1 entries ', met1[0, 0], met2[0, 0])
            #     print('#### metricGradation DEBUG: scale factors', eta2_12, eta2_21)
            #     redMet1 = eta2_21 * met2
            #     redMet2 = eta2_12 * met1
            # else:

            # Intersect metric with a scaled 'grown' metric to get reduced metric
            eta2_12 = 1. / pow(1 + symmetricProduct(met1, v12) * ln_beta, 2)
            eta2_21 = 1. / pow(1 + symmetricProduct(met2, v21) * ln_beta, 2)
            # print('#### metricGradation DEBUG: scale factors', eta2_12, eta2_21)
            # print('#### metricGradation DEBUG: determinants', la.det(met1), la.det(met2))
            redMet1 = localMetricIntersection(met1, eta2_21 * met2)
            redMet2 = localMetricIntersection(met2, eta2_12 * met1)

            # Calculate difference in order to ascertain whether the metric is modified
            diff = np.abs(met1[0, 0] - redMet1[0, 0]) + np.abs(met1[0, 1] - redMet1[0, 1]) \
                   + np.abs(met1[1, 1] - redMet1[1, 1])
            diff /= (np.abs(met1[0, 0]) + np.abs(met1[0, 1]) + np.abs(met1[1, 1]))
            if diff > 1e-3:
                M_grad.dat.data[iVer1][0, 0] = redMet1[0, 0]
                M_grad.dat.data[iVer1][0, 1] = redMet1[0, 1]
                M_grad.dat.data[iVer1][1, 0] = redMet1[1, 0]
                M_grad.dat.data[iVer1][1, 1] = redMet1[1, 1]
                verTag[iVer1] = i+1
                correction = True

            # Repeat above process using other reduced metric
            diff = np.abs(met2[0, 0] - redMet2[0, 0]) + np.abs(met2[0, 1] - redMet2[0, 1]) \
                   + np.abs(met2[1, 1] - redMet2[1, 1])
            diff /= (np.abs(met2[0, 0]) + np.abs(met2[0, 1]) + np.abs(met2[1, 1]))
            if diff > 1e-3:
                M_grad.dat.data[iVer2][0, 0] = redMet2[0, 0]
                M_grad.dat.data[iVer2][0, 1] = redMet2[0, 1]
                M_grad.dat.data[iVer2][1, 0] = redMet2[1, 0]
                M_grad.dat.data[iVer2][1, 1] = redMet2[1, 1]
                verTag[iVer2] = i+1
                correction = True

    return M_grad


def localMetricIntersection(M1, M2):
    """
    Intersect two metrics `M1` and `M2` defined at a particular point in space.
    """
    # print('#### localMetricIntersection DEBUG: attempting to compute sqrtm of matrix with determinant ', la.det(M1))
    sqM1 = sla.sqrtm(M1)
    sqiM1 = la.inv(sqM1)    # Note inverse and square root commute whenever both are defined
    lam, v = la.eig(np.dot(np.transpose(sqiM1), np.dot(M2, sqiM1)))
    M12 = np.dot(v, np.dot([[max(lam[0], 1), 0], [0, max(lam[1], 1)]], np.transpose(v)))
    return np.dot(np.transpose(sqM1), np.dot(M12, sqM1))


def metricIntersection(M1, M2, bdy=None):
    """
    Intersect a metric field, i.e. intersect (globally) over all local metrics.
    
    :arg M1: first metric to be intersected.
    :arg M2: second metric to be intersected.
    :param bdy: specify domain boundary to intersect over.
    :return: intersection of metrics M1 and M2.
    """
    V = M1.function_space()
    assert V == M2.function_space()
    M = Function(V).assign(M1)
    for i in DirichletBC(V, 0, bdy).nodes if bdy is not None else range(V.mesh().num_vertices()):
        M.dat.data[i] = localMetricIntersection(M1.dat.data[i], M2.dat.data[i])
        # print('#### metricIntersection DEBUG: det(Mi) = ', la.det(M1.dat.data[i]))
    return M


def metricConvexCombination(M1, M2, alpha=0.5):
    """
    Alternatively to intersection, pointwise metric information may be combined using a convex combination. Whilst this
    method does not have as clear an interpretation as metric intersection, it has the benefit that the combination may 
    be weighted towards one of the metrics in question.
    
    :arg M1: first metric to be combined.
    :arg M2: second metric to be combined.
    :param alpha: scalar parameter in [0,1].
    :return: convex combination of metrics M1 and M2 with parameter alpha.
    """
    V = M1.function_space()
    assert V == M2.function_space()
    M = Function(V)
    M.dat.data[:] = alpha * M1.dat.data + (1-alpha) * M2.dat.data
    return M


def symmetricProduct(A, b):
    """
    Compute the product of 2-vector `b` with itself, under the scalar product $b^T A b$ defined by the 2x2 matrix `A`.
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
    Take the pointwise maximum (in modulus) of arrays `f` and `g`.
    """
    try:
        assert(len(f.dat.data) == len(g.dat.data))
    except:
        fu = f.function_space().ufl_element()
        gu = g.function_space().ufl_element()
        raise ValueError("Function space mismatch: ", fu.family(), fu.degree(), " vs. ", gu.family(), gu.degree())
    for i in range(len(f.dat.data)):
        if np.abs(g.dat.data[i]) > np.abs(f.dat.data[i]):
            f.dat.data[i] = g.dat.data[i]
    return f


def metricComplexity(M):
    """
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted based thereupon.
    """
    return assemble(sqrt(det(M)) * dx)
