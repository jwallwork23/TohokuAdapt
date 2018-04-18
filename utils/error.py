from thetis import *


from .interpolation import mixedPairInterp


__all__ = ["explicitErrorEstimator", "fluxJumpError"]


def explicitErrorEstimator(q, residual, b, v, maxBathy=False):
    """
    Estimate error locally using an a posteriori error indicator.
    
    :arg q: primal approximation at current timestep.
    :arg residual: approximation of residual for primal equations.
    :arg b: bathymetry profile.
    :arg v: P0 test function over the same function space.
    :param maxBathy: apply bound on bathymetry.
    :return: field of local error indicators.
    """
    V = residual.function_space()
    m = len(V.dof_count)
    mesh = V.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    b0 = Constant(max(b.dat.data)) if maxBathy else Function(V.sub(1)).interpolate(b)

    # Compute element residual term
    resTerm = assemble(v * h * h * inner(residual, residual) * dx) if m == 1 else \
        assemble(v * h * h * sum([inner(residual.split()[k], residual.split()[k]) for k in range(m)]) * dx)

    # Compute boundary residual term on fine mesh (if necessary)
    if q.function_space().mesh() != mesh:
        q = mixedPairInterp(mesh, V, q)[0]
        u, eta = q.split()

    j0 = assemble(dot(v * grad(u[0]), n) * ds)
    j1 = assemble(dot(v * grad(u[1]), n) * ds)
    j2 = assemble(jump(v * b0 * u, n=n) * dS)
    jumpTerm = assemble(v * h * (j0 * j0 + j1 * j1 + j2 * j2) * dx)

    return assemble(sqrt(resTerm + jumpTerm))


def fluxJumpError(q, v):
    """
    Estimate error locally by flux jump.

    :arg q: primal approximation at current timestep.
    :arg v: P0 test function over the same function space.
    :return: field of local error indicators.
    """
    V = q.function_space()
    mesh = V.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    uh, etah = q.split()
    j0 = assemble(jump(v * grad(uh[0]), n=n) * dS)
    j1 = assemble(jump(v * grad(uh[1]), n=n) * dS)
    j2 = assemble(jump(v * grad(etah), n=n) * dS)

    return assemble(v * h * (j0 * j0 + j1 * j1 + j2 * j2) * dx)
