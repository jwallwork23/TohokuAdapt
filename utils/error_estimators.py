from thetis import *


__all__ = ["explicit_error", "flux_jump_error", "difference_quotient_estimator"]


def flux_jump_error(q, v):
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


def ad_interior_residual(solver_obj):
    """
    Evaluate strong residual on element interiors for advection diffusion.
    """
    mu = solver_obj.fields.get('diffusion_h')
    Dt = Constant(solver_obj.options.timestep)
    tracer_old = solver_obj.timestepper.tracer_old.split()
    tracer_new = solver_obj.fields.tracer_2d.split()
    tracer_2d = 0.5 * (tracer_old + tracer_new)
    u = solver_obj.fields.uv_2d

    return (tracer_new - tracer_old) / Dt + dot(u, grad(tracer_2d)) - mu * div(grad(tracer_2d))


def sw_boundary_residual(solver_obj, dual_new=None, dual_old=None):     # TODO: Account for other timestepping schemes
    """
    Evaluate strong residual across element boundaries for (DG) shallow water. To consider adjoint variables, input
    these as `dual_new` and `dual_old`.
    """

    # Collect fields and parameters
    g = physical_constants['g_grav']
    if dual_new is not None and dual_old is not None:
        uv_new, elev_new = dual_new.split()
        uv_old, elev_old = dual_old.split()
    else:
        uv_new, elev_new = solver_obj.fields.solution_2d.split()
        uv_old, elev_old = solver_obj.timestepper.solution_old.split()
    b = solver_obj.fields.bathymetry_2d
    uv_2d = 0.5 * (uv_old + uv_new)         # Use Crank-Nicolson timestepping so that we isolate errors as being
    elev_2d = 0.5 * (elev_old + elev_new)   # related only to the spatial discretisation
    H = b + elev_2d

    # Element boundary residual
    mesh = solver_obj.mesh2d
    P0 = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(P0)
    n = FacetNormal(mesh)
    bres_u1 = Function(P0).interpolate(assemble(jump(Constant(0.5) * g * v * elev_2d, n=n[0]) * dS))
    bres_u2 = Function(P0).interpolate(assemble(jump(Constant(0.5) * g * v * elev_2d, n=n[1]) * dS))
    bres_e = Function(P0).interpolate(assemble(jump(Constant(0.5) * v * H * uv_2d, n=n) * dS))

    return bres_u1, bres_u2, bres_e


def ad_boundary_residual(solver_obj, dual_new=None, dual_old=None):
    """
    Evaluate strong residual across element boundaries for (DG) advection diffusion. To consider adjoint variables, 
    input these as `dual_new` and `dual_old`.
    """

    # Collect fields and parameters

    if dual_new is not None and dual_old is not None:
        tracer_new = dual_new
        tracer_old = dual_old
    else:
        tracer_new = solver_obj.fields.tracer_2d
        tracer_old = solver_obj.timestepper.tracer_old
    tracer_2d = 0.5 * (tracer_old + tracer_new)

    # Element boundary residual
    mesh = solver_obj.mesh2d
    P0 = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(P0)
    n = FacetNormal(mesh)

    return Function(P0).interpolate(assemble(jump(Constant(-1.) * v * grad(tracer_2d), n=n) * dS))


def difference_quotient_estimator(solver_obj, explicit_term, dual, dual_):

    mesh = solver_obj.mesh2d
    h = CellSize(mesh)
    P0 = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(P0)

    if solver_obj.options.tracer_only:
        b_res = ad_boundary_residual(solver_obj)
        adjoint_term = b_res * b_res
    else:
        bres0_a, bres1_a, bres2_a = sw_boundary_residual(solver_obj, dual, dual_)
        adjoint_term = bres0_a * bres0_a + bres1_a * bres1_a + bres2_a * bres2_a
    dq = Function(P0)
    dq.interpolate(assemble(v * explicit_term * adjoint_term / sqrt(h) * dx))
    # print("Explicit error esimate = %.4e" % norm(dq))

    return dq
