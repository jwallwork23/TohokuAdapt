from thetis import *


__all__ = ["explicit_error", "difference_quotient_estimator"]


def ad_interior_residual(solver_obj):   # TODO: Integrate this into Thetis
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

    # Create P0 TestFunction, scaled to take value 1 in each cell. This has the effect of conserving mass upon
    # premultiplying piecewise constant and piecewise linear functions.
    mesh = solver_obj.mesh2d
    P0 = FunctionSpace(mesh, "DG", 0)
    v = Constant(mesh.num_cells()) * TestFunction(P0)  # Scaled to take value 1 in each cell
    n = FacetNormal(mesh)

    # Element boundary residual
    bres_u1 = Function(P0).interpolate(assemble(jump(Constant(0.5) * g * v * elev_2d, n=n[0]) * dS))
    bres_u2 = Function(P0).interpolate(assemble(jump(Constant(0.5) * g * v * elev_2d, n=n[1]) * dS))
    bres_e = Function(P0).interpolate(assemble(jump(Constant(0.5) * v * H * uv_2d, n=n) * dS))

    return bres_u1, bres_u2, bres_e


def ad_boundary_residual(solver_obj, dual_new=None, dual_old=None):     # TODO: Account for other timestepping schemes
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

    # Create P0 TestFunction, scaled to take value 1 in each cell. This has the effect of conserving mass upon
    # premultiplying piecewise constant and piecewise linear functions.
    mesh = solver_obj.mesh2d
    P0 = FunctionSpace(mesh, "DG", 0)
    v = Constant(mesh.num_cells()) * TestFunction(P0)
    n = FacetNormal(mesh)

    return Function(P0).interpolate(assemble(jump(Constant(-1.) * v * grad(tracer_2d), n=n) * dS))


def difference_quotient_estimator(solver_obj, explicit_term, dual, dual_, divide_by_cell_size=False):
    """
    Difference quotient approximation to the dual weighted residual as described on pp.41-42 of 
    [Becker and Rannacher, 2001].
    
    :param solver_obj: Thetis solver object.
    :param explicit_term: explicit error estimator as calculated by `ExplicitErrorCallback`.
    :param dual: adjoint solution at current timestep.
    :param dual_: adjoint solution at previous timestep.
    :param divide_by_cell_size: optionally divide estimator by cell size.
    """

    # Create P0 TestFunction, scaled to take value 1 in each cell. This has the effect of conserving mass upon
    # premultiplying piecewise constant and piecewise linear functions.
    mesh = solver_obj.mesh2d
    P0 = FunctionSpace(mesh, "DG", 0)
    v = Constant(mesh.num_cells()) * TestFunction(P0)

    if solver_obj.options.tracer_only:
        b_res = ad_boundary_residual(solver_obj)
        adjoint_term = b_res * b_res
    else:
        bres0_a, bres1_a, bres2_a = sw_boundary_residual(solver_obj, dual, dual_)
        adjoint_term = bres0_a * bres0_a + bres1_a * bres1_a + bres2_a * bres2_a
    estimator_loc = v * explicit_term * adjoint_term
    if divide_by_cell_size:
        estimator_loc /= sqrt(CellSize(mesh))
    dq = Function(P0)
    dq.interpolate(assemble(v * sqrt(assemble(estimator_loc * dx)) * dx))
    # print("Difference quotient error esimate = %.4e" % norm(dq))

    return dq
