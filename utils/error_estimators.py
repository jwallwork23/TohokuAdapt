from thetis import *


__all__ = ["explicit_error", "local_norm", "difference_quotient_estimator"]


def local_norm(f, norm_type='L2'):
    """
    Calculate the `norm_type`-norm of `f` separately on each element of the mesh.
    """
    typ = norm_type.lower()
    mesh = f.function_space().mesh()
    v = TestFunction(FunctionSpace(mesh, "DG", 0))

    if isinstance(f, FiredrakeFunction):
        if typ == 'l2':
            form = v * inner(f, f) * dx
        elif typ == 'h1':
            form = v * (inner(f, f) * dx + inner(grad(f), grad(f))) * dx
        elif typ == "hdiv":
            form = v * (inner(f, f) * dx + div(f) * div(f)) * dx
        elif typ == "hcurl":
            form = v * (inner(f, f) * dx + inner(curl(f), curl(f))) * dx
        else:
            raise RuntimeError("Unknown norm type '%s'" % norm_type)
    else:
        if typ == 'l2':
            form = v * sum(inner(fi, fi) for fi in f) * dx
        elif typ == 'h1':
            form = v * sum(inner(fi, fi) * dx + inner(grad(fi), grad(fi)) for fi in f) * dx
        elif typ == "hdiv":
            form = v * sum(inner(fi, fi) * dx + div(fi) * div(fi) for fi in f) * dx
        elif typ == "hcurl":
            form = v * sum(inner(fi, fi) * dx + inner(curl(fi), curl(fi)) for fi in f) * dx
        else:
            raise RuntimeError("Unknown norm type '%s'" % norm_type)
    return sqrt(assemble(form))


def local_edge_norm(f, mesh=None, flux_jump=False):
    """
    Integrates `f` over all interior edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    edge_function = Function(P0)
    v = TestFunction(P0)
    if flux_jump:
        n = FacetNormal(mesh)
        edge_function.interpolate(assemble(jump(f, n) * (v('+') + v('-')) * dS))
    else:
        edge_function.interpolate(assemble((f('+') * v('+') + f('-') * v('-')) * dS))

    return edge_function


def local_boundary_norm(f, mesh=None):
    """
    Integrates `f` over all exterior edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    boundary_function = Function(P0)
    v = TestFunction(P0)
    boundary_function.interpolate(assemble(f * v * ds))

    return boundary_function


def sw_boundary_residual(solver_obj, dual_new=None, dual_old=None):     # TODO: Generalise to other BCs and timesteps
    """
    Evaluate strong residual across element boundaries for (DG) shallow water for the tsunami application (where free
    slip boundaries are considered). To consider adjoint variables, input these as `dual_new` and `dual_old`.
    """

    # Collect fields and parameters
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
    # v = Constant(mesh.num_cells()) * TestFunction(P0)  # Scaled to take value 1 in each cell
    v = TestFunction(P0)
    n = FacetNormal(mesh)

    # Construct residuals across edges
    bres_u1 = Function(P0)	# No contribution
    bres_u2 = Function(P0)	# No contribution
    edge_contribution = Constant(0.5) * jump(-H * uv_2d, n=n) * (v('+') + v('-')) * dS
    boundary_contribution = -dot(H * uv_2d, n) * v * ds
    bres_e = Function(P0).interpolate(assemble(edge_contribution + boundary_contribution))

    return bres_u1, bres_u2, bres_e


def ad_boundary_residual(solver_obj, dual_new=None, dual_old=None):     # TODO: Generalise to other BCs and timesteps
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
    # v = Constant(mesh.num_cells()) * TestFunction(P0)
    v = TestFunction(P0)
    n = FacetNormal(mesh)

    # Construct residuals across edges
    edge_contribution = Constant(0.5) * jump(-grad(tracer_2d), n=n) * (v('+') + v('-')) * dS
    boundary_contribution = -dot(grad(tracer_2d), n) * v * ds	# TODO: This is not applied on all boundaries

    return Function(P0).interpolate(assemble(edge_contribution + boundary_contribution))


def difference_quotient_estimator(solver_obj, explicit_term, dual, dual_old, divide_by_cell_size=True):
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
    # v = Constant(mesh.num_cells()) * TestFunction(P0)
    v = TestFunction(P0)

    if solver_obj.options.tracer_only:
        b_res = ad_boundary_residual(solver_obj)
        adjoint_term = b_res * b_res
    else:
        bres0_a, bres1_a, bres2_a = sw_boundary_residual(solver_obj, dual, dual_old)
        adjoint_term = bres0_a * bres0_a + bres1_a * bres1_a + bres2_a * bres2_a
    estimator_loc = v * explicit_term * adjoint_term
    if divide_by_cell_size:
        estimator_loc /= CellSize(mesh)
    dq = Function(P0)
    dq.interpolate(assemble(v * sqrt(assemble(estimator_loc * dx)) * dx))
    print("Difference quotient error estimate = %.4e" % norm(dq))

    return dq
