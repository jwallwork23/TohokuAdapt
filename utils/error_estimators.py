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


def sw_interior_residual(solver_obj):
    """
    Evaluate strong residual on element interiors for shallow water.
    """

    # Collect fields and parameters
    nu = solver_obj.fields.get('viscosity_h')
    Dt = Constant(solver_obj.options.timestep)
    g = physical_constants['g_grav']
    uv_old, elev_old = solver_obj.timestepper.solution_old.split()
    uv_new, elev_new = solver_obj.fields.solution_2d.split()
    b = solver_obj.fields.bathymetry_2d
    f = solver_obj.options.coriolis_frequency
    uv_2d = 0.5 * (uv_old + uv_new)         # Use Crank-Nicolson timestepping so that we isolate errors as being
    elev_2d = 0.5 * (elev_old + elev_new)   # related only to the spatial discretisation
    H = b + elev_2d

    # Momentum equation residual on element interiors
    res_u = (uv_new - uv_old) / Dt + g * grad(elev_2d)
    if solver_obj.options.use_nonlinear_equations:
        res_u += dot(uv_2d, nabla_grad(uv_2d))
    if solver_obj.options.coriolis_frequency is not None:
        res_u += f * as_vector((-uv_2d[1], uv_2d[0]))
    if nu is not None:
        if solver_obj.options.use_grad_depth_viscosity_term:
            res_u -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
        if solver_obj.options.use_grad_div_viscosity_term:
            res_u -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
        else:
            res_u -= div(nu * grad(uv_2d))

    # Continuity equation residual on element interiors
    res_e = (elev_new - elev_old) / Dt + div(H * uv_2d)

    return res_u, res_e


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


def sw_boundary_residual(solver_obj, dual_new=None, dual_old=None):
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


def explicit_error(solver_obj):
    r"""
    Estimate error locally using an a posteriori error indicator [Ainsworth & Oden, 1997], given by

    .. math::
        \|\textbf{R}(\textbf{q}_h)\|_{\mathcal{L}_2(K)}
            + h_K^{-1}\|\textbf{r}(\textbf{q}_h)\|_{\mathcal{L}_2(\partial K)},

    where
    :math:`\textbf{q}_h` is the approximation to the PDE solution,
    :math:`\textbf{R}` denotes the strong residual on element interiors,
    :math:`\textbf{r}` denotes the strong residual on element boundaries,
    :math:`h_K` is the size of mesh element `K`.

    :arg solver_obj: Thetis solver object.
    :return: explicit error estimator. 
    """
    mesh = solver_obj.mesh2d
    P0 = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(P0)
    ee = Function(P0)
    h = CellSize(mesh)

    if solver_obj.options.tracer_only:
        res = ad_interior_residual(solver_obj)
        bres = ad_boundary_residual(solver_obj)
        # print("Interior residual norm = %.4e" % norm(res))
        # print("Boundary residual norm = %.4e" % norm(bres))
        ee.interpolate(assemble(v * (res * res + bres * bres / sqrt(h)) * dx))
    else:
        res_u, res_e = sw_interior_residual(solver_obj)
        bres_u1, bres_u2, bres_e = sw_boundary_residual(solver_obj)
        # print("Interior residual norm = %.4e" % assemble((inner(res_u, res_u) + res_e * res_e) * dx))
        # print("Boundary residual norm = %.4e" % assemble((bres_u1 * bres_u1 + bres_u2 * bres_u2 + res_e * res_e) * dx))
        ee.interpolate(assemble(v * (inner(res_u, res_u) + res_e * res_e
                             + (bres_u1 * bres_u1 + bres_u2 * bres_u2 + bres_e * bres_e) / sqrt(h)) * dx))

    return ee


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
