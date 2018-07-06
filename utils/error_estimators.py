from thetis import *


__all__ = ["sw_strong_residual", "interelement_term", "explicit_error", "flux_jump_error"]


def interelement_term(v, n=None):
    """
    :arg v: Function to be averaged over element boundaries.
    :param n: FacetNormal
    :return: averaged jump discontinuity over element boundary.
    """
    if n == None:
        n = FacetNormal(v.function_space().mesh())
    v = as_ufl(v)
    if len(v.ufl_shape) == 0:
        return 0.5 * (v('+') * n('+') - v('-') * n('-'))
    else:
        return 0.5 * (dot(v('+'), n('+')) - dot(v('-'), n('-')))


def explicit_error(mesh, res_int, res_bdy):
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

    :arg mesh: mesh upon which problem is defined.
    :arg res_int: residual calculated on element interiors.
    :arg res_bdy: residual calculated on element boundaries.
    :return: explicit error estimator. 
    """
    v = TestFunction(FunctionSpace(mesh, "DG", 0))
    h = CellSize(mesh)

    return assemble(v * (inner(res_int, res_int) + inner(res_bdy, res_bdy) / sqrt(h)) * dx)


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


# TODO: Integrate the below into Thetis

def sw_strong_residual(solver_obj, UV_new, ELEV_new, UV_old, ELEV_old, Ve=None, op=TohokuOptions()):
    """
    Construct the strong residual for the semi-discrete shallow water equations at the current timestep,
    using Crank-Nicolson timestepping.

    :param op: option parameters object.
    :return: two components of strong residual on element interiors, along with the element boundary residual.
    """

    # Collect fields and parameters
    nu = solver_obj.fields.get('viscosity_h')
    Dt = Constant(solver_obj.options.timestep)
    g = physical_constants['g_grav']

    # Enrich FE space (if appropriate)
    if op.order_increase:
        uv_old, elev_old = Function(Ve).split()
        uv_new, elev_new = Function(Ve).split()
        uv_old.interpolate(UV_old)
        uv_new.interpolate(UV_new)
        elev_old.interpolate(ELEV_old)
        elev_new.interpolate(ELEV_new)
        b = solver_obj.fields.bathymetry_2d
        f = solver_obj.options.coriolis_frequency
    else:
        uv_old = UV_old
        uv_new = UV_new
        elev_old = ELEV_old
        elev_new = ELEV_new
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

    # Element boundary residual
    mesh = uv_old.function_space().mesh()
    v = TestFunction(FunctionSpace(mesh, "DG", 0))
    n = FacetNormal(mesh)
    # bres_u1 = assemble(jump(Constant(0.5) * g * v * elev_2d, n=n[0]) * dS)
    # bres_u2 = assemble(jump(Constant(0.5) * g * v * elev_2d, n=n[1]) * dS)
    # bres_u = assemble(jump(Constant(0.5) * g * v * elev_2d, n=n) * dS)
    bres_u = Function(VectorFunctionSpace(mesh, "DG", 1))       # TODO: Fix this (Can't integrate vector field)
    bres_e = assemble(jump(Constant(0.5) * v * H * uv_2d, n=n) * dS)    # This gives a scalar P0 field

    return res_u, res_e, bres_u, bres_e
