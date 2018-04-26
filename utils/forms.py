from thetis import *
from thetis_adjoint import *
from firedrake import Expression

from .interpolation import mixedPairInterp
from .options import Options


__all__ = ["strongResidualSW", "formsSW", "interelementTerm", "indicator", "explicitErrorEstimator",
           "fluxJumpError", "timestepCoeffs", "timestepScheme"]


def timestepCoeffs(timestepper):    # TODO: Make this format more conventional / delete it eventually
    """
    :arg timestepper: scheme of choice.
    :return: coefficients for use in scheme.
    """
    if timestepper == 'ExplicitEuler':
        a1 = Constant(0.)
        a2 = Constant(1.)
    elif timestepper == 'ImplicitEuler':
        a1 = Constant(1.)
        a2 = Constant(0.)
    elif timestepper == 'CrankNicolson':
        a1 = Constant(0.5)
        a2 = Constant(0.5)
    else:
        raise NotImplementedError("Timestepping scheme %s not yet considered." % timestepper)

    return a1, a2


def timestepScheme(u, u_, timestepper):
    """
    :arg u: prognostic variable at current timestep. 
    :arg u_: prognostic variable at previous timestep. 
    :arg timestepper: scheme of choice.
    :return: expression for prognostic variable to be used in scheme.
    """
    a1, a2 = timestepCoeffs(timestepper)

    return a1 * u + a2 * u_


def strongResidualSW(q, q_, b, coriolisFreq=None, op=Options()):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param op: parameter holding class.
    :return: strong residual for shallow water equations at current timestep.
    """
    Dt = Constant(op.dt)
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])

    Au = (u - u_) / Dt + op.g * grad(eta) + dot(u, nabla_grad(u))
    Ae = (eta - eta_) / Dt + div((b + eta) * u)
    if coriolisFreq:
        Au += coriolisFreq * as_vector((-u[1], u[0]))

    return Au, Ae


def strongResidual(Ve, solution_old, solver_obj, op=Options()):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.

    :arg Ve: enriched finite element space in which to compute strong residual.
    :arg solution_old: solution tuple for linear shallow water equations at previous timestep.
    :arg solver_obj: thetis solver object containing parameters and fields.
    :param op: parameter holding class.
    :return: strong residual for shallow water equations at current timestep.
    """

    # Collect fields and parameters
    q = Function(Ve).interpolate(solver_obj.fields.solution_2d)
    q_ = Function(Ve).interpolate(solution_old)
    b = Function(FunctionSpace(Ve.mesh(), "CG", 2)).interpolate(solver_obj.fields.bathymetry_2d)
    nu = solver_obj.fields.get('viscosity_h')
    Dt = Constant(solver_obj.options.timestep)
    uv_2d, elev_2d = q.split()
    uv_2d_, elev_2d_ = q_.split()
    H = b * elev_2d

    # Momentum equation
    Au = (uv_2d - uv_2d_) / Dt + Constant(op.g) * grad(elev_2d) # TODO: Other terms to include
    if solver_obj.options.use_nonlinear_equations:
        Au += dot(uv_2d, nabla_grad(uv_2d))
    if solver_obj.options.coriolis_frequency is not None:
        Au += solver_obj.options.coriolis_frequency * as_vector((-uv_2d[1], uv_2d[0]))
    if nu is not None:
        if solver_obj.options.use_grad_depth_viscosity_term:
            Au -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
        if solver_obj.options.use_grad_div_viscosity_term:
            Au -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
        else:
            Au -= div(nu * grad(uv_2d))

    # Continuity equation
    Ae = (elev_2d - elev_2d_) / Dt + div((b + elev_2d) * uv_2d)

    return Au, Ae


def formsSW(q, q_, b, coriolisFreq=None, impermeable=True, op=Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param impermeable: impose Neumann BCs.
    :param op: parameter holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    Dt = Constant(op.dt)
    V = q_.function_space()
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    qt = TestFunction(V)
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])
    a1, a2 = timestepCoeffs(op.timestepper)
    g = op.g

    B = (inner(q, qt)) / Dt * dx + a1 * g * inner(grad(eta), w) * dx        # LHS bilinear form
    B += a1 * (inner(dot(u, nabla_grad(u)), w) + div(eta * u)) * dx
    L = (inner(q_, qt)) / Dt * dx - a2 * g * inner(grad(eta_), w) * dx      # RHS linear functional
    L -= a2 * div(b * u_) * xi * dx                                     # Note: Don't "apply BCs" to linear functional
    L -= a2 * (inner(dot(u_, nabla_grad(u_)), w) + div(eta_ * u_)) * dx
    if impermeable:
        B -= a1 * inner(b * u, grad(xi)) * dx
        if V.sub(0).ufl_element().family() != 'Lagrange':
            B += a1 * jump(b * u * xi, n=FacetNormal(V.mesh())) * dS
    else:
        B += a1 * div(b * u) * xi * dx
    if coriolisFreq:
        B += a1 * coriolisFreq * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * coriolisFreq * inner(as_vector((-u_[1], u_[0])), w) * dx

    return B, L


def interelementTerm(v, n=None):
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


def indicator(V, op=Options()):
    """
    :arg V: Function space to use.
    :param op: options parameter class.
    :return: ('Smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    """
    smooth = True if op.mode == 'tohoku' else False

    # Define extent of region A
    xy = op.xy
    if smooth:
        xd = (xy[1] - xy[0]) / 2
        yd = (xy[3] - xy[2]) / 2
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.' \
              % (xy[0], xy[1], xy[2], xy[3], xy[0] + xd, xd, xy[2] + yd, yd)
    else:
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (xy[0], xy[1], xy[2], xy[3])
    iA = Function(V, name="Region of interest").interpolate(Expression(ind))

    return iA


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
