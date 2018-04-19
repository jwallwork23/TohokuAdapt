from thetis import *
from thetis_adjoint import *
from firedrake import Expression

from .interpolation import mixedPairInterp
from .options import Options
from .timestepping import timestepScheme, timestepCoeffs


__all__ = ["strongResidualSW", "formsSW", "adjointSW", "weakResidualSW", "interelementTerm", "solutionRW", "indicator",
           "explicitErrorEstimator", "fluxJumpError"]


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
    um = timestepScheme(u, u_, op.timestepper)
    em = timestepScheme(eta, eta_, op.timestepper)

    Au = (u - u_) / Dt + op.g * grad(em)
    Ae = (eta - eta_) / Dt + div(b * um)
    if coriolisFreq:
        Au += coriolisFreq * as_vector((-u[1], u[0]))
    if op.nonlinear:
        Au += dot(u, nabla_grad(u))
        Ae += div(em * um)

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
    L = (inner(q_, qt)) / Dt * dx - a2 * g * inner(grad(eta_), w) * dx      # RHS linear functional
    L -= a2 * div(b * u_) * xi * dx                                     # Note: Don't "apply BCs" to linear functional
    if impermeable:
        B -= a1 * inner(b * u, grad(xi)) * dx
        if V.sub(0).ufl_element().family() != 'Lagrange':
            B += a1 * jump(b * u * xi, n=FacetNormal(V.mesh())) * dS
    else:
        B += a1 * div(b * u) * xi * dx
    if coriolisFreq:
        B += a1 * coriolisFreq * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * coriolisFreq * inner(as_vector((-u_[1], u_[0])), w) * dx
    if op.nonlinear:
        B += a1 * (inner(dot(u, nabla_grad(u)), w) + div(eta * u)) * dx
        L -= a2 * (inner(dot(u_, nabla_grad(u_)), w)  + div(eta_ * u_)) * dx

    return B, L


def adjointSW(l, l_, b, switch=Constant(1.), op=Options()): # TODO: This only works in linear case. What about BCs?
    """
    Semi-discrete (time-discretised) weak form adjoint shallow water equations with no normal flow boundary conditions.

    :arg l: solution tuple for adjoint linear shallow water equations.
    :arg l_: solution tuple for adjoint linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    Dt = Constant(op.dt)
    (lu, le) = (as_vector((l[0], l[1])), l[2])
    (lu_, le_) = (as_vector((l_[0], l_[1])), l_[2])
    lt = TestFunction(l.function_space())
    (w, xi) = (as_vector((lt[0], lt[1])), lt[2])
    a1, a2 = timestepCoeffs(op.timestepper)
    iA = indicator(l.function_space().sub(1), op=op)

    B = ((inner(lu, w) + le * xi) / Dt + a1 * b * inner(grad(le), w) - a1 * op.g * inner(lu, grad(xi))) * dx
    L = ((inner(lu_, w) + le_ * xi) / Dt - a2 * b * inner(grad(le_), w) + a2 * op.g * inner(lu_, grad(xi))) * dx
    L -= switch * iA * xi * dx

    return B, L


def weakResidualSW(q, q_, b, coriolisFreq=None, impermeable=True, adjoint=False, switch=Constant(0.), op=Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param nu: coefficient for stress term.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param impermeable: impose impermeable BCs.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    if adjoint:
        B, L = adjointSW(q, q_, b, switch=switch, op=op)
    else:
        B, L = formsSW(q, q_, b, coriolisFreq=coriolisFreq, impermeable=impermeable, op=op)
    return B - L


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


def solutionRW(V, t=0., B=0.395):
    """
    Analytic solution for equatorial Rossby wave test problem, as given by Huang.

    :arg V: Mixed function space upon which to define solutions.
    :arg t: current time.
    :param B: Parameter controlling amplitude of soliton.
    :return: Analytic solution for rossby-wave test problem of Huang.
    """
    x, y = SpatialCoordinate(V.mesh())
    q = Function(V)
    u, eta = q.split()

    A = 0.771 * B * B
    W = FunctionSpace(V.mesh(), V.sub(0).ufl_element().family(), V.sub(0).ufl_element().degree())
    u0 = Function(W).interpolate(
        A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
        * 0.25 * (-9 + 6 * y * y)
        * exp(-0.5 * y * y))
    u1 = Function(W).interpolate(
        -2 * B * tanh(B * (x + 0.4 * t)) *
        A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
        * 2 * y * exp(-0.5 * y * y))
    u.dat.data[:, 0] = u0.dat.data      # TODO: Shouldn't really do this in adjointland
    u.dat.data[:, 1] = u1.dat.data
    eta.interpolate(A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
                    * 0.25 * (3 + 6 * y * y)
                    * exp(-0.5 * y * y))

    return q


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
