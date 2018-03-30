from thetis import *
from thetis_adjoint import *
from firedrake import Expression

from .options import Options
from .timestepping import timestepScheme, timestepCoeffs


__all__ = ["strongResidualSW", "formsSW", "adjointSW", "weakResidualSW", "interelementTerm", "solutionHuang",
           "weakMetricAdvection", "indicator"]


def strongResidualSW(q, q_, b, Dt, coriolisFreq=None, nonlinear=False, op=Options()):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter holding class.
    :return: strong residual for shallow water equations at current timestep.
    """
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    um = timestepScheme(u, u_, op.timestepper)
    em = timestepScheme(eta, eta_, op.timestepper)

    Au = (u - u_) / Dt + op.g * grad(em)
    Ae = (eta - eta_) / Dt + div(b * um)
    if coriolisFreq:
        Au += coriolisFreq * as_vector((-u[1], u[0]))
    if nonlinear:
        Au += dot(u, nabla_grad(u))
        Ae += div(em * um)

    return Au, Ae


def formsSW(q, q_, b, Dt, coriolisFreq=None, nonlinear=False, impermeable=True, op=Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param impermeable: impose Neumann BCs.
    :param op: parameter holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
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
    if nonlinear:
        B += a1 * (inner(dot(u, nabla_grad(u)), w) + div(eta * u)) * dx
        L -= a2 * (inner(dot(u_, nabla_grad(u_)), w)  + div(eta_ * u_)) * dx

    return B, L


def adjointSW(l, l_, b, Dt, mode='shallow-water', switch=Constant(1.), op=Options()):
    """
    Semi-discrete (time-discretised) weak form adjoint shallow water equations with no normal flow boundary conditions.

    :arg l: solution tuple for adjoint linear shallow water equations.
    :arg l_: solution tuple for adjoint linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    (lu, le) = (as_vector((l[0], l[1])), l[2])
    (lu_, le_) = (as_vector((l_[0], l_[1])), l_[2])
    lt = TestFunction(l.function_space())
    (w, xi) = (as_vector((lt[0], lt[1])), lt[2])
    a1, a2 = timestepCoeffs(op.timestepper)
    iA = indicator(l.function_space().sub(1), mode=mode)

    B = ((inner(lu, w) + le * xi) / Dt + a1 * b * inner(grad(le), w) - a1 * op.g * inner(lu, grad(xi))) * dx
    L = ((inner(lu_, w) + le_ * xi) / Dt - a2 * b * inner(grad(le_), w) + a2 * op.g * inner(lu_, grad(xi))) * dx
    L -= switch * iA * xi * dx

    # TODO: Boundary conditions?

    return B, L


def weakResidualSW(q, q_, b, Dt, coriolisFreq=None, nonlinear=False, impermeable=True, adjoint=False,
                   mode='shallow-water', switch=Constant(0.), op=Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param impermeable: impose impermeable BCs.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    if adjoint:
        B, L = adjointSW(q, q_, b, Dt, mode=mode, switch=switch, op=op)
    else:
        B, L = formsSW(q, q_, b, Dt, coriolisFreq=coriolisFreq, nonlinear=nonlinear,
                       impermeable=impermeable, op=op)
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


def weakMetricAdvection(M, M_, w, Dt, timestepper='ImplicitEuler'):
    """
    Advect a metric. Also works for vector fields.

    :arg M: metric at current timestep.
    :arg M_: metric at previous timestep.
    :arg w: wind vector.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param timestepper: time integration scheme used.
    :return: weak residual for metric advection.
    """
    Mt = TestFunction(M.function_space())
    Mm = timestepScheme(M, M_, timestepper)
    F = (inner(M - M_, Mt) / Dt + inner(dot(w, nabla_grad(Mm)), Mt)) * dx
    return F


def indicator(V, mode='tohoku'):
    """
    :arg V: Function space to use.
    :param mode: test problem considered.
    :return: ('Smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    """
    smooth = True if mode == 'tohoku' else False

    # Define extent of region A
    if mode == 'helmholtz1':
        xy = [0., 0.2, 0.4, 0.6]
    elif mode == 'helmholtz2':
        xy = [0.1, 0.3, 0.1, 0.3]
    elif mode == 'advection-diffusion':
        xy = [2.5, 3.5, 0.1, 0.9]
    elif mode == 'shallow-water':
        xy = [0., 0.5 * pi, 0.5 * pi, 1.5 * pi]
    elif mode == 'rossby-wave':
        xy = [0., 8., 10., 14.]
    elif mode == 'tohoku':
        xy = [490e3, 640e3, 4160e3, 4360e3]
    else:
        raise NotImplementedError
    if smooth:
        xd = (xy[1] - xy[0]) / 2
        yd = (xy[3] - xy[2]) / 2
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.' \
              % (xy[0], xy[1], xy[2], xy[3], xy[0] + xd, xd, xy[2] + yd, yd)
    else:
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (xy[0], xy[1], xy[2], xy[3])
    iA = Function(V, name="Region of interest").interpolate(Expression(ind))
    if plot:
        File("plots/adjointBased/kernel.pvd").write(iA)

    return iA


def solutionHuang(V, t=0., B=0.395):
    """
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
    u.dat.data[:,0] = u0.dat.data
    u.dat.data[:,1] = u1.dat.data
    eta.interpolate(A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
                    * 0.25 * (3 + 6 * y * y)
                    * exp(-0.5 * y * y))

    return q
