from firedrake import *

import numpy as np

import utils.options as opt


def timestepCoeffs(timestepper):
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


# TODO: make this format more conventional and consider RK4 timestepping


def timestepScheme(u, u_, timestepper):
    """
    :arg u: prognostic variable at current timestep. 
    :arg u_: prognostic variable at previous timestep. 
    :arg timestepper: scheme of choice.
    :return: expression for prognostic variable to be used in scheme.
    """
    a1, a2 = timestepCoeffs(timestepper)

    return a1 * u + a2 * u_


def strongResidualSW(q, q_, b, Dt, nu=0., rotational=False, nonlinear=False, op=opt.Options()):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter holding class.
    :return: strong residual for shallow water equations at current timestep.
    """
    # TODO: include optionality for BCs
    # TODO: implement Galerkin Least Squares (GLS) stabilisation


    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    um = timestepScheme(u, u_, op.timestepper)
    em = timestepScheme(eta, eta_, op.timestepper)

    Au = (u - u_) / Dt + op.g * grad(em)
    Ae = (eta - eta_) / Dt + div(b * um)
    if nu != 0.:
        Au += div(nu * (grad(um) + transpose(grad(um))))
    if rotational:
        f = op.coriolis0 + op.coriolis1 * SpatialCoordinate(q.function_space().mesh())[1]
        Au += f * as_vector((-u[1], u[0]))
    if nonlinear:
        Au += dot(u, nabla_grad(u))

    return Au, Ae


def formsSW(q, q_, b, Dt, nu=0., coriolisFreq=None, nonlinear=False, allowNormalFlow=True, op=opt.Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    V = q.function_space()
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    qt = TestFunction(V)
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])
    a1, a2 = timestepCoeffs(op.timestepper)
    g = op.g

    B = (inner(u, w) + eta * xi) / Dt * dx  # LHS bilinear form
    L = (inner(u_, w) + eta_ * xi) / Dt * dx  # RHS linear functional

    if V.sub(1).ufl_element().family() == 'Lagrange':
        B += a1 * g * inner(grad(eta), w) * dx
        L -= a2 * g * inner(grad(eta_), w) * dx
    else:
        B -= a1 * g * eta * div(w) * dx
        L += a2 * g * eta_ * div(w) * dx
    if allowNormalFlow:
        B += a1 * div(b * u) * xi * dx
        L -= a2 * div(b * u_) * xi * dx
    else:
        B -= a1 * inner(b * u, grad(xi)) * dx
        L += a2 * inner(b * u_, grad(xi)) * dx
    if nu != 0.:
        B -= a1 * nu * inner(grad(u) + transpose(grad(u)), grad(w)) * dx
        L += a2 * nu * inner(grad(u_) + transpose(grad(u_)), grad(w)) * dx
    if coriolisFreq is not None:
        B += a1 * coriolisFreq * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * coriolisFreq * inner(as_vector((-u_[1], u_[0])), w) * dx
    if nonlinear:
        B += a1 * inner(dot(u, nabla_grad(u)), w) * dx
        L -= a2 * inner(dot(u_, nabla_grad(u_)), w) * dx

    return B, L


def adjointSW(l, l_, b, Dt, mode='shallow-water', switch=Constant(1.), op=opt.Options()):
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

    return B, L


def weakResidualSW(q, q_, b, Dt, nu=0., coriolisFreq=None, nonlinear=False, allowNormalFlow=True, adjoint=False,
                   mode='shallow-water', switch=Constant(0.), op=opt.Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param coriolisFreq: Coriolis parameter for rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    if adjoint:
        B, L = adjointSW(q, q_, b, Dt, mode=mode, switch=switch, op=op)
    else:
        B, L = formsSW(q, q_, b, Dt, nu=nu, coriolisFreq=coriolisFreq, nonlinear=nonlinear,
                       allowNormalFlow=allowNormalFlow, op=op)
    return B - L


# TODO: Conisder 2D linear dispersive SWEs as in Saito '10a


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


def solutionHuang(V, t=0., B=0.395):
    """
    :arg V: Mixed function space upon which to define solutions.
    :arg t: current time.
    :param B: Parameter controlling amplitude of soliton.
    :return: Analytic solution for test problem of Huang.
    """
    x, y = SpatialCoordinate(V.mesh())
    q = Function(V)
    u, eta = q.split()

    A = 0.771 * B * B
    W = FunctionSpace(V.mesh(), V.sub(0).ufl_element().family(), V.sub(0).ufl_element().degree())
    u0 = Function(W).interpolate(
        A * (1 / (cosh(B * (x + B * B * t)) ** 2)) * 0.25 * (-9 + 6 * y * y) * exp(-0.5 * y * y))
    u1 = Function(W).interpolate(
        -2 * B * tanh(B * (x + B * B * t)) * A * (1 / (cosh(B * (x + B * B * t)) ** 2)) * 2 * y * exp(-0.5 * y * y))
    u.dat.data[:,0] = u0.dat.data
    u.dat.data[:,1] = u1.dat.data
    eta.interpolate(A*(1/(cosh(B*(x + B * B * t))**2))*0.25*(3+6*y*y)*exp(-0.5*y*y))

    return q


def strongResidualAD(c, c_, w, Dt, nu=1e-3, timestepper='CrankNicolson'):
    """
    :arg c: concentration solution at current timestep. 
    :arg c_: concentration at previous timestep.
    :arg w: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: diffusivity parameter.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    cm = timestepScheme(c, c_, timestepper)
    return (c - c_) / Dt + inner(w, grad(cm)) - Constant(nu) * div(grad(cm))


def weakResidualAD(c, c_, w, Dt, nu=1e-3, timestepper='CrankNicolson'):
    """
    :arg c: concentration solution at current timestep. 
    :arg c_: concentration at previous timestep.
    :arg w: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: diffusivity parameter.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    cm = timestepScheme(c, c_, timestepper)
    ct = TestFunction(c.function_space())
    return ((c - c_) * ct / Dt + inner(grad(cm), w * ct) + Constant(nu) * inner(grad(cm), grad(ct))) * dx


def adjointAD(l, l_, w, Dt,  timestepper='CrankNicolson', x1=2.75, x2=3.25, y1=0.25, y2=0.75):
    """
    :arg l: adjoint concentration solution at current timestep. 
    :arg l_: adjoint concentration at previous timestep.
    :arg w: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    lm = timestepScheme(l, l_, timestepper)
    lt = TestFunction(l.function_space())
    iA = indicator(l.function_space(), mode='advection-diffusion')
    return ((l - l_) * lt / Dt + inner(grad(lm), w * lt) + iA * lt) * dx


# TODO: not sure this is properly linearised


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
    if mode == 'tohoku':
        x1 = 490e3
        x2 = 640e3
        y1 = 4160e3
        y2 = 4360e3
    elif mode == 'shallow-water':
        x1 = 0.
        x2 = 0.5 * np.pi
        y1 = 0.5 * np.pi
        y2 = 1.5 * np.pi
    elif mode == 'advection-diffusion':
        x1 = 2.5
        x2 = 3.5
        y1 = 0.1
        y2 = 0.9
    elif mode == 'helmholtz1':
        x1 = 0.
        x2 = 0.2
        y1 = 0.4
        y2 = 0.6
    elif mode == 'helmholtz2':
        x1 = 0.1
        x2 = 0.3
        y1 = 0.1
        y2 = 0.3
    else:
        raise NotImplementedError
    if smooth:
        xd = (x2 - x1) / 2
        yd = (y2 - y1) / 2
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.' \
              % (x1, x2, y1, y2, x1 + xd, xd, y1 + yd, yd)
    else:
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (x1, x2, y1, y2)
    iA = Function(V, name="Region of interest").interpolate(Expression(ind))
    if plot:
        File("plots/adjointBased/kernel.pvd").write(iA)

    return iA
