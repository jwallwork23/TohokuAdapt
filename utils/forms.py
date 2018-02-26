from firedrake import *

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


def formsSW(q, q_, qt, b, Dt, nu=0., rotational=False, nonlinear=False, allowNormalFlow=True, op=opt.Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg qt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    V = q.function_space()
    mesh = V.mesh()
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
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
    if rotational:
        f = op.coriolis0 + op.coriolis1 * SpatialCoordinate(mesh)[1]
        B += a1 * f * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * f * inner(as_vector((-u_[1], u_[0])), w) * dx
    if nonlinear:
        B += a1 * inner(dot(u, nabla_grad(u)), w) * dx
        L -= a2 * inner(dot(u_, nabla_grad(u_)), w) * dx

    return B, L


def adjointSW(l, l_, lt, b, Dt, x1=2.5, x2=3.5, y1=0.1, y2=0.9, smooth=False, switch=Constant(1.), op=opt.Options()):
    """
    Semi-discrete (time-discretised) weak form adjoint shallow water equations with no normal flow boundary conditions.

    :arg l: solution tuple for adjoint linear shallow water equations.
    :arg l_: solution tuple for adjoint linear shallow water equations at previous timestep.
    :arg lt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    (lu, le) = (as_vector((l[0], l[1])), l[2])
    (lu_, le_) = (as_vector((l_[0], l_[1])), l_[2])
    (w, xi) = (as_vector((lt[0], lt[1])), lt[2])
    a1, a2 = timestepCoeffs(op.timestepper)
    iA = indicator(l.function_space().sub(1), x1=x1, x2=x2, y1=y1, y2=y2, smooth=smooth)

    B = ((inner(lu, w) + le * xi) / Dt + a1 * b * inner(grad(le), w) - a1 * op.g * inner(lu, grad(xi))) * dx
    L = ((inner(lu_, w) + le_ * xi) / Dt - a2 * b * inner(grad(le_), w) + a2 * op.g * inner(lu_, grad(xi))) * dx
    L -= switch * iA * xi * dx

    return B, L


def weakResidualSW(q, q_, qt, b, Dt, nu=0., rotational=False, nonlinear=False, allowNormalFlow=True, adjoint=False,
                   switch=Constant(0.), op=opt.Options()):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg qt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter-holding class.
    :return: weak residual for shallow water equations at current timestep.
    """
    if adjoint:
        B, L = adjointSW(q, q_, qt, b, Dt, x1=2.5, x2=3.5, y1=0.1, y2=0.9, smooth=False, switch=switch, op=op)
    else:
        B, L = formsSW(q, q_, qt, b, Dt, nu=nu, rotational=rotational, nonlinear=nonlinear,
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


def outwardFlux(v, n=None, inward=False):
    """
    :arg v: Function to be evaluated over element boundaries.
    :param n: FacetNormal
    :param inward: toggle inward flux.
    :return: averaged jump discontinuity over element boundary.
    """
    sign = '-' if inward else '+'
    if n == None:
        n = FacetNormal(v.function_space().mesh())
    v = as_ufl(v)
    if len(v.ufl_shape) == 0:
        return (v(sign) * n(sign))
    else:
        return (dot(v(sign), n(sign)))


def localProblemSW(q, q_, qt, b, Dt, nu=0., rotational=False, nonlinear=False, allowNormalFlow=True, op=opt.Options()):
    """
    Semi-discrete (time-discretised) local variational problem for the shallow water equations with no normal flow 
    boundary conditions, under the element residual method.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg qt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param op: parameter-holding class.
    :return: residual of local problem.
    """
    V = q.function_space()
    n = FacetNormal(V.mesh())
    u, eta = q.split()
    ut, et = qt.split()

    # Establish variational form for residual equation
    B_, L = formsSW(q, q_, qt, b, Dt, nu=nu, rotational=rotational, nonlinear=nonlinear,
                    allowNormalFlow=allowNormalFlow, op=op)
    phi = Function(V, name='Local solution')
    B = formsSW(phi, q_, qt, b, Dt, nu=nu, rotational=rotational, nonlinear=nonlinear,
                allowNormalFlow=allowNormalFlow, op=op)[0]
    F = B + B_ - L + interelementTerm(grad(u) * et, n=n) * dS

    # TODO: test this

    return F


def analyticHuang(V, t=0., B=0.395):
    """
    :arg V: Mixed function space upon which to define solutions.
    :arg t: current time.
    :param B: Parameter controlling amplitude of soliton.
    :return: Initial condition for test problem of Huang.
    """
    x, y = SpatialCoordinate(V.mesh())
    q = Function(V)
    u, eta = q.split()
    # u.interpolate([0.25*(-9+6*y*y)*exp(-0.5*y*y)*0.771*B*B/pow(cosh(B*(x+0.395*B*B*t)), 2),
    #                -0.5*(-9+6*y*y)*exp(-0.5*y*y)*B*tanh(B*(x+0.395*B*B*t))])


    x_phi = " * 0.771 * %f * %f / pow(cosh(%f * (x[0] + 0.395 * %f * %f * %f)), 2)" % (B, B, B, B, B, t)
    x_dphidx = " * -2 * %f * tanh(%f * (x[0] + 0.395 * %f * %f * %f))/ pow(cosh(%f * (x[0] + 0.395 * %f * %f * %f)), 2)" \
               % (B, B, B, B, t, B, B, B, t)

    u.interpolate(Expression(["0.25 * (-9 + 6 * x[1] * x[1]) * exp(-0.5 * x[1] * x[1])" + x_phi,
                              "2 * x[1] * exp(-0.5 * x[1] * x[1])" + x_dphidx]))
    eta.interpolate(Expression("0.25 * (3 + 6 * x[1] * x[1]) * exp(-0.5 * x[1] * x[1])" + x_phi))

    return q

def icHuang(V, B=0.395):
    x, y = SpatialCoordinate(V.mesh())
    q = Function(V)
    u, eta = q.split()

    A = 0.771 * B * B
    W = FunctionSpace(V.mesh(), V.sub(0).ufl_element().family(), V.sub(0).ufl_element().degree())
    u0 = Function(W)
    u1 = Function(W)
    u0.interpolate(A * (1 / ((cosh(B * x) ** 2))) * 0.25 * (-9 + 6 * y * y) * exp(-0.5 * y * y))
    u1.interpolate(-2 * B * tanh(B * x) * A * (1 / ((cosh(B * x) ** 2))) * 2 * y * exp(-0.5 * y * y))
    u.dat.data[:,0] = u0.dat.data
    u.dat.data[:,1] = u1.dat.data
    eta.interpolate(A*(1/((cosh(B*x)**2)))*0.25*(3+6*y*y)*exp(-0.5*y*y))

    return q

def val(X, t):
    """
    From Matt's code. 
    """
    from math import cosh,tanh,exp
    B = 0.395
    A = 0.771*B*B
    v = [0, 0]
    v[0] = A*(1/((cosh(B*X[0]))**(2)))*0.25*(-9+ 6*X[1]**2)*exp(-0.5*X[1]**2)
    v[1] = -2*B*tanh(B*X[0])*A*(1/((cosh(B*X[0])**2)))*2*X[1]*exp(-0.5*X[1]**2)
    return v


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


def weakResidualAD(c, c_, ct, w, Dt, nu=1e-3, timestepper='CrankNicolson'):
    """
    :arg c: concentration solution at current timestep. 
    :arg c_: concentration at previous timestep.
    :arg ct: test function.
    :arg w: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: diffusivity parameter.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    cm = timestepScheme(c, c_, timestepper)
    return ((c - c_) * ct / Dt + inner(grad(cm), w * ct) + Constant(nu) * inner(grad(cm), grad(ct))) * dx


def adjointAD(l, l_, lt, w, Dt,  timestepper='CrankNicolson', x1=2.75, x2=3.25, y1=0.25, y2=0.75):
    """
    :arg l: adjoint concentration solution at current timestep. 
    :arg l_: adjoint concentration at previous timestep.
    :arg lt: test function.
    :arg w: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    lm = timestepScheme(l, l_, timestepper)
    iA = indicator(l.function_space(), x1=x1, x2=x2, y1=y1, y2=y2, smooth=False)
    return ((l - l_) * lt / Dt + inner(grad(lm), w * lt) + iA * lt) * dx


# TODO: not sure this is properly linearised


def weakMetricAdvection(M, M_, Mt, w, Dt, timestepper='ImplicitEuler'):
    """
    Advect a metric. Also works for vector fields.

    :arg M: metric at current timestep.
    :arg M_: metric at previous timestep.
    :arg Mt: test function.
    :arg w: wind vector.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param timestepper: time integration scheme used.
    :return: weak residual for metric advection.
    """
    Mm = timestepScheme(M, M_, timestepper)
    F = (inner(M - M_, Mt) / Dt + inner(dot(w, nabla_grad(Mm)), Mt)) * dx
    return F


def indicator(V, mode='tohoku', smooth=False):
    """
    :arg V: Function space to use.
    :param mode: test problem considered.
    :param smooth: toggle smoothening.
    :return: ('Smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    """
    # Define extent of region A
    if mode == 'tohoku':
        x1 = 490e3
        x2 = 640e3
        y1 = 4160e3
        y2 = 4360e3
    elif mode == 'shallow-water':
        x1 = 2.5
        x2 = 3.5
        y1 = 0.1
        y2 = 0.9
    elif mode == 'advection-diffusion':
        x1 = 2.5
        x2 = 3.5
        y1 = 0.1
        y2 = 0.9
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
