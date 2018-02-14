from firedrake import *


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

# TODO: consider RK4 timestepping


def timestepScheme(u, u_, timestepper):
    """
    :arg u: prognostic variable at current timestep. 
    :arg u_: prognostic variable at previous timestep. 
    :arg timestepper: scheme of choice.
    :return: expression for prognostic variable to be used in scheme.
    """
    a1, a2 = timestepCoeffs(timestepper)

    return a1 * u + a2 * u_


def strongResidualSW(q, q_, b, Dt, nu=0., g=9.81, f0=0., beta=1., rotational=False, nonlinear=False,
                     timestepper='CrankNicolson'):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param g: gravitational acceleration.
    :param f0: 0th order coefficient for asymptotic Coriolis expansion.
    :param beta: 1st order coefficient for asymptotic Coriolis expansion.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param timestepper: scheme of choice.
    :return: strong residual for shallow water equations at current timestep.
    """
    # TODO: include optionality for BCs
    # TODO: implement Galerkin Least Squares (GLS) stabilisation

    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    um = timestepScheme(u, u_, timestepper)
    em = timestepScheme(eta, eta_, timestepper)

    Au = (u - u_) / Dt + g * grad(em)
    Ae = (eta - eta_) / Dt + div(b * um)
    if nu != 0.:
        Au += div(nu * (grad(um) + transpose(grad(um))))
    if rotational:
        f = f0 + beta * SpatialCoordinate(q.function_space().mesh())[1]
        Au += f * as_vector((-u[1], u[0]))
    if nonlinear:
        Au += dot(u, nabla_grad(u))

    return Au, Ae


def formsSW(q, q_, qt, b, Dt, nu=0., g=9.81, f0=0., beta=1., rotational=False, nonlinear=False, allowNormalFlow=True,
            timestepper='CrankNicolson'):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg qt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param g: gravitational acceleration.
    :param f0: 0th order coefficient for asymptotic Coriolis expansion.
    :param beta: 1st order coefficient for asymptotic Coriolis expansion.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param timestepper: scheme of choice.
    :return: weak residual for shallow water equations at current timestep.
    """
    V = q.function_space()
    mesh = V.mesh()
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])
    a1, a2 = timestepCoeffs(timestepper)

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
        f = f0 + beta * SpatialCoordinate(mesh)[1]
        B += a1 * f * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * f * inner(as_vector((-u_[1], u_[0])), w) * dx
    if nonlinear:
        B += a1 * inner(dot(u, nabla_grad(u)), w) * dx
        L -= a2 * inner(dot(u_, nabla_grad(u_)), w) * dx

    return B, L


def adjointSW(l, l_, lt, b, Dt, g=9.81, timestepper='CrankNicolson', x1=2.5, x2=3.5, y1=0.1, y2=0.9, smooth=False,
              switch=Constant(1.)):
    """
    Semi-discrete (time-discretised) weak form adjoint shallow water equations with no normal flow boundary conditions.

    :arg l: solution tuple for adjoint linear shallow water equations.
    :arg l_: solution tuple for adjoint linear shallow water equations at previous timestep.
    :arg lt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param g: gravitational acceleration.
    :param timestepper: scheme of choice.
    :return: weak residual for shallow water equations at current timestep.
    """
    (lu, le) = (as_vector((l[0], l[1])), l[2])
    (lu_, le_) = (as_vector((l_[0], l_[1])), l_[2])
    (w, xi) = (as_vector((lt[0], lt[1])), lt[2])
    a1, a2 = timestepCoeffs(timestepper)
    iA = indicator(l.function_space().sub(1), x1=x1, x2=x2, y1=y1, y2=y2, smooth=smooth)

    B = ((inner(lu, w) + le * xi) / Dt + a1 * b * inner(grad(le), w) - a1 * g * inner(lu, grad(xi))) * dx
    L = ((inner(lu_, w) + le_ * xi) / Dt - a2 * b * inner(grad(le_), w) + a2 * g * inner(lu_, grad(xi))) * dx
    L -= switch * iA * xi * dx

    return B, L


def weakResidualSW(q, q_, qt, b, Dt, nu=0., g=9.81, f0=0., beta=1., rotational=False, nonlinear=False,
                   allowNormalFlow=True, timestepper='CrankNicolson', adjoint=False, switch=Constant(0.)):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg qt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param g: gravitational acceleration.
    :param f0: 0th order coefficient for asymptotic Coriolis expansion.
    :param beta: 1st order coefficient for asymptotic Coriolis expansion.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param timestepper: scheme of choice.
    :return: weak residual for shallow water equations at current timestep.
    """
    if adjoint:
        B, L = adjointSW(q, q_, qt, b, Dt, g=9.81, timestepper='CrankNicolson', x1=2.5, x2=3.5, y1=0.1, y2=0.9,
                         smooth=False, switch=switch)
    else:
        B, L = formsSW(q, q_, qt, b, Dt, nu=nu, g=g, f0=f0, beta=beta, rotational=rotational, nonlinear=nonlinear,
                       allowNormalFlow=allowNormalFlow, timestepper=timestepper)

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


def localProblemSW(q, q_, qt, b, Dt, nu=0., g=9.81, f0=0., beta=1., rotational=False, nonlinear=False,
                   allowNormalFlow=True, timestepper='CrankNicolson'):
    """
    Semi-discrete (time-discretised) local variational problem for the shallow water equations with no normal flow 
    boundary conditions, under the element residual method.

    :arg q: solution tuple for linear shallow water equations.
    :arg q_: solution tuple for linear shallow water equations at previous timestep.
    :arg qt: test function tuple.
    :arg b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param g: gravitational acceleration.
    :param f0: 0th order coefficient for asymptotic Coriolis expansion.
    :param beta: 1st order coefficient for asymptotic Coriolis expansion.
    :param rotational: toggle rotational / non-rotational equations.
    :param nonlinear: toggle nonlinear / linear equations.
    :param timestepper: scheme of choice.
    :return: residual of local problem.
    """
    V = q.function_space()
    n = FacetNormal(V.mesh())
    u, eta = q.split()
    ut, et = qt.split()

    # Establish variational form for residual equation
    B_, L = formsSW(q, q_, qt, b, Dt, nu=nu, g=g, f0=f0, beta=beta, rotational=rotational, nonlinear=nonlinear,
                    allowNormalFlow=allowNormalFlow, timestepper=timestepper)
    phi = Function(V, name='Local solution')
    B = formsSW(phi, q_, qt, b, Dt, nu=nu, g=g, f0=f0, beta=beta, rotational=rotational, nonlinear=nonlinear,
                allowNormalFlow=allowNormalFlow, timestepper=timestepper)[0]
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
    x_phi = " * 0.771 * %f * %f / pow(cosh(%f * (x[0] + 0.395 * %f * %f * %f)), 2)" % (B, B, B, B, B, t)
    x_dphidx = " * -2 * %f * tanh(%f * (x[0] + 0.395 * %f * %f * %f))" % (B, B, B, B, t)
    q = Function(V)
    u, eta = q.split()
    u.interpolate(Expression(["0.25 * (-9 + 6 * x[1] * x[1]) * exp(-0.5 * x[1] * x[1])" + x_phi,
                              "2 * x[1] * exp(-0.5 * x[1] * x[1])" + x_dphidx]))
    eta.interpolate(Expression("0.25 * (3 + 6 * x[1] * x[1]) * exp(-0.5 * x[1] * x[1])" + x_phi))

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


def adjointAD(l, l_, lt, w, Dt, nu=1e-3, timestepper='CrankNicolson', x1=2.75, x2=3.25, y1=0.25, y2=0.75):
    """
    :arg l: adjoint concentration solution at current timestep. 
    :arg l_: adjoint concentration at previous timestep.
    :arg lt: test function.
    :arg w: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: diffusivity parameter.
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


def indicator(V, x1=2.5, x2=3.5, y1=0.1, y2=0.9, smooth=False):
    """
    :arg V: Function space to use.
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :param smooth: toggle smoothening.
    :return: ('Smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    """
    if smooth:
        xd = (x2 - x1) / 2
        yd = (y2 - y1) / 2
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.' \
              % (x1, x2, y1, y2, x1 + xd, xd, y1 + yd, yd)
    else:
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (x1, x2, y1, y2)

    return Function(V).interpolate(Expression(ind))


from firedrake_adjoint import dt, Functional


def objectiveFunctionalAD(c, x1=2.5, x2=3.5, y1=0.1, y2=0.9):
    """
    :arg c: concentration.
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :return: objective functional for advection diffusion problem. 
    """
    return Functional(c * indicator(c.function_space(), x1, x2, y1, y2) * dx * dt)


def objectiveFunctionalSW(q, Tstart=300., Tend=1500., x1=490e3, x2=640e3, y1=4160e3, y2=4360e3,
                          plot=False, smooth=True):
    """
    :arg q: forward solution tuple.
    :param Tstart: first time considered as relevant (s).
    :param Tend: last time considered as relevant (s).
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :param plot: toggle plotting of indicator function.
    :param smooth: toggle 'smoothening' of the indicator function.
    :return: objective functional for shallow water equations. 
    """
    V = q.function_space()
    k = Function(V)
    ku, ke = k.split()
    ke.assign(indicator(V.sub(1), x1, x2, y1, y2))
    if plot:
        File("plots/adjointBased/kernel.pvd").write(ke)

    return Functional(inner(q, k) * dx * dt[Tstart:Tend])
