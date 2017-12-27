from firedrake import *


def timestepCoeffs(timestepper):
    """
    :param timestepper: scheme of choice.
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
    :param u: prognostic variable at current timestep. 
    :param u_: prognostic variable at previous timestep. 
    :param timestepper: scheme of choice.
    :return: expression for prognostic variable to be used in scheme.
    """
    a1, a2 = timestepCoeffs(timestepper)

    return a1 * u + a2 * u_


def strongResidualSW(q, q_, b, Dt, nu=0., g=9.81, f0=0., beta=1., rotational=False, nonlinear=False,
                   timestepper='CrankNicolson'):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.
    
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param b: bathymetry profile.
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

    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param qt: test function tuple.
    :param b: bathymetry profile.
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
    mesh = q.function_space().mesh()
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])
    a1, a2 = timestepCoeffs(timestepper)

    # LHS bilinear form
    B = (inner(u, w) + inner(eta, xi)) / Dt * dx                            # Time derivative component
    B += a1 * (g * inner(grad(eta), w) - inner(b * u, grad(xi))) * dx    # Linear spatial derivative components

    # RHS linear functional
    L = (inner(u_, w) + inner(eta_, xi)) / Dt * dx                          # Time derivative component
    L -= a2 * (g * inner(grad(eta_), w) - inner(b * u_, grad(xi))) * dx   # Linear spatial derivative components
    # TODO: try NOT applying Neumann condition ^^^ in RHS functional

    if allowNormalFlow:
        n = FacetNormal(mesh)
        B += a1 * b * xi * dot(u, n) * ds
        L -= a2 * b * xi * dot(u_, n) * ds
    if nu != 0.:
        B -= a1 * nu * inner(grad(u) + transpose(grad(u)), grad(w)) * dx
        L += a2 * nu * inner(grad(u_) + transpose(grad(u_)), grad(w)) * dx
    if rotational:
        f = f0 + beta * SpatialCoordinate(mesh)[1]
        B += a1 * f * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * f * inner(as_vector((-u_[1], u_[0])), w) * dx
    if nonlinear:
        B += a1 * inner(dot(u, nabla_grad(u)), w) * dx
        L += a2 * inner(dot(u_, nabla_grad(u_)), w) * dx

    return B, L


def weakResidualSW(q, q_, qt, b, Dt, nu=0., g=9.81, f0=0., beta=1., rotational=False, nonlinear=False,
                   allowNormalFlow=True, timestepper='CrankNicolson'):
    """
    Semi-discrete (time-discretised) weak form shallow water equations with no normal flow boundary conditions.
    
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param qt: test function tuple.
    :param b: bathymetry profile.
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
    B, L = formsSW(q, q_, qt, b, Dt, nu=nu, g=g, f0=f0, beta=beta, rotational=rotational, nonlinear=nonlinear,
                   allowNormalFlow=allowNormalFlow, timestepper=timestepper)

    return B - L


def analyticHuang(V, B=0.395, t=0.):
    """
    :param V: Mixed function space upon which to define solutions.
    :param B: Parameter controlling amplitude of soliton.
    :param t: current time.
    :return: Initial condition for test problem of Huang.
    """

    # Establish phi functions
    q = Function(V)
    x_phi = " * 0.771 * %f * %f / pow(cosh(%f * (x[0] + 0.395 * %f * %f * %f)), 2)" % (B, B, B, B, B, t)
    x_dphidx = " * -2 * %f * tanh(%f * (x[0] + 0.395 * %f * %f * %f))" % (B, B, B, B, t)

    # Set components of q
    u, eta = q.split()
    u.interpolate(Expression(["(-9 + 6 * pow(x[1], 2)) * exp(-0.5 * pow(x[1], 2)) / 4" + x_phi,
                              "2 * x[1] * exp(-0.5 * pow(x[1], 2))" + x_dphidx]))
    eta.interpolate(Expression("(3 + 6 * pow(x[1], 2)) * exp(-0.5 * pow(x[1], 2)) / 4" + x_phi))

    return q


def strongResidualAD(c, c_, u, Dt, nu=1e-3, timestepper='CrankNicolson'):
    """
    :param c: concentration solution at current timestep. 
    :param c_: concentration at previous timestep.
    :param u: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: diffusivity parameter.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    cm = timestepScheme(c, c_, timestepper)
    return (c - c_) / Dt + inner(u, grad(cm)) - Constant(nu) * div(grad(cm))


def weakResidualAD(c, c_, ct, u, Dt, nu=1e-3, timestepper='CrankNicolson'):
    """
    :param c: concentration solution at current timestep. 
    :param c_: concentration at previous timestep.
    :param ct: test function.
    :param u: wind field.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: diffusivity parameter.
    :param timestepper: time integration scheme used.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    cm = timestepScheme(c, c_, timestepper)
    return ((c - c_) * ct / Dt - inner(cm * u, grad(ct)) + Constant(nu) * inner(grad(cm), grad(ct))) * dx


def weakMetricAdvection(M, M_, Mt, w, Dt, timestepper='ImplicitEuler'):
    """
    Advect a metric. Also works for vector fields.
    
    :param M: metric at current timestep.
    :param M_: metric at previous timestep.
    :param Mt: test function.
    :param w: wind vector.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param timestepper: time integration scheme used.
    :return: weak residual for metric advection.
    """
    Mm = timestepScheme(M, M_, timestepper)
    F = (inner(M - M_, Mt) / Dt + inner(dot(w, nabla_grad(Mm)), Mt)) * dx
    return F


def indicator(V, x1=2.5, x2=3.5, y1=0.1, y2=0.9, smooth=False):
    """
    :param V: Function space to use.
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
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.'\
              % (x1, x2, y1, y2, x1 + xd, xd, y1 + yd, yd)
    else:
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (x1, x2, y1, y2)

    return Function(V).interpolate(Expression(ind))


from firedrake_adjoint import dt, Functional


def objectiveFunctionalAD(c, x1=2.5, x2=3.5, y1=0.1, y2=0.9):
    """
    :param c: concentration.
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :return: objective functional for advection diffusion problem. 
    """
    return  Functional(c * indicator(c.function_space(), x1, x2, y1, y2) * dx * dt)


def objectiveFunctionalSW(q, Tstart=300., Tend=1500., x1=490e3, x2=640e3, y1=4160e3, y2=4360e3,
                          plot=False, smooth=True):
    """
    :param q: forward solution tuple.
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

    # TODO: `smoothen` in time (?)

    if plot:
        File("plots/adjointBased/kernel.pvd").write(ke)

    return Functional(inner(q, k) * dx * dt[Tstart:Tend])
