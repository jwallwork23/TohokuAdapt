from firedrake import *
from firedrake_adjoint import dt, Functional


def timestepScheme(u, u_, timestepper):
    """
    :param u: prognostic variable at current timestep. 
    :param u_: prognostic variable at previous timestep. 
    :param timestepper: scheme of choice.
    :return: expression for prognostic variable to be used in scheme.
    """
    if timestepper == 'CrankNicolson':
        um = 0.5 * (u + u_)
    elif timestepper == 'ImplicitEuler':
        um = u
    elif timestepper == 'ExplicitEuler':
        um = u_
    else:
        raise NotImplementedError("Timestepping scheme %s not yet considered." % timestepper)
    return um


def objectiveFunctionalSW(q, Tstart=300., Tend=1500., x1=490e3, x2=640e3, y1=4160e3, y2=4360e3,
                          plot=False, smooth=True):
    """
    :param q: forward solution tuple.
    # :param t: current time value (s).
    # :param timestep: time step (s).
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

    # Create a (possibly 'smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    if smooth:
        xd = (x2 - x1) / 2
        yd = (y2 - y1) / 2
        indicator = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
                    'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.'\
                    % (x1, x2, y1, y2, x1 + xd, xd, y1 + yd, yd)
    else:
        indicator = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (x1, x2, y1, y2)
    k = Function(q.function_space())
    ku, ke = k.split()
    ke.interpolate(Expression(indicator))

    # TODO: `smoothen` in time (?)
    # # Modify forcing term to 'smoothen' in time
    # coeff = Constant(1.)
    # if t.dat.data < Tstart + 1.5 * timestep:
    #     coeff.assign(0.5)
    # elif t.dat.data < Tstart + 0.5 * timestep:
    #     coeff.assign(0.)
    #  return (coeff * eta * iA) * dx * dt[Tstart:Tend]

    if plot:
        File("plots/adjointBased/kernel.pvd").write(ke)

    return Functional(inner(q, k) * dx * dt[Tstart:Tend])


def strongResidualSW(q, q_, b, Dt, nu=0., timestepper='CrankNicolson', rotational=False):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep.
    
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param rotational: toggle rotational / non-rotational equations.
    :return: strong residual for shallow water equations at current timestep.
    """
    # TODO: include optionality for BCs

    # TODO: implement Galerkin Least Squares (GLS) stabilisation

    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    um = timestepScheme(u, u_, timestepper)
    em = timestepScheme(eta, eta_, timestepper)

    Au = u - u_ + Dt * 9.81 * grad(em)
    Ae = eta - eta_ + Dt * div(b * um)
    if nu != 0.:
        Au += div(nu * (grad(um) + transpose(grad(um))))
    if rotational:
        Au += as_vector((-u[1], u[0]))

    return Au, Ae


def weakResidualSW(q, q_, qt, b, Dt, nu=0., timestepper='CrankNicolson', rotational=False, momentum=False):
    """
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param qt: test function tuple.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :param rotational: toggle rotational / non-rotational equations.
    :param momentum: toggle momentum form of equations.
    :return: weak residual for shallow water equations at current timestep.
    """
    # TODO: include optionality for BCs

    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])
    um = timestepScheme(u, u_, timestepper)
    em = timestepScheme(eta, eta_, timestepper)

    # TODO: account for momentum form of SWEs, as used in Huang's test case
    # if momentum:
    #   ...
    # else:

    F = (inner(u - u_, w) + inner(eta - eta_, xi) + Dt * (9.81 * inner(grad(em), w) - inner(b * um, grad(xi)))) * dx
    if nu != 0.:
        F -= nu * inner(grad(um) + transpose(grad(um)), grad(w))
    if rotational:
        F += inner(as_vector((-u[1], u[0])), w) * dx

    return F


def objectiveFunctionalAD(c, x1=2.5, x2=3.5, y1=0.1, y2=0.9):
    """
    :param c: concentration.
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :return: objective functional for advection diffusion problem. 
    """

    # Create a 'smoothened' indicator function for region A = [x1, x2] x [y1, y1]
    xd = (x2 - x1) / 2
    yd = (y2 - y1) / 2
    indicator = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
                'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.'\
                % (x1, x2, y1, y2, x1 + xd, xd, y1 + yd, yd)
    iA = Function(c.function_space()).interpolate(Expression(indicator))

    return  Functional(c * iA * dx * dt)


def strongResidualMSW(q, q_, h, Dt, y, f0=0., beta=1., g=1., timestepper='CrankNicolson'):
    """
    Construct the strong residual for the semi-discrete linear shallow water equations at the current timestep, in the 
    nonlinear momentum form used in Huang et al 2008.

    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param h: water depth.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param y: azimuthal coordinate field.
    :param f0: constant coefficient of Coriolis parameter.
    :param beta: linear coefficient of Coriolis parameter.
    :param g: gravitiational acceleration.
    :return: strong residual for shallow water equations in momentum form at current timestep.
    """
    # TODO: include optionality for BCs

    (u, v, eta) = (q[0], q[1], q[2])
    (u_, v_, eta_) = (q_[0], q_[1], q_[2])
    um = timestepScheme(u, u_, timestepper)
    vm = timestepScheme(v, v_, timestepper)
    em = timestepScheme(eta, eta_, timestepper)
    H = h + eta
    H_ = h + eta_
    Hm = h + em
    Hu2 = Hm * um * um
    Huv = Hm * um * vm
    Hv2 = Hm * vm * vm
    f = f0 + beta * y

    Au = Hm * (u - u_) + (H - H_) * um + Dt * (Hu2.dx(0) + Huv.dx(1) - f * Hm * vm + g * Hm * em.dx(0))
    Av = Hm * (v - v_) + (H - H_) * vm + Dt * (Huv.dx(0) + Hv2.dx(1) + f * Hm * um + g * Hm * em.dx(1))
    Ae = eta - eta_ + Dt * div(as_vector(Hm * um, Hm * vm))

    return Au, Av, Ae


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
    return c - c_ + Dt * inner(u, grad(cm)) - Dt * Constant(nu) * div(grad(cm))


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
    return ((c - c_) * ct - Dt * inner(cm * u, grad(ct)) + Dt * Constant(nu) * inner(grad(cm), grad(ct))) * dx


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
    F = (inner(M - M_, Mt) + Dt * inner(dot(w, nabla_grad(Mm)), Mt)) * dx
    return F
