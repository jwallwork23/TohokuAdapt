from firedrake import *


def objectiveFunctionalSW(eta, t, timestep, Tstart=300., Tend=1500., x1=490e3, x2=640e3, y1=4160e3, y2=4360e3):
    """
    :param eta: free surface displacement solution.
    :param t: current time value (s).
    :param timestep: time step (s).
    :param Tstart: first time considered as relevant (s).
    :param Tend: last time considered as relevant (s).
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :return: objective functional for shallow water equations. 
    """

    # Create a 'smoothened' indicator function for region A = [x1, x2] x [y1, y1]
    xd = (x2 - x1) / 2
    yd = (y2 - y1) / 2
    indicator = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
                'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.'\
                % (x1, x2, y1, y2, x1 + xd, xd, y1 + yd, yd)
    iA = Function(eta.function_space()).interpolate(Expression(indicator))

    # Modify forcing term to 'smoothen' in time
    coeff = Constant(1.)
    if t.dat.data < Tstart + 1.5 * timestep:
        coeff.assign(0.5)
    elif t.dat.data < Tstart + 0.5 * timestep:
        coeff.assign(0.)

    from firedrake_adjoint import dt

    return (coeff * eta * iA) * dx * dt[Tstart:Tend]


def strongResidualSW(q, q_, b, Dt, nu=0., timestepper='CrankNicolson'):
    """
    Construct the strong residual for linear shallow water equations at the current timestep, using Crank Nicolson
    timestepping to express as a 'stationary' PDE `Lq-s=0`, where the 'source term' s depends on the data from the
    previous timestep.
    
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :return: strong residual for shallow water equations at current timestep.
    """
    u, eta = q.split()
    u_, eta_ = q_.split()

    if timestepper == 'CrankNicolson':
        um = 0.5 * (u + u_)
        em = 0.5 * (eta + eta_)
    elif timestepper == 'ImplicitEuler':
        um = u
        em = eta
    elif timestepper == 'ExplicitEuler':
        um = u_
        em = eta_
    else:
        raise NotImplementedError

    b = Constant(3000.)  # TODO: don't assume flat bathymetry

    Au = u - u_ + Dt * 9.81 * grad(em)
    Ae = eta - eta_ + Dt * div(b * um)

    if nu != 0.:
        Au += div(nu * (grad(um) + transpose(grad(um))))

    return Au, Ae


# TODO: implement Galerkin Least Squares (GLS) stabilisation

def weakResidualSW(q, q_, qt, b, Dt, nu=0., timestepper='CrankNicolson'):
    """
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param qt: test function tuple.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :return: weak residual for shallow water equations at current timestep.
    """
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])

    if timestepper == 'CrankNicolson':
        um = 0.5 * (u + u_)
        em = 0.5 * (eta + eta_)
    elif timestepper == 'ImplicitEuler':
        um = u
        em = eta
    elif timestepper == 'ExplicitEuler':
        um = u_
        em = eta_
    else:
        raise NotImplementedError

    F = (inner(u - u_, w) + inner(eta - eta_, xi) + 9.81 * inner(grad(em), w) - inner(b * um, grad(xi))) * dx

    if nu != 0.:
        F -= nu * inner(grad(um) + transpose(grad(um)), grad(w))

    return F


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
    if timestepper == 'CrankNicolson':
        cm = 0.5 * (c + c_)
    elif timestepper == 'ImplicitEuler':
        cm = c
    elif timestepper == 'ExplicitEuler':
        cm = c_
    else:
        raise NotImplementedError
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
    if timestepper == 'CrankNicolson':
        cm = 0.5 * (c + c_)
    elif timestepper == 'ImplicitEuler':
        cm = c
    elif timestepper == 'ExplicitEuler':
        cm = c_
    else:
        raise NotImplementedError
    return ((c - c_) * ct - Dt * inner(cm * u, grad(ct)) + Dt * Constant(nu) * inner(grad(cm), grad(ct))) * dx
