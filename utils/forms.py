from firedrake import *
# from firedrake_adjoint import *


dt_meas = dt  # Keep a reference to dt, the time-measure of Firedrake

def objectiveFunctional(eta, t, dt, Tstart=300., x1=490e3, x2=640e3, y1=4160e3, y2=4360e3):
    """
    :param eta: free surface displacement solution.
    :param t: current time value (s).
    :param dt: time step (s).
    :param Tstart: first time considered as relevant (s).
    :param x1: West-most coordinate for region A (m).
    :param x2: East-most coordinate for region A (m).
    :param y1: South-most coordinate for region A (m).
    :param y2: North-most coordinate for region A (m).
    :return: 
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
    if t.dat.data < Tstart + 1.5 * dt:
        coeff.assign(0.5)
    elif t.dat.data < Tstart + 0.5 * dt:
        coeff.assign(0.)

    return (coeff * eta * iA) * dx * dt_meas


def strongResidual(q, q_, b, Dt, nu=0.):
    """
    Construct the strong residual for linear shallow water equations at the current timestep, using Crank Nicolson
    timestepping to express as a 'stationary' PDE `Lq-s=0`, where the 'source term' s depends on the data from the
    previous timestep.
    
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :return: strong residual at current timestep.
    """
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    Au = u - u_ + Dt * 9.81 * grad(etah)
    Ae = eta - eta_ + Dt * div(b * uh)

    if nu != 0.:
        Au += div(nu * (grad(uh) + transpose(grad(uh))))

    return as_vector([Au[i] for i in range(2)] + [Ae])


# TODO: implement Galerkin Least Squares (GLS) stabilisation

def weakResidual(q, q_, qt, b, Dt, nu=0.):
    """
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param qt: test function tuple.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :param nu: coefficient for stress term.
    :return: weak residual at current timestep.
    """
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    (w, xi) = (as_vector((qt[0], qt[1])), qt[2])
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    F = (inner(u - u_, w) + inner(eta - eta_, xi) + 9.81 * inner(grad(etah), w) - inner(b * uh, grad(xi))) * dx

    if nu != 0.:
        F -= nu * inner(grad(uh) + transpose(grad(uh)), grad(w))

    return F
