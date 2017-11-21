from firedrake import *

g = 9.81

def R(q, q_, b, Dt):
    """
    Construct the strong residual for linear shallow water equations at the current timestep, using Crank Nicolson
    timestepping to express as a 'stationary' PDE `Lq-s=0`, where the 'source term' s depends on the data from the
    previous timestep.
    
    :param q: solution tuple for linear shallow water equations.
    :param q_: solution tuple for linear shallow water equations at previous timestep.
    :param b: bathymetry profile.
    :param Dt: timestep expressed as a FiredrakeConstant.
    :return: strong residual at current timestep.
    """
    (u, eta) = (as_vector((q[0], q[1])), q[2])
    (u_, eta_) = (as_vector((q_[0], q_[1])), q_[2])
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    Au = u - u_ + Dt * g * grad(etah)
    Ae = eta - eta_ + Dt * div(b * uh)

    Aui = [Au[i] for i in range(0, 2)]

    return as_vector(Aui + [Ae])