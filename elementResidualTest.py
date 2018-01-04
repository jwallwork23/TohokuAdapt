from firedrake import *
import numpy as np


def interelementTerm(v, n=None):
    """
    :param v: Function to be averaged over element boundaries.
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


# Solve Helmholtz' equation in DG1 space. (By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y))
mesh = UnitSquareMesh(8, 8)
x, y = SpatialCoordinate(mesh)
DG1 = FunctionSpace(mesh, "DG", 1)
f_DG1 = Function(DG1).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
u_h_DG1 = Function(DG1)
v_DG1 = TestFunction(DG1)
B_DG1 = (inner(grad(u_h_DG1), grad(v_DG1)) + u_h_DG1*v_DG1)*dx
L_DG1 = f_DG1*v_DG1*dx
F_DG1 = B_DG1 - L_DG1
solve(F_DG1 == 0, u_h_DG1)
File("plots/u_h.pvd").write(u_h_DG1)

# Interpolate solution into DG2 space
DG2 = FunctionSpace(mesh, "DG", 2)
u_h = Function(DG2, name='Approximation').interpolate(u_h_DG1)

# Define element residual problem
f = Function(DG2).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
e_h = Function(DG2, name='Error estimate')
v = TestFunction(DG2)
B = (inner(grad(e_h), grad(v)) + e_h*v)*dx                          # LHS bilinear form on prognostic variable
B_ = (inner(grad(u_h), grad(v)) + u_h*v)*dx                         # LHS form on data
L = f*v*dx                                                          # RHS linear form
I = interelementTerm(grad(u_h)*v, n=FacetNormal(mesh))*dS           # Interelement flux term
F = B - L + B_ - I
solve(F == 0, e_h)
e_h.dat.data[:] = np.abs(e_h.dat.data)
File("plots/e_h.pvd").write(e_h)

# Generate 'error in the error'
e = Function(DG2, name='True error').interpolate(cos(x*pi*2)*cos(y*pi*2))
e.dat.data[:] = np.abs(e.dat.data - u_h.dat.data)
File("plots/e.pvd").write(e)
errorInError = Function(DG2)
errorInError.dat.data[:] = np.abs(e_h.dat.data - e.dat.data)
File("plots/errorInError.pvd").write(errorInError)