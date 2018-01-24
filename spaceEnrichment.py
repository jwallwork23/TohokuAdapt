from firedrake import *

from time import clock

import utils.adaptivity as adap
import utils.interpolation as inte


# Solve Helmholtz' equation in CG2 space. (By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y))
mesh = UnitSquareMesh(8, 8)
x, y = SpatialCoordinate(mesh)
CG2 = FunctionSpace(mesh, "CG", 2)
f_CG2 = Function(CG2).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
u_H_CG2 = Function(CG2)
v_CG2 = TestFunction(CG2)
B_CG2 = (inner(grad(u_H_CG2), grad(v_CG2)) + u_H_CG2*v_CG2)*dx
L_CG2 = f_CG2*v_CG2*dx
F_CG2 = B_CG2 - L_CG2
solve(F_CG2 == 0, u_H_CG2)
File("plots/approxLowOrder.pvd").write(u_H_CG2)

# Interpolate solution into CG3 space
pTimer = clock()
CG3 = FunctionSpace(mesh, "CG", 3)
u_H = Function(CG3, name='Approximation').interpolate(u_H_CG2)
File("plots/approxHighOrder.pvd").write(u_H)
f_H = Function(CG3).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
v_H = TestFunction(FunctionSpace(mesh, "DG", 0))
R = assemble(v_H * (- div(grad(u_H)) + u_H - f_H) * dx)
R.rename("Residual by order increase")
File("plots/residualOI.pvd").write(R)
pTimer = clock() - pTimer

# Interpolate solution into iso-P2 refined space
hTimer = clock()
mesh_h = adap.isoP2(mesh)
x, y = SpatialCoordinate(mesh_h)
u_h = inte.interp(mesh_h, u_H_CG2)[0]
CG2_h = FunctionSpace(mesh_h, "CG", 2)
f_h = Function(CG2_h).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
v_h = TestFunction(FunctionSpace(mesh_h, "DG", 0))
R_h = assemble(v_h * (- div(grad(u_h)) + u_h - f_h) * dx)
R_H = inte.interp(mesh_h, R_h)[0]
R_H.rename("Residual by refinement")
File("plots/residualRef.pvd").write(R_H)
hTimer = clock() - hTimer

print("""
Enrichment by order increase: %.3fs
Enrichment by refinement:     %.3fs""" % (pTimer, hTimer))
