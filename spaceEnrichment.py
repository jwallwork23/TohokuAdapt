from firedrake import *

# Solve Helmholtz' equation in CG1 space. (By method of reconstructed solns exact soln is u=cos(2*pi*x)cos(2*pi*y))
mesh = UnitSquareMesh(8, 8)
x, y = SpatialCoordinate(mesh)
CG1 = FunctionSpace(mesh, "CG", 1)
f_CG1 = Function(CG1).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
u_h_CG1 = Function(CG1)
v_CG1 = TestFunction(CG1)
B_CG1 = (inner(grad(u_h_CG1), grad(v_CG1)) + u_h_CG1*v_CG1)*dx
L_CG1 = f_CG1*v_CG1*dx
F_CG1 = B_CG1 - L_CG1
solve(F_CG1 == 0, u_h_CG1)
File("plots/approxLowOrder.pvd").write(u_h_CG1)

# Interpolate solution into CG2 space
CG2 = FunctionSpace(mesh, "CG", 2)
u_h = Function(CG2, name='Approximation').interpolate(u_h_CG1)
File("plots/approxHighOrder.pvd").write(u_h)
f = Function(CG2).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

# Form residual in enriched space
v = TestFunction(FunctionSpace(mesh, "DG", 0))
R = assemble(v * (- nabla_grad(grad(u_h)) + u_h - f) * dx)
File("plots/residual.pvd").write(R)

# TODO: compare with enrichment by refinement
