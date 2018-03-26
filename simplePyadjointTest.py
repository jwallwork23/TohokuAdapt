from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock   # Need checkout Sebastians `linear-solver` branch of pyadjoint

import numpy as np

n = 30
mesh = UnitSquareMesh(n, n)
x, y = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
V0 = FunctionSpace(mesh, "CG", 2)

u0 = Function(V0).interpolate(sin(2*pi*x))
u1 = Function(V0).interpolate(cos(2*pi*y))
u = Function(V).interpolate(as_vector([u0, u1]))    # Can't currently annotate Expression objects in firedrake_adjoint
control = Control(u)

u_next = Function(V)
v = TestFunction(V)

nu = Constant(0.0001)
dt = Constant(0.01)
F = (inner((u_next - u)/dt, v)
     + inner(grad(u_next)*u_next, v)
     + nu*inner(grad(u_next), grad(v)))*dx
bc = DirichletBC(V, (0.0, 0.0), "on_boundary")

forwardFile = File("plots/pyadjoint_test/burgers.pvd")
forwardFile.write(u)

t = 0.0
cnt = 0
end = 0.1
Jtemp = assemble(inner(u,u)*dx)
Jlist = [Jtemp]
while (t <= end):
    solve(F == 0, u_next, bc)
    u.assign(u_next)
    t += float(dt)
    cnt += 1

    Jtemp = assemble(inner(u, u)*dx)
    Jlist.append(Jtemp)

    print("t = ", np.round(t, 3))
    forwardFile.write(u)
t -= float(dt)

J = 0
for i in range(1, len(Jlist)):
    J += 0.5*(Jlist[i-1] + Jlist[i])*float(dt)
# def indicator(P1):
#     k = Function(P1, name='Region of interest', annotate=False)
#     k.interpolate(Expression('(x[0] > 2.5) & (x[0] < 3.5) & (x[1] > 0.1) & (x[1] < 0.9) ? 1. : 0.'), annotate=False)
#     return k
# k = form.indicator(V, mode='advection-diffusion')

dual = Function(V, name="Adjoint")
dJdu, dJdnu = compute_gradient(J, [control, Control(nu)])
adjointFile = File("plots/pyadjoint_test/adjoint.pvd")

tape = get_working_tape()
# tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]

for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    adjointFile.write(dual, time=t)
    print("t = ", np.round(t, 3))
    t -= float(dt)
print("Gradient = ", float(dJdnu.dat.data))
