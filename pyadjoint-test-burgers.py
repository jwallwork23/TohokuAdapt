from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock   # Need checkout Sebastians `linear-solver` branch of pyadjoint
from firedrake.expression import Expression     # Can't currently annotate Expression objects in firedrake_adjoint

import numpy as np

n = 30
mesh = UnitSquareMesh(n, n)
x, y = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
Vs = FunctionSpace(mesh, "CG", 2)
u = Function(V).interpolate(Expression([sin(2*pi*x), cos(2*pi*y)]))
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

def indicator(W):
    k = Function(W, name='Region of interest')
    k.interpolate(Expression('(x[0] > 0.) & (x[0] < 0.2) & (x[1] > 0.4) & (x[1] < 0.6) ? 1. : 0.'))
    return k
k = indicator(Vs)

t = 0.0
end = 0.1
Jtemp = assemble(k*inner(u,u)*dx)
Jlist = [Jtemp]
while (t <= end):
    solve(F == 0, u_next, bc)
    u.assign(u_next)
    t += float(dt)

    Jtemp = assemble(k*inner(u, u)*dx)
    Jlist.append(Jtemp)

    print("t = ", np.round(t, 3))
    forwardFile.write(u)
t -= float(dt)

J = 0
for i in range(1, len(Jlist)):
    J += 0.5*(Jlist[i-1] + Jlist[i])*float(dt)

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

File("plots/pyadjoint_test/gradient.pvd").write(dJdu)
print("Gradient = ", float(dJdnu.dat.data))
