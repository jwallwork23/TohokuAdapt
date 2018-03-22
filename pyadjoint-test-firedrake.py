from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock

import numpy as np
from time import clock

import utils.forms as form


# dt_meas = dt

# Establish filenames
dirName = 'plots/pyadjoint_test/'
forwardFile = File(dirName + "forward.pvd")
adjointFile = File(dirName + "adjoint.pvd")

# Load Mesh(es)
lx = 2 * np.pi
n = pow(2, 4)
mesh_H = SquareMesh(n, n, lx, lx)  # Computational mesh
x, y = SpatialCoordinate(mesh_H)
P1_2d = FunctionSpace(mesh_H, "CG", 1)
eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
b = Function(P1_2d).assign(0.1, annotate=False)

# Define initial FunctionSpace and variables of problem and apply initial conditions
V_H = VectorFunctionSpace(mesh_H, "DG", 1) * FunctionSpace(mesh_H, "CG", 2)
q_ = Function(V_H)
u_, eta_ = q_.split()
eta_.interpolate(eta0)
q = Function(V_H)
q.assign(q_)
u, eta = q.split()
u.rename("uv_2d")
eta.rename("elev_2d")

# Specify physical and solver parameters
dt = 0.05
Dt = Constant(dt)
Tstart = 0.5
Tend = 2.5
rm = 5
a1 = a2 = 0.5   # For Crank-Nicolson timestepping
g = 9.81
iStart = int(Tstart / dt)
iEnd = int(np.ceil(Tend / dt))

dual = Function(V_H)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Intialise counters
t = 0.
cnt = 0
save = True

# Define variational problem
u, eta = split(q)
u_, eta_ = split(q_)
qt = TestFunction(V_H)
(w, xi) = (as_vector((qt[0], qt[1])), qt[2])

B = (inner(u, w) + eta * xi) / Dt * dx  # LHS bilinear form
L = (inner(u_, w) + eta_ * xi) / Dt * dx  # RHS linear functional
B -= a1 * g * eta * div(w) * dx
L += a2 * g * eta_ * div(w) * dx
B -= a1 * inner(b * u, grad(xi)) * dx
L += a2 * inner(b * u_, grad(xi)) * dx
forwardProblem = NonlinearVariationalProblem(B-L, q)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters={'mat_type': 'matfree',
                                                                              'snes_type': 'ksponly',
                                                                              'pc_type': 'python',
                                                                              'pc_python_type': 'firedrake.AssembledPC',
                                                                              'assembled_pc_type': 'lu',
                                                                              'snes_lag_preconditioner': -1,
                                                                              'snes_lag_preconditioner_persists': True})
u, eta = q.split()
u_, eta_ = q_.split()

# Define indicator function
k = Function(V_H)
k0, k1 = k.split()
iA = form.indicator(V_H.sub(1), mode='shallow-water')
iA.rename("Region of interest")
File(dirName+"indicator.pvd").write(iA)
k1.assign(iA)

print('Starting fixed mesh primal run (forwards in time)')
primalTimer = clock()
forwardFile.write(u, eta, time=t)
# J = Functional(inner(q, k) * dx * dt_meas)
Jfunc = assemble(inner(k, q_) * dx)
Jfuncs = [Jfunc]
while t < Tend + dt:
    # Solve problem and update solution
    forwardSolver.solve()
    q_.assign(q)

    # Update OF
    Jfunc = assemble(inner(k, q_) * dx)
    Jfuncs.append(Jfunc)

    # # Mark timesteps to be used in adjoint simulation
    # if t == 0.:
    #     adj_start_timestep()
    # else:
    #     adj_inc_timestep(time=t, finished= t >= Tend)

    if cnt % rm == 0:
        forwardFile.write(u, eta, time=t)
        print('t = %.2fs' % t)
    t += dt
    cnt += 1
cnt -=1
t -= dt
cntT = cnt  # Total number of steps
primalTimer = clock() - primalTimer
print('Primal run complete. Run time: %.3fs' % primalTimer)

# Establish OF
J = 0
for i in range(1, len(Jfuncs)):
    J += 0.5*(Jfuncs[i-1] + Jfuncs[i])*dt

# parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
print('\nStarting fixed mesh dual run (backwards in time)')
dualTimer = clock()
# for (variable, solution) in compute_adjoint(J):
#     if save:
#         dual.assign(variable)
#         if cnt % rm == 0:
#             adjointFile.write(dual_u, dual_e, time=t)
#             print('t = %.2fs' % t)
#         cnt -= 1
#         t -= dt
#         save = False
#     else:
#         save = True
#     if cnt == -1:
#         break

dJdnu = compute_gradient(J, Control(b)) # TODO: Perhaps could make a different, more relevant calculation?
tape = get_working_tape()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]

for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    if cnt % rm == 0:
        adjointFile.write(dual_u, dual_e, time=t)
        print('t = %.2fs' % t)
    t -= dt

# TODO: Now test this in Thetis!

dualTimer = clock() - dualTimer
print('Adjoint run complete. Run time: %.3fs' % dualTimer)
cnt += 1
