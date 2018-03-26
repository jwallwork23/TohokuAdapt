from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock

import numpy as np
from time import clock

import utils.forms as form
import utils.options as opt


def formsSWcrankNicolson(q, q_, b, Dt, coriolisFreq=None, nonlinear=False, impermeable=True):
    V = q.function_space()
    u, eta = split(q)
    u_, eta_ = split(q_)
    w, xi = TestFunctions(q.function_space())
    a1 = Constant(0.5)
    a2 = Constant(0.5)
    g = 9.81

    B = ((inner(u, w) + inner(eta, xi)) / Dt * dx + a1 * g * inner(grad(eta), w)) * dx          # LHS bilinear form
    L = ((inner(u_, w) + inner(eta_, xi)) / Dt * dx - a2 * g * inner(grad(eta_), w)) * dx       # RHS linear functional
    L -= a2 * div(b * u_) * xi * dx     # Note: Don't "apply BCs" to linear functional
    if impermeable:
        B -= a1 * inner(b * u, grad(xi)) * dx
        if V.sub(0).ufl_element().family() != 'Lagrange':
            B += a1 * jump(b * u * xi, n=FacetNormal(V.mesh())) * dS
        # TODO: Test this
    else:
        B += a1 * div(b * u) * xi * dx
    if coriolisFreq:
        B += a1 * coriolisFreq * inner(as_vector((-u[1], u[0])), w) * dx
        L -= a2 * coriolisFreq * inner(as_vector((-u_[1], u_[0])), w) * dx
    if nonlinear:
        B += a1 * inner(dot(u, nabla_grad(u)), w) * dx
        L -= a2 * inner(dot(u_, nabla_grad(u_)), w) * dx

    return B, L


def forms(q, q_, Dt, b, op=opt.Options()):
    u, eta = split(q)
    u_, eta_ = split(q_)
    w, xi = TestFunctions(q.function_space())
    g = 1.
    a1 = a2 = Constant(0.5)
    B = (inner(u, w) + eta * xi) / Dt * dx  # LHS bilinear form
    L = (inner(u_, w) + eta_ * xi) / Dt * dx  # RHS linear functional
    B -= a1 * g * eta * div(w) * dx
    L += a2 * g * eta_ * div(w) * dx
    # B += a1 * g * inner(grad(eta), w) * dx
    # B -= a2 * g * inner(grad(eta_), w) * dx
    B -= a1 * inner(b * u, grad(xi)) * dx
    L += a2 * inner(b * u_, grad(xi)) * dx
    return B, L


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

# Define variational problem
u, eta = split(q)
u_, eta_ = split(q_)
B, L = forms(q, q_, Dt, b)
# B, L = formsSWcrankNicolson(q, q_, b, Dt)
F = B - L
forwardProblem = NonlinearVariationalProblem(F, q)
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
Jfunc = assemble(inner(k, q_) * dx)
Jfuncs = [Jfunc]
while t < Tend + dt:
    # Solve problem and update solution
    forwardSolver.solve()
    q_.assign(q)

    # Update OF
    Jfunc = assemble(inner(k, q_) * dx)
    Jfuncs.append(Jfunc)

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

print('\nStarting fixed mesh dual run (backwards in time)')
dualTimer = clock()
dJdb = compute_gradient(J, Control(b)) # TODO: Perhaps could make a different, more relevant calculation?
tape = get_working_tape()
# tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]

for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    if cnt % rm == 0:
        adjointFile.write(dual_u, dual_e, time=t)
        print('t = %.2fs' % t)
    t -= dt

dualTimer = clock() - dualTimer
print('Adjoint run complete. Run time: %.3fs' % dualTimer)
File(dirName + 'gradient.pvd').write(dJdb)
