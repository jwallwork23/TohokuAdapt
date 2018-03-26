from thetis import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock

import numpy as np
from time import clock
import datetime

import utils.error as err
import utils.forms as form
import utils.misc as msc
import utils.options as opt

# Get date and parameters
now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

op = opt.Options(vscale=0.1,
                 family='dg-dg',
                 rm=20,
                 gradate=True,
                 orderChange=1,
                 ndump=10,
                 Tstart=0.5,
                 Tend=2.5,
                 hmin=5e-2,
                 hmax=1.)
Ts = op.Tstart
T = op.Tend
dt = 0.025
dirName = 'plots/pyadjointTest/'

# Set up Mesh
lx = 2 * np.pi
n = pow(2, 4)
mesh_H = SquareMesh(n, n, lx, lx)
x, y = SpatialCoordinate(mesh_H)
P1_2d = FunctionSpace(mesh_H, "CG", 1)
eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
b = Function(P1_2d).assign(0.1)

# Define initial FunctionSpace and variables of problem and apply initial conditions
V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
q = Function(V_H)
uv_2d, elev_2d = q.split()
elev_2d.interpolate(eta0)
uv_2d.rename("uv_2d")
elev_2d.rename("elev_2d")

# Set up adjoint variables
dual = Function(V_H)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Define indicator function
k = Function(V_H)
k0, k1 = k.split()
iA = form.indicator(V_H.sub(1), mode='shallow-water')
iA.rename("Region of interest")
File(dirName+"indicator.pvd").write(iA)
k1.assign(iA)
J = assemble(inner(k, q) * dx)
# Jfunc = assemble(inner(k, q_) * dx)
# Jfuncs = [Jfunc]

# Get solver parameter values and construct solver
msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
solver_obj = solver2d.FlowSolver2d(mesh_H, b)
solver_obj.create_equations()
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.use_grad_div_viscosity_term = False
options.simulation_export_time = dt * op.rm
options.simulation_end_time = T
options.timestepper_type = op.timestepper
options.timestep = dt
options.output_directory = dirName
options.export_diagnostics = True
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

# Output OF values
cb1 = err.ShallowWaterCallback(solver_obj)
cb1.output_dir = dirName
cb1.append_to_log = True
cb1.export_to_hdf5 = False
solver_obj.add_callback(cb1, 'timestep')

# TODO: Callback for adding OF in pyadjoint sense?

# Apply ICs and time integrate
solver_obj.assign_initial_conditions(elev=eta0)
# def selector():
#     t = solver_obj.simulation_time
#     rm = 20
#     dt = options.timestep
#     options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
# solver_obj.iterate(export_func=selector)
solver_obj.iterate()

print("Objective value = ", cb1.__call__()[1])

print('\nStarting fixed mesh dual run (backwards in time)')
dualTimer = clock()
dJdb = compute_gradient(J, Control(b)) # TODO: Perhaps could make a different, more relevant calculation?
tape = get_working_tape()
tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]

t = op.Tend
adjointFile = File(dirName + 'adjoint.pvd')
for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    dual_u, dual_e = dual.split()
    print('t = %.2fs' % t)
    adjointFile.write(dual_u, dual_e, time=t)
    t -= dt
dualTimer = clock() - dualTimer
print('Adjoint run complete. Run time: %.3fs' % dualTimer)
File(dirName + 'gradient.pvd').write(dJdb)
