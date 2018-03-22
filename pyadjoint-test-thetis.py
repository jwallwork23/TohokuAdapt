from thetis import *
from thetis.field_defs import field_metadata
from firedrake_adjoint import *

import numpy as np
from time import clock
import datetime

import utils.adaptivity as adap
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt

# Get date and parameters
now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
dt_meas = dt

class AnnotationCallback(DiagnosticCallback):
    """Base class for callbacks that annotate tape."""
    variable_names = ['Annotation']

    def __init__(self, solver_obj, **kwargs):
        """
        Creates error comparison check callback object

        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        super(AnnotationCallback, self).__init__(solver_obj, **kwargs)

    def __call__(self):
        # Track adjoint data
        t = self.solver_obj.simulation_time
        dt = self.solver_obj.options.timestep
        T = self.solver_obj.options.simulation_end_time
        finished = True if t > T + 0.5 * dt else False      # + 1 because callback happens after incrementation
        if t < 1.5 * dt:                                    # + 1 because callback happens after incrementation
            adj_start_timestep()
            value = 'start'
        else:
            adj_inc_timestep(time=t, finished=finished)
            value = 'end' if finished else 'middle'

        return value, value

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line

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
b = Function(P1_2d).assign(0.1, annotate=False)

# Define initial FunctionSpace and variables of problem and apply initial conditions
V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
q = Function(V_H)
uv_2d, elev_2d = q.split()
elev_2d.interpolate(eta0)
uv_2d.rename("uv_2d")
elev_2d.rename("elev_2d")

# Set up adjoint problem
J = Functional(elev_2d * form.indicator(V_H.sub(1), mode='shallow-water') * dx * dt_meas)
dual = Function(V_H)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Get solver parameter values and construct solver
msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
solver_obj = solver2d.FlowSolver2d(mesh_H, b)
solver_obj.create_equations()
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.use_grad_div_viscosity_term = False
options.simulation_export_time = dt * (op.rm-1)
options.simulation_end_time = T
options.timestepper_type = op.timestepper
options.timestep = dt
options.output_directory = dirName
options.export_diagnostics = True
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

# # Output OF values
# cb1 = err.ShallowWaterCallback(solver_obj)
# cb1.output_dir = dirName
# cb1.append_to_log = True
# cb1.export_to_hdf5 = False
# solver_obj.add_callback(cb1, 'timestep')

# # Track adjoint data
# cb2 = AnnotationCallback(solver_obj)
# cb2.output_dir = dirName
# cb2.append_to_log = False
# cb2.append_to_hdf5 = False
# cb2.export_to_hdf5 = False
# solver_obj.add_callback(cb2, 'timestep')

# Apply ICs and time integrate
solver_obj.assign_initial_conditions(elev=eta0)
def selector():
    t = solver_obj.simulation_time
    rm = 20
    dt = options.timestep
    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
solver_obj.iterate(export_func=selector)

msc.dis('\nStarting fixed mesh dual run (backwards in time)', op.printStats)
cntT = int(np.ceil(T/dt))
cnt = cntT
save = True
parameters["adjoint"]["stop_annotating"] = True  # Stop registering equations
dualTimer = clock()
for (variable, solution) in compute_adjoint(J):
    print(variable, solution)
    if save:
        # Load adjoint data and save to HDF5
        dual.assign(variable, annotate=False)
        dual_u, dual_e = dual.split()
        dual_u.rename('Adjoint velocity')
        dual_e.rename('Adjoint elevation')
        with DumbCheckpoint(dirName + 'hdf5/adjoint_' + msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
            saveAdj.store(dual_u)
            saveAdj.store(dual_e)
            saveAdj.close()
        msc.dis('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT * 100), op.printStats)
        cnt -= 1
        save = False
    else:
        save = True
    if cnt == -1:
        break