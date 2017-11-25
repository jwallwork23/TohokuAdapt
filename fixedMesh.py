from thetis import *
from time import clock

import utils.mesh as msh
import utils.options as opt

# Set up mesh, initialise variables and specify parameters
print('******************** FIXED MESH TSUNAMI SIMULATION ********************')
res = 4     # Coarse resolution
mesh, eta0, b = msh.TohokuDomain(res=res)
print('...... mesh loaded. #Elements : %d. #Vertices : %d. \n' % msh.meshStats(mesh))

# Get solver parameter values and construct solver, with default dg1-dg1 space
tic = clock()
op = opt.Options()
op.checkCFL(b)
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.simulation_export_time = op.dt * op.ndump
options.simulation_end_time = op.T
options.timestepper_type = op.timestepper
options.timestep = op.dt
options.output_directory = 'plots/fixedMesh/' + msh.MeshSetup(res).meshName
options.export_diagnostics = True
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

# Apply ICs and time integrate
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.iterate()
fixedMeshTime = clock() - tic
print('Time elapsed for fixed mesh solver: %.1fs (%.2fmins)' % (fixedMeshTime, fixedMeshTime / 60))