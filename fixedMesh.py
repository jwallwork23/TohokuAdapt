from thetis import *
import numpy as np

import utils.adaptivity as adap
import utils.mesh as msh
import utils.options as opt

# Set up mesh, initialise variables and specify parameters
print('******************** FIXED MESH TSUNAMI SIMULATION ********************\nOptions...')
mesh, eta0, b = msh.TohokuDomain(res=int(input('Mesh coarseness (integer in range 1-5, default 4)?: ') or 4))
print('...... mesh loaded. #Elements : %d. #Vertices : %d. \n' % adap.meshStats(mesh))

# Get solver parameter values and construct solver, with default dg1-dg1 space
op = opt.Options()
op.checkCFL(b)
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.use_nonlinear_equations = False
options.simulation_export_time = op.dt * op.ndump
options.simulation_end_time = op.T
options.timestepper_type = op.timestepper
options.timestep = op.dt
options.output_directory = 'plots/fixedMesh'
options.export_diagnostics = True
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

# Apply ICs and time integrate
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.iterate()
