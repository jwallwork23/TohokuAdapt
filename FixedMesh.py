from thetis import *
import numpy as np

import utils.domain as dom
import utils.options as opt

print('\n******************************** FIXED MESH TSUNAMI SIMULATION ********************************\nOptions...')

# Set up mesh, initialise variables and specify parameters:
mesh, eta0, b = dom.TohokuDomain(res=int(input('Mesh coarseness (integer in range 1-5, default 4)?: ') or 4))
plex = mesh._plex
eStart, eEnd = plex.getHeightStratum(0)
vStart, vEnd = plex.getDepthStratum(0)
nEle = eEnd - eStart
nVer = vEnd - vStart
print('...... mesh loaded. #Vertices : %d. #Elements : %d. \n' % (nVer, nEle))

# Get solver parameter values and construct solver, with default dg1-dg1 space:
op = opt.Options()
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.simulation_export_time = op.dt * op.ndump
options.simulation_end_time = op.T
options.timestepper_type = op.timestepper
options.timestep = op.dt

# Specify outfile directory and HDF5 checkpointing:
options.output_directory = 'plots/fixedMesh'
options.export_diagnostics = True
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

# Apply ICs and time integrate:
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.iterate()
