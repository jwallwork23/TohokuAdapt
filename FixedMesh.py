from thetis import *
import numpy as np

import utils.domain as dom
import utils.options as opt

print('\n******************************** FIXED MESH TSUNAMI SIMULATION ********************************\nOptions...')

# Set up mesh, initialise variables and specify parameters:
mesh, eta0, b = dom.TohokuDomain(res=int(input('Mesh coarseness (integer in range 1-5, default 4)?: ') or 4))
print('...... mesh loaded. Number of vertices : ', len(mesh.coordinates.dat.data))

# Get solver parameter values and construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.simulation_export_time = op.dt * op.ndump
options.simulation_end_time = op.T
options.timestepper_type = 'CrankNicolson'
options.timestep = op.dt
options.output_directory = 'plots/fixedMesh'

# Apply ICs:
solver_obj.assign_initial_conditions(elev=eta0)

# Run the model:
solver_obj.iterate()
