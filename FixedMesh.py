from thetis import *

import numpy as np
from math import radians, sin, cos
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

import utils.domain as dom

print('\n******************************** FIXED MESH TSUNAMI SIMULATION ********************************\nOptions...')

# Set up mesh and initialise variables:
mesh, eta0, b = dom.TohokuDomain(res=int(input('Mesh coarseness (integer in range 1-5, default 4)?: ') or 4))

# Specify time parameters:
dt = float(input('Specify timestep (s) (default 1):') or 1.)
ndump = 60      # Inverse data dump frequency
T = 3600        # Simulation time period (s) of 1 hour

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.simulation_export_time = dt * ndump
options.simulation_end_time = T
options.timestepper_type = 'CrankNicolson'
options.timestep = dt
options.output_directory = 'plots/fixedMesh'

# Apply ICs:
solver_obj.assign_initial_conditions(elev=eta0)

# Run the model:
solver_obj.iterate()
