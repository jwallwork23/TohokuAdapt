from thetis import *

import numpy as np
from math import radians, sin, cos
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

import utils.domain as dom

print('\n******************************** FIXED MESH TSUNAMI SIMULATION ********************************\nOptions...')

# Set up mesh, initialise variables and specify parameters:
mesh, eta0, b = dom.TohokuDomain(res=int(input('Mesh coarseness (integer in range 1-5, default 4)?: ') or 4))
dt = float(input('Specify timestep (s) (default 1):') or 1.)
ndump = 60                                                                  # Inverse data dump frequency
T = float(input('Specify time period (mins) (default 60):') or 60.) * 60    # Default 1 hour simulation time period (s)
wd = bool(input('Press anything except enter to consider wetting-and-drying. '))
print('\n')                                                                 # End of options.

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.simulation_export_time = dt * ndump
options.simulation_end_time = T
options.timestepper_type = 'CrankNicolson'
options.timestep = dt
options.output_directory = 'plots/fixedMesh'
if wd:
    print('Using wetting-and-drying')
    options.use_wetting_and_drying = True

# Apply ICs:
solver_obj.assign_initial_conditions(elev=eta0)

# Run the model:
solver_obj.iterate()
