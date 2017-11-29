from thetis import *
from thetis.field_defs import field_metadata
from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor


# Define initial mesh and mesh statistics placeholders
print('**************** GOAL-BASED ADAPTIVE TSUNAMI SIMULATION ***************\n')
print('GOAL-ORIENTED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = msh.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 4): ') or 4))
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d.' % (nEle, nVer))
dirName = 'plots/goalBased/'
msh.saveMesh(mesh, dirName + 'hdf5/mesh')
approach = input("Choose approach: 'fixdMesh', 'simpleAdapt' or 'goalBased': ") or 'simpleAdapt'

# Get solver parameters
op = opt.Options(vscale=0.2, rm=60)
iso = op.iso
if iso:
    dirName += 'isotropic/'
T = op.T
dt = op.dt
Dt = Constant(dt)
op.checkCFL(b)
ndump = op.ndump

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]            # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Create FunctionSpace upon which to define forms and residuals
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)

# Define Functions to hold residual and adjoint solution data
rho = Function(V)
rho_u, rho_e = rho.split()
rho_u.rename("Velocity residual")
rho_e.rename("Elevation residual")
dual = Function(V)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

if approach in ('fixedMesh', 'goalBased'):
    # Get solver parameter values and construct solver, with default dg1-dg1 space
    tic = clock()
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = False
    options.use_grad_depth_viscosity_term = False
    options.simulation_export_time = op.dt * op.ndump
    options.simulation_end_time = op.T
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = dirName + msh.MeshSetup(res).meshName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

    # Apply ICs and time integrate
    solver_obj.assign_initial_conditions(elev=eta0)
    solver_obj.iterate()
    fixedMeshTime = clock() - tic
    print('Time elapsed for fixed mesh solver: %.1fs (%.2fmins)' % (fixedMeshTime, fixedMeshTime / 60))

# TODO: forward simulation, similar to that in ``fixedMesh``. Make sure to include timesteps for adjoint calculation.
# TODO: calculate and save residual data, as in ``testSuite``.

# TODO: adjoint run, using included timesteps.
# TODO: load residual data and calculate error estimators. Save these data.

# TODO: adaptive run, similar to as in ``simpleAdapt``, but encorporating error estimates, as in ``testSuite``.
# TODO: integrate ``fixedMesh`` and  ``simpleAdapt`` into this script, to de-clutter repo.
