from thetis import *
# from firedrake_adjoint import *

import numpy as np
from time import clock
import datetime

import utils.conversion as conv
import utils.error as err
import utils.mesh as msh
import utils.options as opt

now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
assert (float(physical_constants['g_grav'].dat.data) == 9.81)

def solverSW(startRes, op=opt.Options()):
    dirName = "plots/modelVerification/"

    # Establish Mesh, initial FunctionSpace and variables of problem and apply initial conditions
    mesh_H, eta0, b = msh.TohokuDomain(startRes, wd=op.wd)
    V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
    q = Function(V_H)
    uv_2d, elev_2d = q.split()
    elev_2d.interpolate(eta0)

    # Get Coriolis frequency
    f = Function(FunctionSpace(mesh_H, 'CG', 1))
    if op.rotational:
        Omega = 7.291e-5
        for i, v in zip(range(len(mesh_H.coordinates.dat.data)), mesh_H.coordinates.dat.data):
            f.dat.data[i] = 2 * Omega * np.sin(np.radians(conv.get_latitude(v[0], v[1], 54)))

    # Get timestep, ensuring simulation time is achieved exactly
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    solver_obj.create_equations()
    dt = min(np.abs(solver_obj.compute_time_step().dat.data))
    if dt > 3.:
        dt = 3.
    elif dt > 2.5:
        dt = 2.5
    elif dt > 2.:
        dt = 2.
    elif dt > 1.5:
        dt = 1.5
    elif dt > 1.:
        dt = 1.
    elif dt > 0.5:
        dt = 0.5
    elif dt > 0.25:
        dt = 0.25
    elif dt > 0.2:
        dt = 0.2
    elif dt > 0.1:
        dt = 0.1
    elif dt > 0.05:
        dt = 0.05

    # Get solver parameter values and construct solver
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True if op.nonlinear else False
    options.use_grad_depth_viscosity_term = False
    options.use_grad_div_viscosity_term = False
    options.coriolis_frequency = f
    options.simulation_export_time = dt * op.ndump
    options.simulation_end_time = op.Tend
    options.timestepper_type = op.timestepper
    options.timestep = dt
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
    # options.use_wetting_and_drying = op.wd        # TODO: Consider w&d
    # if op.wd:
    #     options.wetting_and_drying_alpha = alpha

    # Apply ICs
    solver_obj.assign_initial_conditions(elev=eta0)

    # Output objective functional computation error
    cb1 = err.TohokuCallback(solver_obj)
    cb1.output_dir = dirName
    cb1.append_to_log = True
    cb1.export_to_hdf5 = False
    solver_obj.add_callback(cb1, 'timestep')

    # Output gauge timeseries error
    cb2 = err.P02Callback(solver_obj)
    cb2.output_dir = dirName
    cb2.append_to_log = True
    cb2.export_to_hdf5 = False
    solver_obj.add_callback(cb2, 'timestep')

    # Output gauge timeseries error
    cb3 = err.P06Callback(solver_obj)
    cb3.output_dir = dirName
    cb3.append_to_log = True
    cb3.export_to_hdf5 = False
    solver_obj.add_callback(cb3, 'timestep')

    timer = clock()
    solver_obj.iterate()
    timer = clock() - timer
    J_h = err.getOF(dirName)  # Evaluate objective functional
    # TODO: also create a function to read gauge data from log

    return J_h, clock() - timer


if __name__ == '__main__':

    op = opt.Options(family='dg-dg',
                     wd=False,
                     # wd=True,
                     ndump=10)

    for i in (False, True):
        for j in (False, True):
            op.nonlinear = i
            op.rotational = j
            filename = 'outdata/outputs/modelVerification/nonlinear=' + str(i) + '_'
            filename += 'rotational=' + str(j) + '_'
            textfile = open(filename + date + '.txt', 'w+')
            for k in range(11):
                print("\nNONLINEAR = %s, ROTATIONAL = %s, RUN %d\n" % (i, j, k))
                J_h, timing = solverSW(k, op=op)
                textfile.write('%d, %.4e, %.1f\n' % (k, J_h, timing))
            textfile.close()
