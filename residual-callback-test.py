from thetis import *
from thetis.callback import MomentumResidualCallback, ContinuityResidualCallback

from utils.setup import problemDomain, solutionRW
from utils.misc import indexString
from utils.options import Options


def DWR(startRes, op=Options()):
    di = 'plots/' + op.mode + '/residual-test/'

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh_H, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh_H)
    q = Function(V)
    uv_2d, elev_2d = q.split()    # Needed to load data into
    uv_2d.rename('uv_2d')
    elev_2d.rename('elev_2d')

    # Solve fixed mesh primal problem to get residuals and adjoint solutions
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.use_grad_depth_viscosity_term = True if op.mode == 'tohoku' else False
    options.use_lax_friedrichs_velocity = False
    options.coriolis_frequency = f
    options.simulation_export_time = op.rm * op.dt
    options.simulation_end_time = op.Tend
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.timesteps_per_remesh = op.rm
    options.output_directory = di
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = MomentumResidualCallback(solver_obj)
    solver_obj.add_callback(cb1, 'export')
    cb2 = ContinuityResidualCallback(solver_obj)
    solver_obj.add_callback(cb2, 'export')
    solver_obj.bnd_functions['shallow_water'] = BCs
    solver_obj.iterate()
    R0 = cb1.__call__()[0]
    R1 = cb2.__call__()[0]
    File("plots/test1.pvd").write(R0)
    File("plots/test2.pvd").write(R1)

    rho = Function(V)
    rho_u, rho_e = rho.split()
    residualFile = File(di+"residual.pvd")
    for i in range(int(op.cntT/op.rm)):
        indexStr = indexString(i)
        with DumbCheckpoint(di + 'hdf5/MomentumResidual2d_' + indexStr, mode=FILE_READ) as loadRes:
            loadRes.load(rho_u, name="Momentum error")
            loadRes.close()
        with DumbCheckpoint(di + 'hdf5/ContinuityResidual2d_' + indexStr, mode=FILE_READ) as loadRes:
            loadRes.load(rho_e, name="Continuity error")
            loadRes.close()
        residualFile.write(rho_u, rho_e, time=float(i))


if __name__ == "__main__":
    import argparse
    import datetime


    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="Choose test problem from {'shallow-water', 'rossby-wave'} (default 'tohoku')")
    parser.add_argument("-res", help="Resolution index")
    parser.add_argument("-o", help="Output data")
    args = parser.parse_args()

    if args.t is None:
        mode = 'tohoku'
    else:
        mode = args.t
    print("Mode: %s" % mode)

    # Choose mode and set parameter values
    op = Options(mode=mode,
                 approach='DWR',
                 plotpvd=True if args.o else False)

    # Run simulation
    DWR(int(args.res), op=op)
