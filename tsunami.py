from thetis_adjoint import *
from thetis.field_defs import field_metadata
import pyadjoint
from fenics_adjoint.solving import SolveBlock

import numpy as np
from time import clock

from utils.adaptivity import isoP2, metricIntersection, steadyMetric
from utils.callbacks import *
from utils.forms import solutionRW
from utils.interpolation import interp
from utils.mesh import meshStats, problemDomain
from utils.misc import indexString, peakAndDistance
from utils.options import Options


def fixedMesh(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/fixedMesh/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        nEle = meshStats(mesh)[0]
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        if op.mode == 'rossby-wave':
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state

        # Initialise solver
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True if op.nonlinear else False   # TODO: Go nonlinear everywhere?
        options.use_grad_depth_viscosity_term = False
        options.use_grad_div_viscosity_term = False
        options.use_lax_friedrichs_velocity = False                         # TODO: This is a temporary fix
        if op.mode == 'rossby-wave':
            options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.ndump
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.log_output = op.printStats
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        options.use_wetting_and_drying = op.wd
        # if op.wd:                                                         # TODO: Calculate w&d alpha
        #     options.wetting_and_drying_alpha = alpha
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = SWCallback(solver_obj)
        cb1.op = op
        if op.mode == 'tohoku':
            cb2 = P02Callback(solver_obj)
            cb3 = P06Callback(solver_obj)
            solver_obj.add_callback(cb2, 'timestep')
            solver_obj.add_callback(cb3, 'timestep')
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs

        # Solve and extract timeseries / functionals
        solverTimer = clock()
        solver_obj.iterate()
        solverTimer = clock() - solverTimer
        J_h = cb1.quadrature()          # Evaluate objective functional
        integrand = cb1.__call__()[1]   # and get integrand timeseries
        if op.mode == 'tohoku':
            totalVarP02 = cb2.totalVariation()
            totalVarP06 = cb3.totalVariation()

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT/op.ndump)                                   # TODO: fix indexing error
            with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return nEle, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, solverTimer, 0.
        elif op.mode == 'tohoku':
            return nEle, rel, J_h, integrand, totalVarP02, totalVarP06, solverTimer, 0.
        else:
            return nEle, rel, J_h, integrand, solverTimer, 0.


def hessianBased(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/hessianBased/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        if op.mode == 'rossby-wave':
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)

        # Initialise parameters and counters
        nEle, op.nVerT = meshStats(mesh)
        op.nVerT *= op.rescaling  # Target #Vertices
        mM = [nEle, nEle]  # Min/max #Elements
        Sn = nEle
        cnt = 0
        endT = 0.

        while cnt < op.cntT:
            indexStr = indexString(int(cnt / op.ndump))

            # Load variables from disk
            if cnt != 0:
                V = op.mixedSpace(mesh)
                q = Function(V)
                uv_2d, elev_2d = q.split()
                with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            adaptTimer = clock()
            for l in range(op.nAdapt):                                          # TODO: Test this functionality

                # Construct metric
                if op.adaptField != 's':
                    M = steadyMetric(elev_2d, op=op)
                if cnt != 0:  # Can't adapt to zero velocity
                    if op.adaptField != 'f':
                        spd = Function(FunctionSpace(mesh, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                        M2 = steadyMetric(spd, op=op)
                        M = metricIntersection(M, M2) if op.adaptField == 'b' else M2

                # Adapt mesh and interpolate variables
                if cnt != 0 or op.adaptField == 'f':
                    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                    P1 = FunctionSpace(mesh, "CG", 1)
                    elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
                    if op.mode == 'tohoku':
                        b = interp(mesh, b)
                    elif op.mode == 'shallow-water':
                        b = Function(P1).assign(0.1)
                    else:
                        b = Function(P1).assign(1.)
                    uv_2d.rename('uv_2d')
                    elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True if op.nonlinear else False   # TODO: Go nonlinear everywhere?
            adapOpt.use_grad_depth_viscosity_term = False
            adapOpt.use_grad_div_viscosity_term = False
            adapOpt.use_lax_friedrichs_velocity = False                         # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = di
            adapOpt.export_diagnostics = True
            adapOpt.log_output = op.printStats
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.use_wetting_and_drying = op.wd
            # if op.wd:                                             #           # TODO: Calculate w&d alpha
            #     adapOpt.wetting_and_drying_alpha = alpha
            if op.mode == 'rossby-wave':
                adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh)[1])
            field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
            e = exporter.ExportManager(di + 'hdf5',
                                       ['elev_2d', 'uv_2d'],
                                       field_dict,
                                       field_metadata,
                                       export_type='hdf5')
            adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
            adapSolver.i_export = int(cnt / op.ndump)
            adapSolver.iteration = cnt
            adapSolver.simulation_time = startT
            adapSolver.next_export_t = startT + adapOpt.simulation_export_time  # For next export
            for e in adapSolver.exporters.values():
                e.set_next_export_ix(adapSolver.i_export)

            # Establish callbacks and iterate
            cb1 = SWCallback(adapSolver)
            cb1.op = op
            if op.mode == 'tohoku':
                cb2 = P02Callback(adapSolver)
                cb3 = P06Callback(adapSolver)
            adapSolver.add_callback(cb1, 'timestep')
            if cnt != 0:
                cb1.objective_value = integrand
                if op.mode == 'tohoku':
                    cb2.gauge_values = gP02
                    cb3.gauge_values = gP06
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode == 'tohoku':
                adapSolver.add_callback(cb2, 'timestep')
                adapSolver.add_callback(cb3, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            J_h = cb1.quadrature()
            integrand = cb1.__call__()[1]
            if op.mode == 'tohoku':
                gP02 = cb2.__call__()[1]
                gP06 = cb3.__call__()[1]
                totalVarP02 = cb2.totalVariation()
                totalVarP06 = cb3.totalVariation()

            # Get mesh stats
            nEle = meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':                                            # TODO: fix indexing error
            index = int(op.cntT / op.ndump)
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return av, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, solverTimer, adaptTimer
        elif op.mode == 'tohoku':
            return av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, solverTimer, adaptTimer
        else:
            return av, rel, J_h, integrand, solverTimer, adaptTimer


def DWR(startRes, op=Options()):
    di = 'plots/' + op.mode + '/DWR/'
    if op.plotpvd:
        residualFile = File(di + "residual.pvd")
        errorFile = File(di + "errorIndicator.pvd")
        adjointFile = File(di + "adjoint.pvd")

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh_H, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh_H)
    q = Function(V)
    uv_2d, elev_2d = q.split()    # Needed to load data into
    P1 = FunctionSpace(mesh_H, "CG", 1)
    q_ = Function(V)                        # Variable at previous timestep
    uv_2d_, elev_2d_ = q_.split()
    P0 = FunctionSpace(mesh_H, "DG", 0)
    if op.mode == 'rossby-wave':
        peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
    if op.orderChange:                      # Define variables on higher/lower order space
        Ve = op.mixedSpace(mesh_H, orderChange=op.orderChange)
        qe = Function(Ve)
        ue, ee = qe.split()
        qe_ = Function(Ve)
        ue_, ee_ = qe_.split()
        duale = Function(Ve)
        duale_u, duale_e = duale.split()
        be = b                              # TODO: Is this a valid thing to do?
    elif op.refinedSpace:                   # Define variables on an iso-P2 refined space
        mesh_h = isoP2(mesh_H)
        Ve = op.mixedSpace(mesh_h)
        be = problemDomain(mesh=mesh_h, op=op)[3]
        qe = Function(Ve)
        P0 = FunctionSpace(mesh_h, "DG", 0)
    else:                                   # Copy standard variables to mimic enriched space labels
        Ve = V
        qe = q
        be = b
    rho = Function(Ve)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    v = TestFunction(P0)                    # For extracting elementwise error indicators

    # Initialise parameters and counters
    nEle, op.nVerT = meshStats(mesh_H)
    op.nVerT *= op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    endT = 0.
    Dt = Constant(op.dt)
    cntT = int(np.ceil(op.Tend / op.dt))

    # Get initial boundary metric and TODO wetting and drying parameter
    if op.gradate or op.wd:
        H0 = Function(P1).interpolate(CellSize(mesh_H))
    if op.wd:
        g = constructGradient(elev_2d)
        spd = assemble(v * sqrt(inner(g, g)) * dx)
        gs = np.min(np.abs(spd.dat.data))
        print('#### gradient = ', gs)
        ls = np.min([H0.dat.data[i] for i in DirichletBC(P1, 0, 'on_boundary').nodes])
        print('#### ls = ', ls)
        alpha = Constant(gs * ls)
        print('#### alpha = ', alpha.dat.data)

    # Solve fixed mesh primal problem to get residuals and adjoint solutions
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True if op.nonlinear else False
    options.use_grad_depth_viscosity_term = False
    options.use_grad_div_viscosity_term = False
    options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
    if op.mode == 'rossby-wave':
        options.coriolis_frequency = f
    options.simulation_export_time = op.dt * (op.rm - 1)
    options.simulation_end_time = op.Tend
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.timesteps_per_remesh = op.rm
    options.output_directory = di
    options.export_diagnostics = True
    options.log_output = op.printStats
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
    # options.use_wetting_and_drying = op.wd                        # TODO: Establish w&d alpha
    # if op.wd:
    #     options.wetting_and_drying_alpha = alpha
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = ObjectiveSWCallback(solver_obj)
    cb1.op = op
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs
    def selector():
        rm = options.timesteps_per_remesh
        dt = options.timestep
        options.simulation_export_time = dt if int(solver_obj.simulation_time / dt) % rm == 0 else (rm - 1) * dt
    primalTimer = clock()
    solver_obj.iterate(export_func=selector)
    primalTimer = clock() - primalTimer
    J = cb1.assembleOF()                    # Assemble objective functional
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    # Compute gradient
    gradientTimer = clock()
    dJdb = compute_gradient(J, Control(b))                          # TODO: Rewrite pyadjoint to avoid computing this
    gradientTimer = clock() - gradientTimer
    # File(di + 'gradient.pvd').write(dJdb)     # Too memory intensive in Tohoku case
    print("Norm of gradient: %.3e. Time for computation: %.1fs" % (dJdb.dat.norm, gradientTimer))

    # Extract adjoint solutions
    dualTimer = clock()
    tape = get_working_tape()
    solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
    N = len(solve_blocks)
    r = N % op.rm  # Number of extra tape annotations in setup
    for i in range(N - 1, r - 2, -op.rm):
        dual.assign(solve_blocks[i].adj_sol)
        dual_u, dual_e = dual.split()
        dual_u.rename('Adjoint velocity')
        dual_e.rename('Adjoint elevation')
        with DumbCheckpoint(di + 'hdf5/adjoint_' + indexString(int((i - r + 1) / op.rm)), mode=FILE_CREATE) as saveAdj:
            saveAdj.store(dual_u)
            saveAdj.store(dual_e)
            saveAdj.close()
        if op.plotpvd:
            adjointFile.write(dual_u, dual_e, time=op.dt * (i - r + 1))
        if op.printStats:
            print('Adjoint simulation %.2f%% complete' % ((N - i + r - 1) / N * 100))
    dualTimer = clock() - dualTimer
    print('Dual run complete. Run time: %.3fs' % dualTimer)
    cnt = 0


if __name__ == "__main__":
    import argparse
    import datetime


    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'hessianBased', 'DWR'} (default fixedMesh)")
    parser.add_argument("-t", help="Choose test problem from {'shallow-water', 'rossby-wave'} (default Tohoku)")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-w", help="Use wetting and drying")
    args = parser.parse_args()
    approach = args.a
    if args.t is None:
        mode = 'tohoku'
    else:
        mode = args.t
        try:
            assert args.w is None
        except:
            raise ValueError("Wetting and drying not available for test cases.")
    if approach is None:
        approach = 'fixedMesh'
    else:
        assert approach in ('hessianBased', 'DWR')
    solver = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased, 'DWR': DWR}[approach]
    print("Mode: %s, approach: %s" % (mode, approach))

    # Choose mode and set parameter values
    op = Options(mode=mode,
                 # gradate=True if aposteriori and mode == 'tohoku' else False,
                 gradate=False,  # TODO: Fix this for tohoku case
                 plotpvd=True if args.o else False,
                 printStats=True,
                 wd=True if args.w else False)

    # Establish filename
    filename = 'outdata/' + mode + '/fixedMesh'
    if args.w:
        filename += '_w'
    filename += '_' + date
    errorfile = open(filename + '.txt', 'w+')
    integrandFile = open(filename + 'Integrand.txt', 'w+')

    # Run simulations
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    Jlist = np.zeros(len(resolutions))
    if mode == 'tohoku':
        g2list = np.zeros(len(resolutions))
        g6list = np.zeros(len(resolutions))
        gaugeFileP02 = open(filename + 'P02.txt', 'w+')
        gaugeFileP06 = open(filename + 'P06.txt', 'w+')
    for i in resolutions:
        # Get data and save to disk
        if mode == 'rossby-wave':
            av, rel, J_h, integrand, relativePeak, distance, phaseSpd, solverTime, adaptTime = solver(i, op=op)
            print("""Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs
        OF error: %.4e  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm"""
                  % (i, av, J_h, solverTime+adaptTime, rel, relativePeak, distance, phaseSpd))
            errorfile.write('%d, %.4e, %.4f, %.4f, %.4f, %.1f, %.4e\n'
                           % (av, rel, relativePeak, distance, phaseSpd, solverTime+adaptTime, J_h))
        elif mode == 'tohoku':
            av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, solverTime, adaptTime = solver(i, op=op)
            print("""Run %d: Mean element count: %6d Objective %.4e Timing %.1fs 
        OF error: %.4e P02: %.3f P06: %.3f""" % (i, av, J_h, solverTime+adaptTime, rel, totalVarP02, totalVarP06))
            errorfile.write('%d, %.4e, %.3f, %.3f, %.1f, %.4e\n'
                            % (av, rel, totalVarP02, totalVarP06, solverTime+adaptTime, J_h))
            gaugeFileP02.writelines(["%s," % val for val in gP02])
            gaugeFileP02.write("\n")
            gaugeFileP06.writelines(["%s," % val for val in gP06])
            gaugeFileP06.write("\n")
        else:
            av, rel, J_h, integrand, solverTime, adaptTime = solver(i, op=op)
            print('Run %d: Mean element count: %6d Objective: %.4e OF error %.4e Timing %.1fs'
                  % (i, av, J_h, rel, solverTime+adaptTime))
            errorfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, solverTime+adaptTime, J_h))
        integrandFile.writelines(["%s," % val for val in integrand])
        integrandFile.write("\n")

        # Calculate orders of convergence
        Jlist[i] = J_h
        if mode == 'tohoku':
            g2list[i] = totalVarP02
            g6list[i] = totalVarP06
        if i > 1:
            Jconv = (Jlist[i] - Jlist[i - 1]) / (Jlist[i - 1] - Jlist[i - 2])
            if mode == 'tohoku':
                g2conv = (g2list[i] - g2list[i - 1]) / (g2list[i - 1] - g2list[i - 2])
                g6conv = (g6list[i] - g6list[i - 1]) / (g6list[i - 1] - g6list[i - 2])
                print("Orders of convergence... J: %.4f, P02: %.4f, P06: %.4f" % (Jconv, g2conv, g6conv))
            else:
                print("Order of convergence: %.4f" % Jconv)
    errorfile.close()
    if mode == 'tohoku':
        gaugeFileP02.close()
        gaugeFileP06.close()
    integrandFile.close()
