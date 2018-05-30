from firedrake import *

from time import clock

from utils.adaptivity import *
from utils.callbacks import *
from utils.interpolation import interp, mixedPairInterp
from utils.misc import indicator, indexString, peakAndDistance
from utils.options import Options
from utils.setup import problemDomain, RossbyWaveSolution


__all__ = ["fixedMeshAD", "hessianBasedAD", "fixedMesh", "hessianBased"]


def weakResidualAD(c, c_, w, op=Options(mode='advection-diffusion')):
    """
    :arg c: concentration solution at current timestep. 
    :arg c_: concentration at previous timestep.
    :arg w: wind field.
    :param op: Options parameter object.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    if op.timestepper == 'CrankNicolson':
        cm = 0.5 * (c + c_)
    else:
        raise NotImplementedError
    ct = TestFunction(c.function_space())
    F = ((c - c_) * ct / Constant(op.dt) + inner(grad(cm), w * ct)) * dx
    F += Constant(op.viscosity) * inner(grad(cm), grad(ct)) * dx
    return F


def fixedMeshAD(n=3, op=Options(mode='advection-diffusion', approach="fixedMesh")):
    forwardFile = File(op.di + "fixedMesh.pvd")

    # Initialise FunctionSpaces and variables
    mesh, phi0, BCs, w = problemDomain(n, op=op)
    V = FunctionSpace(mesh, "CG", 1)
    phi = Function(V).assign(phi0)
    phi.rename('Concentration')
    phi_next = Function(V, name='Concentration next')
    nEle = mesh.num_cells()
    F = weakResidualAD(phi_next, phi, w, op=op)

    t = 0.
    cnt = 0
    quantities = {}
    fullTimer = 0.
    if op.plotpvd:
        forwardFile.write(phi, time=t)
    iA = indicator(mesh, xy=[3., 0.], radii=0.5, op=op)
    J_list = [assemble(iA * phi * dx)]
    while t < op.Tend:
        # Solve problem at current timestep
        solverTimer = clock()
        solve(F == 0, phi_next, bcs=BCs)
        solverTimer = clock() - solverTimer
        fullTimer += solverTimer
        phi.assign(phi_next)

        J_list.append(assemble(iA * phi * dx))

        if op.plotpvd and (cnt % op.ndump == 0):
            forwardFile.write(phi, time=t)
            print('t = %.1fs' % t)
        t += op.dt
        cnt += 1

    J_h = 0.
    for i in range(1, len(J_list)):
        J_h += 0.5 * (J_list[i] + J_list[i-1]) * op.dt

    quantities['meanElements'] = nEle
    quantities['solverTimer'] = fullTimer
    quantities['J_h'] = J_h

    return quantities


def hessianBasedAD(n=3, op=Options(mode='advection-diffusion', approach="hessianBased")):
    forwardFile = File(op.di + "hessianBased.pvd")

    # Initialise FunctionSpaces and variables
    mesh, phi0, BCs, w = problemDomain(n, op=op)
    V = FunctionSpace(mesh, "CG", 1)
    phi = Function(V).assign(phi0)
    phi.rename('Concentration')
    phi_next = Function(V, name='Concentration next')
    nEle = mesh.num_cells()
    F = weakResidualAD(phi_next, phi, w, op=op)

    # Get adaptivity parameters
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    op.nVerT = mesh.num_vertices() * op.rescaling  # Target #Vertices

    # Initialise counters
    t = 0.
    cnt = 0
    adaptSolveTimer = 0.
    quantities = {}
    iA = indicator(mesh, xy=[3., 0.], radii=0.5, op=op)
    J_list = [assemble(iA * phi * dx)]
    while t <= op.Tend:
        adaptTimer = clock()
        if cnt % op.rm == 0:

            temp = Function(V).assign(phi)
            for l in range(op.nAdapt):

                # Construct metric
                M = steadyMetric(temp, op=op)
                if op.gradate:
                    metricGradation(mesh, M)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                if l < op.nAdapt-1:
                    temp = interp(mesh, temp)

            phi = interp(mesh, phi)
            phi.rename("Concentration")
            V = FunctionSpace(mesh, "CG", 1)
            phi_next = Function(V)
            iA = indicator(mesh, xy=[3., 0.], radii=0.5, op=op)

            # Re-establish bilinear form and set boundary conditions
            BCs, w = problemDomain(mesh=mesh, op=op)[2:]
            F = weakResidualAD(phi_next, phi, w, op=op)

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle

        # Solve problem at current timestep
        solverTimer = clock()
        solve(F == 0, phi_next, bcs=BCs)
        phi.assign(phi_next)
        J_list.append(assemble(iA * phi * dx))
        solverTimer = clock() - solverTimer

        # Print to screen, save data and increment counters
        if op.plotpvd and (cnt % op.ndump == 0):
            forwardFile.write(phi, time=t)
            print('t = %.1fs' % t)
        t += op.dt
        cnt += 1

        if cnt % op.rm == 0:
            av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
            adaptSolveTimer += adaptTimer + solverTimer

    J_h = 0.
    for i in range(1, len(J_list)):
        J_h += 0.5 * (J_list[i] + J_list[i - 1]) * op.dt

    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['J_h'] = J_h

    return quantities


from thetis import *


def fixedMesh(startRes, **kwargs):
    op = kwargs.get('op')

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh)
    if op.mode == 'rossby-wave':            # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])

    # Initialise solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.use_grad_div_viscosity_term = True              # Symmetric viscous stress
    options.use_lax_friedrichs_velocity = False             # TODO: This is a temporary fix
    options.coriolis_frequency = f
    options.simulation_export_time = op.dt * op.ndump
    options.simulation_end_time = op.Tend - 0.5 * op.dt
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = op.di
    if not op.plotpvd:
        options.no_exports = True
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = SWCallback(solver_obj)
    cb1.op = op
    if op.mode != 'tohoku':
        cb2 = MirroredSWCallback(solver_obj)
        cb2.op = op
        solver_obj.add_callback(cb2, 'timestep')
    else:
        cb3 = P02Callback(solver_obj)
        cb4 = P06Callback(solver_obj)
        solver_obj.add_callback(cb3, 'timestep')
        solver_obj.add_callback(cb4, 'timestep')
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs

    # Solve and extract timeseries / functionals
    quantities = {}
    solverTimer = clock()
    solver_obj.iterate()
    solverTimer = clock() - solverTimer
    quantities['J_h'] = cb1.quadrature()          # Evaluate objective functional
    quantities['Integrand'] = cb1.getVals()
    if op.mode != 'tohoku':
        quantities['J_h mirrored'] = cb2.quadrature()
        quantities['Integrand-mirrored'] = cb2.getVals()
    else:
        quantities['TV P02'] = cb3.totalVariation()
        quantities['TV P06'] = cb4.totalVariation()
        quantities['P02'] = cb3.getVals()
        quantities['P06'] = cb4.getVals()

    # Measure error using metrics, as in Huang et al.
    if op.mode == 'rossby-wave':
        peak, distance = peakAndDistance(solver_obj.fields.solution_2d.split()[1], op=op)
        quantities['peak'] = peak/peak_a
        quantities['dist'] = distance/distance_a
        quantities['spd'] = distance /(op.Tend * 0.4)

    # Output mesh statistics and solver times
    quantities['meanElements'] = mesh.num_cells()
    quantities['solverTimer'] = solverTimer
    quantities['adaptSolveTimer'] = 0.

    return quantities


def hessianBased(startRes, **kwargs):
    op = kwargs.get('op')

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh)
    uv_2d, elev_2d = Function(V).split()  # Needed to load data into
    elev_2d.interpolate(eta0)
    uv_2d.interpolate(u0)
    if op.mode == 'rossby-wave':    # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.nVerT = mesh.num_vertices() * op.rescaling   # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    cnt = 0
    t = 0.

    adaptSolveTimer = 0.
    quantities = {}
    while cnt < op.cntT:
        adaptTimer = clock()
        P1 = FunctionSpace(mesh, "CG", 1)

        if op.adaptField != 's':
            height = Function(P1).assign(elev_2d)
        if op.adaptField != 'f':
            spd = Function(P1).interpolate(sqrt(dot(uv_2d, uv_2d)))
        for l in range(op.nAdapt):                  # TODO: Test this functionality

            # Construct metric
            if op.adaptField != 's':
                M = steadyMetric(height, op=op)
            if op.adaptField != 'f' and cnt != 0:   # Can't adapt to zero velocity
                M2 = steadyMetric(spd, op=op)
                M = metricIntersection(M, M2) if op.adaptField == 'b' else M2
            if op.bAdapt and not (op.adaptField != 'f' and cnt == 0):
                M2 = steadyMetric(b, op=op)
                M = M2 if op.adaptField != 'f' and cnt == 0. else metricIntersection(M, M2)     # TODO: Convex combination?

            # Adapt mesh and interpolate variables
            if op.bAdapt or cnt != 0 or op.adaptField == 'f':
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            if l < op.nAdapt-1:
                if op.adaptField != 's':
                    height = interp(mesh, height)
                if op.adaptField != 'f':
                    spd = interp(mesh, spd)

        if cnt != 0 or op.adaptField == 'f':
            elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
            b, BCs, f = problemDomain(mesh=mesh, op=op)[3:]
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
        adaptTimer = clock() - adaptTimer

        # Solver object and equations
        adapSolver = solver2d.FlowSolver2d(mesh, b)
        adapOpt = adapSolver.options
        adapOpt.element_family = op.family
        adapOpt.use_nonlinear_equations = True
        adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
        adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
        adapOpt.simulation_export_time = op.dt * op.ndump
        adapOpt.simulation_end_time = t + op.dt * (op.rm - 0.5)
        adapOpt.timestepper_type = op.timestepper
        adapOpt.timestep = op.dt
        adapOpt.output_directory = op.di
        if not op.plotpvd:
            adapOpt.no_exports = True
        adapOpt.coriolis_frequency = f
        adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
        adapSolver.i_export = int(cnt / op.ndump)
        adapSolver.next_export_t = adapSolver.i_export * adapSolver.options.simulation_export_time
        adapSolver.iteration = cnt
        adapSolver.simulation_time = t
        for e in adapSolver.exporters.values():
            e.set_next_export_ix(adapSolver.i_export)

        # Establish callbacks and iterate
        cb1 = SWCallback(adapSolver)
        cb1.op = op
        if op.mode != 'tohoku':
            cb2 = MirroredSWCallback(adapSolver)
            cb2.op = op
        else:
            cb3 = P02Callback(adapSolver)
            cb4 = P06Callback(adapSolver)
            if cnt == 0:
                initP02 = cb3.init_value
                initP06 = cb4.init_value
        if cnt != 0:
            cb1.objective_value = quantities['Integrand']
            if op.mode != 'tohoku':
                cb2.objective_value = quantities['Integrand-mirrored']
            else:
                cb3.gauge_values = quantities['P02']
                cb3.init_value = initP02
                cb4.gauge_values = quantities['P06']
                cb4.init_value = initP06
        adapSolver.add_callback(cb1, 'timestep')
        if op.mode != 'tohoku':
            adapSolver.add_callback(cb2, 'timestep')
        else:
            adapSolver.add_callback(cb3, 'timestep')
            adapSolver.add_callback(cb4, 'timestep')
        adapSolver.bnd_functions['shallow_water'] = BCs
        solverTimer = clock()
        adapSolver.iterate()
        solverTimer = clock() - solverTimer
        quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
        quantities['Integrand'] = cb1.getVals()
        if op.mode != 'tohoku':
            quantities['J_h mirrored'] = cb2.quadrature()
            quantities['Integrand-mirrored'] = cb2.getVals()
        else:
            quantities['P02'] = cb3.getVals()
            quantities['P06'] = cb4.getVals()
            quantities['TV P02'] = cb3.totalVariation()
            quantities['TV P06'] = cb4.totalVariation()

        # Get mesh stats
        nEle = mesh.num_cells()
        mM = [min(nEle, mM[0]), max(nEle, mM[1])]
        Sn += nEle
        cnt += op.rm
        t += op.dt * op.rm
        av = op.printToScreen(int(cnt/op.rm+1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
        adaptSolveTimer += adaptTimer + solverTimer

        # Extract fields for next step
        uv_2d, elev_2d = adapSolver.fields.solution_2d.split()

    # Measure error using metrics, as in Huang et al.
    if op.mode == 'rossby-wave':
        peak, distance = peakAndDistance(elev_2d, op=op)
        quantities['peak'] = peak / peak_a
        quantities['dist'] = distance / distance_a
        quantities['spd'] = distance / (op.Tend * 0.4)

    # Output mesh statistics and solver times
    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['adaptSolveTimer'] = adaptSolveTimer

    return quantities
