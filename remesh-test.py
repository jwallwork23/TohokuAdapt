from thetis import *
from thetis.field_defs import field_metadata

from time import clock
import numpy as np
import datetime

from utils.adaptivity import *
from utils.callbacks import *
from utils.interpolation import interp, mixedPairInterp
from utils.setup import problemDomain, RossbyWaveSolution
from utils.options import Options


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

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.nVerT = mesh.num_vertices() * op.rescaling   # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    cnt = 0
    endT = 0.

    adaptSolveTimer = 0.
    quantities = {}
    while cnt < op.final_index:
        adaptTimer = clock()
        for l in range(op.adaptations):

            # Construct metric
            if op.adapt_field != 's':
                M = steadyMetric(elev_2d, op=op)
            if cnt != 0:  # Can't adapt to zero velocity
                if op.adapt_field != 'f':
                    spd = Function(FunctionSpace(mesh, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                    M2 = steadyMetric(spd, op=op)
                    M = metricIntersection(M, M2) if op.adapt_field == 'b' else M2
            if op.adapt_on_bathymetry:
                M2 = steadyMetric(b, op=op)
                M = M2 if op.adapt_field != 'f' and cnt == 0. else metricIntersection(M, M2)

            # Adapt mesh and interpolate variables
            if op.adapt_on_bathymetry or cnt != 0 or op.adapt_field == 'f':
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                if cnt != 0:
                    uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
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
        adapOpt.simulation_export_time = op.timestep * op.timesteps_per_export
        startT = endT
        endT += op.timestep * op.timesteps_per_remesh
        adapOpt.simulation_end_time = endT
        adapOpt.timestepper_type = op.timestepper
        adapOpt.timestep = op.timestep
        adapOpt.output_directory = op.di
        adapOpt.export_diagnostics = True
        adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        adapOpt.coriolis_frequency = f
        field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
        e = exporter.ExportManager(op.di + 'hdf5',
                                   ['elev_2d', 'uv_2d'],
                                   field_dict,
                                   field_metadata,
                                   export_type='hdf5')
        adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
        adapSolver.i_export = int(cnt / op.timesteps_per_export)
        adapSolver.iteration = cnt
        adapSolver.simulation_time = startT
        adapSolver.next_export_t = startT + adapOpt.simulation_export_time  # For next export
        for e in adapSolver.exporters.values():
            e.set_next_export_ix(adapSolver.i_export)

        # Establish callbacks and iterate
        cb1 = SWCallback(adapSolver)
        cb1.op = op
        if cnt != 0:
            cb1.objective_value = quantities['Integrand']
        adapSolver.add_callback(cb1, 'timestep')
        adapSolver.bnd_functions['shallow_water'] = BCs
        solverTimer = clock()
        adapSolver.iterate()
        solverTimer = clock() - solverTimer
        quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
        quantities['Integrand'] = cb1.getVals()

        # Get mesh stats
        nEle = mesh.num_cells()
        mM = [min(nEle, mM[0]), max(nEle, mM[1])]
        Sn += nEle
        cnt += op.timesteps_per_remesh
        av = op.adaptation_stats(int(cnt/op.timesteps_per_remesh+1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.timestep)
        adaptSolveTimer += adaptTimer + solverTimer

    # Output mesh statistics and solver times
    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['adaptSolveTimer'] = adaptSolveTimer

    return quantities


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

rm = [12, 24, 48, 72]
for i in range(3):
    outfile = open("outdata/rossby-wave/rmTest_mesh"+str(i)+'_'+date+'.txt', 'w')
    for adaptations in [1, 2, 3, 4]:
        for j in range(len(rm)):
            op = Options(mode='rossby-wave', approach='hessianBased')
            op.timesteps_per_remesh = rm[j]
            op.adaptations = adaptations
            q = hessianBased(i, op=op)
            err = np.abs((op.J - q['J_h'])/op.J)
            timer = q['solverTimer']
            print("Mesh %d: rm = %d, #Elements = %d, J_h = %.4f, error = %.4f, Time = %.2fs"
                  % (i, rm[j], q['meanElements'], q['J_h'], err, timer))
            outfile.write("%d, %d, %d, %.4f, %.4f, %.2f\n" % (i, rm[j], q['meanElements'], q['J_h'], err, timer))
        outfile.write("\n")
    outfile.close()
