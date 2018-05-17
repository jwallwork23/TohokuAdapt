from thetis import *
from firedrake.petsc import PETSc

from time import clock

from utils.adaptivity import isoP2
from utils.interpolation import interp
from utils.setup import problemDomain
from utils.options import Options


def getObjective(mesh, b, op=Options()):

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh, u0, eta0, b, BCs, f = problemDomain(mesh=mesh, b=b, op=op)

    # Initialise solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.use_grad_div_viscosity_term = True  # Symmetric viscous stress
    options.use_lax_friedrichs_velocity = False  # TODO: This is a temporary fix
    options.coriolis_frequency = f
    options.simulation_export_time = op.dt * op.ndump
    options.simulation_end_time = op.Tend
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = op.di
    options.no_exports = True
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = SWCallback(solver_obj)
    cb1.op = op
    if op.mode == 'tohoku':
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
    quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
    quantities['Integrand'] = cb1.getVals()
    if op.mode == 'tohoku':
        quantities['TV P02'] = cb3.totalVariation()
        quantities['TV P06'] = cb4.totalVariation()
        quantities['P02'] = cb3.getVals()
        quantities['P06'] = cb4.getVals()

    # Measure error using metrics, as in Huang et al.
    if op.mode == 'rossby-wave':
        peak, distance = peakAndDistance(solver_obj.fields.solution_2d.split()[1], op=op)
        quantities['peak'] = peak / peak_a
        quantities['dist'] = distance / distance_a
        quantities['spd'] = distance / (op.Tend * 0.4)

    # Output mesh statistics and solver times
    quantities['meanElements'] = mesh.num_cells()
    quantities['solverTimer'] = solverTimer
    quantities['adaptSolveTimer'] = 0.

    return quantities


if __name__ == "__main__":

    op = Options(mode='tohoku')

    mesh, b = problemDomain(0, op=op)[0::3]

    OF = [getObjective(mesh, b, op)]

    for i in range(5):
        mesh = isoP2(mesh)      # Hierarchical refinement
        b = interp(mesh, b)
        OF.append(getObjective(mesh, b, op))
