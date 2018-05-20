from thetis import *
from firedrake.petsc import PETSc

from utils.callbacks import SWCallback
from utils.setup import problemDomain
from utils.options import Options


def getObjective(level, b=None, op=Options()):

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh, u0, eta0, b, BCs, f = problemDomain(level, b=b, op=op)
    print(type(b))
    print(b.dat.data)

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
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs

    # Solve and extract quantities
    quantities = {}
    solver_obj.iterate()
    quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
    quantities['Integrand'] = cb1.getVals()
    quantities['Element count'] = mesh.num_cells()

    return quantities, b


if __name__ == "__main__":

    op = Options(mode='tohoku')
    q, b = getObjective(0, op=op)
    OF = [q['J_h']]
    nEls = [q['Element count']]
    PETSc.Sys.Print("   Objective value %.4e" % OF[0])
    PETSc.Sys.Print("   Element count %d" % nEls[0])

    for level in range(1, 11):
        q = getObjective(level, b, op)[0]
        OF.append(q['J_h'])
        nEls.append(q['Element count'])
        PETSc.Sys.Print("   Objective value %.4e" % OF[-1])
        PETSc.Sys.Print("   Element count %d" % nEls[-1])

    PETSc.Sys.Print("Objective values %s" % OF)
    PETSc.Sys.Print("Element counts %s" % nEls)
