from thetis import *
from firedrake.petsc import PETSc

import numpy as np

from utils.callbacks import SWCallback
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
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs

    # Solve and extract quantities
    quantities = {}
    solver_obj.iterate()
    quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
    quantities['Integrand'] = cb1.getVals()
    quantities['Element count'] = mesh.num_cells()

    return quantities


if __name__ == "__main__":

    op = Options(mode='tohoku')
    mesh, b, hierarchy = problemDomain(0, hierarchy=True, op=op)[0::3]

    q = getObjective(mesh, b, op)
    OF = [q['J_h']]
    nEls = [q['Element count']]
    PETSc.Sys.Print("   Objective value %.4e" % OF[0])
    PETSc.Sys.Print("   Element count %d" % nEls[0])
    bathyfile = open("resources/bathymetry/array.txt", "r")
    bathyfile.readline()

    for i in range(1, 5):
        mesh = hierarchy.__getitem__(i)      # Hierarchical refinement
        b = Function(FunctionSpace(mesh, "CG", 1))
        b.dat.data[:] = np.array(bathyfile.readline().split())
        q = getObjective(mesh, b, op)
        OF.append(q['J_h'])
        nEls.append(q['Element count'])
        PETSc.Sys.Print("   Objective value %.4e" % OF[-1])
        PETSc.Sys.Print("   Element count %d" % nEls[-1])

    PETSc.Sys.Print("Objective values %s" % OF)
    PETSc.Sys.Print("Element counts %s" % nEls)
