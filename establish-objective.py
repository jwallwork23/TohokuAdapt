from thetis import *
from firedrake.petsc import PETSc

import math

from utils.callbacks import SWCallback
from utils.setup import problem_domain
from utils.options import TohokuOptions


def get_objective(level=0, mesh=None, b=None, op=TohokuOptions()):

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh, u0, eta0, b, BCs, f = problem_domain(level, mesh=mesh, b=b, op=op)
    for i in range(len(b.dat.data)):
        if math.isnan(b.dat.data[i]):
            b.dat.data[i] = 30.

    # Initialise solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.use_grad_div_viscosity_term = True      # Symmetric viscous stress
    options.use_lax_friedrichs_velocity = False     # TODO: This is a temporary fix
    options.coriolis_frequency = f
    options.simulation_export_time = op.timestep * op.timesteps_per_export
    options.simulation_end_time = op.end_time
    options.timestepper_type = op.timestepper
    options.timestep = op.timestep
    options.output_directory = op.di
    # options.no_exports = True
    options.no_exports = False
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = SWCallback(solver_obj)
    cb1.op = op
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs

    # Solve and extract quantities
    quantities = {}
    solver_obj.iterate()
    quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
    quantities['Element count'] = mesh.num_cells()

    return quantities, b


if __name__ == "__main__":

    op = TohokuOptions()
    mesh = Mesh("resources/meshes/wd_Tohoku0.msh")
    # coarse_bathy = problem_domain(mesh=mesh, op=op)[3]
    coarse_bathy = Function(FunctionSpace(mesh, "CG", 1)).assign(3000.)
    OF = []
    nEls = []

    for level in range(11):
        q = get_objective(level=level, b=coarse_bathy, op=op)[0]
        OF.append(q['J_h'])
        nEls.append(q['Element count'])
        PETSc.Sys.Print("Run %d" % level)
        PETSc.Sys.Print("   Objective value %.4e" % OF[-1])
        PETSc.Sys.Print("   Element count %d" % nEls[-1])

    PETSc.Sys.Print("Objective values %s" % OF)
    PETSc.Sys.Print("Element counts %s" % nEls)
