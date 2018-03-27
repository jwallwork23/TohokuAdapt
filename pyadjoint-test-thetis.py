from thetis import *
from firedrake_adjoint import *
from firedrake import Expression        # Annotation of Expressions not currently supported in firedrake_adjoint

class ObjectiveCallback(DiagnosticCallback):
    """Base class for callbacks that form objective functionals."""
    variable_names = ['current functional', 'objective functional']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates error comparison check callback object

        :arg scalar_callback: Python function that takes the solver object as an argument and
            returns a scalar quantity of interest
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        super(ObjectiveCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.objective_functional = [scalar_callback()]
        self.append_to_hdf5 = False
        self.append_to_log = False

    def __call__(self):
        value = self.scalar_callback()
        self.objective_functional.append(value)

        return value, self.objective_functional

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line


class ObjectiveSWCallback(ObjectiveCallback):
    """Integrates objective functional in shallow water case."""
    name = 'SW objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def objectiveSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = Function(elev_2d.function_space()) # Spatial integral is over [0, pi/2] x [pi/2, 3pi/2]
            ks.interpolate(Expression('(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' %
                                      (0., pi / 2, pi / 2, 3 * pi / 2)))
            kt = Constant(0.)
            if solver_obj.simulation_time > 0.5:    # Time integral is over [0.5, 2.5]
                kt.assign(1. if solver_obj.simulation_time > 0.5 + 0.5 * solver_obj.options.timestep else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(ObjectiveSWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


# Establish Mesh, initial condition and bathymetry
mesh = SquareMesh(16, 16, 2 * pi, 2 * pi)
x, y = SpatialCoordinate(mesh)
P1_2d = FunctionSpace(mesh, "CG", 1)
eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - pi, 2) + pow(y - pi, 2))))
b = Function(P1_2d).assign(0.1)

# Establish adjoint variables and indicator function
dual = Function(VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1))
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Get solver parameter values and construct solver
dt = 0.025
ndump = 10
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = 'dg-dg'
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.use_grad_div_viscosity_term = False
options.simulation_export_time = dt * ndump
options.simulation_end_time = 2.5
options.timestepper_type = 'CrankNicolson'
options.timestep = dt
options.output_directory = 'plots/pyadjointTest/'
options.export_diagnostics = False
solver_obj.create_equations()
solver_obj.assign_initial_conditions(elev=eta0)
# J = assemble(solver_obj.fields.solution_2d.split()[1] * dx)
cb = ObjectiveSWCallback(solver_obj)        # Extract objective functional at each timestep for use in pyadjoint
solver_obj.add_callback(cb, 'timestep')
solver_obj.iterate()

# Assemble objective functional and compute gradient
Jfuncs = cb.__call__()[1]
J = 0
for i in range(1, len(Jfuncs)):
    J += 0.5*(Jfuncs[i-1] + Jfuncs[i])*dt
dJdb = compute_gradient(J, Control(b))
File('plots/pyadjointTest/gradient.pvd').write(dJdb)
assert(dJdb.dat.norm > 1e-6)   # According to a standalone solver, this norm should be approximately 0.02


from fenics_adjoint.solving import SolveBlock   # Need use Sebastian's `linear-solver` branch of pyadjoint

tape = get_working_tape()
# tape.visualise()

solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
adjointFile = File('plots/pyadjointTest/adjoint.pvd')
for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)        # TODO: adj_sols currently have None type
    dual_u, dual_e = dual.split()
    if i % ndump == 0:
        adjointFile.write(dual_u, dual_e, time=dt*i)
        print('t = %.2fs' % (dt*i))
