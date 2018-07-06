from thetis_adjoint import *
from thetis.callback import DiagnosticCallback, AccumulatorCallback

from .options import TohokuOptions, AdvectionOptions


__all__ = ["SWCallback", "ObjectiveSWCallback", "AdvectionCallback", "ObjectiveAdvectionCallback"]


class SWCallback(AccumulatorCallback):
    """Integrates objective functional for shallow water problem."""
    name = 'SW objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake import assemble

        self.op = TohokuOptions()   # TODO: Make more general
        dt = solver_obj.options.timestep

        def objectiveSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            mesh = solver_obj.fields.solution_2d.function_space().mesh()
            ks = Function(VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1))
            k0, k1 = ks.split()
            iA = self.op.indicator(mesh)
            # File("plots/" + self.op.mode + "/indicator.pvd").write(iA)
            k1.assign(iA)
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.start_time - 0.5 * dt:      # Slightly smooth transition
                kt.assign(1. if solver_obj.simulation_time > self.op.start_time + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(SWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class AdvectionCallback(AccumulatorCallback):
    """Integrates objective functional."""
    name = 'advection objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake import assemble

        self.op = AdvectionOptions()   # TODO: Make more general
        dt = solver_obj.options.timestep

        def objectiveAD():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            mesh = solver_obj.fields.tracer_2d.function_space().mesh()
            ks = Function(FunctionSpace(mesh, "DG", 1))
            iA = self.op.indicator(mesh)
            # File("plots/" + self.op.mode + "/indicator.pvd").write(iA)
            ks.assign(iA)
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.start_time - 0.5 * dt:      # Slightly smooth transition
                kt.assign(1. if solver_obj.simulation_time > self.op.start_time + 0.5 * dt else 0.5)

            return assemble(kt * ks * solver_obj.fields.tracer_2d * dx)

        super(AdvectionCallback, self).__init__(objectiveAD, solver_obj, **kwargs)


class ObjectiveSWCallback(AccumulatorCallback):
    """Writes objective functional values to tape."""
    name = 'SW objective tape'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake_adjoint import assemble

        self.op = TohokuOptions()   # TODO: Make more general
        self.mirror = False
        dt = solver_obj.options.timestep

        def objectiveSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            mesh = solver_obj.fields.solution_2d.function_space().mesh()
            ks = Function(VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1))
            k0, k1 = ks.split()
            iA = self.op.indicator(mesh)
            # File("plots/" + self.op.mode + "/indicator.pvd").write(iA)
            k1.assign(iA)
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.start_time - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > self.op.start_time + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(ObjectiveSWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class ObjectiveAdvectionCallback(AccumulatorCallback):
    """Writes objective functional values to tape."""
    name = 'advection objective tape'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake_adjoint import assemble

        self.op = AdvectionOptions()   # TODO: Make more general
        dt = solver_obj.options.timestep

        def objectiveAD():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            mesh = solver_obj.fields.tracer_2d.function_space().mesh()
            ks = Function(FunctionSpace(mesh, "DG", 1))
            iA = self.op.indicator(mesh)
            # File("plots/" + self.op.mode + "/indicator.pvd").write(iA)
            ks.assign(iA)
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.start_time - 0.5 * dt:      # Slightly smooth transition
                kt.assign(1. if solver_obj.simulation_time > self.op.start_time + 0.5 * dt else 0.5)

            return assemble(kt * ks * solver_obj.fields.tracer_2d * dx)

        super(ObjectiveAdvectionCallback, self).__init__(objectiveAD, solver_obj, **kwargs)
