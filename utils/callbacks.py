from thetis_adjoint import *

from .options import TohokuOptions, AdvectionOptions


__all__ = ["SWCallback", "AdvectionCallback"]


class SWCallback(callback.AccumulatorCallback):
    """Integrates objective functional for shallow water problem."""
    name = 'SW objective functional'

    def __init__(self, solver_obj, parameters=None, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg parameters: class containing parameters, including time period of interest.
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        if parameters is None:
            self.parameters = TohokuOptions()
        else:
            self.parameters = parameters
        if self.parameters.approach in ("DWP", "DWR"):
            from firedrake_adjoint import assemble
        else:
            from firedrake import assemble
        self.outfile = File(self.parameters.directory() + "indicator.pvd")
        
        def objectiveSW():	# TODO: Why does self.parameters give None?
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            mesh = solver_obj.fields.solution_2d.function_space().mesh()
            ks = Function(VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1))
            k0, k1 = ks.split()
            iA = parameters.indicator(mesh)
            t = solver_obj.simulation_time
            dt = solver_obj.options.timestep
            if parameters.plot_pvd and solver_obj.iteration % parameters.timesteps_per_export == 0:
                self.outfile.write(iA, time=t)
            k1.assign(iA)
            kt = Constant(0.)

            # Slightly smooth transition
            if parameters.start_time - 0.5 * dt < t < parameters.start_time + 0.5 * dt:
                kt.assign(0.5)
            elif parameters.start_time + 0.5 * dt < t < parameters.end_time - 0.5 * dt:
                kt.assign(1.)
            elif parameters.end_time - 0.5 * dt < t < parameters.end_time + 0.5 * dt:
                kt.assign(0.5)
            else:
                kt.assign(0.)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(SWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class AdvectionCallback(callback.AccumulatorCallback):
    """Integrates objective functional for advection diffusion problem."""
    name = 'advection objective functional'

    def __init__(self, solver_obj, parameters=None, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg parameters: class containing parameters, including time period of interest.
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        if parameters is None:
            self.parameters = AdvectionOptions()
        else:
            self.parameters = parameters
        if self.parameters.approach in ("DWP", "DWR"):
            from firedrake_adjoint import assemble
        else:
            from firedrake import assemble
        self.outfile = File(self.parameters.directory() + "indicator.pvd")

        def objectiveAD():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            mesh = solver_obj.fields.tracer_2d.function_space().mesh()
            ks = Function(FunctionSpace(mesh, "DG", 1))
            iA = self.parameters.indicator(mesh)
            t = solver_obj.simulation_time
            dt = solver_obj.options.timestep
            if self.parameters.plot_pvd and solver_obj.iteration % self.parameters.timesteps_per_export == 0:
                self.outfile.write(iA, time=t) 
            ks.assign(iA)
            kt = Constant(0.)

            # Slightly smooth transition
            if self.parameters.start_time - 0.5 * dt < t < self.parameters.start_time + 0.5 * dt:
                kt.assign(0.5)
            elif self.parameters.start_time + 0.5 * dt < t < self.parameters.end_time - 0.5 * dt:
                kt.assign(1.)
            elif self.parameters.end_time - 0.5 * dt < t < self.parameters.end_time + 0.5 * dt:
                kt.assign(0.5)
            else:
                kt.assign(0.)

            return assemble(kt * ks * solver_obj.fields.tracer_2d * dx)

        super(AdvectionCallback, self).__init__(objectiveAD, solver_obj, parameters, **kwargs)
