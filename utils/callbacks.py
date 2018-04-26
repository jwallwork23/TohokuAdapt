from thetis_adjoint import *
from thetis.callback import FunctionalCallback, GaugeCallback

from .timeseries import gaugeTV
from .forms import indicator
from .options import Options


__all__ = ["SWCallback", "ObjectiveSWCallback", "P02Callback", "P06Callback"]


class SWCallback(FunctionalCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake import assemble

        self.op = Options()
        dt = solver_obj.options.timestep

        def objectiveSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            V = solver_obj.fields.solution_2d.function_space()
            ks = Function(V)
            k0, k1 = ks.split()
            k1.assign(indicator(V.sub(1), op=self.op))
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.Tstart - 0.5 * dt:      # Slightly smooth transition
                kt.assign(1. if solver_obj.simulation_time > self.op.Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(SWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class ObjectiveSWCallback(FunctionalCallback):
    """Integrates objective functional in shallow water case."""
    name = 'SW objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake_adjoint import assemble

        self.op = Options()
        dt = solver_obj.options.timestep

        def objectiveSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            V = solver_obj.fields.solution_2d.function_space()
            ks = Function(V)
            k0, k1 = ks.split()
            k1.assign(indicator(V.sub(1), op=self.op))
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > self.op.Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(ObjectiveSWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class P02Callback(GaugeCallback):
    """Evaluates at gauge P02."""
    name = 'gauge P02'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def extractP02():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]

            return elev_2d.at(Options().gaugeCoord("P02"))

        super(P02Callback, self).__init__(extractP02, solver_obj, **kwargs)

    def totalVariation(self):
        return gaugeTV(self.gauge_values, gauge="P02")



class P06Callback(GaugeCallback):
    """Evaluates at gauge P06."""
    name = 'gauge P06'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def extractP06():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]

            return elev_2d.at(Options().gaugeCoord("P06"))

        super(P06Callback, self).__init__(extractP06, solver_obj, **kwargs)

    def totalVariation(self):
        return gaugeTV(self.gauge_values, gauge="P06")
