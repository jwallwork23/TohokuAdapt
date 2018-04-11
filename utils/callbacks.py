from thetis_adjoint import *
from thetis.callback import DiagnosticCallback

from .forms import indicator
from .options import Options


__all__ = ["FunctionalCallback", "TohokuCallback", "ShallowWaterCallback", "RossbyWaveCallback",
           "GaugeCallback", "P02Callback", "P06Callback", "ObjectiveCallback", "ObjectiveTohokuCallback",
           "ObjectiveSWCallback", "ObjectiveRWCallback"]


class FunctionalCallback(DiagnosticCallback):
    """Base class for callbacks that integrate a scalar quantity in time and space. Time integration is achieved
    using the trapezium rule."""
    variable_names = ['current integral', 'objective value']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates error comparison check callback object

        :arg scalar_callback: Python function that takes the solver object as an argument and
            returns a scalar quantity of interest
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        super(FunctionalCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.objective_value = [scalar_callback()]
        self.append_to_hdf5 = False
        self.append_to_log = False

    def __call__(self):
        value = self.scalar_callback()
        self.objective_value.append(value)

        return value, self.objective_value

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line

    def quadrature(self):
        dt = self.options.timestep
        func = self.objective_value
        J = 0
        for i in range(1, len(func)):
            J += 0.5 * (func[i] + func[i-1]) * dt
        return J


class TohokuCallback(FunctionalCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake import assemble

        def indicatorTohoku():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = indicator(elev_2d.function_space(), mode='tohoku')
            kt = Constant(0.)
            dt = solver_obj.options.timestep
            Tstart = solver_obj.options.period_of_interest_start
            if solver_obj.simulation_time > Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > Tstart + 0.5 * dt else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(TohokuCallback, self).__init__(indicatorTohoku, solver_obj, **kwargs)


class ShallowWaterCallback(FunctionalCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake import assemble

        def indicatorSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = indicator(elev_2d.function_space(), mode='shallow-water')
            kt = Constant(0.)
            dt = solver_obj.options.timestep
            Tstart = solver_obj.options.period_of_interest_start
            if solver_obj.simulation_time > Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > Tstart + 0.5 * dt else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(ShallowWaterCallback, self).__init__(indicatorSW, solver_obj, **kwargs)


class RossbyWaveCallback(FunctionalCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake import assemble

        def indicatorRW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = indicator(elev_2d.function_space(), mode='rossby-wave')
            kt = Constant(0.)
            dt = solver_obj.options.timestep
            Tstart = solver_obj.options.period_of_interest_start
            if solver_obj.simulation_time > Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > Tstart + 0.5 * dt else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(RossbyWaveCallback, self).__init__(indicatorRW, solver_obj, **kwargs)


class GaugeCallback(DiagnosticCallback):
    """Base class for callbacks that evaluate a scalar quantity at a gauge location."""
    variable_names = ['current value', 'gauge values']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates gauge callback object

        :arg scalar_callback: Python function that takes the solver object as an argument and
            returns a scalar quantity of interest
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        super(GaugeCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.init_value = self.scalar_callback()
        self.gauge_values = [0.]
        self.append_to_hdf5 = False
        self.append_to_log = False

    def __call__(self):
        value = self.scalar_callback()
        self.gauge_values.append(value - self.init_value)
        return value, self.gauge_values

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line


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


class ObjectiveTohokuCallback(ObjectiveCallback):
    """Integrates objective functional in Tohoku case."""
    name = 'Tohoku objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake_adjoint import assemble


        def objectiveTohoku():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            V = solver_obj.fields.solution_2d.function_space()
            ks = Function(V)
            k0, k1 = ks.split()
            k1.assign(indicator(V.sub(1), mode='tohoku'))
            kt = Constant(0.)
            dt = solver_obj.options.timestep
            Tstart = solver_obj.options.period_of_interest_start
            if solver_obj.simulation_time > Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(ObjectiveTohokuCallback, self).__init__(objectiveTohoku, solver_obj, **kwargs)


class ObjectiveSWCallback(ObjectiveCallback):
    """Integrates objective functional in shallow water case."""
    name = 'SW objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake_adjoint import assemble

        def objectiveSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            V = solver_obj.fields.solution_2d.function_space()
            ks = Function(V)
            k0, k1 = ks.split()
            k1.assign(indicator(V.sub(1), mode='shallow-water'))
            kt = Constant(0.)
            dt = solver_obj.options.timestep
            Tstart = solver_obj.options.period_of_interest_start
            if solver_obj.simulation_time > Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(ObjectiveSWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class ObjectiveRWCallback(ObjectiveCallback):
    """Integrates objective functional in equatorial Rossby wave case."""
    name = 'Rossby wave objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        from firedrake_adjoint import assemble

        def objectiveRW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            V = solver_obj.fields.solution_2d.function_space()
            ks = Function(V)
            k0, k1 = ks.split()
            k1.assign(indicator(V.sub(1), mode='rossby-wave'))
            kt = Constant(0.)
            dt = solver_obj.options.timestep
            Tstart = solver_obj.options.period_of_interest_start
            if solver_obj.simulation_time > Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(ObjectiveRWCallback, self).__init__(objectiveRW, solver_obj, **kwargs)
