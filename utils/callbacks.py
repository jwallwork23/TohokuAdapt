from thetis_adjoint import *
from thetis.callback import DiagnosticCallback

from .timeseries import gaugeTV
from .forms import indicator
from .options import Options


__all__ = ["FunctionalCallback", "SWCallback", "ObjectiveCallback", "ObjectiveSWCallback",
           "GaugeCallback", "P02Callback", "P06Callback"]


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
        self.dt = solver_obj.options.timestep

    def __call__(self):
        value = self.scalar_callback()
        self.objective_value.append(value)

        return value, self.objective_value

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line

    def quadrature(self):
        func = self.objective_value
        J = 0
        for i in range(1, len(func)):
            J += 0.5 * (func[i] + func[i-1]) * self.dt
        return J


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
        self.dt = solver_obj.options.timestep

    def __call__(self):
        value = self.scalar_callback()
        self.objective_functional.append(value)

        return value, self.objective_functional

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line

    def assembleOF(self):
        func = self.objective_functional
        J = 0
        for i in range(1, len(func)):
            J += 0.5 * (func[i - 1] + func[i]) * self.dt
        return J


class ObjectiveSWCallback(ObjectiveCallback):
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
        self.init_value = scalar_callback()
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
