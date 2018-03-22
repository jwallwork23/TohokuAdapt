from thetis import *
from thetis.callback import DiagnosticCallback
# from firedrake_adjoint import adj_start_timestep, adj_inc_timestep

import numpy as np

from . import forms
from . import interpolation
from . import options


__all__ = ["IntegralCallback", "TohokuCallback", "ShallowWaterCallback", "RossbyWaveCallback", "getOF",
           "explicitErrorEstimator", "DWR", "fluxJumpError", "basicErrorEstimator", "totalVariation"]


class IntegralCallback(DiagnosticCallback):
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
        super(IntegralCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.objective_value = 0.5 * scalar_callback() * solver_obj.options.timestep

    def __call__(self):
        # Output OF value
        t = self.solver_obj.simulation_time
        dt = self.solver_obj.options.timestep
        # T = self.solver_obj.options.simulation_end_time
        value = self.scalar_callback() * dt
        if t > self.solver_obj.options.simulation_end_time - 0.5 * dt:
            value *= 0.5
        self.objective_value += value

        # # Track adjoint data
        # if t < 0.5 * dt:
        #     adj_start_timestep()
        # else:
        #     adj_inc_timestep(time=t, finished=True if t > T - 0.5 * dt else False)

        return value, self.objective_value

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line


class TohokuCallback(IntegralCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def indicatorTohoku():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = forms.indicator(elev_2d.function_space(), mode='tohoku')
            kt = Constant(0.)
            if solver_obj.simulation_time > 300.:   # TODO: make this more general
                kt.assign(1. if solver_obj.simulation_time >
                                300. + 0.5 * solver_obj.options.timestep else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(TohokuCallback, self).__init__(indicatorTohoku, solver_obj, **kwargs)


class ShallowWaterCallback(IntegralCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def indicatorSW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = forms.indicator(elev_2d.function_space(), mode='shallow-water')
            kt = Constant(0.)
            if solver_obj.simulation_time > 0.5:    # TODO: make this more general
                kt.assign(
                    1. if solver_obj.simulation_time >
                          0.5 + 0.5 * solver_obj.options.timestep else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(ShallowWaterCallback, self).__init__(indicatorSW, solver_obj, **kwargs)


class RossbyWaveCallback(IntegralCallback):
    """Integrates objective functional."""
    name = 'objective functional'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def indicatorRW():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            elev_2d = solver_obj.fields.solution_2d.split()[1]
            ks = forms.indicator(elev_2d.function_space(), mode='rossby-wave')
            kt = Constant(0.)
            if solver_obj.simulation_time > 30.:    # TODO: make this more general
                kt.assign(
                    1. if solver_obj.simulation_time >
                          30. + 0.5 * solver_obj.options.timestep else 0.5)

            return assemble(elev_2d * ks * kt * dx)

        super(RossbyWaveCallback, self).__init__(indicatorRW, solver_obj, **kwargs)


class GaugeCallback(DiagnosticCallback):
    """Base class for callbacks that evaluate a scalar quantity at a gauge location."""
    variable_names = ['current value', 'gauge value']

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

    def __call__(self):
        self.gauge_value = self.scalar_callback() - self.init_value
        return self.gauge_value, float(self.gauge_value)

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
            loc = options.Options().gaugeCoord("P02")

            return elev_2d.at(loc)

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

            return elev_2d.at(options.Options().gaugeCoord("P06"))

        super(P06Callback, self).__init__(extractP06, solver_obj, **kwargs)


def getOF(dirName):
    """
    :arg dirName: directory in which log file is saved
    :return: final value of objective functional.
    """
    l = len([line for line in open(dirName + 'log', 'r')])
    logfile = open(dirName + 'log', 'r')
    i = 0
    for line in logfile:
        i += 1
        if i == l-1:
            text_ = line.split()
        elif i == l:
            text = line.split()
            J_h = text[-1] if text[0] == 'objective' else text_[-1]

    return float(J_h)


def explicitErrorEstimator(q, residual, b, v, maxBathy=False):
    """
    Estimate error locally using an a posteriori error indicator.
    
    :arg q: primal approximation at current timestep.
    :arg residual: approximation of residual for primal equations.
    :arg b: bathymetry profile.
    :arg v: P0 test function over the same function space.
    :param maxBathy: apply bound on bathymetry.
    :return: field of local error indicators.
    """
    V = residual.function_space()
    m = len(V.dof_count)
    mesh = V.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    b0 = Constant(max(b.dat.data)) if maxBathy else Function(V.sub(1)).interpolate(b)

    # Compute element residual term
    resTerm = assemble(v * h * h * inner(residual, residual) * dx) if m == 1 else \
        assemble(v * h * h * sum([inner(residual.split()[k], residual.split()[k]) for k in range(m)]) * dx)

    # Compute boundary residual term on fine mesh
    qh = interpolation.mixedPairInterp(mesh, V, q)[0]       # TODO: not needed if changing order
    uh, etah = qh.split()

    j0 = assemble(dot(v * grad(uh[0]), n) * ds)
    j1 = assemble(dot(v * grad(uh[1]), n) * ds)
    j2 = assemble(jump(v * b0 * uh, n=n) * dS)
    # jumpTerm = assemble(v * h * j2 * j2 * dx)

    # j0 = assemble(jump(v * grad(uh[0]), n=n) * dS)
    # j1 = assemble(jump(v * grad(uh[1]), n=n) * dS)
    # j2 = assemble(jump(v * grad(etah), n=n) * dS)
    jumpTerm = assemble(v * h * (j0 * j0 + j1 * j1 + j2 * j2) * dx)

    return assemble(sqrt(resTerm + jumpTerm))


def fluxJumpError(q, v):
    """
    Estimate error locally by flux jump.

    :arg q: primal approximation at current timestep.
    :arg v: P0 test function over the same function space.
    :return: field of local error indicators.
    """
    V = q.function_space()
    mesh = V.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    uh, etah = q.split()
    j0 = assemble(jump(v * grad(uh[0]), n=n) * dS)
    j1 = assemble(jump(v * grad(uh[1]), n=n) * dS)
    j2 = assemble(jump(v * grad(etah), n=n) * dS)

    return assemble(v * h * (j0 * j0 + j1 * j1 + j2 * j2) * dx)


def DWR(residual, adjoint, v):
    """
    :arg residual: approximation of residual for primal equations. 
    :arg adjoint: approximate solution of adjoint equations.
    :arg v: P0 test function over the same function space.
    :return: dual weighted residual.
    """
    m = len(residual.function_space().dof_count)
    n = len(adjoint.function_space().dof_count)
    assert(m == n)
    if n == 1:
        return assemble(v * inner(residual, adjoint) * dx)
    else:
        return assemble(v * sum([inner(residual.split()[k], adjoint.split()[k]) for k in range(n)]) * dx)


def basicErrorEstimator(primal, dual, v):
    """
    :arg primal: approximate solution of primal equations. 
    :arg dual: approximate solution of dual equations.
    :arg v: P0 test function over the same function space.
    :return: error estimate as in DL16.
    """
    m = len(primal.function_space().dof_count)
    n = len(dual.function_space().dof_count)
    assert(m == n)
    if n == 1:
        return assemble(v * inner(primal, dual) * dx)
    else:
        return assemble(v * sum([inner(primal.split()[k], dual.split()[k]) for k in range(n)]) * dx)


def totalVariation(data):
    """
    :arg data: (one-dimensional) timeseries record.
    :return: total variation thereof.
    """
    TV = 0
    iStart = 0
    for i in range(len(data)):
        if i == 1:
            sign = (data[i] - data[i-1]) / np.abs(data[i] - data[i-1])
        elif i > 1:
            sign_ = sign
            sign = (data[i] - data[i - 1]) / np.abs(data[i] - data[i - 1])
            if sign != sign_:
                TV += np.abs(data[i-1] - data[iStart])
                iStart = i-1
                if i == len(data)-1:
                    TV += np.abs(data[i] - data[i-1])
            elif i == len(data)-1:
                TV += np.abs(data[i] - data[iStart])
    return TV
