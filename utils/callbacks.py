from thetis_adjoint import *
from thetis.callback import DiagnosticCallback, FunctionalCallback, GaugeCallback

from .interpolation import interp
from .misc import indicator
from .options import Options
from .timeseries import gaugeTV


__all__ = ["SWCallback", "ObjectiveSWCallback", "P02Callback", "P06Callback", "EnrichedErrorCallback",
           "HigherOrderResidualCallback"]


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


class EnrichedErrorCallback(DiagnosticCallback):    # TODO: When fixed, move to Thetis branch
    """Base class for callbacks that evaluate an error quantity (such as the strong residual) related to the prognostic
    equation in an enriched finite element space of higher order."""
    variable_names = ['error', 'normed error']

    def __init__(self, tuple_callback, solver_obj, enriched_space, **kwargs):
        """
        Creates error callback object

        :arg scalar_or_vector_callback: Python function that takes the solver object as an argument and
            returns the equation residual at the current timestep.
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        kwargs.setdefault('export_to_hdf5', False)  # Needs a different functionality as output is a Function
        kwargs.setdefault('append_to_log', True)
        super(EnrichedErrorCallback, self).__init__(solver_obj, **kwargs)
        self.tuple_callback = tuple_callback        # Error quantifier with 2 components: momentum and continuity
        self.error = Function(enriched_space)
        self.normed_error = 0.
        self.index = 0
        self.di = solver_obj.options.output_directory

    def __call__(self):
        t0, t1 = self.tuple_callback()
        err_u, err_e = self.error.split()
        err_u.interpolate(t0)
        err_e.interpolate(t1)
        err_u.rename("Momentum error")
        err_e.rename("Continuity error")

        self.normed_error = self.error.dat.norm
        indexStr = (5 - len(str(self.index))) * '0' + str(self.index)
        with DumbCheckpoint(self.di + 'hdf5/Error2d_' + indexStr, mode=FILE_CREATE) as saveRes:
            saveRes.store(err_u)
            saveRes.store(err_e)
            saveRes.close()
        self.index += 1
        return self.error, self.normed_error

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line


class HigherOrderResidualCallback(EnrichedErrorCallback):   # TODO: When fixed, move to Thetis branch
    """Computes strong residual in an enriched finite element space of higher degree for the shallow water case."""
    name = 'strong residual'

    def __init__(self, solver_obj, enriched_space, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def residualSW():   # TODO: More terms to include
            """
            Construct the strong residual for the semi-discrete shallow water equations at the current timestep.

            :return: strong residual for shallow water equations at current timestep.
            """
            UV_old, ELEV_old = solver_obj.timestepper.solution_old.split()
            UV_2d, ELEV_2d = solver_obj.fields.solution_2d.split()

            # Enrich finite element space
            uv_old, elev_old = Function(enriched_space).split()
            uv_old.interpolate(UV_old)
            elev_old.interpolate(ELEV_old)
            uv_2d, elev_2d = Function(enriched_space).split()
            uv_2d.interpolate(UV_2d)
            elev_2d.interpolate(ELEV_2d)

            # Collect fields and parameters
            nu = solver_obj.fields.get('viscosity_h')
            Dt = Constant(solver_obj.options.timestep)
            H = solver_obj.fields.bathymetry_2d + elev_2d
            g = physical_constants['g_grav']

            # Construct residual
            res_u = (uv_2d - uv_old) / Dt + g * grad(elev_2d)
            if solver_obj.options.use_nonlinear_equations:
                res_u += dot(uv_2d, nabla_grad(uv_2d))
            if solver_obj.options.coriolis_frequency is not None:
                res_u += solver_obj.options.coriolis_frequency * as_vector((-uv_2d[1], uv_2d[0]))
            if nu is not None:
                if solver_obj.options.use_grad_depth_viscosity_term:
                    res_u -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
                if solver_obj.options.use_grad_div_viscosity_term:
                    res_u -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
                else:
                    res_u -= div(nu * grad(uv_2d))

            res_e = (elev_2d - elev_old) / Dt + div((solver_obj.fields.bathymetry_2d + elev_2d) * uv_2d)

            return res_u, res_e

        super(HigherOrderResidualCallback, self).__init__(residualSW, solver_obj, enriched_space, **kwargs)


class RefinedResidualCallback(EnrichedErrorCallback):
    """Computes strong residual in an enriched finite element space (with refined mesh) for the shallow water case."""
    name = 'strong residual'

    def __init__(self, solver_obj, enriched_space, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def residualSW():   # TODO: More terms to include
            """
            Construct the strong residual for the semi-discrete shallow water equations at the current timestep.

            :return: strong residual for shallow water equations at current timestep.
            """
            UV_old, ELEV_old = solver_obj.timestepper.solution_old.split()
            UV_2d, ELEV_2d = solver_obj.fields.solution_2d.split()

            # Enrich finite element space
            uv_old, elev_old = interp(enriched_space.mesh(), UV_old, ELEV_old)
            uv_2d, elev_2d = interp(enriched_space.mesh(), UV_2d, ELEV_2d)

            # Collect fields and parameters
            nu = solver_obj.fields.get('viscosity_h')
            Dt = Constant(solver_obj.options.timestep)
            H = solver_obj.fields.bathymetry_2d + elev_2d
            g = physical_constants['g_grav']

            # Construct residual
            res_u = (uv_2d - uv_old) / Dt + g * grad(elev_2d)
            if solver_obj.options.use_nonlinear_equations:
                res_u += dot(uv_2d, nabla_grad(uv_2d))
            if solver_obj.options.coriolis_frequency is not None:
                res_u += solver_obj.options.coriolis_frequency * as_vector((-uv_2d[1], uv_2d[0]))
            if nu is not None:
                if solver_obj.options.use_grad_depth_viscosity_term:
                    res_u -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
                if solver_obj.options.use_grad_div_viscosity_term:
                    res_u -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
                else:
                    res_u -= div(nu * grad(uv_2d))

            res_e = (elev_2d - elev_old) / Dt + div((solver_obj.fields.bathymetry_2d + elev_2d) * uv_2d)

            return res_u, res_e

        super(RefinedResidualCallback, self).__init__(residualSW, solver_obj, enriched_space, **kwargs)
