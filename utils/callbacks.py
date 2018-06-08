from thetis_adjoint import *
from thetis.callback import DiagnosticCallback, FunctionalCallback

from .interpolation import *
from .options import TohokuOptions, Options
from .timeseries import gaugeTV


__all__ = ["SWCallback", "MirroredSWCallback", "ObjectiveSWCallback", "P02Callback", "P06Callback", "strongResidualSW",
           "ResidualCallback", "EnrichedErrorCallback", "HigherOrderResidualCallback", "RefinedResidualCallback"]


class SWCallback(FunctionalCallback):
    """Integrates objective functional."""
    name = 'objective functional'

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
            File("plots/" + self.op.mode + "/indicator.pvd").write(iA)
            k1.assign(iA)
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.Tstart - 0.5 * dt:      # Slightly smooth transition
                kt.assign(1. if solver_obj.simulation_time > self.op.Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(SWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class MirroredSWCallback(FunctionalCallback):
    """Integrates objective functional."""
    name = 'mirrored objective functional'

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
            k1.assign(iA)   # TODO: need select mirrored version
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.Tstart - 0.5 * dt:      # Slightly smooth transition
                kt.assign(1. if solver_obj.simulation_time > self.op.Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(MirroredSWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class ObjectiveSWCallback(FunctionalCallback):
    """Integrates objective functional in shallow water case."""
    name = 'SW objective functional'

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
            File("plots/" + self.op.mode + "/indicator.pvd").write(iA)
            k1.assign(iA)
            kt = Constant(0.)
            if solver_obj.simulation_time > self.op.Tstart - 0.5 * dt:
                kt.assign(1. if solver_obj.simulation_time > self.op.Tstart + 0.5 * dt else 0.5)

            return assemble(kt * inner(ks, solver_obj.fields.solution_2d) * dx)

        super(ObjectiveSWCallback, self).__init__(objectiveSW, solver_obj, **kwargs)


class GaugeCallback(DiagnosticCallback):    # TODO: This is probably superfluous. Could just use DetectorsCallback?
    """Base class for callbacks that evaluate a scalar quantity at a particular gauge location over all time.
    Evaluations are based around the initial value being zero."""
    variable_names = ['current value', 'gauge values']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates gauge callback object

        :arg scalar_callback: Python function that takes the solver object as an argument and
            returns a single point value of a field related to the fluid state.
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        kwargs.setdefault('export_to_hdf5', False)
        kwargs.setdefault('append_to_log', False)
        super(GaugeCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.init_value = scalar_callback()
        self.gauge_values = [0.]
        self.ix = 0

    def __call__(self):
        value = self.scalar_callback()
        if self.ix != 0:
            self.gauge_values.append(value - self.init_value)
        self.ix += 1
        return value, self.gauge_values

    def getVals(self):
        return self.gauge_values

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


# def strongResidualSW(solver_obj, Ve=None, op=Options()):  # TODO: More terms to include
#     """
#     Construct the strong residual for the semi-discrete shallow water equations at the current timestep,
#     using Crank-Nicolson timestepping.
#
#     :param op: option parameters object.
#     :return: strong residual for shallow water equations at current timestep.
#     """
#     if solver_obj.timestepper.solution_old is None:     # i.e. at first timestep
#         Q_old = solver_obj.fields.solution_2d
#     else:
#         Q_old = solver_obj.timestepper.solution_old
#     UV_old, ELEV_old = Q_old.split()
#     Q_new = solver_obj.fields.solution_2d
#     UV_new, ELEV_new = Q_new.split()
#
#     if op.orderChange:
#         uv_old, elev_old = Function(Ve).split()
#         uv_new, elev_new = Function(Ve).split()
#         uv_old.interpolate(UV_old)
#         uv_new.interpolate(UV_new)
#         elev_old.interpolate(ELEV_old)
#         elev_new.interpolate(ELEV_new)
#     elif op.refinedSpace:
#         q_old, q_new = mixedPairInterp(Ve, Q_old, Q_new)
#         uv_old, elev_old = q_old.split()
#         uv_new, elev_new = q_new.split()
#     else:
#         uv_old = UV_old
#         uv_new = UV_new
#         elev_old = ELEV_old
#         elev_new = ELEV_new
#
#     uv_2d = 0.5 * (uv_old + uv_new)         # Use Crank-Nicolson timestepping so that we isolate errors as being
#     elev_2d = 0.5 * (elev_old + elev_new)   # related only to the spatial discretisation
#
#     # Collect fields and parameters
#     nu = solver_obj.fields.get('viscosity_h')
#     Dt = Constant(solver_obj.options.timestep)
#     H = solver_obj.fields.bathymetry_2d + elev_2d
#     g = physical_constants['g_grav']
#
#     # Construct residual        # TODO: How to consider boundary integrals resulting from IBP?
#     res_u = (uv_new - uv_old) / Dt + g * grad(elev_2d)
#     if solver_obj.options.use_nonlinear_equations:
#         res_u += dot(uv_2d, nabla_grad(uv_2d))
#     if solver_obj.options.coriolis_frequency is not None:
#         res_u += solver_obj.options.coriolis_frequency * as_vector((-uv_2d[1], uv_2d[0]))
#     if nu is not None:
#         if solver_obj.options.use_grad_depth_viscosity_term:
#             res_u -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
#         if solver_obj.options.use_grad_div_viscosity_term:
#             res_u -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
#         else:
#             res_u -= div(nu * grad(uv_2d))
#
#     res_e = (elev_new - elev_old) / Dt + div((solver_obj.fields.bathymetry_2d + elev_2d) * uv_2d)
#
#     return res_u, res_e

def strongResidualSW(solver_obj, UV_new, ELEV_new, UV_old, ELEV_old, Ve=None, op=Options()):
    """
    Construct the strong residual for the semi-discrete shallow water equations at the current timestep,
    using Crank-Nicolson timestepping.

    :param op: option parameters object.
    :return: strong residual for shallow water equations at current timestep.
    """

    # Collect fields and parameters
    nu = solver_obj.fields.get('viscosity_h')
    Dt = Constant(solver_obj.options.timestep)
    g = physical_constants['g_grav']

    # Enrich FE space (if appropriate)
    if op.orderChange:
        uv_old, elev_old = Function(Ve).split()
        uv_new, elev_new = Function(Ve).split()
        uv_old.interpolate(UV_old)
        uv_new.interpolate(UV_new)
        elev_old.interpolate(ELEV_old)
        elev_new.interpolate(ELEV_new)
        b = solver_obj.fields.bathymetry_2d
        f = solver_obj.options.coriolis_frequency
    elif op.refinedSpace:
        uv_old, elev_old, uv_new, elev_new = interp(Ve.mesh(), UV_old, ELEV_old, UV_new, ELEV_new)
        b = interp(Ve.mesh(), solver_obj.fields.bathymetry_2d)
        if nu is not None:
            nu = interp(Ve.mesh(), nu)
        if solver_obj.options.coriolis_frequency is not None:
            f = interp(Ve.mesh(), solver_obj.options.coriolis_frequency)
    else:
        uv_old = UV_old
        uv_new = UV_new
        elev_old = ELEV_old
        elev_new = ELEV_new
        b = solver_obj.fields.bathymetry_2d
        f = solver_obj.options.coriolis_frequency
    uv_2d = 0.5 * (uv_old + uv_new)         # Use Crank-Nicolson timestepping so that we isolate errors as being
    elev_2d = 0.5 * (elev_old + elev_new)   # related only to the spatial discretisation
    H = b + elev_2d

    # Construct residual        # TODO: How to consider boundary integrals resulting from IBP?
    res_u = (uv_new - uv_old) / Dt + g * grad(elev_2d)
    if solver_obj.options.use_nonlinear_equations:
        res_u += dot(uv_2d, nabla_grad(uv_2d))
    if solver_obj.options.coriolis_frequency is not None:
        res_u += f * as_vector((-uv_2d[1], uv_2d[0]))
    if nu is not None:
        if solver_obj.options.use_grad_depth_viscosity_term:
            res_u -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
        if solver_obj.options.use_grad_div_viscosity_term:
            res_u -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
        else:
            res_u -= div(nu * grad(uv_2d))

    res_e = (elev_new - elev_old) / Dt + div(H * uv_2d)

    return res_u, res_e


class ErrorCallback(DiagnosticCallback):
    """Base class for callbacks that evaluate an error quantity (such as the strong residual) related to the prognostic
    equation."""
    variable_names = ['error', 'normed error']

    def __init__(self, tuple_callback, solver_obj, **kwargs):
        """
        Creates error callback object

        :arg scalar_or_vector_callback: Python function that takes the solver object as an argument and
            returns the equation residual at the current timestep.
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        kwargs.setdefault('export_to_hdf5', False)  # Needs a different functionality as output is a Function
        kwargs.setdefault('append_to_log', True)
        super(ErrorCallback, self).__init__(solver_obj, **kwargs)
        self.tuple_callback = tuple_callback        # Error quantifier with 2 components: momentum and continuity
        self.error = Function(solver_obj.fields.solution_2d.function_space())
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


class ResidualCallback(ErrorCallback):
    """Computes strong residual for the shallow water case."""
    name = 'strong residual'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """

        def residualSW():   # TODO: More terms to include
            """
            Construct the strong residual for the semi-discrete shallow water equations at the current timestep,
            using Crank-Nicolson timestepping.

            :return: strong residual for shallow water equations at current timestep.
            """
            uv_old, elev_old = solver_obj.timestepper.solution_old.split()
            uv_new, elev_new = solver_obj.fields.solution_2d.split()
            uv_2d = 0.5 * (uv_old + uv_new)         # Use Crank-Nicolson timestepping so that we isolate errors as
            elev_2d = 0.5 * (elev_old + elev_new)   # being related only to the spatial discretisation

            # Collect fields and parameters
            nu = solver_obj.fields.get('viscosity_h')
            Dt = Constant(solver_obj.options.timestep)
            H = solver_obj.fields.bathymetry_2d + elev_2d
            g = physical_constants['g_grav']

            # Construct residual        TODO: How to consider boundary integrals resulting from IBP?
            res_u = (uv_new - uv_old) / Dt + g * grad(elev_2d)
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

            res_e = (elev_new - elev_old) / Dt + div((solver_obj.fields.bathymetry_2d + elev_2d) * uv_2d)

            return res_u, res_e

        super(ResidualCallback, self).__init__(residualSW, solver_obj, **kwargs)


class EnrichedErrorCallback(DiagnosticCallback):
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

        # TODO: Could we build in the following functionality?
                # Load residuals from this remesh period (if existent)
                # Time integrate in the sense of adding current residual and multiplying by timestep
                # Store partially time integrated residual for next callback
        # Not only would it have the effect of considering the 'future' residuals, but it would also weight the residual
        #   more heavily than the dual. If necessary an averaging procedure could be applied

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


class HigherOrderResidualCallback(EnrichedErrorCallback):
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
            UV_new, ELEV_new = solver_obj.fields.solution_2d.split()

            # Enrich finite element space
            uv_old, elev_old = Function(enriched_space).split()
            uv_old.interpolate(UV_old)
            elev_old.interpolate(ELEV_old)
            uv_new, elev_new = Function(enriched_space).split()
            uv_new.interpolate(UV_new)
            elev_new.interpolate(ELEV_new)
            uv_2d = 0.5 * (uv_old + uv_new)  # Use Crank-Nicolson timestepping so that we isolate errors as
            elev_2d = 0.5 * (elev_old + elev_new)  # being related only to the spatial discretisation

            # Collect fields and parameters
            nu = solver_obj.fields.get('viscosity_h')
            Dt = Constant(solver_obj.options.timestep)
            H = solver_obj.fields.bathymetry_2d + elev_2d
            g = physical_constants['g_grav']

            # Construct residual        TODO: How to consider boundary integrals resulting from IBP?
            res_u = (uv_new - uv_old) / Dt + g * grad(elev_2d)
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

            res_e = (elev_new - elev_old) / Dt + div((solver_obj.fields.bathymetry_2d + elev_2d) * uv_2d)

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
            UV_new, ELEV_new = solver_obj.fields.solution_2d.split()
            nu = solver_obj.fields.get('viscosity_h')

            # Enrich finite element space
            uv_old, elev_old, uv_new, elev_new = interp(enriched_space.mesh(), UV_old, ELEV_old, UV_new, ELEV_new)
            uv_2d = 0.5 * (uv_old + uv_new)  # Use Crank-Nicolson timestepping so that we isolate errors as
            elev_2d = 0.5 * (elev_old + elev_new)  # being related only to the spatial discretisation
            b = interp(enriched_space.mesh(), solver_obj.fields.bathymetry_2d)
            if nu is not None:
                nu = interp(enriched_space.mesh(), nu)

            # Collect fields and parameters
            Dt = Constant(solver_obj.options.timestep)
            H = b + elev_2d
            g = physical_constants['g_grav']

            # Construct residual        TODO: How to consider boundary integrals resulting from IBP?
            res_u = (uv_new - uv_old) / Dt + g * grad(elev_2d)
            if solver_obj.options.use_nonlinear_equations:
                res_u += dot(uv_2d, nabla_grad(uv_2d))
            if solver_obj.options.coriolis_frequency is not None:
                f = interp(enriched_space.mesh(), solver_obj.options.coriolis_frequency)
                res_u += f * as_vector((-uv_2d[1], uv_2d[0]))
            if nu is not None:
                if solver_obj.options.use_grad_depth_viscosity_term:
                    res_u -= dot(nu * grad(H), (grad(uv_2d) + sym(grad(uv_2d))))
                if solver_obj.options.use_grad_div_viscosity_term:
                    res_u -= div(nu * (grad(uv_2d) + sym(grad(uv_2d))))
                else:
                    res_u -= div(nu * grad(uv_2d))

            res_e = (elev_new - elev_old) / Dt + div((b + elev_2d) * uv_2d)

            return res_u, res_e

        super(RefinedResidualCallback, self).__init__(residualSW, solver_obj, enriched_space, **kwargs)
