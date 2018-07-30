from thetis import *
from thetis import FiredrakeConstant as Constant
from thetis.configuration import *
from firedrake import Expression

import numpy as np

from .conversion import from_latlon

__all__ = ["TohokuOptions", "RossbyWaveOptions", "KelvinWaveOptions", "GaussianOptions", "AdvectionOptions"]


class AdaptOptions(FrozenConfigurable):
    name = 'Common parameters for TohokuAdapt project'

    # Mesh adaptivity parameters
    approach = Unicode('FixedMesh',
                       help="Mesh adaptive approach considered, from {'FixedMesh', 'HessianBased', 'DWP', 'DWR'}"
                       ).tag(config=True)
    gradate = Bool(False, help='Toggle metric gradation.').tag(config=True)
    adapt_on_bathymetry = Bool(False, help='Toggle adaptation based on bathymetry field.').tag(config=True)
    plot_pvd = Bool(False, help='Toggle plotting of fields.').tag(config=True)
    plot_metric = Bool(False, help='Toggle plotting of metric field.').tag(config=True)
    max_element_growth = PositiveFloat(1.4, help="Metric gradation scaling parameter.").tag(config=True)
    max_anisotropy = PositiveFloat(100., help="Maximum tolerated anisotropy.").tag(config=True)
    num_adapt = NonNegativeInteger(1, help="Number of mesh adaptations per remeshing.").tag(config=True)
    order_increase = Bool(False, help="Interpolate adjoint solution into higher order space.").tag(config=True)
    normalisation = Unicode('lp', help="Normalisation approach, from {'lp', 'manual'}.").tag(config=True)
    hessian_recovery = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    norm_order = NonNegativeInteger(2, help="Degree p of Lp norm used.")
    family = Unicode('dg-dg', help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)
    min_norm = PositiveFloat(1e-6).tag(config=True)
    max_norm = PositiveFloat(1e9).tag(config=True)

    def final_index(self):
        """Final timestep index"""
        return int(np.ceil(self.end_time / self.timestep))

    def first_export(self):
        """First exported timestep of period of interest"""
        return int(self.start_time / (self.timesteps_per_export * self.timestep))

    def final_export(self):
        """Final exported timestep of period of interest"""
        return int(self.final_index() / self.timesteps_per_export)

    def final_mesh_index(self):
        """Final mesh index"""
        return int(self.final_index() / self.timesteps_per_remesh)

    def exports_per_remesh(self):
        """Number of exports per mesh adaptation"""
        assert self.timesteps_per_remesh % self.timesteps_per_export == 0
        return int(self.timesteps_per_remesh / self.timesteps_per_export)

    def indicator(self, mesh):
        """Indicator function associated with region(s) of interest"""
        P1 = FunctionSpace(mesh, "DG", 1)
        iA = Function(P1, name="Region of interest")

        if np.shape(self.radii)[0] == 1:
            expr = Expression("pow(x[0] - x0, 2) + pow(x[1] - y0, 2) < r + eps ? 1 : 0",
                              x0=self.loc[0], y0=self.loc[1], r=pow(self.radii[0], 2), eps=1e-10)
        elif np.shape(self.radii)[0] > 1:
            assert len(self.loc)/2 == len(self.radii)
            e = "(pow(x[0] - {x0:f}, 2) + pow(x[1] - {y0:f}, 2) < {r:f} + {eps:f})".format(x0=self.loc[0],
                                                                                           y0=self.loc[1],
                                                                                           r=pow(self.radii[0], 2),
                                                                                           eps=1e-10)
            for i in range(1, len(self.radii)):
                e += "|| (pow(x[0] - {x0:f}, 2) + pow(x[1] - {y0:f}, 2) < {r:f} + {eps:f})".format(x0=self.loc[2*i],
                                                                                                   y0=self.loc[2*i+1],
                                                                                                   r=pow(self.radii[i], 2),
                                                                                                   eps=1e-10)
            expr = Expression(e)
        else:
            raise ValueError("Indicator function radii input not recognised.")

        iA.interpolate(expr)

        return iA


    def mixed_space(self, mesh, enrich=False):
        """
        :param mesh: mesh upon which to build mixed space.
        :return: mixed VectorFunctionSpace x FunctionSpace as specified by ``self.family``.
        """
        d1 = 1
        d2 = 2 if self.family == 'dg-cg' else 1
        if enrich:
            d1 += self.order_increase
            d2 += self.order_increase
        return VectorFunctionSpace(mesh, "DG", d1) * FunctionSpace(mesh, "DG" if self.family == 'dg-dg' else "CG", d2)

    def adaptation_stats(self, mn, adaptTimer, solverTime, nEle, Sn, mM, t):
        """
        :arg mn: mesh number.
        :arg adaptTimer: time taken for mesh adaption.
        :arg solverTime: time taken for solver.
        :arg nEle: current number of elements.
        :arg Sn: sum over #Elements.
        :arg mM: tuple of min and max #Elements.
        :arg t: current simuation time.
        :return: mean element count.
        """
        av = Sn / mn
        print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Adapt time : %4.2fs Solver time : %4.2fs     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s\n""" %
              (mn, 100 * t / self.end_time, adaptTimer, solverTime, nEle, av, mM[0], mM[1]))
        return av

    def directory(self):
        return 'plots/' + self.mode + '/' + self.approach + '/'


class TohokuOptions(AdaptOptions):
    name = 'Parameters for the Tohoku problem'
    mode = 'Tohoku'

    # Solver parameters
    timesteps_per_export = NonNegativeInteger(10, help="Timesteps per data dump").tag(config=True)
    timesteps_per_remesh = NonNegativeInteger(30, help="Timesteps per mesh adaptation").tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('s', help="Adaptation field of interest, from {'s', 'f', 'b'}.").tag(config=True)
    timestep = PositiveFloat(5., help="Timestep").tag(config=True)
    start_time = PositiveFloat(300., help="Start time of period of interest").tag(config=True)
    end_time = PositiveFloat(1800., help="End time of period of interest").tag(config=True)
    h_min = PositiveFloat(10., help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(1e5, help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    diffusivity = NonNegativeFloat(1e-3, allow_none=True, help="Diffusion coefficient").tag(config=True)
    solver_parameters = PETScSolverParameters({'ksp_type': 'gmres',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'multiplicative'}).tag(config=True)

    # Physical parameters
    coriolis = Unicode('sin', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(9.81, help="Gravitational acceleration").tag(config=True)
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)

    # Data extraction
    gauges = List(trait=Unicode, default_value=["P02", "P06"], help="Gauges at which to extract timeseries").tag(config=True)

    def lat(self, gauge):
        return {"P02": 38.5002, "P06": 38.6340, "801": 38.2, "802": 39.3, "803": 38.9, "804": 39.7, "806": 37.0,
                "Fukushima Daiichi": 37.4213, "Onagawa": 38.3995, "Fukushima Daini": 37.3166, "Tokai": 36.4664,
                "Hamaoka": 34.6229, "Tohoku": 41.1880, "Tokyo": 35.6895}[gauge]

    def lon(self, gauge):
        return {"P02": 142.5016, "P06": 142.5838, "801": 141.7, "802": 142.1, "803": 141.8, "804": 142.2, "806": 141.2,
                "Fukushima Daiichi": 141.0281, "Onagawa": 141.5008, "Fukushima Daini": 141.0249, "Tokai": 140.6067,
                "Hamaoka": 138.1433, "Tohoku": 141.3903, "Tokyo": 139.6917}[gauge]

    def gauge_coordinates(self, gauge):
        """
        :param gauge: Tide / pressure gauge name, from {P02, P06, 801, 802, 803, 804, 806}.
        :return: UTM coordinate for chosen gauge.
        """
        E, N = from_latlon(self.lat(gauge), self.lon(gauge), force_zone_number=54)
        return E, N

    def meshSize(self, i):
        return (5918, 7068, 8660, 10988, 14160, 19082, 27280, 41730, 72602, 160586, 681616)[i]

    # Region of importance
    radii = List(trait=Float, default_value=[50e3],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=list(from_latlon(37.4213, 141.0281)),
               help="Important locations, written as a list.").tag(config=True)
    J = Float(1.3347e+12, help="Objective functional value on a fine mesh").tag(config=True)


class RossbyWaveOptions(AdaptOptions):
    name = 'Parameters for the equatorial Rossby wave test problem'
    mode = 'RossbyWave'

    # Solver parameters
    timesteps_per_export = NonNegativeInteger(12, help="Timesteps per data dump").tag(config=True)
    timesteps_per_remesh = NonNegativeInteger(48, help="Timesteps per mesh adaptation").tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('s', help="Adaptation field of interest, from {'s', 'f', 'b'}.").tag(config=True)
    timestep = PositiveFloat(0.05, help="Timestep").tag(config=True)
    start_time = PositiveFloat(30., help="Start time of period of interest").tag(config=True)
    end_time = PositiveFloat(120., help="End time of period of interest").tag(config=True)
    h_min = PositiveFloat(1e-3, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(10., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.255, help="Scaling parameter for target number of vertices.").tag(config=True)
    # Here ``rescaling`` is adjusted based on using oversized domain to account for periodicity
    solver_parameters = PETScSolverParameters({'ksp_type': 'gmres',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'multiplicative'}).tag(config=True)

    # Physical parameters
    coriolis = Unicode('beta', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(1., help="Gravitational acceleration").tag(config=True)

    # Region of importance
    radii = List(trait=Float, default_value=[np.sqrt(3), np.sqrt(3)],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[0., 0., -48., 0.],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(5.3333, help="Objective functional value on a fine mesh").tag(config=True)
    J_mirror = Float(5.3333, help="Objective functional value for mirrored problem on a fine mesh").tag(config=True)


class KelvinWaveOptions(AdaptOptions):
    name = 'Parameters for the equatorial Kelvin wave test problem'
    mode = 'KelvinWave'

    # Solver parameters
    timesteps_per_export = NonNegativeInteger(12, help="Timesteps per data dump").tag(config=True)
    timesteps_per_remesh = NonNegativeInteger(48, help="Timesteps per mesh adaptation").tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('s', help="Adaptation field of interest, from {'s', 'f', 'b'}.").tag(config=True)
    timestep = PositiveFloat(0.05, help="Timestep").tag(config=True)
    start_time = PositiveFloat(10., help="Start time of period of interest").tag(config=True)
    end_time = PositiveFloat(36., help="End time of period of interest").tag(config=True)
    h_min = PositiveFloat(1e-3, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(10., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    solver_parameters = PETScSolverParameters({'ksp_type': 'gmres',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'multiplicative'}).tag(config=True)

    # Physical parameters
    coriolis = Unicode('beta', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(1., help="Gravitational acceleration").tag(config=True)

    # Region of importance
    radii = List(trait=Float, default_value=[np.sqrt(3)],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[15., 0.],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(5.3333, help="Objective functional value on a fine mesh").tag(config=True)
    J_mirror = Float(5.3333, help="Objective functional value for mirrored problem on a fine mesh").tag(config=True)


class GaussianOptions(AdaptOptions):
    name = 'Parameters for the shallow water test problem with Gaussian initial condition'
    mode = 'GaussianTest'

    # Solver parameters
    timesteps_per_export = NonNegativeInteger(6, help="Timesteps per data dump").tag(config=True)
    timesteps_per_remesh = NonNegativeInteger(12, help="Timesteps per mesh adaptation").tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('s', help="Adaptation field of interest, from {'s', 'f', 'b'}.").tag(config=True)
    timestep = PositiveFloat(0.05, help="Timestep").tag(config=True)
    start_time = PositiveFloat(0.6, help="Start time of period of interest").tag(config=True)
    end_time = PositiveFloat(3., help="End time of period of interest").tag(config=True)
    h_min = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(1., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    solver_parameters = PETScSolverParameters({'ksp_type': 'gmres',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'multiplicative'}).tag(config=True)

    # Physical parameters
    coriolis = Unicode('beta', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(9.81, help="Gravitational acceleration").tag(config=True)

    # Region of importance
    radii = List(trait=Float, default_value=[np.sqrt(0.3)],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[0., np.pi],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(1.6160e-4, help="Objective functional value on a fine mesh").tag(config=True)


class AdvectionOptions(AdaptOptions):
    name = 'Parameters for advection diffusion test problem'
    mode = 'AdvectionDiffusion'

    # Solver parameters
    timesteps_per_export = NonNegativeInteger(20, help="Timesteps per data dump").tag(config=True)
    timesteps_per_remesh = NonNegativeInteger(40, help="Timesteps per mesh adaptation").tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    timestep = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = PositiveFloat(10., help="Start time of period of interest").tag(config=True)
    end_time = PositiveFloat(60., help="End time of period of interest").tag(config=True)
    h_min = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(5., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    diffusivity = NonNegativeFloat(0.1, allow_none=True, help="Diffusion coefficient").tag(config=True)
    bell_x0 = Float(0.5, help="x-coordinate corresponding to tracer source centre").tag(config=True)
    bell_y0 = Float(5., help="y-coordinate corresponding to tracer source centre").tag(config=True)
    bell_r0 = Float(0.457, help="Radius of tracer source").tag(config=True)
    u_mag = FiredrakeScalarExpression(Constant(1.), help="(Estimate of) maximum advective speed").tag(config=True)
    tracer_family = Unicode('cg', help="Finite element family for tracer flow, from {'dg', 'dg'}.").tag(config=True)
    solver_parameters = PETScSolverParameters({'ksp_type': 'gmres',
                                               'pc_type': 'sor'}).tag(config=True)
    # solver_parameters = PETScSolverParameters({'ksp_type': 'preonly',
    #                                            'pc_type': 'bjacobi',
    #                                            'sub_pc_type': 'ilu'}).tag(config=True)

    # Region of importance
    radii = List(trait=Float, default_value=[0.5],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[40., 7.5],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(0.1871, help="Objective functional value on a fine mesh").tag(config=True)    # On 64,000 elements

    # Cross sections
    plot_cross_section = Bool(False, help="Plot horizontal and vertical slices through profile.").tag(config=True)
    h_slice = List(trait=Tuple, default_value=[(x, 5.) for x in np.linspace(0.01, 49.99, 101)],
                   help="List of coordinates corresponding to a horizontal slice of the domain").tag(config=True)
    v_slice = List(trait=Tuple, default_value=[(30., y) for y in np.linspace(0.002, 9.998, 101)],
                   help="List of coordinates corresponding to a vertical slice of the domain").tag(config=True)
