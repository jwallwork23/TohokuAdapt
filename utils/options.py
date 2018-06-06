from thetis import *
from thetis.configuration import *
from firedrake import Expression

import numpy as np

from .conversion import from_latlon

__all__ = ["Options", "TohokuOptions", "RossbyWaveOptions", "GaussianOptions", "AdvectionDiffusionOptions"]


class AdaptOptions(FrozenConfigurable):
    name = 'Common parameters for TohokuAdapt project'

    # Mesh adaptivity parameters
    approach = Unicode('fixedMesh',
                       help="Mesh adaptive approach considered, from {'fixedMesh', 'hessianBased', 'DWP', 'DWR'}"
                       ).tag(config=True)
    gradate = Bool(False, help='Toggle metric gradation.').tag(config=True)
    bAdapt = Bool(False, help='Toggle adaptation based on bathymetry field.').tag(config=True)
    plotpvd = Bool(False, help='Toggle plotting of fields.').tag(config=True)
    maxGrowth = PositiveFloat(1.4, help="Metric gradation scaling parameter.").tag(config=True)
    maxAnisotropy = PositiveFloat(100., help="Maximum tolerated anisotropy.").tag(config=True)
    nAdapt = NonNegativeInteger(1, help="Number of mesh adaptations per remeshing.").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    orderChange = NonNegativeInteger(0, help="Change in polynomial degree for residual approximation.").tag(config=True)
    refinedSpace = Bool(False, help="Refine space too compute errors and residuals.").tag(config=True)
    adaptField = Unicode('s', help="Adaptation field of interest, from {'s', 'f', 'b'}.").tag(config=True)
    normalisation = Unicode('lp', help="Normalisation approach, from {'lp', 'manual'}.").tag(config=True)
    hessMeth = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    normOrder = NonNegativeInteger(2, help="Degree p of Lp norm used.")
    family = Unicode('dg-dg', help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)

    def cntT(self):
        return int(np.ceil(self.Tend / self.dt))  # Final timestep index

    def iStart(self):
        return int(self.Tstart / (self.ndump * self.dt))  # First exported timestep of period of interest

    def iEnd(self):
        return int(self.cntT() / self.ndump)  # Final exported timestep of period of interest

    def rmEnd(self):
        return int(self.cntT() / self.rm)  # Final mesh index

    def dumpsPerRemesh(self):
        assert self.rm % self.ndump == 0
        return int(self.rm / self.ndump)

    def indicator(self, mesh):
        try:
            P1 = FunctionSpace(mesh, "DG", 1)
            iA = Function(P1, name="Region of interest")

            if len(np.shape(self.radii)) == 0:
                expr = Expression("pow(x[0] - x0, 2) + pow(x[1] - y0, 2) < r + eps ? 1 : 0",
                                  x0=self.loc[0], y0=self.loc[1], r=pow(self.radii, 2), eps=1e-10)
            elif len(np.shape(self.radii)) == 1:
                assert len(self.loc)/2 == len(self.radii)
                e = "(pow(x[0] - %f, 2) + pow(x[1] - %f, 2) < %f + %f)" \
                    % (self.loc[0], self.loc[1], pow(self.radii[0], 2), 1e-10)
                for i in range(1, len(self.radii)):
                    e += "&& (pow(x[0] - %f, 2) + pow(x[1] - %f, 2) < %f + %f)" \
                         % (self.loc[2*i], self.loc[2*i+1], pow(self.radii[i], 2), 1e-10)
                expr = Expression(e)
            else:
                raise ValueError("Indicator function radii input not recognised.")
        except:
            raise ValueError("Radius or location of region of importance not currently given.")

        iA.interpolate(expr)

        return iA


    def mixedSpace(self, mesh, enrich=False):
        """
        :param mesh: mesh upon which to build mixed space.
        :return: mixed VectorFunctionSpace x FunctionSpace as specified by ``self.family``.
        """
        d1 = 1
        d2 = 2 if self.family == 'dg-cg' else 1
        if enrich:
            d1 += self.orderChange
            d2 += self.orderChange
        return VectorFunctionSpace(mesh, "DG", d1) * FunctionSpace(mesh, "DG" if self.family == 'dg-dg' else "CG", d2)

    def printToScreen(self, mn, adaptTimer, solverTime, nEle, Sn, mM, t):
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
              (mn, 100 * t / self.Tend, adaptTimer, solverTime, nEle, av, mM[0], mM[1]))
        return av


class TohokuOptions(AdaptOptions):
    name = 'Parameters for the Tohoku problem'
    mode = 'tohoku'

    # Solver parameters
    ndump = NonNegativeInteger(10, help="Timesteps per data dump").tag(config=True)
    rm = NonNegativeInteger(30, help="Timesteps per mesh aapdaptation").tag(config=True)
    nVerT = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    dt = PositiveFloat(5., help="Timestep").tag(config=True)
    Tstart = PositiveFloat(300., help="Start time of period of interest").tag(config=True)
    Tend = PositiveFloat(1500., help="End time of period of interest").tag(config=True)
    hmin = PositiveFloat(10., help="Minimum element size").tag(config=True)
    hmax = PositiveFloat(1e5, help="Maximum element size").tag(config=True)
    minNorm = PositiveFloat(1.).tag(config=True)    # TODO: Not sure about this
    maxScaling = PositiveFloat(5e5).tag(config=True)  # TODO: Not sure about this

    # Physical parameters
    coriolis = Unicode('sin', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(9.81, help="Gravitational acceleration").tag(config=True)
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)
    viscosity = NonNegativeFloat(1e-3, help="Planetary rotation rate").tag(config=True)

    def gaugeLocation(self, gauge):
        return {"P02": (38.5002, 142.5016), "P06": (38.6340, 142.5838),
                "801": (38.2, 141.7), "802": (39.3, 142.1), "803": (38.9, 141.8), "804": (39.7, 142.2),
                "806": (37.0, 141.2), "Fukushima": (37.4213, 141.0281)}[gauge]

    def meshSize(self, i):
        return (5918, 7068, 8660, 10988, 14160, 19082, 27280, 41730, 72602, 160586, 681616)[i]

    def di(self):
        return 'plots/tohoku/' + self.approach + '/'

    # Region of importance
    radii = List(trait=Float, default_value=[50e3],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[37.4213, 141.0281],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(1.335e+12, help="Objective functional value on a fine mesh").tag(config=True)


class RossbyWaveOptions(AdaptOptions):
    name = 'Parameters for the equatorial Rossby wave test problem'
    mode = 'rossby-wave'

    # Solver parameters
    ndump = NonNegativeInteger(12, help="Timesteps per data dump").tag(config=True)
    rm = NonNegativeInteger(48, help="Timesteps per mesh adaptation").tag(config=True)
    nVerT = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    dt = PositiveFloat(0.05, help="Timestep").tag(config=True)
    Tstart = PositiveFloat(10., help="Start time of period of interest").tag(config=True)
    Tend = PositiveFloat(36., help="End time of period of interest").tag(config=True)
    hmin = PositiveFloat(1e-3, help="Minimum element size").tag(config=True)
    hmax = PositiveFloat(10., help="Maximum element size").tag(config=True)
    minNorm = PositiveFloat(1e-4).tag(config=True)  # TODO: Not sure about this
    maxScaling = PositiveFloat(5e5).tag(config=True)  # TODO: Not sure about this

    # Physical parameters
    coriolis = Unicode('beta', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(1., help="Gravitational acceleration").tag(config=True)

    def di(self):
        return 'plots/rossby-wave/' + self.approach + '/'

    # Region of importance
    radii = List(trait=Float, default_value=[np.sqrt(3)],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[-15., 0.],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(5.3333, help="Objective functional value on a fine mesh").tag(config=True)


class GaussianOptions(AdaptOptions):
    name = 'Parameters for the shallow water test problem with Gaussian initial condition'
    mode = 'shallow-water'

    # Solver parameters
    ndump = NonNegativeInteger(6, help="Timesteps per data dump").tag(config=True)
    rm = NonNegativeInteger(12, help="Timesteps per mesh adaptation").tag(config=True)
    nVerT = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    dt = PositiveFloat(0.05, help="Timestep").tag(config=True)
    Tstart = PositiveFloat(0.6, help="Start time of period of interest").tag(config=True)
    Tend = PositiveFloat(3., help="End time of period of interest").tag(config=True)
    hmin = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    hmax = PositiveFloat(1., help="Maximum element size").tag(config=True)
    minNorm = PositiveFloat(1e-6).tag(config=True)      # TODO: Not sure about this
    maxScaling = PositiveFloat(5e9).tag(config=True)    # TODO: Not sure about this

    # Physical parameters
    coriolis = Unicode('beta', help="Type of Coriolis parameter, from {'sin', 'beta', 'f', 'off'}.").tag(config=True)
    g = PositiveFloat(9.81, help="Gravitational acceleration").tag(config=True)

    def di(self):
        return 'plots/shallow-water/' + self.approach + '/'

    # Region of importance
    radii = List(trait=Float, default_value=[np.sqrt(0.3)],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[0., np.pi],
               help="Important locations, written as a list.").tag(config=True)
    J = Float(1.6160e-4, help="Objective functional value on a fine mesh").tag(config=True)


class AdvectionDiffusionOptions(AdaptOptions):
    name = 'Parameters for advection diffusion test problem'
    mode = 'advection-diffusion'

    # Solver parameters
    ndump = NonNegativeInteger(5, help="Timesteps per data dump").tag(config=True)
    rm = NonNegativeInteger(10, help="Timesteps per mesh adaptation").tag(config=True)
    nVerT = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    dt = PositiveFloat(0.05, help="Timestep").tag(config=True)
    Tstart = PositiveFloat(0.4, help="Start time of period of interest").tag(config=True)
    Tend = PositiveFloat(2.4, help="End time of period of interest").tag(config=True)
    hmin = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    hmax = PositiveFloat(1., help="Maximum element size").tag(config=True)
    minNorm = PositiveFloat(1e-5).tag(config=True)  # TODO: Not sure about this
    maxScaling = PositiveFloat(5e5).tag(config=True)  # TODO: Not sure about this

    # Region of importance
    radii = List(trait=Float, default_value=[0.2],
                 help="Radius of indicator function around location of interest.").tag(config=True)
    loc = List(trait=Float, default_value=[3.75, 0.5],
               help="Important locations, written as a list.").tag(config=True)

    def di(self):
        return 'plots/advection-diffusion/' + self.approach + '/'


class Options:
    def __init__(self,
                 mode='tohoku',
                 approach='fixedMesh',
                 family='dg-dg',
                 timestepper='CrankNicolson',
                 maxAnisotropy=100.,
                 gradate=False,
                 bAdapt=False,
                 plotpvd=False,
                 maxGrowth=1.4,
                 nAdapt=1,
                 orderChange=0,
                 refinedSpace=False,
                 coriolis='sin',
                 rescaling=0.85,
                 ndump=None,
                 rm=None,
                 nVerT=1000):
        """
        :param mode: problem considered.
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param timestepper: timestepping scheme, from {'ForwardEuler', 'BackwardEuler', 'CrankNicolson'}.
        :param approach: meshing strategy, from {'fixedMesh', 'hessianBased', 'DWP', 'DWR'}.
        :param coriolis: Type of Coriolis term, from {'off', 'f', 'beta', 'sin'}.
        :param rescaling: Scaling parameter for target number of vertices.
        :param maxAnisotropy: maximum tolerated aspect ratio.
        :param gradate: Toggle metric gradation.
        :param bAdapt: intersect metrics with Hessian w.r.t. bathymetry.
        :param plotpvd: toggle saving solution fields to .pvd.
        :param maxGrowth: metric gradation scaling parameter.
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh. (Should be an integer multiple of ndump.)
        :param nAdapt: number of mesh adaptions per mesh regeneration.
        :param nVerT: target number of vertices.
        :param orderChange: change in polynomial degree for residual approximation.
        :param refinedSpace: refine space too compute errors and residuals.
        """

        # Model parameters
        self.mode = mode
        try:
            assert approach in ('fixedMesh', 'hessianBased', 'DWP', 'DWR', 'DWR_ho', 'DWR_r')
            self.approach = approach
        except:
            raise ValueError('Meshing strategy %s not recognised' % approach)
        try:
            assert coriolis in ('off', 'f', 'beta', 'sin')
            self.coriolis = coriolis
        except:
            raise ValueError('Coriolis term type %s not recognised' % coriolis)

        # Solver parameters
        try:
            assert family in ('dg-dg', 'dg-cg', 'cg-cg')
            self.family = family
        except:
            raise ValueError('Mixed function space %s not recognised.' % family)
        try:
            assert timestepper in ('ForwardEuler', 'BackwardEuler', 'CrankNicolson')
            self.timestepper = timestepper
        except:
            raise NotImplementedError

        # Default parameter choices
        if self.mode == 'shallow-water':
            self.Tstart = 0.0
            self.Tend = 3.
            self.hmin = 1e-4
            self.hmax = 1.
            self.minNorm = 1e-6
            if rm is None:
                self.rm = 12
            self.dt = 0.05
            if ndump is None:
                self.ndump = 6
            # self.J = 1.1184e-3,   # On mesh of 524,288 elements
            self.J = 1.6160e-04     # On mesh of 131,072 elements
            # self.xy = [0., 0.5 * np.pi, 0.5 * np.pi, 1.5 * np.pi]
            # self.xy2 = [1.5 * np.pi, 2 * np.pi, 0.5 * np.pi, 1.5 * np.pi]
            self.xy = [0., np.pi]
            self.xy2 = [2 * np.pi, np.pi]
            self.radius = np.sqrt(0.3)
            self.g = 9.81
        elif self.mode == 'rossby-wave':
            self.Tstart = 10.
            self.Tend = 36.
            self.hmin = 1e-3
            self.hmax = 10.
            self.minNorm = 1e-4
            if rm is None:
                self.rm = 48
                # self.rm = 720        # TODO: experiment with different values
            self.dt = 0.05
            if ndump is None:
                self.ndump = 12
            self.g = 1.
            # self.J = 5.7613                     # On mesh of 9,437,184 elements, using asymptotic solution
            self.J = 5.3333
            # self.xy = [-16., -14., -3., 3.]
            # self.xy2 = [14., 16., -3., 3.]
            self.xy = [-15., 2.]
            self.xy2 = [15., 2.]
            self.radius = np.sqrt(3.)  # TODO: Change this to 2. and redo
            self.J_mirror = 6.1729e-06  # On mesh of 2,359,296 elements, using asymptotic solution
        elif self.mode in ('tohoku', 'model-verification'):
            # Gauge locations in latitude-longitude coordinates and mesh element counts
            self.glatlon = {"P02": (38.5002, 142.5016),
                            "P06": (38.6340, 142.5838),
                            "801": (38.2, 141.7),
                            "802": (39.3, 142.1),
                            "803": (38.9, 141.8),
                            "804": (39.7, 142.2),
                            "806": (37.0, 141.2)}
            self.meshSizes = (5918, 7068, 8660, 10988, 14160, 19082, 27280, 41730, 72602, 160586, 681616)
            self.latFukushima = 37.4213  # Latitude of Daiichi Nuclear Power Plant
            self.lonFukushima = 141.0281  # Longitude of Daiichi Nuclear Power Plant
            self.Omega = 7.291e-5  # Planetary rotation rate

            self.dt = 5.
            # self.Tstart = 300.
            self.Tstart = 600.
            # self.Tend = 1500.
            self.Tend = 1800.
            if ndump is None:
                self.ndump = 10
            if rm is None:
                self.rm = 30
            self.hmin=10.
            self.hmax=1e5
            self.minNorm = 1.
            self.J = {'off': 1.324e+13,
                      'f': 1.309e+13,
                      'beta': 1.288e+13,
                      'sin': 1.305e+13}[self.coriolis]  # (to 4.s.f.) On mesh of 158,596 elements
            # self.xy = [490e3, 640e3, 4160e3, 4360e3]
            self.xy = from_latlon(self.latFukushima, self.lonFukushima, force_zone_number=54)[:2]
            self.xy2 = [0., 0., 0., 0.]
            self.radius = 50e3          # NOTE: P02 and P06 do not fall within this radius. 806 does
            # self.radius = None
            self.g = 9.81
        elif self.mode == 'advection-diffusion':
            self.dt = 0.05
            self.Tend = 2.4
            self.hmin = 1e-4
            self.hmax = 1.
            if rm is None:
                self.rm = 10
            if ndump is None:
                self.ndump = 10

        # Adaptivity parameters
        if self.approach == 'hessianBased':
            self.adaptField = 's'       # Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
            self.normalisation = 'lp'   # Metric normalisation using Lp norm. 'manual' also available.
            self.normOrder = 2
            self.hessMeth = 'dL2'       # Hessian recovery by double L2 projection. 'parts' also available
        else:
            if self.mode == 'shallow-water':
                self.maxScaling = 5e9       # TODO: What is the actual interpretation?
            else:
                self.maxScaling = 5e5       # Maximum scale factor for error estimator
        if self.approach == "DWR":
            # self.rescaling = 0.6        # Chosen small enough to ensure accuracy for a small element count
            self.rescaling = 0.85
        else:
            try:
                assert rescaling > 0
                self.rescaling = rescaling
            except:
                raise ValueError('Invalid value of %.3f for scaling parameter. rescaling > 0 is required.' % rescaling)
        try:
            assert maxAnisotropy > 0
            self.maxAnisotropy = maxAnisotropy
        except:
            raise ValueError('Invalid anisotropy value %.1f. a > 0 is required.' % maxAnisotropy)

        # Misc options
        for i in (gradate, plotpvd, refinedSpace, bAdapt):
            assert(isinstance(i, bool))
        self.gradate = gradate
        self.bAdapt = bAdapt
        self.plotpvd = plotpvd
        self.refinedSpace = refinedSpace
        try:
            assert (maxGrowth > 1)
            self.maxGrowth = maxGrowth
        except:
            raise ValueError('Invalid value for growth parameter.')
        if self.mode is not None and self.approach is not None:
            self.di = 'plots/' + self.mode + '/' + self.approach + '/'
        else:
            self.di = 'plots/'

        # Timestepping and (more) adaptivity parameters
        for i in (orderChange, nVerT, nAdapt):
            assert isinstance(i, int)
        if ndump is not None:
            self.ndump = ndump
        if rm is not None:
            self.rm = rm
        self.nAdapt = nAdapt
        self.orderChange = orderChange
        self.nVerT = nVerT
        if self.refinedSpace:
            assert self.orderChange == 0
        if self.orderChange != 0:
            assert not self.refinedSpace

        self.viscosity = 1e-3

        # Derived timestep indices
        try:
            for i in (self.Tend, self.dt, self.Tstart, self.ndump, self.rm):
                assert isinstance(i, (float, int))
            self.cntT = int(np.ceil(self.Tend / self.dt))               # Final timestep index
            self.iStart = int(self.Tstart / (self.ndump * self.dt))     # First exported timestep of period of interest
            self.iEnd = int(self.cntT / self.ndump)                     # Final exported timestep of period of interest
            self.rmEnd = int(self.cntT / self.rm)                       # Final mesh index
            try:
                assert self.rm % self.ndump == 0
                self.dumpsPerRemesh = int(self.rm / self.ndump)
            except:
                raise ValueError("Timesteps per data dump should divide timesteps per remesh")
        except:
            pass

        # Specify FunctionSpaces
        self.degree1 = 1
        self.degree2 = 2 if self.family == 'dg-cg' else 1
        self.space1 = "DG"
        self.space2 = "DG" if self.family == 'dg-dg' else "CG"

        # Plotting dictionaries
        self.labels = ("Fixed mesh", "Hessian based", "DWP", "DWR")
        self.styles = {self.labels[0]: 's', self.labels[1]: '^', self.labels[2]: 'x', self.labels[3]: 'o'}
        self.stamps = {self.labels[0]: 'fixedMesh', self.labels[1]: 'hessianBased', self.labels[2]: 'DWP',
                       self.labels[3]: 'DWR'}


    def gaugeCoord(self, gauge):
        """
        :param gauge: Tide / pressure gauge name, from {P02, P06, 801, 802, 803, 804, 806}.
        :return: UTM coordinate for chosen gauge.
        """
        E, N = from_latlon(self.glatlon[gauge][0], self.glatlon[gauge][1], force_zone_number=54)[:2]
        return E, N


    def mixedSpace(self, mesh, enrich=False):
        """
        :param mesh: mesh upon which to build mixed space.
        :return: mixed VectorFunctionSpace x FunctionSpace as specified by ``self.family``.
        """
        deg1 = self.degree1
        deg2 = self.degree2
        if enrich:
            deg1 += self.orderChange
            deg2 += self.orderChange
        return VectorFunctionSpace(mesh, self.space1, deg1) * FunctionSpace(mesh, self.space2, deg2)


    def printToScreen(self, mn, adaptTimer, solverTime, nEle, Sn, mM, t):
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
                  (mn, 100 * t / self.Tend, adaptTimer, solverTime, nEle, av, mM[0], mM[1]))
        return av
