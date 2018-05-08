from thetis import *

import numpy as np

__all__ = ["Options"]


class Options:
    def __init__(self,
                 mode='tohoku',
                 family='dg-dg',
                 timestepper='CrankNicolson',
                 approach='fixedMesh',
                 coriolis='off',
                 rescaling=0.85,
                 hmin=500.,
                 hmax=1e5,
                 minNorm=1.,
                 maxAnisotropy=100,
                 gradate=False,
                 bAdapt=False,
                 plotpvd=False,
                 maxGrowth=1.4,
                 dt=5.,
                 ndump=10,
                 rm=30,
                 nAdapt=1,
                 nVerT=1000,
                 orderChange=0,
                 refinedSpace=False):
        """
        :param mode: problem considered.
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param timestepper: timestepping scheme, from {'ForwardEuler', 'BackwardEuler', 'CrankNicolson'}.
        :param approach: meshing strategy, from {'fixedMesh', 'hessianBased', 'DWP', 'DWR'}.
        :param coriolis: Type of Coriolis term, from {'off', 'f', 'beta', 'sin'}.
        :param rescaling: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param minNorm: Minimal tolerated norm for error estimates.
        :param maxAnisotropy: maximum tolerated aspect ratio.
        :param gradate: Toggle metric gradation.
        :param bAdapt: intersect metrics with Hessian w.r.t. bathymetry.
        :param plotpvd: toggle saving solution fields to .pvd.
        :param maxGrowth: metric gradation scaling parameter.
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh. (Should be an integer multiple of ndump.)
        :param nAdapt: number of mesh adaptions per mesh regeneration.
        :param nVerT: target number of vertices.
        :param orderChange: change in polynomial degree for residual approximation.
        :param refinedSpace: refine space too compute errors and residuals.
        """

        # Model parameters
        try:
            assert approach in ('fixedMesh', 'hessianBased', 'DWP', 'DWR')
            self.approach = approach
        except:
            raise ValueError('Meshing strategy %s not recognised' % approach)
        try:
            assert mode in ('tohoku', 'shallow-water', 'rossby-wave', 'model-verification')
            self.mode = mode
        except:
            raise ValueError('Test problem %s not recognised.' % mode)
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
        self.params = {'mat_type': 'matfree',
                       'snes_type': 'ksponly',
                       'pc_type': 'python',
                       'pc_python_type': 'firedrake.AssembledPC',
                       'assembled_pc_type': 'lu',
                       'snes_lag_preconditioner': -1,
                       'snes_lag_preconditioner_persists': True}

        # Adaptivity parameters
        if self.approach == 'hessianBased':
            self.adaptField = 's'       # Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
            self.normalisation = 'lp'   # Metric normalisation using Lp norm. 'manual' also available.
            self.normOrder = 2
            self.targetError = 1e-3
            self.hessMeth = 'dL2'       # Hessian recovery by double L2 projection. 'parts' also available
        else:
            self.maxScaling = 1e6       # Maximum scale factor for error estimator  TODO: choose for Tohoku
        if self.approach == "DWR":
            self.rescaling = 0.1        # Chosen small enough to ensure accuracy for a small element count
        else:
            try:
                assert rescaling > 0
                self.rescaling = rescaling
            except:
                raise ValueError('Invalid value of %.3f for scaling parameter. rescaling > 0 is required.' % rescaling)
        try:
            assert (hmin > 0) and (hmax > hmin) and (minNorm > 0)
            self.hmin = hmin
            self.hmax = hmax
            self.minNorm = minNorm
        except:
            raise ValueError('Invalid min/max element sizes. hmax > hmin > 0 is required.')
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

        # Timestepping and (more) adaptivity parameters
        self.dt = dt
        for i in (ndump, rm, orderChange, nVerT, nAdapt):
            assert isinstance(i, int)
            try:
                assert rm % ndump == 0
            except:
                raise ValueError("`rm` should be an integer multiple of `ndump`.")
        self.ndump = ndump
        self.rm = rm
        self.nAdapt = nAdapt
        self.orderChange = orderChange
        self.nVerT = nVerT

        # Override default parameter choices for SW and RW cases:
        if self.mode == 'shallow-water':
            self.Tstart = 0.0
            self.Tend = 3.
            self.hmin = 1e-4
            self.hmax = 1.
            self.minNorm = 1e-6
            self.rm = 6
            self.dt = 0.05
            self.ndump = 2
            self.J = 1.1184e-3,   # On mesh of 524,288 elements
            self.xy = [0., 0.5 * np.pi, 0.5 * np.pi, 1.5 * np.pi]
            self.xy2 = [1.5 * np.pi, 2 * np.pi, 0.5 * np.pi, 1.5 * np.pi]
            self.g = 9.81
        elif self.mode == 'rossby-wave':
            self.coriolis = 'beta'
            self.Tstart = 10.
            self.Tend = 36.
            self.hmin = 5e-3
            self.hmax = 10.
            self.minNorm = 1e-4
            self.rm = 24
            self.dt = 0.05
            self.ndump = 12
            self.g = 1.
            self.J = 5.7085       # On mesh of 2,359,296 elements, using asymptotic solution
            self.xy = [-16., -14., -3., 3.]
            self.xy2 = [14., 16., -3., 3.]
        elif self.mode in ('tohoku', 'model-verification'):
            self.Tstart = 300.
            self.Tend = 1500.
            self.J = 1.240e+13  #  On mesh of 681,666 elements     TODO: Check
            self.xy = [490e3, 640e3, 4160e3, 4360e3]
            self.g = 9.81
        if self.approach in ("DWP", "DWR"):
            self.rm *= 2

            # Gauge locations in latitude-longitude coordinates and mesh element counts
            self.glatlon = {"P02": (38.5002, 142.5016), "P06": (38.6340, 142.5838), "801": (38.2, 141.7),
                            "802": (39.3, 142.1), "803": (38.9, 141.8), "804": (39.7, 142.2), "806": (37.0, 141.2)}
            self.meshSizes = (5918, 7068, 8660, 10988, 14160, 19082, 27280, 41730, 72602, 160586, 681616)
            self.latFukushima = 37.050419   # Latitude of Fukushima
            self.Omega = 7.291e-5           # Planetary rotation rate

        # Derived timestep indices
        self.cntT = int(np.ceil(self.Tend / self.dt))               # Final timestep index
        self.iStart = int(self.Tstart / (self.ndump * self.dt))     # First exported timestep of period of interest
        self.iEnd = int(self.cntT / self.ndump)                     # Final exported timestep of period of interest
        self.rmEnd = int(self.cntT / self.rm)                       # Final mesh index

        # Specify FunctionSpaces
        self.degree1 = 2 if self.family == 'cg-cg' else 1
        self.degree2 = 2 if self.family == 'dg-cg' else 1
        self.space1 = "CG" if self.family == 'cg-cg' else "DG"
        self.space2 = "DG" if self.family == 'dg-dg' else "CG"

        # Plotting dictionaries
        self.labels = ("Fixed mesh", "Hessian based", "DWP", "DWR")
        self.styles = {self.labels[0]: 's', self.labels[1]: '^', self.labels[2]: 'x', self.labels[3]: 'o'}
        self.stamps = {self.labels[0]: 'fixedMesh', self.labels[1]: 'hessianBased', self.labels[2]: 'DWF',
                       self.labels[3]: 'DWR'}


    def gaugeCoord(self, gauge):
        """
        :param gauge: Tide / pressure gauge name, from {P02, P06, 801, 802, 803, 804, 806}.
        :return: UTM coordinate for chosen gauge.
        """
        from .conversion import from_latlon

        E, N, zn, zl = from_latlon(self.glatlon[gauge][0], self.glatlon[gauge][1], force_zone_number=54)
        return E, N


    def mixedSpace(self, mesh):
        """
        :param mesh: mesh upon which to build mixed space.
        :return: mixed VectorFunctionSpace x FunctionSpace as specified by ``self.family``.
        """
        deg1 = self.degree1 + self.orderChange
        deg2 = self.degree2 + self.orderChange
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
