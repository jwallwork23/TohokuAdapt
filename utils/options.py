from thetis import *

import numpy as np

__all__ = ["Options"]


class Options:
    def __init__(self,
                 mode='tohoku',
                 family='dg-dg',
                 timestepper='CrankNicolson',
                 approach='fixedMesh',
                 rescaling=0.85,
                 hmin=500.,
                 hmax=1e5,
                 minNorm=1.,
                 maxAnisotropy=100,
                 gradate=False,
                 rotational=False,
                 bAdapt=False,
                 regen=False,
                 plotpvd=False,
                 maxGrowth=1.4,
                 dt=0.5,
                 ndump=50,
                 rm=50,         # TODO: Given that dt is now small, experiment with increasing this number
                 nAdapt=1,
                 nVerT=1000,
                 orderChange=0,
                 refinedSpace=False):
        """
        :param mode: problem considered.
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param timestepper: timestepping scheme, from {'ForwardEuler', 'BackwardEuler', 'CrankNicolson'}.
        :param approach: meshing strategy, from {'fixedMesh', 'hessianBased', 'DWP', 'DWR'}.
        :param rescaling: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param minNorm: Minimal tolerated norm for error estimates.
        :param maxAnisotropy: maximum tolerated aspect ratio.
        :param gradate: Toggle metric gradation.
        :param rotational: Toggle rotational / non-rotational equations.
        :param bAdapt: intersect metrics with Hessian w.r.t. bathymetry.
        :param regen: regenerate error estimates based on saved data.
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
        for i in (gradate, rotational, plotpvd, refinedSpace, bAdapt, regen):
            assert(isinstance(i, bool))
        self.gradate = gradate
        self.rotational = rotational
        self.bAdapt = bAdapt
        self.regen = regen
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

        # Physical parameters
        self.Omega = 7.291e-5  # Planetary rotation rate

        # Override default parameter choices for SW and RW cases:
        if self.mode == 'shallow-water':
            self.Tstart = 0.1
            self.Tend = 2.5
            self.hmin = 1e-4
            self.hmax = 1.
            self.minNorm = 1e-6
            self.rm = 5
            self.dt = 0.05
            self.ndump = 5
            self.J = 1.1184e-3,                                         # On mesh of 524,288 elements
            self.xy = [0., 0.5 * np.pi, 0.5 * np.pi, 1.5 * np.pi]
            self.g = 9.81
        elif self.mode == 'rossby-wave':
            self.Tstart = 20.
            self.Tend = 45.60
            self.hmin = 5e-3
            self.hmax = 10.
            self.minNorm = 1e-4
            self.rm = 24
            self.dt = 0.05
            self.ndump = 12
            self.g = 1.
            self.J = 1.                                                 # TODO: establish this
            self.xy = [-24., -20., -2., 2.]
        elif self.mode in ('tohoku', 'model-verification'):
            self.Tstart = 300.
            self.Tend = 1500.
            self.J = 1.240e+13                                          # On mesh of 681,666 elements     TODO: Check
            self.xy = [490e3, 640e3, 4160e3, 4360e3]
            self.g = 9.81

            # Gauge locations in latitude-longitude coordinates and mesh element counts
            self.glatlon = {"P02": (38.5002, 142.5016), "P06": (38.6340, 142.5838), "801": (38.2, 141.7),
                            "802": (39.3, 142.1), "803": (38.9, 141.8), "804": (39.7, 142.2), "806": (37.0, 141.2)}
            self.meshSizes = (5918, 7068, 8660, 10988, 14160, 19082, 27280, 41730, 72602, 160586, 681616)

        # Derived timestep indices
        self.cntT = int(np.ceil(self.Tend / self.dt))               # Final timestep index
        self.iStart = int(self.Tstart / (self.ndump * self.dt))     # First exported timestep of period of interest
        self.iEnd = int(self.cntT / self.ndump)                     # Final exported timestep of period of interest

        # Specify FunctionSpaces
        self.degree1 = 2 if self.family == 'cg-cg' else 1
        self.degree2 = 2 if self.family == 'dg-cg' else 1
        self.space1 = "CG" if self.family == 'cg-cg' else "DG"
        self.space2 = "DG" if self.family == 'dg-dg' else "CG"

        # Plotting dictionaries     TODO: Remove explicit and implicit options, change 'adjoint based'
        self.labels = ("Fixed mesh", "Hessian based", "Explicit", "Implicit", "Adjoint based", "Goal based")
        self.styles = {self.labels[0]: 's', self.labels[1]: '^', self.labels[2]: 'x', self.labels[3]: 'o',
                       self.labels[4]: '*', self.labels[5]: 'h'}
        self.stamps = {self.labels[0]: 'fixedMesh', self.labels[1]: 'hessianBased', self.labels[2]: 'explicit',
                       self.labels[3]: 'implicit', self.labels[4]: 'DWF', self.labels[5]: 'DWR'}


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
