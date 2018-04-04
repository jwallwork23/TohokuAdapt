from thetis import *

from .conversion import from_latlon


__all__ = ["Options"]


class Options:
    def __init__(self,
                 mode='tohoku',
                 family='dg-cg',
                 vscale=0.85,
                 hmin=500.,
                 hmax=1e6,
                 a=100,
                 ntype='lp',
                 p=2,
                 mtype='s',                 # Best approach for tsunami modelling
                 iso=False,
                 gradate=False,
                 nonlinear=False,
                 rotational=False,
                 bootstrap=False,
                 printStats=True,
                 capBathymetry=True,        # TODO: change this under W&D
                 hessMeth='dL2',
                 beta=1.4,
                 gamma=1.,
                 g=9.81,
                 outputMetric=False,
                 plotpvd=True,
                 Tstart=300.,
                 Tend=1500.,
                 dt=0.5,
                 ndump=50,
                 rm=50,
                 orderChange=0,
                 refinedSpace=False,
                 timestepper='CrankNicolson',
                 wd=False):
        """
        :param mode: problem considered.
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param vscale: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param a: maximum tolerated aspect ratio.
        :param ntype: Normalisation approach: 'lp' or 'manual'.
        :param p: norm order in the Lp normalisation approach, where ``p => 1`` and ``p = infty`` is an option.
        :param mtype: Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
        :param iso: Toggle isotropic / anisotropic algorithm.
        :param gradate: Toggle metric gradation.
        :param nonlinear: Toggle nonlinear / linear equations.
        :param rotational: Toggle rotational / non-rotational equations.
        :param bootstrap: implement mesh bootstrapping to establish initial mesh.
        :param printStats: print to screen during simulation.
        :param capBathymetry: under no wetting-and-drying.
        :param hessMeth: Method of Hessian reconstruction: 'dL2' or 'parts'.
        :param beta: metric gradation scaling parameter.
        :param gamma: metric rescaling parameter.
        :param g: gravitational acceleration.
        :param outputMetric: toggle saving metric to PVD.
        :param plotpvd: toggle saving solution fields to PVD.
        :param Tstart: Lower time range limit (s), before which we can assume the wave won't reach the shore.
        :param Tend: Simulation duration (s).
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh. (Should be an integer multiple of ndump.)
        :param orderChange: change in polynomial degree for residual approximation.
        :param refinedSpace: refine space too compute errors and residuals.
        :param timestepper: timestepping scheme.
        :param wd: toggle wetting and drying.
        """
        self.mode = mode
        try:
            assert mode in ('tohoku', 'shallow-water', 'rossby-wave')
        except:
            raise ValueError('Test problem not recognised.')

        # Adaptivity parameters
        self.family = family
        try:
            assert family in ('dg-dg', 'dg-cg', 'cg-cg')
        except:
            raise ValueError('Mixed function space not recognised.')
        self.vscale = vscale
        try:
            assert vscale > 0
        except:
            raise ValueError('Invalid value for scaling parameter. vscale > 0 is required.')
        self.hmin = hmin
        self.hmax = hmax
        try:
            assert (hmin > 0) & (hmax > hmin)
        except:
            raise ValueError('Invalid min/max element sizes. hmax > hmin > 0 is required.')
        self.a = a
        try:
            assert a > 0
        except:
            raise ValueError('Invalid anisotropy value. a > 0 is required.')
        self.ntype = ntype
        try:
            assert ntype in ('lp', 'manual')
        except:
            raise ValueError('Normalisation approach ``%s`` not recognised.' % ntype)
        self.p = p
        try:
            assert p > 0
        except:
            raise ValueError('Invalid value for p. p > 0 is required.')
        self.mtype = mtype
        try:
            assert mtype in ('f', 's', 'b')
        except:
            raise ValueError('Field for adaption ``%s`` not recognised.' % mtype)
        self.iso = iso
        self.gradate = gradate
        self.nonlinear = nonlinear
        self.rotational = rotational
        self.bootstrap = bootstrap
        self.printStats = printStats
        self.capBathymetry = capBathymetry
        self.outputMetric = outputMetric
        self.plotpvd = plotpvd
        self.wd = wd
        assert(type(gradate) == type(nonlinear) == type(rotational) == type(iso) == type(bootstrap)
               == type(printStats) == type(capBathymetry) == type(outputMetric) == type(plotpvd) == type(wd) ==
               type(refinedSpace) == bool)
        self.hessMeth = hessMeth
        try:
            assert hessMeth in ('dL2', 'parts')
        except:
            raise ValueError('Hessian reconstruction method ``%s`` not recognised.' % hessMeth)
        self.beta = beta
        self.gamma = gamma
        try:
            assert (beta > 1) & (gamma > 0)
        except:
            raise ValueError('Invalid value for scaling parameter.')

        # Physical parameters
        self.g = g
        try:
            assert(g > 0)
        except:
            raise ValueError('Unphysical physical parameters!')

        # Timestepping parameters
        self.Tstart = Tstart
        self.Tend = Tend
        self.dt = dt
        assert(type(Tstart) == type(Tend) == type(dt) == float)
        self.ndump = ndump
        self.rm = rm
        self.orderChange = orderChange
        self.refinedSpace = refinedSpace
        assert(type(ndump) == type(rm) == type(orderChange) == int)
        self.timestepper = timestepper

        # Solver parameters for ``firedrake-tsunami`` case
        self.params = {'mat_type': 'matfree',
                       'snes_type': 'ksponly',
                       'pc_type': 'python',
                       'pc_python_type': 'firedrake.AssembledPC',
                       'assembled_pc_type': 'lu',
                       'snes_lag_preconditioner': -1,
                       'snes_lag_preconditioner_persists': True}

        # Override default parameter choices for SW and RW cases:
        if self.mode == 'shallow-water':
            self.Tstart = 0.1
            self.Tend = 2.5
            self.hmin = 1e-4
            self.hmax = 1.
            self.rm = 5
            self.dt = 0.05
            self.ndump = 5
        elif self.mode == 'rossby-wave':
            self.Tstart = 20.
            self.Tend = 60.
            self.hmin = 5e-3
            self.hmax = 10.
            self.rm = 24
            self.dt = 0.05
            self.ndump = 12
            self.nonlinear = True
            self.g = 1.

        # Define FunctionSpaces
        self.degree1 = 2 if family == 'cg-cg' else 1
        self.degree2 = 2 if family == 'dg-cg' else 1
        self.space1 = "CG" if family == 'cg-cg' else "DG"
        self.space2 = "DG" if family == 'dg-dg' else "CG"

        # Gauge locations in latitude-longitude coordinates
        self.glatlon = {"P02": (38.5002, 142.5016), "P06": (38.6340, 142.5838), "801": (38.2, 141.7),
                        "802": (39.3, 142.1), "803": (38.9, 141.8), "804": (39.7, 142.2), "806": (37.0, 141.2)}

        # Plotting dictionaries
        self.labels = ("Fixed mesh", "Hessian based", "Explicit", "Implicit", "Adjoint based", "Goal based")
        self.styles = {self.labels[0]: 's', self.labels[1]: '^', self.labels[2]: 'x', self.labels[3]: 'o',
                       self.labels[4]: '*', self.labels[5]: 'h'}
        self.stamps = {self.labels[0]: 'fixedMesh', self.labels[1]: 'hessianBased', self.labels[2]: 'explicit',
                       self.labels[3]: 'implicit', self.labels[4]: 'DWF', self.labels[5]: 'DWR'}


    def J(self, mode):
        """
        :param mode: test problem choice.
        :return: 'exact' objective functional value, converged to 3 s.f.
        """
        dat = {'tohoku': 1.2185e+13,            # On mesh of 196,560 elements     TODO: Verify this
               'shallow-water': 1.1184e-3,      # On mesh of 524,288 elements
               }                                                                # TODO: rossby-wave test case
        if mode in dat.keys():
            return dat[mode]
        else:
            raise NotImplementedError


    def gaugeCoord(self, gauge):
        """
        :param gauge: Tide / pressure gauge name, from {P02, P06, 801, 802, 803, 804, 806}.
        :return: UTM coordinate for chosen gauge.
        """
        E, N, zn, zl = from_latlon(self.glatlon[gauge][0], self.glatlon[gauge][1], force_zone_number=54)
        return E, N


    def mixedSpace(self, mesh, orderChange=0):
        deg1 = self.degree1 + orderChange
        deg2 = self.degree2 + orderChange
        return VectorFunctionSpace(mesh, self.space1, deg1) * FunctionSpace(mesh, self.space2, deg2)


    def printToScreen(self, mn, outerTime, innerTime, nEle, Sn, mM, t, dt):
        """
        :arg mn: mesh number.
        :arg outerTime: time taken so far.
        :arg innerTime: time taken for this step.
        :arg nEle: current number of elements.
        :arg Sn: sum over #Elements.
        :arg mM: tuple of min and max #Elements.
        :arg t: current simuation time.
        :arg dt: current timestep.
        :returns: mean element count.
        """
        av = Sn / mn
        if self.printStats:
            print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Elapsed time : %4.2fs (This step : %4.2fs)     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s
Current timestep : %4.3fs\n""" %
                  (mn, 100 * t / self.Tend, outerTime, innerTime, nEle, av, mM[0], mM[1], dt))
        return av
