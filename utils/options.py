from thetis import *


__all__ = ["Options"]


class Options:
    def __init__(self,
                 mode='tohoku',
                 family='dg-dg',
                 rescaling=0.85,
                 hmin=500.,
                 hmax=1e6,
                 maxAnisotropy=100,
                 normalisation='lp',
                 normOrder=2,
                 adaptField='s',                 # Best approach for tsunami modelling
                 iso=False,
                 gradate=False,
                 nonlinear=False,
                 rotational=False,
                 printStats=True,
                 hessMeth='dL2',
                 maxGrowth=1.4,
                 g=9.81,
                 plotpvd=True,
                 Tstart=300.,
                 Tend=1500.,
                 dt=0.5,
                 ndump=50,
                 rm=50,
                 nAdapt=1,
                 nVerT=1000,
                 orderChange=0,
                 refinedSpace=False,
                 timestepper='CrankNicolson',
                 wd=False):
        """
        :param mode: problem considered.
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param rescaling: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param maxAnisotropy: maximum tolerated aspect ratio.
        :param normalisation: Normalisation approach: 'lp' or 'manual'.
        :param normOrder: norm order in the Lp normalisation approach, where ``p => 1`` and ``p = infty`` is an option.
        :param adaptField: Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
        :param iso: Toggle isotropic / anisotropic algorithm.
        :param gradate: Toggle metric gradation.
        :param nonlinear: Toggle nonlinear / linear equations.
        :param rotational: Toggle rotational / non-rotational equations.
        :param printStats: print to screen during simulation.
        :param hessMeth: Method of Hessian reconstruction: 'dL2' or 'parts'.
        :param maxGrowth: metric gradation scaling parameter.
        :param g: gravitational acceleration.
        :param plotpvd: toggle saving solution fields to .pvd.
        :param Tstart: Lower time range limit (s), before which we can assume the wave won't reach the shore.
        :param Tend: Simulation duration (s).
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh. (Should be an integer multiple of ndump.)
        :param nAdapt: number of mesh adaptions per mesh regeneration.
        :param nVerT: target number of vertices.
        :param orderChange: change in polynomial degree for residual approximation.
        :param refinedSpace: refine space too compute errors and residuals.
        :param timestepper: timestepping scheme.
        :param wd: toggle wetting and drying.
        """
        try:
            assert mode in ('tohoku', 'shallow-water', 'rossby-wave')
            self.mode = mode
        except:
            raise ValueError('Test problem not recognised.')
        try:
            assert family in ('dg-dg', 'dg-cg', 'cg-cg')
            self.family = family
        except:
            raise ValueError('Mixed function space not recognised.')
        try:
            assert rescaling > 0
            self.rescaling = rescaling
        except:
            raise ValueError('Invalid value for scaling parameter. rescaling > 0 is required.')
        try:
            assert (hmin > 0) & (hmax > hmin)
            self.hmin = hmin
            self.hmax = hmax
        except:
            raise ValueError('Invalid min/max element sizes. hmax > hmin > 0 is required.')
        try:
            assert maxAnisotropy > 0
            self.maxAnisotropy = maxAnisotropy
        except:
            raise ValueError('Invalid anisotropy value. a > 0 is required.')
        try:
            assert normalisation in ('lp', 'manual')
            self.normalisation = normalisation
        except:
            raise ValueError('Normalisation approach ``%s`` not recognised.' % normalisation)
        if normalisation == 'manual':
            self.targetError = 1e-3
        try:
            assert normOrder > 0
            self.normOrder = normOrder
        except:
            raise ValueError('Invalid value for p. p > 0 is required.')
        try:
            assert adaptField in ('f', 's', 'b')
            self.adaptField = adaptField
        except:
            raise ValueError('Field for adaption ``%s`` not recognised.' % adaptField)
        for i in (gradate, nonlinear, rotational, iso, printStats, plotpvd, wd, refinedSpace):
            assert(isinstance(i, bool))
        self.iso = iso
        self.gradate = gradate
        self.nonlinear = nonlinear
        self.rotational = rotational
        self.printStats = printStats
        self.plotpvd = plotpvd
        self.wd = wd
        try:
            assert hessMeth in ('dL2', 'parts')
            self.hessMeth = hessMeth
        except:
            raise ValueError('Hessian reconstruction method ``%s`` not recognised.' % hessMeth)
        try:
            assert (maxGrowth > 1)
            self.maxGrowth = maxGrowth
        except:
            raise ValueError('Invalid value for growth parameter.')

        # Physical parameters
        self.Omega = 7.291e-5   # Planetary rotation rate
        try:
            assert g > 0
            self.g = g          # Gravitational acceleration
        except:
            raise ValueError('Unphysical physical parameters!')

        # Timestepping parameters
        for i in (Tstart, Tend, dt):
            assert isinstance(i, float)
        self.Tstart = Tstart
        self.Tend = Tend
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
        try:
            assert timestepper in ('CrankNicolson', 'ImplicitEuler', 'ExplicitEuler')
            self.timestepper = timestepper
        except:
            raise NotImplementedError

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

        self.meshSizes = (5916, 7062, 8666, 10980, 14166, 19094, 27262, 41712, 72612, 150590, 681666)


    def J(self, mode):
        """
        :param mode: test problem choice.
        :return: 'exact' objective functional value, converged to 3 s.f.
        """
        dat = {'tohoku': 1.2185e+13,            # On mesh of 196,560 elements     TODO: Verify this by modelVerification
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
        from .conversion import from_latlon

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
