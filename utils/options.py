from firedrake import *

import numpy as np

from . import conversion
from . import misc

class Options:
    def __init__(self,
                 coarseness=4,
                 family='dg-cg',
                 vscale=0.85,
                 hmin=500.,
                 hmax=1e6,
                 a=100,
                 ntype='lp',
                 p=2,
                 mtype='s',         # Best approach for tsunami modelling
                 iso=False,
                 advect=False,
                 gradate=False,
                 window=False,
                 tAdapt=False,
                 bAdapt=False,
                 bootstrap=False,
                 outputOF=True,
                 printStats=True,
                 hessMeth='dL2',
                 beta=1.4,
                 gamma=1.,
                 outputMetric=False,
                 plotpvd=True,
                 gauges=False,
                 Tstart=300.,
                 Tend=1500.,
                 dt=1.,
                 ndump=15,
                 rm=30,
                 orderChange=0):
        """
        :param coarseness: mesh coarseness to use, where 1 is x-fine and 5 is x-coarse.
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param vscale: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param a: maximum tolerated aspect ratio.
        :param ntype: Normalisation approach: 'lp' or 'manual'.
        :param p: norm order in the Lp normalisation approach, where ``p => 1`` and ``p = infty`` is an option.
        :param mtype: Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
        :param iso: Toggle isotropic / anisotropic algorithm.
        :param advect: Toggle metric advection.
        :param gradate: Toggle metric gradation.
        :param window: generate error estimators over a time window of relevance.
        :param bAdapt: adapt initial mesh to bathymetry profile.
        :param tAdapt: implement adaptive timestepping.
        :param bootstrap: implement mesh bootstrapping to establish initial mesh.
        :param outputOF: print objective functional value to screen.
        :param printStats: print to screen during simulation.
        :param hessMeth: Method of Hessian reconstruction: 'dL2' or 'parts'.
        :param beta: metric gradation scaling parameter.
        :param gamma: metric rescaling parameter.
        :param outputMetric: toggle saving metric to PVD.
        :param plotpvd: toggle saving solution fields to PVD.
        :param gauges: toggle saving of elevation to HDF5 for timeseries analysis. 
        :param Tstart: Lower time range limit (s), before which we can assume the wave won't reach the shore.
        :param Tend: Simulation duration (s).
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh. (Should be an integer multiple of ndump.)
        :param orderChange: change in polynomial degree for residual approximation.
        """
        # Initial mesh parameters
        self.coarseness = coarseness
        try:
            assert coarseness in range(1, 9)
        except:
            raise ValueError('Please choose an integer between 0 and 8.')

        # Adaptivity parameters
        self.family = family
        try:
            assert family in ('dg-dg', 'dg-cg')
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
        self.advect = advect
        self.gradate = gradate
        self.window = window
        self.tAdapt = tAdapt
        self.bAdapt = bAdapt
        self.bootstrap = bootstrap
        self.outputOF = outputOF
        self.printStats = printStats
        self.outputMetric = outputMetric
        self.plotpvd = plotpvd
        self.gauges = gauges
        assert(type(advect) == type(gradate) == type(window) == type(iso) == type(tAdapt) == type(bAdapt)
               == type(bootstrap) == type(outputOF) == type(printStats) == type(outputMetric) == type(plotpvd)
               == type(gauges) == bool)
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
        self.g = 9.81           # Gravitational acceleration (m s^{-2})

        # Timestepping parameters
        self.Tstart = Tstart
        self.Tend = Tend
        self.dt = dt
        assert(type(Tstart) == type(Tend) == type(dt) == float)
        self.ndump = ndump
        self.rm = rm
        self.orderChange = orderChange
        assert(type(ndump) == type(rm) == type(orderChange) == int)
        self.timestepper = 'CrankNicolson'

        # Solver parameters
        self.params = {'mat_type': 'matfree',
                       'snes_type': 'ksponly',
                       'pc_type': 'python',
                       'pc_python_type': 'firedrake.AssembledPC',
                       'assembled_pc_type': 'lu',
                       'snes_lag_preconditioner': -1,
                       'snes_lag_preconditioner_persists': True}
        self.degree1 = 1
        self.degree2 = 1
        self.space1 = 'DG'
        if family == 'dg-cg':
            self.degree2 += 1
            self.space2 = 'CG'
        else:
            self.space2 = 'DG'

        # Gauge locations in latitude-longitude coordinates
        self.glatlon = {"P02": (38.5002, 142.5016), "P06": (38.6340, 142.5838), "801": (38.2, 141.7),
                        "802": (39.3, 142.1), "803": (38.9, 141.8), "804": (39.7, 142.2), "806": (37.0, 141.2)}

        # Plotting dictionaries
        labels = ('Coarse mesh', 'Medium mesh', 'Fine mesh', 'Hessian based', 'Explicit', 'Adjoint based', 'Goal based')
        self.labels = labels
        self.styles = {labels[0]: 's', labels[1]: '^', labels[2]: 'x', labels[3]: 'o', labels[4]: 'h', labels[5]: '*',
                       labels[6]: '+'}
        self.stamps = {labels[0]: 'fixedMesh', labels[1]: 'fixedMesh', labels[2]: 'fixedMesh', labels[3]: 'hessianBased',
                       labels[4]: 'explicit', labels[5]: 'adjointBased', labels[6]: 'goalBased'}

        # Mesh element counts currently generated in QMESH
        self.meshes = (6176, 8782, 11020, 16656, 20724, 33784, 52998, 81902, 129442, 196560, 450386, 691750)


    def J(self, mode):
        """
        :param mode: test problem choice.
        :return: 'exact' objective functional value, converged to 3 s.f.
        """
        dat = {'tohoku': 2.4391e+13, 'shallow-water': 0.00224}  # TODO: advection-diffusion and rossby-wave
        if mode in dat.keys():
            return dat[mode]
        else:
            raise NotImplementedError


    def gaugeCoord(self, gauge):
        """
        :param gauge: Tide / pressure gauge name, from {P02, P06, 801, 802, 803, 804, 806}.
        :return: UTM coordinate for chosen gauge.
        """
        E, N, zn, zl = conversion.from_latlon(self.glatlon[gauge][0], self.glatlon[gauge][1], force_zone_number=54)
        return E, N

    def checkCFL(self, b):
        """
        :param b: bathymetry profile considered.
        """
        cdt = self.hmin / np.sqrt(self.g * max(b.dat.data))
        if self.dt > cdt:
            print('WARNING: chosen timestep dt = %.4fs exceeds recommended value of %.4fs' % (self.dt, cdt))
            if input('Hit enter if happy to proceed.'):
                exit(23)

    def printToScreen(self, mn, outerTime, innerTime, nEle, Sn, N, t, dt):
        """
        :arg mn: mesh number.
        :arg outerTime: time taken so far.
        :arg innerTime: time taken for this step.
        :arg nEle: current number of elements.
        :arg Sn: sum over #Elements.
        :arg N: tuple of min and max #Elements.
        :arg t: current simuation time.
        :arg dt: current timestep.
        :returns: mean element count.
        """
        av = Sn / (mn+1)
        if self.printStats:
            print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Elapsed time : %4.2fs (This step : %4.2fs)     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s
Current timestep : %4.3fs\n""" %
                  (mn, 100 * t / self.Tend, outerTime, innerTime, nEle, av, N[0], N[1], dt))
        return av
