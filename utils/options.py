from firedrake import *

import numpy as np

from . import conversion
from . import storage


class Options:
    def __init__(self,
                 family='dg-cg',
                 vscale=0.85,
                 hmin=500.,
                 hmax=1e7,
                 a=100,
                 ntype='lp',
                 p=2,
                 mtype='f',
                 iso=False,
                 hessMeth='dL2',
                 beta=1.4,
                 T=1500.,
                 dt=1.,
                 ndump=15,
                 rm=30,
                 Ts=300.):
        """
        :param family: mixed function space family, from {'dg-dg', 'dg-cg'}.
        :param vscale: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param a: maximum tolerated aspect ratio.
        :param ntype: Normalisation approach: 'lp' or 'manual'.
        :param p: norm order in the Lp normalisation approach, where ``p => 1`` and ``p = infty`` is an option.
        :param mtype: Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
        :param iso: Toggle isotropic / anisotropic algorithm.
        :param hessMeth: Method of Hessian reconstruction: 'dL2' or 'parts'.
        :param beta: Metric gradation scaling parameter.
        :param T: Simulation duration (s).
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh. (Should be an integer multiple of ndump.)
        :param Ts: Lower time range limit (s), before which we can assume the wave won't reach the shore.
        """

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
        assert type(iso) == bool
        self.hessMeth = hessMeth
        try:
            assert hessMeth in ('dL2', 'parts')
        except:
            raise ValueError('Hessian reconstruction method ``%s`` not recognised.' % hessMeth)
        self.beta = beta
        try:
            assert beta > 1
        except:
            raise ValueError('Invalid value for scaling parameter. beta > 1 is required.')

        # Physical parameters
        self.g = 9.81           # Gravitational acceleration (m s^{-2})

        # Timestepping parameters
        self.T = T
        self.dt = dt
        self.ndump = ndump
        self.rm = rm
        self.Ts = Ts
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
        self.glatlon = {'P02': (38.5002, 142.5016), 'P06': (38.6340, 142.5838), '801': (38.2, 141.7),
                        '802': (39.3, 142.1), '803': (38.9, 141.8), '804': (39.7, 142.2), '806': (37.0, 141.2)}

        # Plotting dictionaries
        self.plotDir = {1: 'fixedMesh/coarse', 2: 'fixedMesh/medium', 3: 'fixedMesh/fine',
                        4: 'simpleAdapt', 5: 'adjointBased'}
        self.labels = {1: 'Coarse mesh', 2: 'Medium mesh', 3: 'Fine mesh', 4: 'Simple adaptive', 5: 'Adjoint based'}
        self.styles = {1: 's', 2: '^', 3: 'x', 4: 'o', 5: '*'}

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

    def loadFromDisk(self, mesh, mn, dirName, elev0):
        """
        
        :param mesh: mesh on which data is stored.
        :param mn: mesh number.
        :param dirName: name of directory for storage.
        :param elev0: initial free surface.
        :return: saved free surface elevation and fluid velocity, along with mesh index.
        """
        # Enforce initial conditions on discontinuous space / load variables from disk
        W = VectorFunctionSpace(mesh, self.space1, self.degree1) * FunctionSpace(mesh, self.space2, self.degree2)
        uv_2d = Function(W.sub(0))
        elev_2d = Function(W.sub(1))
        index = mn * int(self.rm / self.ndump)
        indexStr = storage.indexString(index)
        if mn == 0:
            elev_2d.interpolate(elev0)
            uv_2d.interpolate(Expression((0, 0)))
        else:
            with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as el:
                el.load(elev_2d, name='elev_2d')
                el.close()
            with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as ve:
                ve.load(uv_2d, name='uv_2d')
                ve.close()
        return elev_2d, uv_2d, index

    def printToScreen(self, mn, outerTime, innerTime, nEle, Sn, N):
        """
        :param mn: mesh number.
        :param outerTime: time taken so far.
        :param innerTime: time taken for this step.
        :param nEle: current number of elements.
        :param Sn: sum over #Elements.
        :param N: tuple of min and max #Elements.
        """
        print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Elapsed time : %4.2fs (This step : %4.2fs)     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s\n""" %
              (mn, (100 * mn * self.rm * self.dt) / self.T, outerTime, innerTime, nEle, Sn / mn, N[0], N[1]))
