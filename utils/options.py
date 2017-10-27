from . import conversion


class Options:
    def __init__(self,
                 vscale=0.85,
                 hmin=500.,
                 hmax=1e7,
                 ntype='lp',
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
        :param vscale: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param ntype: Normalisation approach: 'lp' or 'manual'.
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
        self.vscale = vscale
        self.hmin = hmin
        self.hmax = hmax
        self.ntype = ntype
        self.mtype = mtype
        self.iso = iso
        self.hessMeth = hessMeth
        self.beta = beta

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

        # Gauge locations in latitude-longitude coordinates
        self.glatlon = {'P02': (38.5002, 142.5016), 'P06': (38.6340, 142.5838), '801': (38.2, 141.7),
                        '802': (39.3, 142.1), '803': (38.9, 141.8), '804': (39.7, 142.2), '806': (37.0, 141.2)}

    def gaugeCoord(self, gauge):
        """
        :param gauge: Tide / pressure gauge name, from {P02, P06, 801, 802, 803, 804, 806}.
        :return: UTM coordinate for chosen gauge.
        """
        E, N, zn, zl = conversion.from_latlon(self.glatlon[gauge][0], self.glatlon[gauge][1], force_zone_number=54)
        return E, N