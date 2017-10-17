class Options:
    def __init__(self,
                 vscale = 0.85,
                 hmin = 500.,
                 hmax = 1e7,
                 ntype = 'lp',
                 mtype = 'b',
                 matOut = False,
                 iso = False,
                 hessMeth = 'dL2',
                 T = 1500.,
                 dt = 1.,
                 ndump = 15,
                 rm = 30,
                 g = 9.81):
        """
        :param vscale: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param ntype: Normalisation approach: 'lp' or 'manual'.
        :param mtype: Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
        :param matOut: Toggle outputting of Hessian and metric.
        :param iso: Toggle isotropic / anisotropic algorithm.
        :param hessMeth: Method of Hessian reconstruction: 'dL2' or 'parts'.
        :param T: Simulation duration (s).
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh.
        :param g: Gravitational acceleration (m s^{-2}).
        """

        # Adaptivity parameters:
        self.vscale = vscale
        self.hmin = hmin
        self.hmax = hmax
        self.ntype = ntype
        self.mtype = mtype
        self.matOut = matOut
        self.iso = iso
        self.hessMeth = hessMeth

        # Solver parameters:
        self.T = T
        self.dt = dt
        self.ndump = ndump
        self.rm = rm

        # Physical parameters:
        self.g = g
