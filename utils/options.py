class Options:
    def __init__(self,
                 vscale = 0.85,
                 hmin = 500,
                 hmax = 1e7,
                 ntype = 'lp',
                 mtype = 'b',
                 T = 1500,
                 dt = 1.,
                 ndump = 15,
                 rm = 30):
        """
        :param vscale: Scaling parameter for target number of vertices.
        :param hmin: Minimal tolerated element size (m).
        :param hmax: Maximal tolerated element size (m).
        :param ntype: Normalisation approach: 'lp' or 'manual'.
        :param mtype: Adapt w.r.t 's'peed, 'f'ree surface or 'b'oth.
        :param T: Simulation duration (s).
        :param dt: Timestep (s).
        :param ndump: Timesteps per data dump.
        :param rm: Timesteps per remesh.
        """

        # Adaptivity parameters:
        self.vscale = vscale
        self.hmin = hmin
        self.hmax = hmax
        self.ntype = ntype
        self.mtype = mtype

        # Solver parameters:
        self.T = T
        self.dt = dt
        self.ndump = ndump
        self.rm = rm