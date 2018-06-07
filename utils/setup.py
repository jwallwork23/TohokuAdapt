import numpy as np


__all__ = ["MeshSetup", "problemDomain", "RossbyWaveSolution", "__main__"]


class MeshSetup:
    def __init__(self, level, wd=False):

        # Get mesh descriptions
        self.level = level
        self.dirName = 'resources/meshes/'
        try:
            assert(isinstance(level, int) and (level >= 0) and (level < 11))
        except:
            raise ValueError('Invalid input. Refinement level should be an integer from 0-10.')
        self.meshName = 'Tohoku' + str(level)
        if wd:
            self.meshName = 'wd_' + self.meshName

        # Define gradations (in metres)
        self.innerGradation1 = np.linspace(7500., 1000., 11)[level]
        self.outerGradation1 = np.linspace(25000., 2000., 11)[level]
        self.innerGradation2 = np.linspace(10000., 2000., 11)[level]
        self.outerGradation2 = np.linspace(25000., 4000., 11)[level]

        # Define gradation distances (in degrees)
        self.gradationDistance1 = np.linspace(0.5, 1., 11)[level]
        self.gradationDistance2 = np.linspace(0.5, 1., 11)[level]

    def generateMesh(self, wd=False):
        """
        Generate mesh using QMESH. This script is based on work by Alexandros Advis et al, 2017.
        """

        # Reading in the shapefile describing the domain boundaries, and creating a GMSH file.
        bdyLoc = 'resources/boundaries/'
        boundaries = qmesh.vector.Shapes()
        if wd:
            if self.level > 5:
                bdyfile = bdyLoc+'wd_final_bdys.shp'
                coastfile = bdyLoc+'grad_box.shp'
            else:
                bdyfile = bdyLoc+'coarse_setup.shp'
                coastfile = bdyLoc+'box_coarse.shp'
        else:
            bdyfile = bdyLoc+'final_bdys.shp'
            coastfile = bdyLoc+'coastline.shp'
        boundaries.fromFile(bdyfile)
        loopShapes = qmesh.vector.identifyLoops(boundaries, isGlobal=False, defaultPhysID=1000, fixOpenLoops=True)
        polygonShapes = qmesh.vector.identifyPolygons(loopShapes, meshedAreaPhysID=1,
                                                      smallestNotMeshedArea=5e6, smallestMeshedArea=2e8)
        polygonShapes.writeFile(bdyLoc+'polygons.shp')

        if wd:
            boundaries2 = qmesh.vector.Shapes()
            boundaries2.fromFile(bdyLoc+'poly_bdys.shp')
            loopShapes2 = qmesh.vector.identifyLoops(boundaries2, isGlobal=False, defaultPhysID=1000, fixOpenLoops=True)
            polygonShapes2 = qmesh.vector.identifyPolygons(loopShapes2, meshedAreaPhysID=1,
                                                          smallestNotMeshedArea=5e6, smallestMeshedArea=2e8)
            polygonShapes2.writeFile(bdyLoc + 'coast_poly.shp')

        # Create raster for mesh gradation towards coastal region of importance
        fukushimaCoast = qmesh.vector.Shapes()
        fukushimaCoast.fromFile(bdyLoc + 'fukushima.shp')
        gradationRaster_fukushimaCoast = qmesh.raster.gradationToShapes()
        gradationRaster_fukushimaCoast.setShapes(fukushimaCoast)
        gradationRaster_fukushimaCoast.setRasterBounds(135., 149., 30., 45.)
        gradationRaster_fukushimaCoast.setRasterResolution(300, 300)
        gradationRaster_fukushimaCoast.setGradationParameters(self.innerGradation1, self.outerGradation1,
                                                              self.gradationDistance1, 0.05)
        gradationRaster_fukushimaCoast.calculateLinearGradation()
        gradationRaster_fukushimaCoast.writeNetCDF(self.dirName+'gradationFukushima.nc')

        # Create raster for mesh gradation towards rest of coast (Could be a polygon, line or point)
        gebcoCoastlines = qmesh.vector.Shapes()
        gebcoCoastlines.fromFile(coastfile)
        gradationRaster_gebcoCoastlines = qmesh.raster.gradationToShapes()
        gradationRaster_gebcoCoastlines.setShapes(gebcoCoastlines)
        gradationRaster_gebcoCoastlines.setRasterBounds(135., 149., 30., 45.)
        gradationRaster_gebcoCoastlines.setRasterResolution(300, 300)
        gradationRaster_gebcoCoastlines.setGradationParameters(self.innerGradation2, self.outerGradation2,
                                                               self.gradationDistance2)
        gradationRaster_gebcoCoastlines.calculateLinearGradation()
        gradationRaster_gebcoCoastlines.writeNetCDF(self.dirName+'gradationCoastlines.nc')

        # Create overall mesh metric
        meshMetricRaster = qmesh.raster.meshMetricTools.minimumRaster([gradationRaster_fukushimaCoast,
                                                                       gradationRaster_gebcoCoastlines])
        meshMetricRaster.writeNetCDF(self.dirName+'meshMetricRaster.nc')

        # Create domain object and write GMSH files
        domain = qmesh.mesh.Domain()
        domain.setTargetCoordRefSystem('EPSG:32654', fldFillValue=1000.)
        domain.setGeometry(loopShapes, polygonShapes)
        domain.setMeshMetricField(meshMetricRaster)

        # Meshing
        domain.gmsh(geoFilename=self.dirName + self.meshName + '.geo',
                    fldFilename=self.dirName + self.meshName + '.fld',
                    mshFilename=self.dirName + self.meshName + '.msh')
        # NOTE: default meshing algorithm is Delaunay. To use a frontal approach, include "gmshAlgo='front2d' "

    def convertMesh(self):
        """
        Convert mesh coordinates using QMESH. This script is based on work by Alexandros Advis et al, 2017.
        """
        TohokuMesh = qmesh.mesh.Mesh()
        TohokuMesh.readGmsh(self.dirName + self.meshName + '.msh', 'EPSG:3857')
        TohokuMesh.writeShapefile(self.dirName + self.meshName + '.shp')


if __name__ == "__main__":
    import qmesh
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Generate all meshes in the hierarchy")
    parser.add_argument("-m", help="Generate a particular mesh, using index in range 0:10")
    parser.add_argument("-w", help="Use wetting and drying")
    args = parser.parse_args()
    if args.a is None:
        assert args.m is not None
    if args.m is None:
        assert args.a is not None

    generateAll = True if args.a else False
    if generateAll:
        for i in range(11):
            ms = MeshSetup(i, wd=args.w)
            qmesh.setLogOutputFile(ms.dirName+'generateMesh.log')   # Store QMESH log for later reference
            qmesh.initialise()                                      # Initialise QGIS API
            ms.generateMesh(wd=args.w)                              # Generate the mesh
            ms.convertMesh()                                        # Convert to shapefile, for visualisation with QGIS
    else:
        ms = MeshSetup(int(args.m), wd=args.w)
        qmesh.setLogOutputFile(ms.dirName+'generateMesh.log')
        qmesh.initialise()
        ms.generateMesh(wd=args.w)
        ms.convertMesh()
    exit(0)         # Unix tradition is that exit code 0 is success and anything else is a failure
else:
    from thetis import *
    from thetis_adjoint import *
    from firedrake.petsc import PETSc

    import scipy.interpolate as si
    from scipy.io.netcdf import NetCDFFile
    from time import clock

    from .conversion import earth_radius, get_latitude, vectorlonlat_to_utm
    from .interpolation import interp
    from .misc import indicator
    from .options import Options


def problemDomain(level=0, mesh=None, b=None, hierarchy=False, op=Options(mode='tohoku')):
    """
    Set up problem domain.
    
    :arg level: refinement level, where 0 is coarsest.
    :param mesh: user specified mesh, if already generated.
    :param b: user specified bathymetry, if already generated.
    :param hierarchy: extract 5 level MeshHierarchy.
    :param op: options parameter object.
    :return: associated mesh, initial conditions, bathymetry field, boundary conditions and Coriolis parameter. 
    """
    newmesh = mesh == None
    if op.mode == 'tohoku':
        getBathy = b is None
        if mesh is None:
            # ms = MeshSetup(level, op.wd)
            ms = MeshSetup(level, False)
            mesh = Mesh(ms.dirName + ms.meshName + '.msh')
        if hierarchy:
            mh = MeshHierarchy(mesh, 5)
        meshCoords = mesh.coordinates.dat.data
        P1 = FunctionSpace(mesh, 'CG', 1)
        eta0 = Function(P1, name='Initial free surface displacement')
        u0 = Function(VectorFunctionSpace(mesh, "CG", 1))
        if getBathy:
            b = Function(P1, name='Bathymetry profile')

        # Read and interpolate initial surface data (courtesy of Saito)
        nc1 = NetCDFFile('resources/initialisation/surf_zeroed.nc', mmap=False)
        lon1 = nc1.variables['lon'][:]
        lat1 = nc1.variables['lat'][:]
        x1, y1 = vectorlonlat_to_utm(lat1, lon1, force_zone_number=54)      # Our mesh mainly resides in UTM zone 54
        elev1 = nc1.variables['z'][:, :]
        interpolatorSurf = si.RectBivariateSpline(y1, x1, elev1)
        eta0vec = eta0.dat.data
        assert meshCoords.shape[0] == eta0vec.shape[0]

        # Read and interpolate bathymetry data (courtesy of GEBCO)
        nc2 = NetCDFFile('resources/bathymetry/tohoku.nc', mmap=False)
        lon2 = nc2.variables['lon'][:]
        lat2 = nc2.variables['lat'][:-1]
        x2, y2 = vectorlonlat_to_utm(lat2, lon2, force_zone_number=54)
        elev2 = nc2.variables['elevation'][:-1, :]
        if getBathy:
            interpolatorBath = si.RectBivariateSpline(y2, x2, elev2)
        b_vec = b.dat.data
        try:
            assert meshCoords.shape[0] == b_vec.shape[0]
        except:
            b = interp(mesh, b)

        # Interpolate data onto initial surface and bathymetry profiles
        for i, p in enumerate(meshCoords):
            eta0vec[i] = interpolatorSurf(p[1], p[0])
            if getBathy:
                depth = - eta0vec[i] - interpolatorBath(p[1], p[0])
                # b_vec[i] = depth if op.wd else max(depth, 30)
                b_vec[i] = max(depth, 30)   # Post-process the bathymetry to have a minimum depth of 30m
        BCs = {}

        # Establish Coriolis parameter
        f = Function(P1)
        if op.coriolis == 'sin':
            for i, v in zip(range(len(mesh.coordinates.dat.data)), mesh.coordinates.dat.data):
                f.dat.data[i] = 2 * op.Omega * np.sin(np.radians(get_latitude(v[0], v[1], 54, northern=True)))
        elif op.coriolis in ('beta', 'f'):
            f0 = 2 * op.Omega * np.sin(np.radians(op.latFukushima))
            if op.coriolis == 'f':
                f.assign(f0)
            else:
                beta = 2 * op.Omega * np.cos(np.radians(op.latFukushima)) / earth_radius(op.latFukushima)
                for i, v in zip(range(len(mesh.coordinates.dat.data)), mesh.coordinates.dat.data):
                    f.dat.data[i] = f0 + beta * v[1]
        nu = Function(P1).assign(1e-3)

    elif op.mode == 'shallow-water':
        n = pow(2, level)
        lx = 2 * pi
        if mesh is None:
            mesh = SquareMesh(n, n, lx, lx)
        P1 = FunctionSpace(mesh, "CG", 1)
        x, y = SpatialCoordinate(mesh)
        eta0 = Function(P1).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        u0 = Function(VectorFunctionSpace(mesh, "CG", 1))
        b = Function(P1).assign(0.1)
        BCs = {}
        f = Function(P1)
        nu = None
    elif op.mode == 'rossby-wave':
        n = pow(2, level-1)
        lx = 48
        ly = 24
        if mesh is None:
            mesh = RectangleMesh(lx * n, ly * n, lx, ly)
            xy = Function(mesh.coordinates)
            xy.dat.data[:, :] -= [lx / 2, ly / 2]
            mesh.coordinates.assign(xy)
        P1 = FunctionSpace(mesh, "CG", 1)
        b = Function(P1).assign(1.)
        q = RossbyWaveSolution(op.mixedSpace(mesh), order=1, op=op).__call__()
        u0, eta0 = q.split()
        BCs = {1: {'uv': Constant(0.)}, 2: {'uv': Constant(0.)}, 3: {'uv': Constant(0.)}, 4: {'uv': Constant(0.)}}
        f = Function(P1).interpolate(SpatialCoordinate(mesh)[1])
        nu = None
    elif op.mode == 'advection-diffusion':
        if mesh is None:
            mesh = RectangleMesh(4 * level, level, 4, 1)
        x, y = SpatialCoordinate(mesh)
        P1 = FunctionSpace(mesh, "CG", 1)
        phi0 = Function(P1).interpolate(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04))
        BCs = DirichletBC(P1, 0, 'on_boundary')
        w = Function(VectorFunctionSpace(mesh, "CG", 1), name='Wind field').interpolate(Expression([1, 0]))

    if newmesh:
        PETSc.Sys.Print("Setting up mesh across %d processes" % COMM_WORLD.size)
        PETSc.Sys.Print("  rank %d owns %d elements and can access %d vertices" \
                        % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()), comm=COMM_SELF)

    if hierarchy:
        return mesh, u0, eta0, b, BCs, f, nu, mh
    elif op.mode == 'advection-diffusion':
        return mesh, phi0, BCs, w
    else:
        return mesh, u0, eta0, b, BCs, f, nu


class RossbyWaveSolution:
    """
    Class for constructing the analytic solution of the Rossby wave test case on a given FunctionSpace.
    
    Hermite polynomials taken from the Matlab code found at https://marine.rutgers.edu/po/tests/soliton/hermite.txt
    """
    def __init__(self, function_space, order=1, op=Options(mode='rossby-wave')):
        """
        :arg function_space: mixed FunctionSpace in which to construct the Hermite polynomials.
        """
        try:
            assert order in (0, 1)
            self.order = order
            self.function_space = function_space
            self.soliton_amplitude = 0.395
            x, y = SpatialCoordinate(self.function_space.mesh())
            self.x = x
            self.y = y
        except:
            raise NotImplementedError("Only zeroth and first order analytic solutions considered for this problem.")
        try:
            assert op.mode == 'rossby-wave'
            self.op = op
        except:
            raise ValueError("Analyic solution only available for 'rossby-wave' test case.")

    def coeffs(self):
        """
        Initialise Hermite coefficients.
        """
        u = np.zeros(28)
        v = np.zeros(28)
        eta = np.zeros(28)

        #  Hermite series coefficients for u:
        u[0] = 1.789276
        u[2] = 0.1164146
        u[4] = -0.3266961e-3
        u[6] = -0.1274022e-2
        u[8] = 0.4762876e-4
        u[10] = -0.1120652e-5
        u[12] = 0.1996333e-7
        u[14] = -0.2891698e-9
        u[16] = 0.3543594e-11
        u[18] = -0.3770130e-13
        u[20] = 0.3547600e-15
        u[22] = -0.2994113e-17
        u[24] = 0.2291658e-19
        u[26] = -0.1178252e-21

        #  Hermite series coefficients for v:
        v[3] = -0.6697824e-1
        v[5] = -0.2266569e-2
        v[7] = 0.9228703e-4
        v[9] = -0.1954691e-5
        v[11] = 0.2925271e-7
        v[13] = -0.3332983e-9
        v[15] = 0.2916586e-11
        v[17] = -0.1824357e-13
        v[19] = 0.4920951e-16
        v[21] = 0.6302640e-18
        v[23] = -0.1289167e-19
        v[25] = 0.1471189e-21

        #  Hermite series coefficients for eta:
        eta[0] = -3.071430
        eta[2] = -0.3508384e-1
        eta[4] = -0.1861060e-1
        eta[6] = -0.2496364e-3
        eta[8] = 0.1639537e-4
        eta[10] = -0.4410177e-6
        eta[12] = 0.8354759e-9
        eta[14] = -0.1254222e-9
        eta[16] = 0.1573519e-11
        eta[18] = -0.1702300e-13
        eta[20] = 0.1621976e-15
        eta[22] = -0.1382304e-17
        eta[24] = 0.1066277e-19
        eta[26] = -0.1178252e-21

        return {'u': u, 'v': v, 'eta': eta}

    def polynomials(self):
        """
        Get Hermite polynomials
        """
        polys = [Constant(1.), 2 * self.y]
        for i in range(2, 28):
            polys.append(2 * self.y * polys[i - 1] - 2 * (i - 1) * polys[i - 2])

        return polys

    def xi(self, t=0.):
        """
        :arg t: current time.
        :return: time shifted x-coordinate.
        """
        c = -1./3.
        if self.order == 1:
            c -= 0.395 * self.soliton_amplitude * self.soliton_amplitude
        return self.x - c * t

    def phi(self, t=0.):
        """
        :arg t: current time.
        :return: sech^2 term.
        """
        B = self.soliton_amplitude
        A = 0.771 * B * B

        return A * (1 / (cosh(B * self.xi(t)) ** 2))

    def dphidx(self, t=0.):
        """
        :arg t: current time. 
        :return: tanh * phi term.
        """
        B = self.soliton_amplitude
        return -2 * B * self.phi(t) * tanh(B * self.xi(t))

    def psi(self):
        """
        :arg t: current time. 
        :return: exp term.
        """
        return exp(-0.5 * self.y * self.y)

    def zerothOrderTerms(self, t=0.):
        """
        :arg t: current time.
        :return: zeroth order analytic solution for test problem of Huang.
        """
        return {'u' : self.phi(t) * 0.25 * (-9 + 6 *  self.y * self.y) * self.psi(),
                'v': 2 * self.y * self.dphidx(t) * self.psi(),
                'eta': self.phi(t) * 0.25 * (3 + 6 * self.y * self.y) * self.psi()}

    def firstOrderTerms(self, t=0.):
        """
        :arg t: current time.
        :return: first order analytic solution for test problem of Huang.
        """
        C = - 0.395 * self.soliton_amplitude * self.soliton_amplitude
        phi = self.phi(t)
        coeffs = self.coeffs()
        polys = self.polynomials()
        terms = self.zerothOrderTerms(t)
        terms['u'] += C * phi * 0.5625 * (3 + 2 * self.y * self.y) * self.psi()     # NOTE: This last psi is not included
        terms['u'] += phi * phi * self.psi() * sum(coeffs['u'][i] * polys[i] for i in range(28))
        terms['v'] += self.dphidx(t) * phi * self.psi() * sum(coeffs['v'][i] * polys[i] for i in range(28))
        terms['eta'] += C * phi * 0.5625 * (-5 + 2 * self.y * self.y) * self.psi()
        terms['eta'] += phi * phi * self.psi() * sum(coeffs['eta'][i] * polys[i] for i in range(28))

        return terms

    def plot(self):
        """
        Plot initial condition and final state.
        """
        outFile = File("plots/rossby-wave/analytic/analytic.pvd")
        for t in (0., self.op.Tend):
            q = self.__call__(t)
            u, eta = q.split()
            u.rename("Depth averaged velocity")
            eta.rename("Elevation")
            print("t = %.4f, |u| = %.4f, |eta| = %.4f" % (t,u.dat.norm, eta.dat.norm))
            outFile.write(u, eta, time=t)

    def integrate(self, mirror=False):
        """
        :return: list containing time integrand values at each timestep.
        """
        t = 0.
        cnt = 0
        vals = []

        # Set up spatial and temporal kernel functions
        mesh = self.function_space.mesh()
        ks = Function(VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1))
        k0, k1 = ks.split()
        k1.assign(indicator(mesh, mirror=mirror, radii=self.op.radius, op=self.op))
        kt = Constant(0.)

        # Time integrate
        tic = clock()
        while t < self.op.Tend + 0.5 * self.op.dt:
            q = self.__call__(t)
            if t > self.op.Tstart - 0.5 * self.op.dt:  # Slightly smoothed transition
                kt.assign(1. if t > self.op.Tstart + 0.5 * self.op.dt else 0.5)
            vals.append(assemble(kt * inner(ks, q) * dx))
            if cnt % self.op.ndump == 0:
                tic = clock() - tic
                print("t = %.2fs, CPU time: %.2fs" % (t, tic))
                tic = clock()
            t += self.op.dt
            cnt += 1
        return vals

    def __call__(self, t=0.):
        """
        :arg t: current time.
        :return: semi-analytic solution for Rossby wave test case of Huang.
        """
        terms = self.zerothOrderTerms(t) if self.order == 0 else self.firstOrderTerms(t)

        q = Function(self.function_space)
        u, eta = q.split()
        u.rename("Depth averaged velocity")
        eta.rename("Elevation")

        W = FunctionSpace(self.function_space.mesh(), self.function_space.sub(0).ufl_element().family(),
                          self.function_space.sub(0).ufl_element().degree())
        u0 = Function(W).interpolate(terms['u'])
        u1 = Function(W).interpolate(terms['v'])
        u.dat.data[:, 0] = u0.dat.data  # TODO: This is perhaps not such a good idea in parallel
        u.dat.data[:, 1] = u1.dat.data
        eta.interpolate(terms['eta'])

        return q
