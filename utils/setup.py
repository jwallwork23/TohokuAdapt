import numpy as np


__all__ = ["MeshSetup", "problemDomain", "HermiteCoefficients", "solutionRW", "integrateRW", "__main__"]


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
            ms.generateMesh(wd=args.w)                             # Generate the mesh
            ms.convertMesh()                                        # Convert to shapefile, for visualisation with QGIS
    else:
        ms = MeshSetup(args.m, wd=args.w)
        qmesh.setLogOutputFile(ms.dirName+'generateMesh.log')
        qmesh.initialise()
        ms.generateMesh(wd=args.w)
        ms.convertMesh()

    exit(23)
else:
    from thetis import *
    from thetis_adjoint import *

    import scipy.interpolate as si
    from scipy.io.netcdf import NetCDFFile

    from .conversion import earth_radius, get_latitude, vectorlonlat_to_utm
    from .misc import indicator
    from .options import Options


def problemDomain(level=0, mesh=None, op=Options(mode='tohoku')):
    """
    Set up problem domain.
    
    :arg level: refinement level, where 0 is coarsest.
    :param mesh: user specified mesh, if already generated.
    :param op: options parameter object.
    :return: associated mesh, initial conditions, bathymetry field, boundary conditions and Coriolis parameter. 
    """
    if op.mode == 'tohoku':
        if mesh == None:
            # ms = MeshSetup(level, op.wd)
            ms = MeshSetup(level, False)
            mesh = Mesh(ms.dirName + ms.meshName + '.msh')
        meshCoords = mesh.coordinates.dat.data
        P1 = FunctionSpace(mesh, 'CG', 1)
        eta0 = Function(P1, name='Initial free surface displacement')
        u0 = Function(VectorFunctionSpace(mesh, "CG", 1))
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
        interpolatorBath = si.RectBivariateSpline(y2, x2, elev2)
        b_vec = b.dat.data
        assert meshCoords.shape[0] == b_vec.shape[0]

        # Interpolate data onto initial surface and bathymetry profiles
        for i, p in enumerate(meshCoords):
            eta0vec[i] = interpolatorSurf(p[1], p[0])
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
        q = solutionRW(op.mixedSpace(mesh))
        u0, eta0 = q.split()
        BCs = {1: {'uv': Constant(0.)}, 2: {'uv': Constant(0.)}, 3: {'uv': Constant(0.)}, 4: {'uv': Constant(0.)}}
        f = Function(P1).interpolate(SpatialCoordinate(mesh)[1])
    else:
        raise NotImplementedError

    return mesh, u0, eta0, b, BCs, f


class HermiteCoefficients:
    """
    Class containing Hermite expansion coefficients for the Rossby wave test case first order solution.
    """
    def __init__(self):

        u = np.zeros(28)
        v = np.zeros(28)
        h = np.zeros(28)

        #  Hermite series coefficients for U:

        u[1]=1.789276   # TODO: Should this start at zero?
        u[3]=0.1164146
        u[5]=-0.3266961e-3
        u[7]=-0.1274022e-2
        u[9]=0.4762876e-4
        u[11]=-0.1120652e-5
        u[13]=0.1996333e-7
        u[15]=-0.2891698e-9
        u[17]=0.3543594e-11
        u[19]=-0.3770130e-13
        u[21]=0.3547600e-15
        u[23]=-0.2994113e-17
        u[25]=0.2291658e-19
        u[27]=-0.1178252e-21

        #  Hermite series coefficients for V:

        v[4]=-0.6697824e-1
        v[6]=-0.2266569e-2
        v[8]=0.9228703e-4
        v[10]=-0.1954691e-5
        v[12]=0.2925271e-7
        v[14]=-0.3332983e-9
        v[16]=0.2916586e-11
        v[18]=-0.1824357e-13
        v[20]=0.4920951e-16
        v[22]=0.6302640e-18
        v[24]=-0.1289167e-19
        v[26]=0.1471189e-21

        #  Hermite series coefficients for H:

        h[1]=-3.071430
        h[3]=-0.3508384e-1
        h[5]=-0.1861060e-1
        h[7]=-0.2496364e-3
        h[9]=0.1639537e-4
        h[11]=-0.4410177e-6
        h[13]=0.8354759e-9
        h[15]=-0.1254222e-9
        h[17]=0.1573519e-11
        h[19]=-0.1702300e-13
        h[21]=0.1621976e-15
        h[23]=-0.1382304e-17
        h[25]=0.1066277e-19
        h[27]=-0.1178252e-21

        self.uCoeffs = u
        self.vCoeffs = v
        self.hCoeffs = h

    def __call__(self):
        return self.uCoeffs, self.vCoeffs, self.hCoeffs


def solutionRW(V, t=0., B=0.395):   # TODO: Consider 1st order soln
    """
    Analytic solution for equatorial Rossby wave test problem, as given by Huang.

    :arg V: Mixed function space upon which to define solutions.
    :arg t: current time.
    :param B: Parameter controlling amplitude of soliton.
    :return: Analytic solution for rossby-wave test problem of Huang.
    """
    x, y = SpatialCoordinate(V.mesh())
    q = Function(V)
    u, eta = q.split()
    u.rename("Depth averaged velocity")
    eta.rename("Elevation")

    A = 0.771 * B * B
    W = FunctionSpace(V.mesh(), V.sub(0).ufl_element().family(), V.sub(0).ufl_element().degree())
    u0 = Function(W).interpolate(
        A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
        * 0.25 * (-9 + 6 * y * y)
        * exp(-0.5 * y * y))
    u1 = Function(W).interpolate(
        -2 * B * tanh(B * (x + 0.4 * t)) *
        A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
        * 2 * y * exp(-0.5 * y * y))
    u.dat.data[:, 0] = u0.dat.data      # TODO: Shouldn't really do this in adjointland
    u.dat.data[:, 1] = u1.dat.data
    eta.interpolate(A * (1 / (cosh(B * (x + 0.4 * t)) ** 2))
                    * 0.25 * (3 + 6 * y * y)
                    * exp(-0.5 * y * y))

    return q


def integrateRW(V, op=Options()):   # TODO: Consider 1st order soln
    t = 0.
    vals = []
    ks = Function(V)
    k0, k1 = ks.split()
    k1.assign(indicator(V.sub(1), op=op))
    kt = Constant(0.)
    while t < op.Tend - 0.5 * op.dt:
        q = solutionRW(V, t=t)
        if t > op.Tstart - 0.5 * op.dt:  # Slightly smooth transition
            kt.assign(1. if t > op.Tstart + 0.5 * op.dt else 0.5)
        vals.append(assemble(kt * inner(ks, q) * dx))
        print("t = %.2fs" % t)
        t += op.dt
    return vals
