import numpy as np


__all__ = ["MeshSetup", "problemDomain", "meshStats", "__main__"]


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

    from .conversion import vectorlonlat_to_utm, get_latitude
    from .forms import solutionRW
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
            b_vec[i] = max(depth, 30)

        # Post-process the bathymetry to have a minimum depth of 30m and if no wetting-and-drying
        BCs = {}
        if op.rotational:           # TODO: Try f- or beta-plane approximation
            f = Function(FunctionSpace(mesh, 'CG', 1))
            for i, v in zip(range(len(mesh.coordinates.dat.data)), mesh.coordinates.dat.data):
                f.dat.data[i] = 2 * op.Omega * np.sin(np.radians(get_latitude(v[0], v[1], 54, northern=True)))
        else:
            f = None
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
        f = None
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


def meshStats(mesh):
    """
    :arg mesh: current mesh.
    :return: number of cells and vertices on the mesh.
    """
    plex = mesh._plex
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    return cEnd - cStart, vEnd - vStart
