class MeshSetup:
    def __init__(self, res=3):

        # Get mesh descriptions
        self.res = res
        self.dirName = 'resources/meshes/'
        try:
            self.meshName = {1: 'TohokuXFine', 2: 'TohokuFine', 3: 'TohokuMedium', 4: 'TohokuCoarse',
                             5: 'TohokuXCoarse'}[res]
        except:
            raise ValueError('Resolution value not recognised. Choose an integer in the range 1-5.')

        # Define gradations (in metres)
        self.innerGradation1 = {1: 1000., 2: 1000., 3: 3000., 4: 5000., 5: 7500.}[res]
        self.outerGradation1 = {1: 2500., 2: 4000., 3: 10000., 4: 15000., 5: 25000.}[res]
        self.innerGradation2 = {1: 2500., 2: 3000., 3: 6000., 4: 10000., 5: 10000.}[res]
        self.outerGradation2 = {1: 5000., 2: 8000., 3: 10000., 4: 15000., 5: 25000.}[res]

        # Define gradation distances (in degrees)
        self.gradationDistance1 = 1.
        self.gradationDistance2 = {1: 1., 2: 1., 3: 1., 4: 0.5, 5: 0.5}[res]

    def generateMesh(self):
        """
        Generate mesh using QMESH. This script is based on work by Alexandros Advis et al, 2017.
        """

        # Reading in the shapefile describing the domain boundaries, and creating a GMSH file.
        bdyLoc = 'resources/boundaries/'
        boundaries = qmesh.vector.Shapes()
        boundaries.fromFile(bdyLoc + 'final_bdys.shp')
        loopShapes = qmesh.vector.identifyLoops(boundaries, isGlobal=False, defaultPhysID=1000, fixOpenLoops=True)
        polygonShapes = qmesh.vector.identifyPolygons(loopShapes, meshedAreaPhysID=1,
                                                      smallestNotMeshedArea=5e6, smallestMeshedArea=2e8)
        polygonShapes.writeFile(self.dirName + 'polygons.shp')

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
        gradationRaster_fukushimaCoast.writeNetCDF(self.dirName + 'gradationFukushima.nc')

        # Create raster for mesh gradation towards rest of coast
        gebcoCoastlines = qmesh.vector.Shapes()
        gebcoCoastlines.fromFile(bdyLoc + 'coastline.shp')
        gradationRaster_gebcoCoastlines = qmesh.raster.gradationToShapes()
        gradationRaster_gebcoCoastlines.setShapes(gebcoCoastlines)
        gradationRaster_gebcoCoastlines.setRasterBounds(135., 149., 30., 45.)
        gradationRaster_gebcoCoastlines.setRasterResolution(300, 300)
        gradationRaster_gebcoCoastlines.setGradationParameters(self.innerGradation2, self.outerGradation2,
                                                               self.gradationDistance2)
        gradationRaster_gebcoCoastlines.calculateLinearGradation()
        gradationRaster_gebcoCoastlines.writeNetCDF(self.dirName + 'gradationCoastlines.nc')

        # Create overall mesh metric
        meshMetricRaster = qmesh.raster.meshMetricTools.minimumRaster([gradationRaster_fukushimaCoast,
                                                                       gradationRaster_gebcoCoastlines])
        meshMetricRaster.writeNetCDF(self.dirName + 'meshMetricRaster.nc')

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


if __name__ == '__main__':
    import qmesh

    ms = MeshSetup(input('Specify mesh coarseness (1 = x-fine, 5 = x-coarse): ') or 3)
    qmesh.setLogOutputFile(ms.dirName + 'generateMesh.log')     # Store QMESH log for later reference
    qmesh.initialise()                                          # Initialise QGIS API
    ms.generateMesh()                                           # Generate the mesh
    ms.convertMesh()                                            # Convert to shapefile, for visualisation with QGIS
else:
    from firedrake import *
    from firedrake.petsc import PETSc

    import scipy.interpolate as si
    from scipy.io.netcdf import NetCDFFile
    from . import conversion


def TohokuDomain(res=3):
    """
    Load the mesh, initial condition and bathymetry profile for the 2D ocean domain of the Tohoku tsunami problem.
    
    :param res: mesh resolution value, ranging from 'extra coarse' (1) to extra fine (5).
    :return: associated mesh, initial condition and bathymetry field. 
    """

    # Define mesh and an associated elevation function space and establish initial condition and bathymetry functions
    ms = MeshSetup(res)
    mesh = Mesh(ms.dirName + ms.meshName + '.msh')
    meshCoords = mesh.coordinates.dat.data
    P1 = FunctionSpace(mesh, 'CG', 1)
    eta0 = Function(P1, name='Initial free surface displacement')
    b = Function(P1, name='Bathymetry profile')

    # Read and interpolate initial surface data (courtesy of Saito)
    nc1 = NetCDFFile('resources/initialisation/surf.nc', mmap=False)
    lon1 = nc1.variables['x'][:]
    lat1 = nc1.variables['y'][:]
    x1, y1 = conversion.vectorlonlat2utm(lat1, lon1, force_zone_number=54)      # Our mesh mainly resides in UTM zone 54
    elev1 = nc1.variables['z'][:, :]
    interpolatorSurf = si.RectBivariateSpline(y1, x1, elev1)
    eta0vec = eta0.dat.data
    assert meshCoords.shape[0] == eta0vec.shape[0]

    # Read and interpolate bathymetry data (courtesy of GEBCO)
    nc2 = NetCDFFile('resources/bathymetry/tohoku.nc', mmap=False)
    lon2 = nc2.variables['lon'][:]
    lat2 = nc2.variables['lat'][:-1]
    x2, y2 = conversion.vectorlonlat2utm(lat2, lon2, force_zone_number=54)
    elev2 = nc2.variables['elevation'][:-1, :]
    interpolatorBath = si.RectBivariateSpline(y2, x2, elev2)
    b_vec = b.dat.data
    assert meshCoords.shape[0] == b_vec.shape[0]

    # Interpolate data onto initial surface and bathymetry profiles
    for i, p in enumerate(meshCoords):
        eta0vec[i] = interpolatorSurf(p[1], p[0])
        b_vec[i] = - interpolatorSurf(p[1], p[0]) - interpolatorBath(p[1], p[0])

    # Post-process the bathymetry to have a minimum depth of 30m and plot
    b.assign(conditional(lt(30, b), b, 30))
    File('plots/initialisation/surf.pvd').write(eta0)
    File('plots/initialisation/bathymetry.pvd').write(b)

    return mesh, eta0, b


def meshStats(mesh):
    """
    :param mesh: current mesh.
    :return: number of cells and vertices on the mesh.
    """
    plex = mesh._plex
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    return cEnd - cStart, vEnd - vStart


def saveMesh(mesh, filename):
    """
    :param mesh: mesh to be saved.
    :param filename: name to be given, including directory location.
    """
    viewer = PETSc.Viewer().createHDF5(filename + '.h5', 'w')
    viewer(mesh._plex)


def loadMesh(filename):
    """
    :param filename: 
    :return: mesh loaded from .h5.
    """
    plex = PETSc.DMPlex().create()
    plex.createFromFile(filename + '.h5')
    return Mesh(plex)