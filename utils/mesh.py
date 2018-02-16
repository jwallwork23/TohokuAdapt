class MeshSetup:
    def __init__(self, nEle=6176):

        # Get mesh descriptions
        self.nEle = nEle
        self.dirName = 'resources/meshes/'
        try:
            self.meshName = {691750: 'Tohoku691750', 450386: 'Tohoku450386', 196560: 'Tohoku196560',
                             129442: 'Tohoku129442',81902: 'Tohoku81902', 52998: 'Tohoku52998',
                             33784: 'Tohoku33784', 20724: 'Tohoku20724', 16656: 'Tohoku16656',
                             11020: 'Tohoku11020', 8782: 'Tohoku8782', 6176: 'Tohoku6176'}[nEle]
        except:
            raise ValueError('Number of elements not recognised.')

        # Define gradations (in metres)
        self.innerGradation1 = {691750: 900., 450386: 1000., 196560: 1000., 129442: 1200.,
                                81902: 1500., 52998: 2000., 33784: 3000., 20724: 4000.,
                                16656: 4500., 11020: 5500., 8782: 6000., 6176: 7500., }[nEle]
        self.outerGradation1 = {691750: 2000., 450386: 2500., 196560: 4000., 129442: 5000.,
                                81902: 6500., 52998: 8000., 33784: 10000., 20724: 12500.,
                                16656: 14000., 11020: 17500., 8782: 20000., 6176: 25000.,}[nEle]
        self.innerGradation2 = {691750: 2000., 450386: 2500., 196560: 3000., 129442: 3500.,
                                81902: 4000., 52998: 5000., 33784: 6000., 20724: 8000.,
                                16656: 9000., 11020: 10000., 8782: 10000., 6176: 10000.,}[nEle]
        self.outerGradation2 = {691750: 4000., 450386: 5000., 196560: 8000., 129442: 7500.,
                                81902: 7000., 52998: 9000., 33784: 10000., 20724: 12500.,
                                16656: 14000., 11020: 17500., 8782: 20000., 6176: 25000.,}[nEle]

        # Define gradation distances (in degrees)
        self.gradationDistance1 = 1.
        self.gradationDistance2 = {691750: 1., 450386: 1., 196560: 1., 129442: 1.,
                                   81902: 1., 52998: 1., 33784: 1., 20724: 0.75,
                                   16656: 0.65, 11020: 0.5, 8782: 0.5, 6176: 0.5}[nEle]

    def generateMesh(self, wd=False):
        """
        Generate mesh using QMESH. This script is based on work by Alexandros Advis et al, 2017.
        """

        # Reading in the shapefile describing the domain boundaries, and creating a GMSH file.
        bdyLoc = 'resources/boundaries/'
        boundaries = qmesh.vector.Shapes()
        if wd:
            boundaries.fromFile(bdyLoc+'wd_final_bdys.shp')
            self.meshName = 'wd_'+self.meshName
        else:
            boundaries.fromFile(bdyLoc+'final_bdys.shp')
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
        gradationRaster_fukushimaCoast.writeNetCDF(self.dirName + 'gradationFukushima.nc')

        # Create raster for mesh gradation towards rest of coast
        gebcoCoastlines = qmesh.vector.Shapes()
        if wd:
            gebcoCoastlines.fromFile(bdyLoc+'coast_poly.shp')
        else:
            gebcoCoastlines.fromFile(bdyLoc+'coastline.shp')
        gradationRaster_gebcoCoastlines = qmesh.raster.gradationToShapes()
        gradationRaster_gebcoCoastlines.setShapes(gebcoCoastlines)      # Could be a polygon, line or point
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

    ms = MeshSetup(input('Choose #Elements from:\n '
                         '{6176, 8782, 11020, 16656, 20724, 33784, 52998, 81902, 129442, 196560, 450386, 691750}:\n')
                   or 6176)
    qmesh.setLogOutputFile(ms.dirName + 'generateMesh.log')     # Store QMESH log for later reference
    qmesh.initialise()                                          # Initialise QGIS API
    wd = bool(input("Press 0 for a standard mesh or 1 to generate a mesh for wetting and drying. "))
    ms.generateMesh(wd=wd)                                      # Generate the mesh
    ms.convertMesh()                                            # Convert to shapefile, for visualisation with QGIS
else:
    from firedrake import *
    from firedrake.petsc import PETSc

    import scipy.interpolate as si
    from scipy.io.netcdf import NetCDFFile
    from . import conversion


def TohokuDomain(nEle=6176, mesh=None, output=False, capBathymetry=True):
    """
    Load the mesh, initial condition and bathymetry profile for the 2D ocean domain of the Tohoku tsunami problem.
    
    :arg nEle: number of elements considered.
    :param mesh: user specified mesh, if already generated.
    :param output: toggle plotting of bathymetry and initial surface.
    :param capBathymetry: in the case of no wetting-and-drying.
    :return: associated mesh, initial condition and bathymetry field. 
    """

    # Define mesh and an associated elevation function space and establish initial condition and bathymetry functions
    if mesh == None:
        ms = MeshSetup(nEle)
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
    if capBathymetry:
        b.assign(conditional(lt(30, b), b, 30))
    if output:
        File('plots/initialisation/surf.pvd').write(eta0)
        File('plots/initialisation/bathymetry.pvd').write(b)

    return mesh, eta0, b


def meshStats(mesh):
    """
    :arg mesh: current mesh.
    :return: number of cells and vertices on the mesh.
    """
    plex = mesh._plex
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    return cEnd - cStart, vEnd - vStart


def saveMesh(mesh, filename):
    """
    :arg mesh: Mesh to be saved to HDF5.
    :arg filename: filename to be given, including directory location.
    """
    viewer = PETSc.Viewer().createHDF5(filename + '.h5', 'w')
    viewer(mesh._plex)


# def loadMesh(filename):
#     """
#     :arg filename: mesh filename to load from, including directory location.
#     :return: Mesh, as loaded from HDF5.
#     """
#     plex = PETSc.DMPlex().create()
#     plex.createFromFile(filename + '.h5')
#     return Mesh(plex)
