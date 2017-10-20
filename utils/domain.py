from firedrake import *
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

from . import conversion


def TohokuDomain(res=3):
    """
    Set up a mesh of the 2D ocean domain as in the Tohoku tsunami problem (courtesy of QMESH), along with the associated 
    initial condition and bathymetry profile.
    
    :param res: mesh resolution value, ranging from 'extra coarse' (1) to extra fine (5).
    :return: associated mesh, initial condition and bathymetry field. 
    """

    # Define mesh and an associated elevation function space and establish initial condition and bathymetry functions:
    if res == 1:
        mesh = Mesh('resources/meshes/TohokuXFine.msh')     # 226,967 vertices, ~45 seconds per timestep
        print('WARNING: chosen mesh resolution can be extremely computationally intensive')
        if input('Are you happy to proceed? (y/n)') == 'n':
            exit(23)
    elif res == 2:
        mesh = Mesh('resources/meshes/TohokuFine.msh')      # 97,343 vertices, ~1 second per timestep
    elif res == 3:
        mesh = Mesh('resources/meshes/TohokuMedium.msh')    # 25,976 vertices, ~0.25 seconds per timestep
    elif res == 4:
        mesh = Mesh('resources/meshes/TohokuCoarse.msh')    # 7,194 vertices, ~0.07 seconds per timestep
    elif res == 5:
        mesh = Mesh('resources/meshes/TohokuXCoarse.msh')   # 3,126 vertices, ~0.03 seconds per timestep
    else:
        raise ValueError('Please try again, choosing an integer in the range 1-5.')
    meshCoords = mesh.coordinates.dat.data
    P1 = FunctionSpace(mesh, 'CG', 1)
    eta0 = Function(P1, name='Initial free surface displacement')
    b = Function(P1, name='Bathymetry profile')

    # Read and interpolate initial surface data (courtesy of Saito):
    nc1 = NetCDFFile('resources/initialisation/surf.nc', mmap=False)
    lon1 = nc1.variables['x'][:]
    lat1 = nc1.variables['y'][:]
    x1, y1 = conversion.vectorlonlat2utm(lat1, lon1, force_zone_number=54)      # Our mesh mainly resides in UTM zone 54
    elev1 = nc1.variables['z'][:, :]
    interpolatorSurf = si.RectBivariateSpline(y1, x1, elev1)
    eta0vec = eta0.dat.data
    assert meshCoords.shape[0] == eta0vec.shape[0]

    # Read and interpolate bathymetry data (courtesy of GEBCO):
    nc2 = NetCDFFile('resources/bathymetry/tohoku.nc', mmap=False)
    lon2 = nc2.variables['lon'][:]
    lat2 = nc2.variables['lat'][:-1]
    x2, y2 = conversion.vectorlonlat2utm(lat2, lon2, force_zone_number=54)
    elev2 = nc2.variables['elevation'][:-1, :]
    interpolatorBath = si.RectBivariateSpline(y2, x2, elev2)
    b_vec = b.dat.data
    assert meshCoords.shape[0] == b_vec.shape[0]

    # Interpolate data onto initial surface and bathymetry profiles:
    for i, p in enumerate(meshCoords):
        eta0vec[i] = interpolatorSurf(p[1], p[0])
        b_vec[i] = - interpolatorSurf(p[1], p[0]) - interpolatorBath(p[1], p[0])

    # Post-process the bathymetry to have a minimum depth of 30m:
    b.assign(conditional(lt(30, b), b, 30))

    # Plot initial surface and bathymetry profiles:
    File('plots/initialisation/surf.pvd').write(eta0)
    File('plots/initialisation/bathymetry.pvd').write(b)

    return mesh, eta0, b
