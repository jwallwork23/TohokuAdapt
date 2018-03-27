import math
import numpy as np


# Top matter courtesy of Tobias Bieniek, 2012.
class OutOfRangeError(ValueError):
    pass

__all__ = ["to_latlon", "from_latlon", "vectorlonlat2utm", "get_latitude", "latitude_to_zone_letter",
           "latlon_to_zone_number", "zone_number_to_central_longitude", "vectorlonlat2tangentxy", "earth_radius",
           "lonlat2tangentxy", "lonlat2tangent_pair", "vectorlonlat2utm", "mesh_converter", "xy2barycentric",
           "rescaleMesh"]

K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = math.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None, force_longitude=False):
    """
    Convert UTM coordinates to latitude-longitude, courtesy of Tobias Bieniek, 2012 (with some minor edits).
    
    :arg easting: eastward-measured Cartesian geographic distance.
    :arg northing: northward-measured Cartesian geographic distance.
    :arg zone_number: UTM zone number (increasing eastward).
    :param zone_letter: UTM zone letter (increasing alphabetically northward).
    :param northern: specify northern or southern hemisphere.
    :return: latitude-longitude coordinate pair.
    """
    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if not force_longitude:
        if not 100000 <= easting < 1000000:
            raise OutOfRangeError('easting out of range (must be between 100,000 m and 999,999 m)')
    if not 0 <= northing <= 10000000:
        raise OutOfRangeError('northing out of range (must be between 0 m and 10,000,000 m)')
    if not 1 <= zone_number <= 60:
        raise OutOfRangeError('zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise OutOfRangeError('zone letter out of range (must be between C and X)')

        northern = (zone_letter >= 'N')

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu + P2 * math.sin(2 * mu) + P3 * math.sin(4 * mu) + P4 * math.sin(6 * mu) + P5 * math.sin(8 * mu))

    p_sin = math.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = math.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = math.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = _E * p_cos**2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) * (d2 / 2 - d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d - d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    return math.degrees(latitude), math.degrees(longitude) + zone_number_to_central_longitude(zone_number)


def get_latitude(easting, northing, zone_number, zone_letter=None, northern=None):
    """
    Convert UTM coordinates to latitude alone.

    :arg easting: eastward-measured Cartesian geographic distance.
    :arg northing: northward-measured Cartesian geographic distance.
    :arg zone_number: UTM zone number (increasing eastward).
    :param zone_letter: UTM zone letter (increasing alphabetically northward).
    :param northern: specify northern or southern hemisphere.
    :return: latitude coordinate.
    """
    return to_latlon(easting, northing, zone_number, zone_letter, northern, force_longitude=True)[0]


def from_latlon(latitude, longitude, force_zone_number=None):
    """
    Convert latitude-longitude coordinates to UTM, courtesy of Tobias Bieniek, 2012.
    
    :arg latitude: northward anglular position, origin at the Equator.
    :arg longitude: eastward angular position, with origin at the Greenwich Meridian.
    :param force_zone_number: force coordinates to fall within a particular UTM zone.
    :return: UTM coordinate 4-tuple.
    """
    if not -80.0 <= latitude <= 84.0:
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not -180.0 <= longitude <= 180.0:
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')

    lat_rad = math.radians(latitude)
    lat_sin = math.sin(lat_rad)
    lat_cos = math.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    zone_letter = latitude_to_zone_letter(latitude)

    lon_rad = math.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = math.radians(central_lon)

    n = R / math.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad - M2 * math.sin(2 * lat_rad) + M3 * math.sin(4 * lat_rad) - M4 * math.sin(6 * lat_rad))

    easting = K0 * n * (a + a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 + a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if latitude < 0:
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def latitude_to_zone_letter(latitude):
    """
    Convert latitude UTM letter, courtesy of Tobias Bieniek, 2012.
    
    :arg latitude: northward anglular position, origin at the Equator.
    :return: UTM zone letter (increasing alphabetically northward).
    """
    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def latlon_to_zone_number(latitude, longitude):
    """
    Convert a latitude-longitude coordinate pair to UTM zone, courtesy of Tobias Bieniek, 2012.
    
    :arg latitude: northward anglular position, origin at the Equator.
    :arg longitude: eastward angular position, with origin at the Grenwich Meridian.
    :return: UTM zone number (increasing eastward).
    """
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180) / 6) + 1


def zone_number_to_central_longitude(zone_number):
    """
    Convert a UTM zone number to the corresponding central longitude, courtesy of Tobias Bieniek, 2012.
    
    :arg zone_number: UTM zone number (increasing eastward).
    :return: central eastward angular position of the UTM zone, with origin at the Grenwich Meridian.
    """
    return (zone_number - 1) * 6 - 180 + 3


def vectorlonlat2utm(latitude, longitude, force_zone_number):
    """
    Convert a vector of longitude-latitude coordinate pairs to UTM coordinates.
    
    :arg latitude: northward anglular position, origin at the Equator.
    :arg longitude: eastward angular position, with origin at the Grenwich Meridian.
    :param force_zone_number: 
    :return: force coordinates to fall within a particular UTM zone.
    """
    x = np.zeros((len(longitude), 1))
    y = np.zeros((len(latitude), 1))
    assert (len(x) == len(y))
    for i in range(len(x)):
        x[i], y[i], zn, zl = from_latlon(latitude[i], longitude[i], force_zone_number=force_zone_number)
        # print 'Coords ', x[i], y[i], 'Zone ', zn, zl      # For debugging purposes
    return x, y


def earth_radius(latitude):
    """
    :arg latitude: latitudinal coordinate.
    :return: radius of the Earth at this latitude.
    """
    k = 1. / 298.257  # Earth flatness constant
    a = 6378136.3  # Semi-major axis of the Earth (m)
    return (1 - k * (math.sin(math.radians(latitude)) ** 2)) * a


def lonlat2tangentxy(latitude, longitude, latitude0, longitude0):
    """
    Project latitude-longitude coordinates onto a tangent plane at (lon0, lat0), in metric Cartesian coordinates (x,y).

    :arg latitude: latitudinal coordinate for projection.
    :arg longitude: longitudinal coordinate for projection.
    :param latitude0: latitudinal tangent coordinate. 
    :param longitude0: longitudinal tangent coordinate.
    :return: Cartesian coordinates on tangent plane.
    """
    re = earth_radius(latitude)
    rphi = re * math.cos(math.radians(latitude))
    x = rphi * math.sin(math.radians(longitude - longitude0))
    y = rphi * (1 - math.cos(math.radians(longitude - longitude0))) * math.sin(math.radians(latitude0)) \
        + re * math.sin(math.radians(latitude - latitude0))
    return x, y


def lonlat2tangent_pair(latitude, longitude, latitude0, longitude0):
    """
    Project latitude-longitude coordinates onto a tangent plane at (lon0, lat0), in metric Cartesian coordinates (x,y),
    with output given as a pair.

    :arg latitude: latitudinal coordinate for projection.
    :arg longitude: longitudinal coordinate for projection.
    :param latitude0: latitudinal tangent coordinate. 
    :param longitude0: longitudinal tangent coordinate.
    :return: Cartesian coordinates on tangent plane.
    """
    x, y = lonlat2tangentxy(latitude, longitude, latitude0, longitude0)
    return [x, y]


def vectorlonlat2tangentxy(latitude, longitude, latitude0, longitude0):
    """
    Project a vector of latitude-longitude coordinates onto a tangent plane at (lon0, lat0), in metric Cartesian 
    coordinates (x,y).

    :arg latitude: latitudinal coordinate for projection.
    :arg longitude: longitudinal coordinate for projection.
    :param latitude0: latitudinal tangent coordinate. 
    :param longitude0: longitudinal tangent coordinate.
    :return: vector of Cartesian coordinates on tangent plane.
    """
    x = np.zeros((len(longitude), 1))
    y = np.zeros((len(latitude), 1))
    assert (len(x) == len(y))
    for i in range(len(x)):
        x[i], y[i] = lonlat2tangentxy(latitude[i], longitude[i], longitude0, latitude0)
    return x, y


def mesh_converter(meshfile, latitude0, longitude0):
    """
    Project a mesh file from latitude-longitude coordinates onto a tangent plane at (lon0, lat0), in metric Cartesian 
    coordinates (x,y).

    :arg meshfile: .msh file to be converted.
    :param latitude0: latitudinal tangent coordinate. 
    :param longitude0: longitudinal tangent coordinate.
    :return: corresponding mesh file in Cartesian coordinates on tangent plane.
    """
    mesh1 = open(meshfile, 'r')
    mesh2 = open('resources/meshes/CartesianTohoku.msh', 'w')
    i = 0
    mode = 0
    cnt = 0
    n = -1
    for line in mesh1:
        i += 1
        if i == 5:
            mode += 1
        if mode == 1:  # Read number
            n = int(line)  # Number of nodes
            mode += 1
        elif mode == 2:  # Edit nodes
            xy = line.split()
            xy[1], xy[2] = lonlat2tangentxy(float(xy[2]), float(xy[1]), latitude0, longitude0)
            xy[1] = str(xy[1])
            xy[2] = str(xy[2])
            line = ' '.join(xy)
            line += '\n'
            cnt += 1
            if cnt == n:
                assert int(xy[0]) == n  # Check all nodes have been covered
                mode += 1  # The end of the nodes has been reached
        mesh2.write(line)
    mesh1.close()
    mesh2.close()


def xy2barycentric(crdM, crdTri, i):
    """
    Compute the barycentric coordinate of M in triangle Tri = P0, P1, P2 with respect to the ith vertex 
    crd = det(MPj, MPk) / det(PiPj, PiPk). Courtesy of Nicolas Barral, 2016.
    
    :arg crdM: coordinate for conversion.
    :arg crdTri: vertices of the triangle.
    :param i: vertex index.
    :return: 
    """

    # Get other indices using a consistent numbering order:
    j = (i + 1) % 3
    k = (i + 2) % 3

    res1 = (crdTri[j][0] - crdM[0]) * (crdTri[k][1] - crdM[1]) - (crdTri[k][0] - crdM[0]) * (crdTri[j][1] - crdM[1])
    res2 = (crdTri[j][0] - crdTri[i][0]) * (crdTri[k][1] - crdTri[i][1]) - \
           (crdTri[k][0] - crdTri[i][0]) * (crdTri[j][1] - crdTri[i][1])
    res = res1 / res2

    return res


def rescaleMesh(mesh):
    """
    :arg mesh: mesh to be converted.
    :return: mesh rescaled to [-1, 1] x [-1, 1]. 
    """
    xy = Function(mesh.coordinates)
    xmin = min(xy.dat.data[:, 0])
    xmax = max(xy.dat.data[:, 0])
    ymin = min(xy.dat.data[:, 1])
    ymax = max(xy.dat.data[:, 1])
    xdiff = xmax - xmin
    ydiff = ymax - ymin
    cx = (xmax + xmin) / xdiff
    cy = (ymax + ymin) / ydiff

    for i in range(len(xy.dat.data)):
        xy.dat.data[i, 0] = 2 * xy.dat.data[i, 0] / xdiff - cx
        xy.dat.data[i, 1] = 2 * xy.dat.data[i, 1] / ydiff - cy
    mesh.coordinates.assign(xy)

    return mesh
