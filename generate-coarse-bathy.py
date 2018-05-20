from thetis import *

from utils.interpolation import interp
from utils.setup import problemDomain
from utils.options import Options


bathyfile = open("resources/bathymetry/array.txt", "w")
op = Options(mode='tohoku')
mesh, b, hierarchy = problemDomain(0, hierarchy=True, op=op)[0::3]
bathyfile.writelines(["%s," % val for val in b.dat.data])
bathyfile.write('\n')

for i in range(1, 5):
    mesh = hierarchy.__getitem__(i)
    b = interp(mesh, b)
    bathyfile.writelines(["%s," % val for val in b.dat.data])
    bathyfile.write('\n')
bathyfile.close()
