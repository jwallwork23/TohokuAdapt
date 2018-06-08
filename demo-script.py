from thetis_adjoint import *

from utils.conversion import from_latlon
from utils.options import TohokuOptions
from utils.setup import problemDomain
from utils.solvers import tsunami

# Set up problem
op = TohokuOptions(approach='DWR')
mesh, u0, eta0, b, BCs, f, nu = problemDomain(8, op=op)
op.loc = from_latlon(op.lat("Tokai"), op.lon("Tokai")) \
         + from_latlon(op.lat("Fukushima Daiichi"), op.lon("Fukushima Daiichi")) \
         + from_latlon(op.lat("Fukushima Daini"), op.lon("Fukushima Daini")) \
         + from_latlon(op.lat("Onagawa"), op.lon("Onagawa")) \
         + from_latlon(op.lat("Hamaoka"), op.lon("Hamaoka")) \
         + from_latlon(op.lat("Tohoku"), op.lon("Tohoku"))
op.radii = [50e3, 50e3, 50e3, 50e3, 50e3, 50e3]

# Solve
q = tsunami(mesh, u0, eta0, b, BCs=BCs, f=f, nu=nu, op=op)
print("Objective functional value = %.4e" % q['J_h'])
print("CPU time = %.2f" % q['solverTime'])
